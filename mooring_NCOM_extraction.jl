# mooring_NCOM_extraction.jl
#
# Extract (interpolate) NCOM output at a list of mooring lon/lat locations.
#
# For each mooring (one Distributed worker per mooring, via pmap):
# Step 1: read the parent NCOM grid and find the smallest (i,j) index box
#         around the mooring (cf. h2r_bry_subgrid.m), so only that
#         sub-region has to be read from the model output.
# Step 2: build the interpolation operator once with interp_bry.jl
#         (horizontal triangle interp + vertical linear interp onto zout).
# Step 3: loop over the daily NCOM NetCDF files, interpolate temp, salt,
#         u, v (3D) and ssh (2D) at every hour, and save NCOM_<name>.nc.
#
# The mooring line transect (straight line M1 -> CPIES7, every dx_km, same
# points as in the HYCOM / GLOBAL scripts) is one more job, done the same way:
# it is split into pieces of ~piece_km, each with its own small sub-grid, so
# only a narrow band along the line is read (not the whole bounding box).
# Saved as NCOM_M1_CPIES7.nc: variables (z, x, time) + u_along/u_cross, ssh (x, time).
#
# Usage:
#   julia mooring_NCOM_extraction.jl              -> all moorings + transect, n_workers in parallel
#   julia mooring_NCOM_extraction.jl M2           -> only M2   (same as: ... 3)
#   julia mooring_NCOM_extraction.jl M1 CPIES4    -> M1 and CPIES4
#   julia mooring_NCOM_extraction.jl transect     -> only the transect (same as: ... 15)
# or under a SLURM array job (#SBATCH --array=1-15) the task id picks the mooring
# (15 = the transect).

using Distributed

n_workers = 4   # moorings run at the same time; mostly limited by disk reads, not CPU

# number of moorings in this run is known from the command line already:
# one mooring (one argument or a SLURM array task) runs on the main process only
n_moor = !isempty(ARGS) ? length(ARGS) : haskey(ENV, "SLURM_ARRAY_TASK_ID") ? 1 : typemax(Int)
if n_moor > 1 && nprocs() == 1
    addprocs(min(n_workers, n_moor))
end
println("running with $(nworkers()) worker process(es)")

# packages + interp_bry.jl first, in their own @everywhere: the module has to
# exist before the block below (which uses it) is evaluated.
@everywhere using NCDatasets, Dates, NearestNeighbors
# interp_bry.jl functions are always called with the module name (InterpBry.get_bry_coef, ...),
# never via `using`, so re-running the script in the same session / on the same
# workers always uses the freshly included module (and picks up edits to interp_bry.jl).
@everywhere include("/home/hsinyi/Documents/Julia/function/interp_bry.jl")
@everywhere include("/home/hsinyi/Documents/Julia/function/mooring_transect.jl")

@everywhere begin

    # -----------------------------------------------------------------------
    # User input (needed on every worker)
    # -----------------------------------------------------------------------
    parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"
    datadir     = "/home/hsinyi/roms_data/NCOM_DATA_NC/"     # yyyymmddHH_{ts,uv,ssh}.nc
    outdir      = "/home/hsinyi/roms_data/NCOM_mooring/"     # one file per mooring: NCOM_<name>.nc

    # output depths (m, negative down, must be ASCENDING for get_1d_coef);
    # depths below the local seafloor are set to NaN
    zout = collect(-1000.0:5.0:0.0)

    dates = Date(2022, 8, 22):Day(1):Date(2022, 11, 30)    # one file set per day

    pad = 3   # extra grid cells around the mooring: 1 is enough for the triangle stencil in interp_bry.jl,
              # the extra cells give coastal moorings more wet neighbours for the land-fill step
    piece_km = 20.0   # transect: length of each piece with its own sub-grid

    # -----------------------------------------------------------------------
    # Sub-grid search
    # -----------------------------------------------------------------------
    """
        nearest_ij(lonp, latp, lon0, lat0) -> (i, j, dist_km)

    Index of the parent grid point closest to (lon0, lat0) (local flat-earth distance).
    """
    function nearest_ij(lonp, latp, lon0, lat0)
        coslat = cosd(lat0)
        dmin, imin = Inf, CartesianIndex(1, 1)
        @inbounds for I in CartesianIndices(lonp)
            d = ((lonp[I] - lon0) * coslat)^2 + (latp[I] - lat0)^2
            if d < dmin
                dmin, imin = d, I
            end
        end
        return imin[1], imin[2], sqrt(dmin) * 111.195
    end

    """
        mooring_subgrid(lonp, latp, lons, lats; pad=2)

    Return (i0, i1, j0, j1, inear, jnear): the index limits of the minimal parent
    sub-grid containing all mooring points (plus `pad` cells), and the nearest
    (i, j) of each mooring in the full grid.
    """
    function mooring_subgrid(lonp, latp, lons, lats; pad=2)
        Li, Lj = size(lonp)
        n = length(lons)
        inear, jnear = zeros(Int, n), zeros(Int, n)
        for k in 1:n
            i, j, dkm = nearest_ij(lonp, latp, lons[k], lats[k])
            # point outside the parent grid -> nearest point lies on the boundary
            if i in (1, Li) || j in (1, Lj)
                @warn "Mooring $k ($(lons[k]), $(lats[k])) is on/outside the parent grid edge (nearest point $(round(dkm, digits=2)) km away)"
            end
            inear[k], jnear[k] = i, j
        end
        i0 = max(minimum(inear) - pad, 1);  i1 = min(maximum(inear) + pad, Li)
        j0 = max(minimum(jnear) - pad, 1);  j1 = min(maximum(jnear) + pad, Lj)
        return i0, i1, j0, j1, inear, jnear
    end

    # -----------------------------------------------------------------------
    # Land / below-seafloor filling
    # interp_bry.jl ignores maskp in the 3D path, so every NaN has to be removed
    # from the parent fields before interpolating.
    # -----------------------------------------------------------------------
    """
        fill_index(masks, lons, lats) -> src

    For every sub-grid column, the linear index of the column to take data from:
    itself if wet, otherwise the nearest wet column (fillmask.m equivalent).
    """
    function fill_index(masks, lons, lats)
        src = collect(1:length(masks))
        wet = findall(vec(masks) .== 1)
        dry = findall(vec(masks) .!= 1)
        isempty(dry) && return src
        isempty(wet) && error("No wet points in the sub-grid; increase pad")
        coslat = cosd(sum(lats) / length(lats))
        tree = KDTree(permutedims(hcat(vec(lons)[wet] .* coslat, vec(lats)[wet])))
        for c in dry
            k, _ = nn(tree, [lons[c] * coslat, lats[c]])
            src[c] = wet[k]
        end
        return src
    end

    """
        prep_field(F, src; dz_below=0.0) -> Fp

    NCOM 3D field from NCDatasets order (i, j, k), k=1 at the surface, to the
    (k, i, j) order with k ASCENDING (bottom -> surface) that get_bry_coef wants.
    Land columns are copied from their nearest wet column (`src`), and NaN/missing
    values below the seafloor are filled downward from the deepest valid value.
    For the depth array use `dz_below > 0` so the filled depths keep decreasing
    (get_1d_coef needs strictly monotonic levels).
    """
    function prep_field(F, src; dz_below=0.0)
        Mi, Mj, K = size(F)
        Fc = reshape(F, Mi * Mj, K)
        Fp = Array{Float64}(undef, K, Mi, Mj)
        Fv = reshape(Fp, K, Mi * Mj)
        for c in 1:Mi*Mj
            s = src[c]
            last = NaN
            for k in 1:K
                v = coalesce(Fc[s, k], NaN)
                v = isfinite(v) ? Float64(v) : last - dz_below
                Fv[K-k+1, c] = v
                last = v
            end
        end
        return Fp
    end

    prep_field2d(F, src) = Float64.(coalesce.(vec(F), NaN))[src]

    # MT in the NCOM files is days since 1900-12-31 00:00
    mt2datetime(mt) = DateTime(1900, 12, 31) + Millisecond(round(Int, mt * 86_400_000))

    # -----------------------------------------------------------------------
    # One piece = a set of points with its own sub-grid and interpolation operators
    # -----------------------------------------------------------------------
    """
        build_piece(lonp, latp, maskp, plon, plat) -> piece

    Sub-grid around the points (plon, plat) and the interpolation operators
    (built once, reused for every variable and time).
    """
    function build_piece(lonp, latp, maskp, plon, plat)
        i0, i1, j0, j1, _, _ = mooring_subgrid(lonp, latp, plon, plat; pad=pad)
        lons  = lonp[i0:i1, j0:j1]
        lats  = latp[i0:i1, j0:j1]
        masks = maskp[i0:i1, j0:j1]

        # sub-grid depths: zm3 (i,j,k) at layer centers, k=1 at the surface, missing below kb / on land
        zm3s, hs = NCDataset(parent_grid) do ds
            ds["zm3"][i0:i1, j0:j1, :], ds["h"][i0:i1, j0:j1]
        end

        src = fill_index(masks, lons, lats)
        zp  = prep_field(zm3s, src; dz_below=1.0)

        A = InterpBry.get_bry_coef(lons, lats, zp, plon, plat, zout)          # 3D: (Nz*Np) x (K*Mi*Mj)
        elem2, coef2, _ = InterpBry.Interp3D.Interp2D.get_2d_coef(lons, lats, reshape(plon, :, 1), reshape(plat, :, 1))
        h = InterpBry.Interp3D.Interp2D.apply_2d_coef(coef2, elem2, prep_field2d(hs, src))   # seafloor depth (negative)
        return (i = i0:i1, j = j0:j1, src = src, A = A, elem2 = elem2, coef2 = coef2,
                np = length(plon), h = h, below = zout .< permutedims(h))   # below: (Nz, Np) under the seafloor
    end

    interp3d(p, F) = (Fc = InterpBry.apply_bry_coef(p.A, prep_field(F, p.src), length(zout), p.np); Fc[p.below] .= NaN; Fc)
    interp2d(p, F) = InterpBry.Interp3D.Interp2D.apply_2d_coef(p.coef2, p.elem2, prep_field2d(F, p.src))

    # -----------------------------------------------------------------------
    # One job (a mooring, or the transect `tr`): pieces -> time loop -> save
    # -----------------------------------------------------------------------
    """
        extract_mooring(name, lon, lat; tr=nothing) -> fname

    Interpolate NCOM temp, salt, u, v (at `zout`) and ssh to one mooring
    (lon, lat), or to every point of the transect `tr` (see transect_points),
    for every hour in `dates`, and save them to `outdir/NCOM_<name>.nc`.
    """
    function extract_mooring(name, lon, lat; tr=nothing)
        t_start = time()

        # parent grid; NCDatasets gives lon(dim_j, dim_i) as lon[i, j], size (Li, Lj)
        lonp, latp, maskp = NCDataset(parent_grid) do ds
            Float64.(nomissing(ds["lon"][:, :], NaN)),
            Float64.(nomissing(ds["lat"][:, :], NaN)),
            Float64.(nomissing(ds["mask"][:, :], 0.0))
        end

        plon, plat = tr === nothing ? ([lon], [lat]) : (tr.lon, tr.lat)
        np_piece = tr === nothing ? 1 : max(1, round(Int, piece_km / (tr.dist[2] - tr.dist[1])))
        pieces = [build_piece(lonp, latp, maskp, plon[r], plat[r])
                  for r in Iterators.partition(eachindex(plon), np_piece)]
        for p in pieces
            println("$name: sub-grid i=$(p.i), j=$(p.j), h = $(round.(extrema(p.h), digits=1)) m")
        end
        h_out = reduce(vcat, [p.h for p in pieces])

        # Loop over the daily files. Each file holds 25 hourly records (00..24 h);
        # record 25 is the same time as record 1 of the next day, so it is
        # skipped except for the last day.
        # Each record: (Nz, Np) for the 3D variables, (Np,) for ssh.
        # The transect is kept (and saved) as Float32: ~0.6 GB per variable for 100 days.
        T = tr === nothing ? Float64 : Float32
        time_out = DateTime[]
        temp_out, salt_out, u_out, v_out = Matrix{T}[], Matrix{T}[], Matrix{T}[], Matrix{T}[]
        ssh_out = Vector{T}[]
        all3d(var, t) = reduce(hcat, [interp3d(p, var[p.i, p.j, :, t]) for p in pieces])

        for d in dates
            tag = Dates.format(d, "yyyymmdd") * "00"
            ds_ts  = NCDataset(joinpath(datadir, tag * "_ts.nc"))
            ds_uv  = NCDataset(joinpath(datadir, tag * "_uv.nc"))
            ds_ssh = NCDataset(joinpath(datadir, tag * "_ssh.nc"))
            try
                MT = ds_ts["MT"][:]
                recs = d == last(dates) ? eachindex(MT) : eachindex(MT)[1:end-1]
                for t in recs
                    push!(time_out, mt2datetime(MT[t]))
                    push!(temp_out, all3d(ds_ts["layer_temperature"], t))
                    push!(salt_out, all3d(ds_ts["layer_salinity"], t))
                    push!(u_out,    all3d(ds_uv["u_velocity"], t))
                    push!(v_out,    all3d(ds_uv["v_velocity"], t))
                    push!(ssh_out,  reduce(vcat, [interp2d(p, ds_ssh["ssh"][p.i, p.j, t]) for p in pieces]))
                end
            finally
                close(ds_ts); close(ds_uv); close(ds_ssh)
            end
            println("$name: done $tag  ($(length(time_out)) records)")
        end

        fname = joinpath(outdir, "NCOM_$(name).nc")
        if tr !== nothing
            # transect: (z, x, time)
            save_transect(fname, tr, zout, time_out,
                          ["temp" => (stack(temp_out), "temperature", "degC"),
                           "salt" => (stack(salt_out), "salinity", "psu"),
                           "u"    => (stack(u_out), "eastward velocity", "m/s"),
                           "v"    => (stack(v_out), "northward velocity", "m/s")];
                          ssh = stack(ssh_out), h = h_out,
                          title  = "NCOM output interpolated to the $(tr.from) - $(tr.to) transect",
                          source = "mooring_NCOM_extraction.jl, grid $parent_grid, data $datadir")
            println("$name: $(round((time() - t_start) / 60, digits=1)) min")
            return fname
        end

        # mooring: 3D variables (z, time), ssh (time)
        Nz = length(zout)
        zt(x) = stack(x)[:, 1, :]
        mkpath(outdir)
        NCDataset(fname, "c") do ds
            ds.attrib["title"]   = "NCOM output interpolated to mooring $name"
            ds.attrib["station"] = name
            ds.attrib["source"]  = "mooring_NCOM_extraction.jl, grid $parent_grid, data $datadir"
            defDim(ds, "z", Nz); defDim(ds, "time", length(time_out))

            defVar(ds, "lon", lon, (), attrib=["units" => "degrees_east"])
            defVar(ds, "lat", lat, (), attrib=["units" => "degrees_north"])
            defVar(ds, "h", h_out[1], (),
                   attrib=["long_name" => "interpolated NCOM water depth", "units" => "m", "positive" => "up"])
            defVar(ds, "z", zout, ("z",), attrib=["units" => "m", "positive" => "up"])
            defVar(ds, "time", time_out, ("time",), attrib=["units" => "hours since 2022-01-01 00:00:00"])

            dims3 = ("z", "time")
            defVar(ds, "temp", zt(temp_out), dims3, attrib=["long_name" => "temperature", "units" => "degC"])
            defVar(ds, "salt", zt(salt_out), dims3, attrib=["long_name" => "salinity", "units" => "psu"])
            defVar(ds, "u", zt(u_out), dims3, attrib=["long_name" => "eastward velocity", "units" => "m/s"])
            defVar(ds, "v", zt(v_out), dims3, attrib=["long_name" => "northward velocity", "units" => "m/s"])
            defVar(ds, "ssh", first.(ssh_out), ("time",), attrib=["long_name" => "sea surface height", "units" => "m"])
        end
        println("$name: saved $fname  ($(round((time() - t_start) / 60, digits=1)) min)")
        return fname
    end
end

# ---------------------------------------------------------------------------
# Main process only: mooring list, selection, and hand-out to the workers
# ---------------------------------------------------------------------------
mooring_file = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"

transect_ends = ("M1", "CPIES7")   # mooring line transect, from -> to
dx_km         = 1.0                # transect point spacing (same in all the model scripts)

# mooring locations (lon in degrees_east, -180..180 like the grid; lat in degrees_north),
# taken from the global attributes of mooring_file, e.g.
#   :M1_mor_info = "indices for M1_mor in roms_grd_900m.nc , location:(-45.4802;2.5415)"
# (order = variable order in the file: french, M1..M4, CPIES1..9).
# The transect is appended as one more job, after the moorings.
moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
tr = transect_points(moor_name, moor_lon, moor_lat, transect_ends...; dx_km = dx_km)
println("transect $(tr.name): $(length(tr.lon)) points, $(round(tr.dist[end], digits=1)) km, moorings on it: ",
        join(tr.moor, ", "))
push!(moor_name, "transect"); push!(moor_lon, NaN); push!(moor_lat, NaN)

# which moorings to run (index into the list above, or the name), see Usage at the top
function find_mooring(a, names)
    k = something(tryparse(Int, a), findfirst(==(a), names), 0)
    1 <= k <= length(names) || error("Unknown mooring '$a'; choose 1-$(length(names)) or one of: $(join(names, ", "))")
    return k
end
moor_sel = !isempty(ARGS)                        ? [find_mooring(a, moor_name) for a in ARGS] :
           haskey(ENV, "SLURM_ARRAY_TASK_ID")    ? [find_mooring(ENV["SLURM_ARRAY_TASK_ID"], moor_name)] :
                                                   collect(eachindex(moor_name))
println("Moorings in this run : ", join(moor_name[moor_sel], ", "))

# hand the moorings out to the workers, one mooring per worker at a time;
# an error in one mooring is reported but does not stop the others
# the transect is the longest job: start it first
sort!(moor_sel, by = k -> moor_name[k] != "transect")
results = pmap(k -> moor_name[k] == "transect" ? extract_mooring(tr.name, NaN, NaN; tr = tr) :
                                                extract_mooring(moor_name[k], moor_lon[k], moor_lat[k]),
               moor_sel;
               on_error = e -> (@warn "mooring failed" exception = e; e))
failed = [moor_name[moor_sel[n]] for n in eachindex(results) if results[n] isa Exception]
println(isempty(failed) ? "all moorings done" : "FAILED: " * join(failed, ", "))

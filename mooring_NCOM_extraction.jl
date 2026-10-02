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
# Usage:
#   julia mooring_NCOM_extraction.jl              -> all moorings, n_workers in parallel
#   julia mooring_NCOM_extraction.jl M2           -> only M2   (same as: ... 3)
#   julia mooring_NCOM_extraction.jl M1 CPIES4    -> M1 and CPIES4
# or under a SLURM array job (#SBATCH --array=1-14) the task id picks the mooring.

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
# exist before the block below (which uses it) is evaluated
@everywhere using NCDatasets, Dates, NearestNeighbors
@everywhere include("/home/hsinyi/Documents/Julia/function/interp_bry.jl")

@everywhere begin
    using .InterpBry
    const Interp2D = InterpBry.Interp3D.Interp2D

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
    # One mooring: sub-grid -> interpolation operator -> time loop -> save
    # -----------------------------------------------------------------------
    """
        extract_mooring(name, lon, lat) -> fname

    Interpolate NCOM temp, salt, u, v (at `zout`) and ssh to one mooring for
    every hour in `dates`, and save them to `outdir/NCOM_<name>.nc`.
    """
    function extract_mooring(name, lon, lat)
        t_start = time()

        # parent grid; NCDatasets gives lon(dim_j, dim_i) as lon[i, j], size (Li, Lj)
        lonp, latp, maskp = NCDataset(parent_grid) do ds
            Float64.(nomissing(ds["lon"][:, :], NaN)),
            Float64.(nomissing(ds["lat"][:, :], NaN)),
            Float64.(nomissing(ds["mask"][:, :], 0.0))
        end

        i0, i1, j0, j1, inear, jnear = mooring_subgrid(lonp, latp, [lon], [lat]; pad=pad)
        println("$name: sub-grid i=$i0:$i1, j=$j0:$j1, nearest (i,j)=($(inear[1]),$(jnear[1])), ",
                "mask=$(maskp[inear[1], jnear[1]])")

        lons  = lonp[i0:i1, j0:j1]
        lats  = latp[i0:i1, j0:j1]
        masks = maskp[i0:i1, j0:j1]

        # sub-grid depths: zm3 (i,j,k) at layer centers, k=1 at the surface, missing below kb / on land
        zm3s, hs = NCDataset(parent_grid) do ds
            ds["zm3"][i0:i1, j0:j1, :], ds["h"][i0:i1, j0:j1]
        end

        src = fill_index(masks, lons, lats)
        zp  = prep_field(zm3s, src; dz_below=1.0)

        # interpolation operators (built once, reused for every variable and time)
        Nz = length(zout)
        A = get_bry_coef(lons, lats, zp, [lon], [lat], zout)        # 3D: Nz x (K*Mi*Mj)
        elem2, coef2, _ = Interp2D.get_2d_coef(lons, lats, fill(lon, 1, 1), fill(lat, 1, 1))
        interp2d(F) = Interp2D.apply_2d_coef(coef2, elem2, prep_field2d(F, src))[1]

        h_moor = interp2d(hs)                 # seafloor depth at the mooring (negative)
        below  = zout .< h_moor               # output depths under the seafloor
        println("$name: h = $(round(h_moor, digits=1)) m")

        interp3d(F) = (Fc = vec(apply_bry_coef(A, prep_field(F, src), Nz, 1)); Fc[below] .= NaN; Fc)

        # Loop over the daily files. Each file holds 25 hourly records (00..24 h);
        # record 25 is the same time as record 1 of the next day, so it is
        # skipped except for the last day.
        time_out = DateTime[]
        temp_out, salt_out, u_out, v_out = Vector{Float64}[], Vector{Float64}[],
                                           Vector{Float64}[], Vector{Float64}[]
        ssh_out = Float64[]

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
                    push!(temp_out, interp3d(ds_ts["layer_temperature"][i0:i1, j0:j1, :, t]))
                    push!(salt_out, interp3d(ds_ts["layer_salinity"][i0:i1, j0:j1, :, t]))
                    push!(u_out,    interp3d(ds_uv["u_velocity"][i0:i1, j0:j1, :, t]))
                    push!(v_out,    interp3d(ds_uv["v_velocity"][i0:i1, j0:j1, :, t]))
                    push!(ssh_out,  interp2d(ds_ssh["ssh"][i0:i1, j0:j1, t]))
                end
            finally
                close(ds_ts); close(ds_uv); close(ds_ssh)
            end
            println("$name: done $tag  ($(length(time_out)) records)")
        end

        # save: 3D variables (z, time), ssh (time)
        mkpath(outdir)
        fname = joinpath(outdir, "NCOM_$(name).nc")
        NCDataset(fname, "c") do ds
            ds.attrib["title"]   = "NCOM output interpolated to mooring $name"
            ds.attrib["station"] = name
            ds.attrib["source"]  = "mooring_NCOM_extraction.jl, grid $parent_grid, data $datadir"
            defDim(ds, "z", Nz); defDim(ds, "time", length(time_out))

            defVar(ds, "lon", lon, (), attrib=["units" => "degrees_east"])
            defVar(ds, "lat", lat, (), attrib=["units" => "degrees_north"])
            defVar(ds, "h", h_moor, (),
                   attrib=["long_name" => "interpolated NCOM water depth", "units" => "m", "positive" => "up"])
            defVar(ds, "z", zout, ("z",), attrib=["units" => "m", "positive" => "up"])
            defVar(ds, "time", time_out, ("time",), attrib=["units" => "hours since 2022-01-01 00:00:00"])

            dims3 = ("z", "time")
            defVar(ds, "temp", stack(temp_out), dims3, attrib=["long_name" => "temperature", "units" => "degC"])
            defVar(ds, "salt", stack(salt_out), dims3, attrib=["long_name" => "salinity", "units" => "psu"])
            defVar(ds, "u", stack(u_out), dims3, attrib=["long_name" => "eastward velocity", "units" => "m/s"])
            defVar(ds, "v", stack(v_out), dims3, attrib=["long_name" => "northward velocity", "units" => "m/s"])
            defVar(ds, "ssh", ssh_out, ("time",), attrib=["long_name" => "sea surface height", "units" => "m"])
        end
        println("$name: saved $fname  ($(round((time() - t_start) / 60, digits=1)) min)")
        return fname
    end
end

# ---------------------------------------------------------------------------
# Main process only: mooring list, selection, and hand-out to the workers
# ---------------------------------------------------------------------------
mooring_file = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"

# mooring locations (lon in degrees_east, -180..180 like the grid; lat in degrees_north),
# taken from the global attributes of mooring_file, e.g.
#   :M1_mor_info = "indices for M1_mor in roms_grd_900m.nc , location:(-45.4802;2.5415)"
# (order = variable order in the file: french, M1..M4, CPIES1..9)
moor_name, moor_lon, moor_lat = NCDataset(mooring_file) do ds
    names = [replace(v, "_mor" => "") for v in keys(ds) if endswith(v, "_mor")]
    locs  = [match(r"location:\(([-\d.]+);([-\d.]+)\)", ds.attrib["$(n)_mor_info"]) for n in names]
    names, [parse(Float64, m[1]) for m in locs], [parse(Float64, m[2]) for m in locs]
end

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
results = pmap(k -> extract_mooring(moor_name[k], moor_lon[k], moor_lat[k]), moor_sel;
               on_error = e -> (@warn "mooring failed" exception = e; e))
failed = [moor_name[moor_sel[n]] for n in eachindex(results) if results[n] isa Exception]
println(isempty(failed) ? "all moorings done" : "FAILED: " * join(failed, ", "))

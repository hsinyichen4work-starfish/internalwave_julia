# mooring_extract_gaussianfilt.jl
#
# Same extraction as mooring_NCOM_extraction.jl / extract_mooring_HYCOM.jl, but
# every model field is first smoothed with a 2D spatial Gaussian filter
# (L_km, default 100 km) BEFORE it is interpolated to the moorings / transect.
# CMEMS has no internal tides, NCOM and HYCOM do; the short-wavelength internal
# tide signal (mode-1 ~100-150 km) is damped by the filter, so the filtered
# NCOM / HYCOM can be compared with CMEMS more fairly.
#
# The filter (per level, per time record):
#   - one lon/lat box around all moorings + the transect is read, padded by
#     truncate*sigma so the filter footprint of every point is inside the box
#   - masked (normalised) convolution: gaussfilt (Buijsman, julia-codes/functions/gaussfilt.jl)
#     is applied separably (along i, then along j) to F*wet and to wet, and the
#     ratio is taken, so land / below-seafloor points do not leak into the ocean
#   - NCOM is ~0.9 km: the box is first block-summed onto an nc x nc coarser grid
#     (sigma ~ 8 coarse cells), filtered there and interpolated back (the filtered
#     field is smooth on ~sigma, so this costs nothing in accuracy). HYCOM (1/25°) uses nc = 1.
#   - points that were land / below the seafloor stay NaN
# The filter is always applied at constant depth:
#   NCOM levels are squeezed toward the bottom in shallow water (an 880 m column
#   still has 100 levels), so the 100 km footprint would mix depths across the
#   shelf break if level k were filtered. Each column of the box is therefore
#   first interpolated linearly onto zout, then filtered on every zout level, then
#   interpolated horizontally to the points (interp_bry.jl triangle weights, land
#   columns from the nearest wet column, below-seafloor filled from above, as in
#   mooring_NCOM_extraction.jl).
#   HYCOM is on fixed z-levels already: filtered on its 40 native levels, then
#   bilinear on the 4 corners, as in extract_mooring_HYCOM.jl.
#
# Output: same file names and layout as the unfiltered scripts, in a separate directory
#   <outdir_NCOM>/NCOM_<name>.nc, NCOM_M1_CPIES7.nc
#   <outdir_HYCOM>/HYCOM_<name>.nc, HYCOM_M1_CPIES7.nc
#
# Usage (NCOM: days are spread over Distributed workers, each worker filters with a few threads;
#        HYCOM: main process only, it is small):
#   julia -t 8 mooring_extract_gaussianfilt.jl NCOM                -> all moorings + transect
#   julia -t 8 mooring_extract_gaussianfilt.jl HYCOM M2 french     -> only those (name or index)
#   julia -t 8 mooring_extract_gaussianfilt.jl NCOM transect       -> only the transect

using Distributed

n_workers          = 16  # NCOM days read at the same time; disk reads scale up to ~16 readers
threads_per_worker = 2   # threads for the filter inside each worker (levels in parallel)

(isempty(ARGS) || !(uppercase(ARGS[1]) in ("NCOM", "HYCOM"))) &&
    error("first argument must be the model: NCOM or HYCOM")
model = uppercase(ARGS[1])
args  = ARGS[2:end]
if model == "NCOM" && nprocs() == 1
    addprocs(n_workers; exeflags = "--threads=$threads_per_worker")
end

@everywhere using NCDatasets, Dates, NearestNeighbors
# interp_bry.jl functions are always called with the module name (InterpBry.get_2d_coef, ...)
@everywhere include("/home/hsinyi/Documents/Julia/function/interp_bry.jl")
@everywhere include("/home/hsinyi/Documents/Julia/function/mooring_transect.jl")
@everywhere include("/home/mbui/Documents/julia-codes/functions/gaussfilt.jl")

@everywhere begin

# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------
L_km     = 100.0   # filter length scale
fwhm     = true    # true: L_km is the full width at half maximum (sigma = L_km/2.3548 = 42.5 km)
                   # false: L_km is sigma itself (much stronger, also damps the mesoscale)
truncate = 3.0     # kernel cut at +-truncate*sigma (exp(-4.5) = 1% weight at the cut)
nc_per_sigma = 8   # coarse grid: ~sigma/8 per coarse cell (nc = 1 when the grid is already coarse)

mooring_file  = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"
transect_ends = ("M1", "CPIES7")   # mooring line transect, from -> to
dx_km         = 1.0                # transect point spacing (same in all the model scripts)

ftag = "gf$(round(Int, L_km))km"   # tag used in the output directory names

# NCOM (as in mooring_NCOM_extraction.jl)
parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"
datadir_NCOM = "/home/hsinyi/roms_data/NCOM_DATA_NC/"          # yyyymmddHH_{ts,uv,ssh}.nc
outdir_NCOM  = "/home/hsinyi/roms_data/NCOM_mooring_$(ftag)/"
zout  = collect(-1000.0:5.0:0.0)    # output depths (m, negative down, ASCENDING)
dates = Date(2022, 8, 22):Day(1):Date(2022, 11, 30)
pad      = 3      # extra grid cells around each mooring for the interpolation stencil
piece_km = 20.0   # transect: length of each piece with its own sub-grid

# HYCOM (as in extract_mooring_HYCOM.jl)
datadir_HYCOM = "/home/mbui/ModelOutput/NCOM/HYCOM_data/"
outdir_HYCOM  = "/home/hsinyi/roms_data/HYCOM_mooring_$(ftag)/"

σ_km = fwhm ? L_km / (2 * sqrt(2 * log(2))) : L_km

# ---------------------------------------------------------------------------
# Masked 2D Gaussian filter
# ---------------------------------------------------------------------------
"""
    gauss2d(F, σi, σj, nc; truncate) -> Ff

Gaussian filter of one level `F` (i, j), NaN = land / below the seafloor.
σi, σj in grid cells of F. F is block-summed onto an nc x nc coarser grid,
F*wet and wet are smoothed separably with gaussfilt (along i, then along j),
their ratio is interpolated back bilinearly. gaussfilt renormalises the kernel
at the box edges; that factor is the same for F*wet and wet, so it cancels.
Points that are NaN in F stay NaN.
"""
function gauss2d(F, σi, σj, nc; truncate = 3.0)
    Mi, Mj = size(F)
    Ni, Nj = cld(Mi, nc), cld(Mj, nc)
    num, den = zeros(Ni, Nj), zeros(Ni, Nj)
    @inbounds for j in 1:Mj, i in 1:Mi
        v = F[i, j]
        isnan(v) && continue
        I, J = (i - 1) ÷ nc + 1, (j - 1) ÷ nc + 1
        num[I, J] += v; den[I, J] += 1
    end
    for M in (num, den)
        for J in 1:Nj; M[:, J] = gaussfilt(M[:, J], σi / nc; truncate = truncate); end
        for I in 1:Ni; M[I, :] = gaussfilt(M[I, :], σj / nc; truncate = truncate); end
    end
    G = [d > 0 ? n / d : NaN for (n, d) in zip(num, den)]

    # back to the fine grid: coarse cell I is centred on fine index (I-1)*nc + (nc+1)/2
    nc == 1 && return [isnan(F[i, j]) ? NaN : G[i, j] for i in 1:Mi, j in 1:Mj]
    pos(i, N) = (x = clamp((i - (nc + 1) / 2) / nc + 1, 1, N); I = min(floor(Int, x), N - 1); (I, x - I))
    Ff = fill(NaN, Mi, Mj)
    @inbounds for j in 1:Mj
        J, b = Nj > 1 ? pos(j, Nj) : (1, 0.0)
        for i in 1:Mi
            isnan(F[i, j]) && continue
            I, a = Ni > 1 ? pos(i, Ni) : (1, 0.0)
            s, ws = 0.0, 0.0
            for (di, dj, w) in ((0, 0, (1 - a) * (1 - b)), (1, 0, a * (1 - b)), (0, 1, (1 - a) * b), (1, 1, a * b))
                (w == 0 || I + di > Ni || J + dj > Nj) && continue
                g = G[I + di, J + dj]
                isnan(g) && continue
                s += w * g; ws += w
            end
            ws > 0 && (Ff[i, j] = s / ws)
        end
    end
    return Ff
end

"""
    gauss3d(B, σi, σj, nc; truncate) -> Bf

gauss2d on every level of the box B (i, j, k), levels in parallel threads.
"""
function gauss3d(B, σi, σj, nc; truncate = 3.0)
    Bf = similar(B, Float64)
    Threads.@threads for k in axes(B, 3)
        Bf[:, :, k] = gauss2d(view(B, :, :, k), σi, σj, nc; truncate = truncate)
    end
    return Bf
end

"""
    filter_box(i0, i1, j0, j1, σi, σj, nc, Li, Lj) -> (i0, i1, j0, j1)

Pad the index box i0:i1, j0:j1 by the filter footprint (truncate*σ + nc cells),
clipped to the grid (Li, Lj), and round its size up to a multiple of nc where possible.
"""
function filter_box(i0, i1, j0, j1, σi, σj, nc, Li, Lj)
    mi, mj = ceil(Int, truncate * σi) + nc, ceil(Int, truncate * σj) + nc
    i0, i1 = max(i0 - mi, 1), min(i1 + mi, Li)
    j0, j1 = max(j0 - mj, 1), min(j1 + mj, Lj)
    i1 = min(i0 + nc * cld(i1 - i0 + 1, nc) - 1, Li)
    j1 = min(j0 + nc * cld(j1 - j0 + 1, nc) - 1, Lj)
    return i0, i1, j0, j1
end

# ===========================================================================
# NCOM
# ===========================================================================
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
    mooring_subgrid(lonp, latp, lons, lats; pad=2) -> (i0, i1, j0, j1)

Index limits of the minimal parent sub-grid containing all points (plus `pad` cells).
"""
function mooring_subgrid(lonp, latp, lons, lats; pad=2)
    Li, Lj = size(lonp)
    ij = [nearest_ij(lonp, latp, lons[k], lats[k]) for k in eachindex(lons)]
    for (k, (i, j, dkm)) in enumerate(ij)
        if i in (1, Li) || j in (1, Lj)
            @warn "Point $k ($(lons[k]), $(lats[k])) is on/outside the parent grid edge (nearest point $(round(dkm, digits=2)) km away)"
        end
    end
    i0 = max(minimum(first.(ij)) - pad, 1);  i1 = min(maximum(first.(ij)) + pad, Li)
    j0 = max(minimum(getindex.(ij, 2)) - pad, 1);  j1 = min(maximum(getindex.(ij, 2)) + pad, Lj)
    return i0, i1, j0, j1
end

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
    zlev_coef(Zb, hb, wet) -> (kk, ww)

Vertical interpolation of every box column onto zout. `Zb` (i, j, k) layer depths,
k=1 at the surface, NaN below kb. For each zout level n: the model level
kk[n,i,j] above it and the weight ww of that level (1-ww for kk+1).
Above the top level: top value; between the deepest level and the seafloor:
deepest value; kk = 0 (-> NaN) on land and below the seafloor.
"""
function zlev_coef(Zb, hb, wet)
    Mi, Mj, _ = size(Zb)
    kk, ww = zeros(Int, length(zout), Mi, Mj), ones(length(zout), Mi, Mj)
    for j in 1:Mj, i in 1:Mi
        z = view(Zb, i, j, :)
        kb = count(isfinite, z)
        (wet[i, j] && kb > 0 && hb[i, j] < 0) || continue
        for (n, zo) in enumerate(zout)
            zo < hb[i, j] && continue
            if zo >= z[1]
                kk[n, i, j] = 1
            elseif zo <= z[kb]
                kk[n, i, j] = kb
            else
                k = findlast(>=(zo), view(z, 1:kb))       # z[k] >= zo > z[k+1]
                kk[n, i, j], ww[n, i, j] = k, (zo - z[k+1]) / (z[k] - z[k+1])
            end
        end
    end
    return kk, ww
end

"""
    to_zlev(F, kk, ww) -> Fz

NCOM box (i, j, k) -> (i, j, zout level) with the weights of zlev_coef.
"""
function to_zlev(F, kk, ww)
    Nz, Mi, Mj = size(kk)
    Fz = fill(NaN, Mi, Mj, Nz)
    @inbounds for j in 1:Mj, i in 1:Mi, n in 1:Nz
        k = kk[n, i, j]
        k == 0 && continue
        w = ww[n, i, j]
        Fz[i, j, n] = w == 1 ? F[i, j, k] : w * F[i, j, k] + (1 - w) * F[i, j, k+1]
    end
    return Fz
end

prep_field2d(F, src) = Float64.(coalesce.(vec(F), NaN))[src]

# MT in the NCOM files is days since 1900-12-31 00:00
mt2datetime(mt) = DateTime(1900, 12, 31) + Millisecond(round(Int, mt * 86_400_000))

"""
    ncom_setup(jobs; verbose=false) -> S

Filter box (S.I, S.J), levels read (1:S.kmax), vertical weights onto zout
(S.kk, S.ww), filter scales and horizontal weights per piece of every job.
Built once per process (see ncom_day).
"""
function ncom_setup(jobs; verbose = false)
    lonp, latp, maskp, dxp, dyp = NCDataset(parent_grid) do ds
        Float64.(nomissing(ds["lon"][:, :], NaN)),
        Float64.(nomissing(ds["lat"][:, :], NaN)),
        Float64.(nomissing(ds["mask"][:, :], 0.0)),
        Float64.(nomissing(ds["dx"][:, :], NaN)),
        Float64.(nomissing(ds["dy"][:, :], NaN))
    end
    Li, Lj = size(lonp)
    boxes = [[mooring_subgrid(lonp, latp, plon, plat; pad=pad) for (plon, plat) in job.parts] for job in jobs]

    # filter box (I0:I1, J0:J1) around all the sub-grids
    b = reduce(vcat, boxes)
    i0, i1 = minimum(getindex.(b, 1)), maximum(getindex.(b, 2))
    j0, j1 = minimum(getindex.(b, 3)), maximum(getindex.(b, 4))
    dxkm = sum(filter(isfinite, dxp[i0:i1, j0:j1])) / count(isfinite, dxp[i0:i1, j0:j1]) / 1000
    dykm = sum(filter(isfinite, dyp[i0:i1, j0:j1])) / count(isfinite, dyp[i0:i1, j0:j1]) / 1000
    σi, σj = σ_km / dxkm, σ_km / dykm
    nc = max(1, floor(Int, min(σi, σj) / nc_per_sigma))
    I0, I1, J0, J1 = filter_box(i0, i1, j0, j1, σi, σj, nc, Li, Lj)
    I, J = I0:I1, J0:J1

    # vertical interpolation onto zout for every box column; levels read: down to
    # the first level below zout[1] in every wet column (all levels in shallow
    # columns, where they are squeezed toward the bottom)
    Zb, hb = NCDataset(parent_grid) do ds
        Float64.(coalesce.(ds["zm3"][I, J, :], NaN)), Float64.(coalesce.(ds["h"][I, J], NaN))
    end
    wetb = maskp[I, J] .== 1
    kmax = maximum(something(findfirst(<(zout[1]), view(Zb, i, j, :)), count(isfinite, view(Zb, i, j, :)))
                   for i in axes(Zb, 1), j in axes(Zb, 2) if wetb[i, j])
    kk, ww = zlev_coef(Zb[:, :, 1:kmax], hb, wetb)
    verbose && println("NCOM: dx = $(round(dxkm, digits=3)) km, dy = $(round(dykm, digits=3)) km, ",
            "sigma = $(round(σi, digits=1)) x $(round(σj, digits=1)) cells, coarse factor nc = $nc")
    verbose && println("NCOM: filter box i=$I, j=$J ($(length(I)) x $(length(J)) columns), levels 1:$kmax read, ",
            "filtered on $(length(zout)) zout levels")

    # horizontal interpolation weights per piece, indices relative to the filter box
    function build_piece(bx, plon, plat)
        pi0, pi1, pj0, pj1 = bx
        ib, jb = (pi0:pi1) .- (I0 - 1), (pj0:pj1) .- (J0 - 1)
        lons, lats, masks = lonp[pi0:pi1, pj0:pj1], latp[pi0:pi1, pj0:pj1], maskp[pi0:pi1, pj0:pj1]
        src = fill_index(masks, lons, lats)
        elem2, coef2, _ = InterpBry.Interp3D.Interp2D.get_2d_coef(lons, lats, reshape(plon, :, 1), reshape(plat, :, 1))
        h = InterpBry.Interp3D.Interp2D.apply_2d_coef(coef2, elem2, prep_field2d(hb[ib, jb], src))
        return (i = ib, j = jb, src = src, elem2 = elem2, coef2 = coef2,
                np = length(plon), h = h, below = zout .< permutedims(h))
    end
    pieces = [[build_piece(bx, plon, plat) for (bx, (plon, plat)) in zip(boxes[n], job.parts)] for (n, job) in enumerate(jobs)]
    for (job, ps) in zip(jobs, pieces)
        verbose && println("$(job.name): $(length(ps)) piece(s), h = $(round.(extrema(reduce(vcat, [p.h for p in ps])), digits=1)) m")
    end
    return (I = I, J = J, kmax = kmax, kk = kk, ww = ww, σi = σi, σj = σj, nc = nc, pieces = pieces)
end

# filtered box on zout levels (i, j, Nz) -> (Nz, np): land columns from the nearest
# wet column, below the seafloor from the level above, then 2D weights per level
function interp3d(p, B)
    Fs = B[p.i, p.j, :]
    Nz = size(Fs, 3)
    Fc = reshape(Fs, :, Nz)[p.src, :]
    for n in Nz-1:-1:1, c in axes(Fc, 1)
        isnan(Fc[c, n]) && (Fc[c, n] = Fc[c, n+1])
    end
    Fo = stack((InterpBry.Interp3D.Interp2D.apply_2d_coef(p.coef2, p.elem2, Fc[:, n]) for n in 1:Nz); dims = 1)
    Fo[p.below] .= NaN
    return Fo
end
interp2d(p, F) = InterpBry.Interp3D.Interp2D.apply_2d_coef(p.coef2, p.elem2, prep_field2d(F[p.i, p.j], p.src))

const vars3d = [("temp", "ts", "layer_temperature"), ("salt", "ts", "layer_salinity"),
                ("u", "uv", "u_velocity"), ("v", "uv", "v_velocity")]

# ncom_setup result per process, keyed by the job list (built on the first day a worker gets)
NCOM_SETUP = Dict{Any,Any}()

"""
    ncom_day(d, jobs) -> (time, out)

One daily file set: read the box, interpolate onto zout, filter and interpolate
to the points of every job, for every record. `out[n][var]` is the list of
records of job n: (Nz, np) for the 3D variables, (np,) for ssh.
Each file holds 25 hourly records (00..24 h); record 25 is record 1 of the next
day, so it is skipped except for the last day.
"""
function ncom_day(d, jobs)
    t0 = time()
    S = get!(() -> ncom_setup(jobs), NCOM_SETUP, jobs)
    I, J, kmax, kk, ww, σi, σj, nc, pieces = S
    # the transect is kept as Float32 (~0.6 GB per variable for 100 days)
    time_out = DateTime[]
    out = [Dict{String,Vector{Array{job.tr === nothing ? Float64 : Float32}}}(v => [] for v in ("temp", "salt", "u", "v", "ssh"))
           for job in jobs]
    tag = Dates.format(d, "yyyymmdd") * "00"
    dsk = Dict(k => NCDataset(joinpath(datadir_NCOM, tag * "_$k.nc")) for k in ("ts", "uv", "ssh"))
    try
        MT = dsk["ts"]["MT"][:]
        recs = d == last(dates) ? eachindex(MT) : eachindex(MT)[1:end-1]
        for t in recs
            push!(time_out, mt2datetime(MT[t]))
            for (v, kind, ncvar) in vars3d
                B = gauss3d(to_zlev(dsk[kind][ncvar][I, J, 1:kmax, t], kk, ww), σi, σj, nc; truncate = truncate)
                for (o, ps) in zip(out, pieces)
                    push!(o[v], reduce(hcat, [interp3d(p, B) for p in ps]))
                end
            end
            Sf = gauss2d(dsk["ssh"]["ssh"][I, J, t], σi, σj, nc; truncate = truncate)
            for (o, ps) in zip(out, pieces)
                push!(o["ssh"], reduce(vcat, [interp2d(p, Sf) for p in ps]))
            end
        end
    finally
        foreach(close, values(dsk))
    end
    println("NCOM: done $tag on worker $(myid())  ($(length(time_out)) records, $(round(time() - t0, digits=1)) s)")
    return time_out, out
end

end   # @everywhere


# ---------------------------------------------------------------------------
# Command line (main process): moorings, same selection as the original scripts
# ---------------------------------------------------------------------------

moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
tr = transect_points(moor_name, moor_lon, moor_lat, transect_ends...; dx_km = dx_km)

function find_mooring(a, names)
    k = something(tryparse(Int, a), findfirst(==(a), names), 0)
    1 <= k <= length(names) || error("Unknown mooring '$a'; choose 1-$(length(names)), one of: $(join(names, ", ")), or transect")
    return k
end
do_transect = isempty(args) || "transect" in args
moor_sel = isempty(args) ? collect(eachindex(moor_name)) :
           [find_mooring(a, moor_name) for a in args if a != "transect"]
println("$model, Gaussian filter $(fwhm ? "FWHM" : "sigma") = $L_km km (sigma = $(round(σ_km, digits=1)) km), ",
        "$(Threads.nthreads()) thread(s)")
println("Moorings in this run : ", join(moor_name[moor_sel], ", "),
        do_transect ? "  + transect $(tr.name) ($(length(tr.lon)) points)" : "")

# ===========================================================================
# NCOM (main process): jobs, days handed out to the workers, save
# ===========================================================================
function run_NCOM()
    t_start = time()
    # jobs: each mooring, and the transect, split into pieces with their own sub-grid
    jobs = NamedTuple[(name = moor_name[k], lon = moor_lon[k], lat = moor_lat[k], tr = nothing,
             parts = [([moor_lon[k]], [moor_lat[k]])]) for k in moor_sel]
    if do_transect
        np_piece = max(1, round(Int, piece_km / (tr.dist[2] - tr.dist[1])))
        push!(jobs, (name = tr.name, lon = NaN, lat = NaN, tr = tr,
                     parts = [(tr.lon[r], tr.lat[r]) for r in Iterators.partition(eachindex(tr.lon), np_piece)]))
    end

    # setup on the main process too: prints it, stops early on a problem, gives h for the files
    S = NCOM_SETUP[jobs] = ncom_setup(jobs; verbose = true)
    pieces = S.pieces
    println("NCOM: $(length(dates)) days on $(nworkers()) worker(s) x $threads_per_worker thread(s)")

    # one day per task; pmap returns the days in order. A failed day stops the run
    # (nothing is saved), so there are never silent gaps in time.
    res = pmap(d -> ncom_day(d, jobs), CachingPool(workers()), dates)
    time_out = reduce(vcat, first.(res))
    out = [Dict(v => reduce(vcat, [r[2][n][v] for r in res]) for v in ("temp", "salt", "u", "v", "ssh"))
           for n in eachindex(jobs)]
    println("NCOM: $(length(time_out)) records, $(round((time() - t_start) / 60, digits=1)) min")

    # save, same layout as mooring_NCOM_extraction.jl
    mkpath(outdir_NCOM)
    filt_note = "2D Gaussian filter before interpolation: $(fwhm ? "FWHM" : "sigma") = $L_km km " *
                "(sigma = $(round(σ_km, digits=2)) km), truncated at $(truncate) sigma, masked (land excluded)"
    src_note = "mooring_extract_gaussianfilt.jl, grid $parent_grid, data $datadir_NCOM"
    for (job, o, ps) in zip(jobs, out, pieces)
        h_out = reduce(vcat, [p.h for p in ps])
        if job.tr !== nothing
            fname = joinpath(outdir_NCOM, "NCOM_$(job.tr.name).nc")
            save_transect(fname, job.tr, zout, time_out,
                          ["temp" => (stack(o["temp"]), "temperature", "degC"),
                           "salt" => (stack(o["salt"]), "salinity", "psu"),
                           "u"    => (stack(o["u"]), "eastward velocity", "m/s"),
                           "v"    => (stack(o["v"]), "northward velocity", "m/s")];
                          ssh = stack(o["ssh"]), h = h_out,
                          title  = "NCOM output, Gaussian filtered ($L_km km), interpolated to the $(job.tr.from) - $(job.tr.to) transect",
                          source = src_note, note = filt_note)
            continue
        end
        fname = joinpath(outdir_NCOM, "NCOM_$(job.name).nc")
        zt(x) = stack(x)[:, 1, :]
        NCDataset(fname, "c") do ds
            ds.attrib["title"]   = "NCOM output, Gaussian filtered ($L_km km), interpolated to mooring $(job.name)"
            ds.attrib["station"] = job.name
            ds.attrib["source"]  = src_note
            ds.attrib["note"]    = filt_note
            defDim(ds, "z", length(zout)); defDim(ds, "time", length(time_out))

            defVar(ds, "lon", job.lon, (), attrib=["units" => "degrees_east"])
            defVar(ds, "lat", job.lat, (), attrib=["units" => "degrees_north"])
            defVar(ds, "h", h_out[1], (),
                   attrib=["long_name" => "interpolated NCOM water depth", "units" => "m", "positive" => "up"])
            defVar(ds, "z", zout, ("z",), attrib=["units" => "m", "positive" => "up"])
            defVar(ds, "time", time_out, ("time",), attrib=["units" => "hours since 2022-01-01 00:00:00"])

            dims3 = ("z", "time")
            defVar(ds, "temp", zt(o["temp"]), dims3, attrib=["long_name" => "temperature", "units" => "degC"])
            defVar(ds, "salt", zt(o["salt"]), dims3, attrib=["long_name" => "salinity", "units" => "psu"])
            defVar(ds, "u", zt(o["u"]), dims3, attrib=["long_name" => "eastward velocity", "units" => "m/s"])
            defVar(ds, "v", zt(o["v"]), dims3, attrib=["long_name" => "northward velocity", "units" => "m/s"])
            defVar(ds, "ssh", first.(o["ssh"]), ("time",), attrib=["long_name" => "sea surface height", "units" => "m"])
        end
        println("$(job.name): saved $fname")
    end
    println("NCOM: all done ($(round((time() - t_start) / 60, digits=1)) min)")
end

# ===========================================================================
# HYCOM
# ===========================================================================
"""
    bilinear_corners(lon, lat, lon0, lat0) -> (corners, w)

The 4 (i, j) grid points around (lon0, lat0) and their bilinear weights.
`lon` is the periodic 0..360 HYCOM axis (lon0 may be -180..180), `lat` ascending.
"""
function bilinear_corners(lon, lat, lon0, lat0)
    lat[1] <= lat0 <= lat[end] || error("lat $lat0 is outside the grid: $(lat[1])..$(lat[end])")
    x = mod(lon0 - lon[1], 360) + lon[1]
    i  = searchsortedlast(lon, x)
    i2 = i == length(lon) ? 1 : i + 1                     # wrap across 360 -> 0
    dlon = mod(lon[i2] - lon[i], 360)
    j = clamp(searchsortedlast(lat, lat0), 1, length(lat) - 1)
    a = (x - lon[i]) / dlon
    b = (lat0 - lat[j]) / (lat[j+1] - lat[j])
    corners = [(i, j), (i2, j), (i, j + 1), (i2, j + 1)]
    w = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b]
    return corners, w
end

"""
    wet_average(cols, w) -> v

Weighted average of the corner values `cols` (one column per corner), ignoring
NaN corners and rescaling the remaining weights; NaN if all corners are NaN.
"""
function wet_average(cols, w)
    v = fill(NaN, size(cols, 1))
    for k in axes(cols, 1)
        s, ws = 0.0, 0.0
        for c in axes(cols, 2)
            isnan(cols[k, c]) && continue
            s += w[c] * cols[k, c]; ws += w[c]
        end
        ws > 0 && (v[k] = s / ws)
    end
    return v
end

function run_HYCOM()
    # snapshots: group the files by "<yyyymmddhh>_t<tau>", keep the ones with all three kinds
    kinds = ("ssh", "ts3z", "uv3z")
    file_re = r"^gofs35_hycom_glbz04_(\d{10}_t\d{4})_(ssh|ts3z|uv3z)\.nc$"
    snap = Dict{String,Dict{String,String}}()
    for f in readdir(datadir_HYCOM)
        m = match(file_re, f)
        m === nothing && continue
        get!(snap, m[1], Dict{String,String}())[m[2]] = joinpath(datadir_HYCOM, f)
    end
    for (tag, fs) in snap
        missing_kinds = [k for k in kinds if !haskey(fs, k)]
        isempty(missing_kinds) || (@warn "$tag: no $(join(missing_kinds, ", ")) file, skipped"; delete!(snap, tag))
    end
    tags = sort(collect(keys(snap)))
    isempty(tags) && error("no complete HYCOM snapshots in $datadir_HYCOM")
    println("$(length(tags)) snapshots: ", join(tags, ", "))

    # grid: lon 0..360 (periodic), lat ascending, 40 z-levels (positive down)
    lon, lat, dep = NCDataset(snap[tags[1]]["ts3z"]) do ds
        Float64.(ds["lon"][:]), Float64.(ds["lat"][:]), Float64.(ds["depth"][:])
    end
    Nz = length(dep)

    moor = map(moor_sel) do k
        corners, w = bilinear_corners(lon, lat, moor_lon[k], moor_lat[k])
        (name = moor_name[k], lon = moor_lon[k], lat = moor_lat[k], corners = corners, w = w)
    end
    tr_cw = do_transect ? [bilinear_corners(lon, lat, x, y) for (x, y) in zip(tr.lon, tr.lat)] : []

    # filter box: every corner of every point, padded by the filter footprint
    all_corners = vcat([m.corners for m in moor]..., first.(tr_cw)...)
    i0, i1 = extrema(first.(all_corners))
    j0, j1 = extrema(last.(all_corners))
    i1 - i0 < length(lon) ÷ 2 || error("points on both sides of lon 0/360; the box read is not set up for that")
    latm = (lat[j0] + lat[j1]) / 2
    dxkm = (lon[2] - lon[1]) * 111.195 * cosd(latm)
    dykm = (lat[j0+1] - lat[j0]) * 111.195
    σi, σj = σ_km / dxkm, σ_km / dykm
    nc = max(1, floor(Int, min(σi, σj) / nc_per_sigma))
    i0, i1, j0, j1 = filter_box(i0, i1, j0, j1, σi, σj, nc, length(lon), length(lat))
    println("HYCOM: dx = $(round(dxkm, digits=2)) km, dy = $(round(dykm, digits=2)) km, ",
            "sigma = $(round(σi, digits=1)) x $(round(σj, digits=1)) cells, coarse factor nc = $nc")
    println("HYCOM: filter box i=$i0:$i1, j=$j0:$j1 ($(i1-i0+1) x $(j1-j0+1) columns)")

    # read + filter the box of a 3D variable -> (i, j, depth), or of a 2D one -> (i, j, 1); missing -> NaN
    read_box(var, t) = ndims(var) == 4 ? Float64.(coalesce.(var[i0:i1, j0:j1, :, t], NaN)) :
                                         reshape(Float64.(coalesce.(var[i0:i1, j0:j1, t], NaN)), i1 - i0 + 1, j1 - j0 + 1, 1)
    interp_point(B, corners, w) = wet_average(stack([B[i - i0 + 1, j - j0 + 1, :] for (i, j) in corners]), w)

    varlist = [("temp", "ts3z", "water_temp"), ("salt", "ts3z", "salinity"),
               ("u", "uv3z", "water_u"), ("v", "uv3z", "water_v"), ("ssh", "ssh", "surf_el")]
    time_out = DateTime[]
    out = Dict(m.name => Dict(v => Vector{Float64}[] for (v, _, _) in varlist) for m in moor)
    tr_out = Dict(v => Matrix{Float64}[] for (v, _, _) in varlist)    # (depth, x) per record

    for tag in tags
        fs = snap[tag]
        ds_ts, ds_uv, ds_ssh = NCDataset(fs["ts3z"]), NCDataset(fs["uv3z"]), NCDataset(fs["ssh"])
        try
            @assert Float64.(ds_uv["lon"][:]) == lon && Float64.(ds_uv["lat"][:]) == lat "$tag: uv3z on a different grid"
            t_all = ds_ts["time"][:]
            @assert ds_uv["time"][:] == t_all && ds_ssh["time"][:] == t_all "$tag: ts3z/uv3z/ssh have different times"
            for t in eachindex(t_all)
                push!(time_out, t_all[t])
                dsk = Dict("ts3z" => ds_ts, "uv3z" => ds_uv, "ssh" => ds_ssh)
                for (v, kind, ncvar) in varlist
                    B = gauss3d(read_box(dsk[kind][ncvar], t), σi, σj, nc; truncate = truncate)
                    for m in moor
                        push!(out[m.name][v], interp_point(B, m.corners, m.w))
                    end
                    do_transect && push!(tr_out[v], stack([interp_point(B, c, w) for (c, w) in tr_cw]))
                end
            end
        finally
            close(ds_ts); close(ds_uv); close(ds_ssh)
        end
        println("done $tag  ($(length(time_out)) records)")
    end

    # sort by valid time, drop duplicates (the first one is kept)
    it = unique(i -> time_out[i], sortperm(time_out))
    length(it) < length(time_out) && @warn "$(length(time_out) - length(it)) duplicate times dropped"
    time_out = time_out[it]

    # save, same layout as extract_mooring_HYCOM.jl: (z, time), z negative down and ASCENDING
    mkpath(outdir_HYCOM)
    filt_note = "2D Gaussian filter before interpolation: $(fwhm ? "FWHM" : "sigma") = $L_km km " *
                "(sigma = $(round(σ_km, digits=2)) km), truncated at $(truncate) sigma, masked (land excluded); " *
                "irregular time sampling: one record per available HYCOM snapshot"
    to_zt(x) = reverse(stack(x)[:, it], dims = 1)        # (depth, time) surface-first -> bottom-first
    for m in moor
        o = out[m.name]
        fname = joinpath(outdir_HYCOM, "HYCOM_$(m.name).nc")
        NCDataset(fname, "c") do ds
            ds.attrib["title"]   = "GOFS 3.5 HYCOM glbz04, Gaussian filtered ($L_km km), interpolated to mooring $(m.name)"
            ds.attrib["station"] = m.name
            ds.attrib["source"]  = "mooring_extract_gaussianfilt.jl, data $datadir_HYCOM"
            ds.attrib["note"]    = filt_note
            defDim(ds, "z", Nz); defDim(ds, "time", length(time_out))

            defVar(ds, "lon", m.lon, (), attrib = ["units" => "degrees_east"])
            defVar(ds, "lat", m.lat, (), attrib = ["units" => "degrees_north"])
            defVar(ds, "z", reverse(-dep), ("z",), attrib = ["units" => "m", "positive" => "up"])
            defVar(ds, "time", time_out, ("time",), attrib = ["units" => "hours since 2022-01-01 00:00:00"])

            dims3 = ("z", "time")
            defVar(ds, "temp", to_zt(o["temp"]), dims3, attrib = ["long_name" => "temperature", "units" => "degC"])
            defVar(ds, "salt", to_zt(o["salt"]), dims3, attrib = ["long_name" => "salinity", "units" => "psu"])
            defVar(ds, "u", to_zt(o["u"]), dims3, attrib = ["long_name" => "eastward velocity", "units" => "m/s"])
            defVar(ds, "v", to_zt(o["v"]), dims3, attrib = ["long_name" => "northward velocity", "units" => "m/s"])
            defVar(ds, "ssh", vec(stack(o["ssh"]))[it], ("time",), attrib = ["long_name" => "sea surface height", "units" => "m"])
        end
        println("$(m.name): saved $fname  ($(time_out[1]) to $(time_out[end]), $(length(time_out)) records)")
    end

    if do_transect
        to_zxt(x) = reverse(stack(x)[:, :, it], dims = 1)
        longname = Dict("temp" => ("temperature", "degC"), "salt" => ("salinity", "psu"),
                        "u" => ("eastward velocity", "m/s"), "v" => ("northward velocity", "m/s"))
        save_transect(joinpath(outdir_HYCOM, "HYCOM_$(tr.name).nc"), tr, reverse(-dep), time_out,
                      [v => (to_zxt(tr_out[v]), longname[v]...) for v in ("temp", "salt", "u", "v")];
                      ssh    = stack(tr_out["ssh"])[1, :, it],
                      title  = "GOFS 3.5 HYCOM glbz04, Gaussian filtered ($L_km km), interpolated to the $(tr.from) - $(tr.to) transect",
                      source = "mooring_extract_gaussianfilt.jl, data $datadir_HYCOM",
                      note   = filt_note)
    end
end

model == "NCOM" ? run_NCOM() : run_HYCOM()

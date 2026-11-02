# Read one ROMS variable over a time span and a region, from many output files, into one array.
# Time span from time_dir (function/time_dir.jl); each file is opened once and read in one block.
# Needs `using NCDatasets, Dates, Printf` in the calling script, and uvp_masks.jl (loaded by load_all_hpc.jl).
#   td  = time_dir(data_folder; tstart = DateTime(2022, 9, 1), tend = DateTime(2022, 9, 30))
#   ssh = data_load(data_folder, "zeta", td)                                   # whole grid, land as stored (0)
#   ssh = data_load(data_folder, "zeta", td; grid_file)                        # land -> NaN (mask_rho/_u/_v)
#   ix, jy = region_index(grid_file; lon = (-47, -44), lat = (2, 5))           # index box around a lon/lat box
#   u   = data_load(data_folder, "u", td; ix, jy, iz = 128, grid_file)         # surface u in that box
#   ssh = data_load(data_folder, "zeta", td; ix = 1:10:686, jy = 1:10:856)     # every 10th point
# Returns an array in Julia order (xi, eta, time) for 2-D fields or (xi, eta, s, time) for 3-D fields
# (an integer iz drops the s dimension). ix / jy index the variable's own grid (xi_u has one point
# less than xi_rho, eta_v one less than eta_rho).

function data_load(data_folder, var, td; ix = Colon(), jy = Colon(), iz = Colon(), grid_file = nothing,
                   T = Float32, verbose = true)
    t0 = time()
    dims, sz, vt = NCDataset(joinpath(data_folder, td.files[td.TL[td.steps[1]][1]])) do ds
        dimnames(ds[var]), size(ds[var]), eltype(ds[var].var)          # Julia order, time last
    end
    nd = length(dims) - 1                                               # spatial dimensions (2 or 3)
    nd in (2, 3) || error("$var: expected (xi, eta[, s], time), got $dims")
    idx = nd == 2 ? (ix, jy) : (ix, jy, iz)
    idx = map((i, n) -> i isa Colon ? (1:n) : i, idx, sz[1:nd])         # Colon -> explicit range
    blk = map(i -> minimum(i):maximum(i), idx)                          # contiguous block read from disk
    sub = map((i, b) -> i isa Integer ? (i - first(b) + 1) : (i .- first(b) .+ 1), idx, blk)   # picked inside the block
    out = Array{T}(undef, map(length, filter(i -> !(i isa Integer), collect(idx)))..., td.nt)
    o = ntuple(_ -> Colon(), ndims(out) - 1)

    byfile = records_by_file(td)
    bufs = Dict{Int, Array{vt}}()                                       # read buffers by record count, reused
    for (q, (fi, recs)) in enumerate(byfile)
        ks = first.(recs); kr = minimum(ks):maximum(ks)
        z = get!(() -> Array{vt}(undef, map(length, blk)..., length(kr)), bufs, length(kr))
        NCDataset(ds -> NCDatasets.load!(ds[var].var, z, blk..., kr), joinpath(data_folder, td.files[fi]))
        for (k, n) in recs; out[o..., n] .= view(z, sub..., k - first(kr) + 1); end
        verbose && q % 50 == 0 && @printf("  %s: %d / %d files, %.1f min\n", var, q, length(byfile), (time() - t0) / 60)
    end

    if grid_file !== nothing                                            # land -> NaN, mask of the variable's grid
        mr = NCDataset(g -> g["mask_rho"].var[:, :], grid_file)        # u/v masks from mask_rho (uvp_masks.jl)
        mk = "xi_u" in dims ? uvp_masks(mr)[1] : "eta_v" in dims ? uvp_masks(mr)[2] : mr
        land = mk[idx[1], idx[2]] .== 0
        for p in findall(land); out[p, ntuple(_ -> Colon(), ndims(out) - 2)...] .= T(NaN); end
    end
    verbose && @printf("read %s %s: %.2f GB, %.1f min\n", var, size(out), sizeof(out) / 1e9, (time() - t0) / 60)
    return out
end

# Index ranges (ix, jy) of the rho points inside a lon/lat box (degrees; lon as -180..180 or 0..360).
# On the rotated grid this is the smallest index box that contains all of them (a few points outside
# the lon/lat box come along at the corners).
function region_index(grid_file; lon, lat)
    lo, la = NCDataset(g -> (g["lon_rho"][:, :], g["lat_rho"][:, :]), grid_file)
    inbox = (mod.(lo .- mod(lon[1], 360), 360) .<= mod(lon[2] - lon[1], 360)) .& (lat[1] .<= la .<= lat[2])
    any(inbox) || error("no grid points in lon $lon, lat $lat")
    p = findall(inbox)
    return minimum(first.(Tuple.(p))):maximum(first.(Tuple.(p))), minimum(last.(Tuple.(p))):maximum(last.(Tuple.(p)))
end

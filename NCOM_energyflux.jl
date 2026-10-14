using Distributed

# worker processes for the parallel run at the bottom. Added only once, so re-running this block in the same session is safe
# n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "6")) - 1)   # -1 reserves a CPU for this main process; each worker needs ~75 GB
n_workers = 1
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # several processes read the same NCOM files; workers inherit this
# each worker is told how much memory it may use before it must clean up. Without this Julia sizes its cleanup to the
# whole machine's memory, and leftovers from earlier pieces pile up.
# share = (memory - 4 GB for this main process) * 0.85 / workers;  under Slurm SLURM_MEM_PER_NODE (MB), otherwise free memory
job_mem_mb = haskey(ENV, "SLURM_MEM_PER_NODE") ? parse(Int, ENV["SLURM_MEM_PER_NODE"]) : round(Int, Sys.free_memory() / 2^20)
heap_mb = floor(Int, (job_mem_mb - 4000) * 0.85 / n_workers)
# nprocs() == 1 && addprocs(n_workers; exeflags = "--heap-size-hint=$(heap_mb)M")

println("memory $(job_mem_mb) MB -> heap size hint $(heap_mb) MB per worker")
println("running with $(nprocs() - 1) worker processes")

# packages and functions every process needs (plotting stays on the main process only)
@everywhere begin
    using NCDatasets, Dates, Statistics, DSP, GibbsSeaWater
    include("/home/hsinyi/Documents/Julia/function/butter_filters.jl")   # bandpass_butter, lowhighpass_butter (need DSP)
end
using CairoMakie
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")

grid_fname = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"   # NCOM parent grid
datadir = "/home/hsinyi/roms_data/NCOM_DATA_NC/"   # NCOM daily files: <yyyymmddhh>_ts.nc, _uv.nc, _ssh.nc, ...
nc_dum_save = "/home/hsinyi/roms_data/NCOM_energyflux_dum"
mkpath(nc_dum_save)

##
# NCOM grid. All fields (T, S, u, v, ssh) sit on this one unstaggered grid, [xi, eta] = [dim_i, dim_j].
# The vertical levels are fixed in time: layer_thickness in the daily _lthick.nc files is the same every day and equals
# dzm here, so depth and layer thickness come straight from the grid file.
#   zm3, dzm [xi, eta, z]   depth of the level centres [m, negative] and layer thickness [m]
#   level k = 1 is the surface ... k = kb the bottom; below kb (and over land) they are missing -> NaN / 0 here
# Each piece reads its own tile of these from the grid file (see energyflux_piece); here only the check below.
mask, h, dzm = NCDataset(grid_fname) do ds
    coalesce.(ds["mask"][:, :], 0f0),
    abs.(coalesce.(ds["h"][:, :], 0f0)),   # h is negative-down in the file; positive depth here
    coalesce.(ds["dzm"][:, :, :], 0f0)
end
@assert all(dropdims(sum(Float64, dzm, dims = 3), dims = 3)[mask .== 1] .≈ h[mask .== 1])   # layers fill the water column exactly
dzm = nothing

# one day = one set of files; only days that have both _ts.nc and _uv.nc are used
days = sort([replace(f, "_ts.nc" => "") for f in readdir(datadir) if occursin(r"^\d{10}_ts\.nc$", f)])
filter!(d -> isfile(joinpath(datadir, d * "_uv.nc")), days)
ts_files = joinpath.(datadir, days .* "_ts.nc")
uv_files = joinpath.(datadir, days .* "_uv.nc")

# Split the grid into tiles that line up with the files' storage chunks (249 x 267 in xi x eta). The files are read a
# whole chunk at a time, so a tile smaller than a chunk would read data it then throws away (4x the reading for tiles
# of a quarter chunk, out of 66 GB files). NCOM has no staggered grid, so tiles need no halo (unlike the ROMS version,
# where u2rho / v2rho needed one): each tile reads exactly the points it owns.
#   i, j   global index ranges of the tile
#   per tile, each array [xi, eta, z, time] over a 20-day window is ~12 GB; a piece peaks at ~6 of them (~75 GB),
#   so choose n_workers (top of the script) with that in mind
#   nsplit: each chunk is cut into nsplit x nsplit tiles (1 = one tile per chunk). Tiles stay inside one chunk, so
#   memory per piece drops by nsplit^2 while the reading grows only by about nsplit^2 too, not more
function tile_ranges(nx, ny, cx, cy; nsplit = 1)
    function split1d(n, c)
        out = UnitRange{Int}[]
        for a in 1:c:n   # one chunk: a .. b
            b = min(a + c - 1, n)
            e = round.(Int, range(a - 1, b, length = nsplit + 1))
            append!(out, [e[k]+1:e[k+1] for k in 1:nsplit])
        end
        out
    end
    [(i = x, j = y) for x in split1d(nx, cx), y in split1d(ny, cy)]
end

chunk_xi, chunk_eta = NCDataset(ts_files[1]) do ds; chunking(ds["layer_temperature"])[2][1:2] end
nsplit = 1   # 1: 5 x 5 tiles (~75 GB per piece);  2: 10 x 10 tiles of half a chunk (~19 GB per piece, ~2-4x the reading)
tiles = tile_ranges(size(mask)..., chunk_xi, chunk_eta; nsplit)   # ndomx × ndomy matrix of NamedTuples
ndomx, ndomy = size(tiles)
println("$ndomx x $ndomy tiles (each chunk of $chunk_xi x $chunk_eta points cut into $nsplit x $nsplit)")

##
# (1) time axis of the whole run — only MT (days since 1900-12-31) is read here, no 3D fields.
#     Consecutive daily files overlap at midnight (record 25 of one day == record 1 of the next), so the records of
#     each file that repeat a time already taken from the previous file are dropped.
#     keep_each[k]: the records of file k that are used (numbered within the file)
t_ref = DateTime(1900, 12, 31)
mt_each = [NCDataset(f) do ds; Float64.(ds["MT"][:]) end for f in ts_files]
keep_each = Vector{UnitRange{Int}}(undef, length(mt_each))
for k in eachindex(mt_each)
    first_new = k == 1 ? 1 : something(findfirst(>(last(mt_each[k-1][keep_each[k-1]]) + 1e-6), mt_each[k]), length(mt_each[k]) + 1)
    keep_each[k] = first_new:length(mt_each[k])
end
nt_each = length.(keep_each)   # records per file actually used
t_end = cumsum(nt_each)
nt = t_end[end]
ncom_time_all = reduce(vcat, [mt[kp] for (mt, kp) in zip(mt_each, keep_each)]) .* 86400   # [s since 1900-12-31]
ncom_date_all = t_ref .+ Millisecond.(round.(Int, ncom_time_all .* 1000))
dt = median(diff(ncom_time_all))   # output interval [s]
all(d -> isapprox(d, dt, rtol = 1e-6), diff(ncom_time_all)) ||
    @warn "MT is not evenly spaced (missing day or duplicated record) — the filters assume a uniform dt" extrema(diff(ncom_time_all))
println("NCOM: $nt records in $(length(ts_files)) days, dt = $(dt / 3600) hr, ",
        ncom_date_all[1], " -> ", ncom_date_all[end], "  (", round((ncom_time_all[end] - ncom_time_all[1]) / 86400, digits = 2), " days)")

# (2) moving windows of win_days, consecutive windows sharing overlap_days.
#     windows[w] is a range of global record indices. If the regular stepping leaves records uncovered at the end,
#     one more window is added flush with the last record (so it overlaps its neighbour by more than overlap_days).
win_days, overlap_days = 20, 6
nwin = round(Int, win_days * 86400 / dt)                      # records per window
nstep = round(Int, (win_days - overlap_days) * 86400 / dt)    # records between window starts
nt >= nwin || error("run has only $nt records, shorter than one $win_days-day window ($nwin records)")
win_starts = collect(1:nstep:nt-nwin+1)
last(win_starts) + nwin - 1 < nt && push!(win_starts, nt - nwin + 1)
windows = [s:s+nwin-1 for s in win_starts]

# (3) which records each window keeps after filtering (the ends of a filtered record are contaminated).
#     keeps[w] is a range of global record indices. Each overlap is split at its midpoint, so every record is taken
#     from the window where it sits farthest from an edge, and the kept parts meet exactly: no gap, no duplicate.
#     At the two ends of the run there is no neighbour, so edge_days are simply dropped there.
edge_days = overlap_days / 2
nedge = round(Int, edge_days * 86400 / dt)
mids = [(first(windows[w+1]) + last(windows[w])) ÷ 2 for w in 1:length(windows)-1]   # last record window w keeps
keeps = [(w == 1 ? first(windows[1]) + nedge : mids[w-1] + 1):(w == length(windows) ? last(windows[end]) - nedge : mids[w])
         for w in eachindex(windows)]
for (w, win) in enumerate(windows)
    ov = w == 1 ? 0 : length(intersect(win, windows[w-1]))
    println("window $w: records $win  ", ncom_date_all[first(win)], " -> ", ncom_date_all[last(win)],
            w == 1 ? "" : "  (overlap with previous: $(round(ov * dt / 86400, digits = 2)) days)")
    println("          keeps $(keeps[w])  ", ncom_date_all[first(keeps[w])], " -> ", ncom_date_all[last(keeps[w])],
            "  (drops $(round((first(keeps[w]) - first(win)) * dt / 86400, digits = 2)) days at the start, ",
            "$(round((last(win) - last(keeps[w])) * dt / 86400, digits = 2)) at the end)")
end

##
# settings for the flux calculation

# time filters applied to p', u', v'. To change a band or add one, edit this table — everything below loops over it.
#   kind "band": T = (short, long), keeps periods in between
#   kind "high": T = cutoff, keeps periods shorter than T        kind "low": keeps periods longer than T
#   T in hours; N = Butterworth order (run forward and backward, so no phase shift and twice the roll-off)
filters = [
    (name = "M2",   kind = "band", T = (9, 15), N = 4),
    (name = "hp24", kind = "high", T = 24,      N = 4),
]

# rho' = rho - its time mean at the same level. NCOM's levels stay at fixed depths, so this is the anomaly at a fixed
# depth — the same thing the "fixed_z" option of the ROMS version computes. (No s_level / fixed_z choice is needed here.)

# everything energyflux_piece needs, in one bundle, so the function uses no global variables
# (that is what lets it be sent to other processes later)
cfg = (; datadir, grid_fname, nc_dum_save, ts_files, uv_files, keep_each, nt_each, t_end, ncom_time_all, ncom_date_all, dt,
       windows, keeps, tiles, ndomx, ndomy, filters)

##
# helper functions and the flux calculation itself — defined on every process
@everywhere begin

    # thickness-weighted depth mean of a 4D field: sum_s(f * Hz) / sum_s(Hz)
    #   f [x, y, z, time], Hz [x, y, z] (NCOM's layers do not change in time)  ->  [x, y, 1, time]
    #   the z dimension is kept (length 1) so the result subtracts straight from f:  f .- depth_mean(f, Hz)
    #   land columns (sum(Hz) = 0) get 0 instead of 0/0
    function depth_mean(f, Hz)
        size(f)[1:3] == size(Hz) || error("depth_mean: f is $(size(f)) but Hz is $(size(Hz))")
        H = sum(Hz, dims = 3)
        sum(f .* Hz, dims = 3) ./ max.(H, eps(Float32)) .* (H .> 0)
    end

    # filter x along its last dimension (time) with one row of `filters`; dt_hr = sampling interval in hours.
    # Done one slab of the first dimension at a time, so the Float64 copies the filter makes stay small.
    function filter_time(x, flt, dt_hr)
        nt = size(x, ndims(x))
        out = similar(x)
        for i in axes(x, 1)
            xi = selectdim(x, 1, i)                           # [..., time]
            y = permutedims(reshape(xi, :, nt))               # [time, points] — butter_filters work along dim 1
            yf = flt.kind == "band" ? bandpass_butter(y, flt.T[1], flt.T[2], dt_hr, flt.N) :
                                      lowhighpass_butter(y, flt.T, dt_hr, flt.N, flt.kind)
            selectdim(out, 1, i) .= reshape(permutedims(yf), size(xi))
        end
        out
    end

    # NCOM marks land and below-bottom points with NaN; anything never written reads as the netCDF default fill (~1e37).
    # Sets every dry point (wet[xi, eta, z] = false) of f[xi, eta, z, time] to 0, so NaN cannot spread into sums, and
    # returns how many wet points held NaN / fill (bad data; also set to 0)
    function clean_dry!(f, wet)
        nbad = 0
        @inbounds for I in CartesianIndices(f)
            ok = abs(f[I]) < 1f30
            wet[I[1], I[2], I[3]] && !ok && (nbad += 1)
            (wet[I[1], I[2], I[3]] && ok) || (f[I] = 0)
        end
        nbad
    end

    # in-situ density [kg m-3] from potential temperature and practical salinity (TEOS-10, same as density_insitu in
    # dbry_fun.jl), written into rho. temp, salt, rho [xi, eta, z, time]; p_lev[xi, eta, z] = sea pressure of each
    # level [dbar], fixed in time because NCOM's levels are. Points with wet = false are set to 0.
    function density!(rho, temp, salt, p_lev, lon, lat, wet)
        @inbounds for n in axes(temp, 4), k in axes(temp, 3), j in axes(temp, 2), i in axes(temp, 1)
            if wet[i, j, k]
                SA = gsw_sa_from_sp(max(Float64(salt[i, j, k, n]), 0.0), p_lev[i, j, k], lon[i, j], lat[i, j])
                CT = gsw_ct_from_pt(SA, Float64(temp[i, j, k, n]))
                rho[i, j, k, n] = gsw_rho(SA, CT, p_lev[i, j, k])
            else
                rho[i, j, k, n] = 0
            end
        end
        rho
    end

    # file one piece is saved to
    piece_fname(ix, iy, iw, cfg) =
        joinpath(cfg.nc_dum_save, "ncom_energyflux_w$(lpad(iw, 2, '0'))_tile_$(lpad(ix, 2, '0'))_$(lpad(iy, 2, '0')).nc")

    ##
    # the flux calculation for one piece: tile (ix, iy) over window iw.
    # Reads T, S, u, v -> rho -> rho' -> hydrostatic p -> removes depth means -> time filters -> p'u', p'v' -> depth integral,
    # freeing or overwriting each big array as soon as it is no longer needed (peak ≈ 6 arrays the size of temp),
    # and saves the tile over the window's kept records to one nc file in cfg.nc_dum_save. Returns the file name.
    # A piece whose file already exists is skipped unless overwrite = true, so an interrupted batch can simply be rerun.
    function energyflux_piece(ix, iy, iw, cfg; overwrite = false)
        (; ts_files, uv_files, keep_each, nt_each, t_end, ncom_time_all, ncom_date_all, dt, windows, keeps, tiles, filters) = cfg
        tag = "[w$(lpad(iw, 2, '0')) tile $(lpad(ix, 2, '0')),$(lpad(iy, 2, '0'))] "
        nc_fname = piece_fname(ix, iy, iw, cfg)
        if isfile(nc_fname) && !overwrite
            println(tag, "already saved, skipping: ", nc_fname)
            return nc_fname
        end
        t = tiles[ix, iy]
        win = windows[iw]
        # start clean: drop whatever the previous piece on this worker left behind and hand the memory back to the system
        GC.gc()
        Sys.islinux() && ccall(:malloc_trim, Cint, (Cint,), 0)

        # ---- vertical grid of this tile (fixed in time), from the grid file
        #   Hz[xi, eta, z]      layer thickness [m], 0 below the bottom and over land; sums over z to h
        #   wet[xi, eta, z]     true where there is water
        #   p_lev[xi, eta, z]   sea pressure at each level centre [dbar], for the equation of state
        Hz, z_lev, lon_t, lat_t = NCDataset(cfg.grid_fname) do ds
            coalesce.(ds["dzm"][t.i, t.j, :], 0f0), coalesce.(ds["zm3"][t.i, t.j, :], NaN32),
            coalesce.(ds["lon"][t.i, t.j], NaN), coalesce.(ds["lat"][t.i, t.j], NaN)
        end
        wet = Hz .> 0
        nz = size(Hz, 3)
        p_lev = [wet[I] ? gsw_p_from_z(Float64(z_lev[I]), lat_t[I[1], I[2]]) : 0.0 for I in CartesianIndices(Hz)]
        any(wet) || println(tag, "all land")

        # ---- read T, S, u, v for this tile over this window, one day file at a time   [xi, eta, z, time]
        #   T and S are turned into density as soon as each day is read, so only rho_t (not T and S) is kept for the window.
        #   NCOM's u, v are already east / north on the same points as T and S, so no u2rho / v2rho or rotation is needed
        println(tag, "rho_t: ", (length(t.i), length(t.j), nz, length(win)), "  ≈ ",
                round(length(t.i) * length(t.j) * nz * length(win) * 4 / 2^30, digits = 1), " GB  (u, v the same each)")
        rho_t = Array{Float32}(undef, length(t.i), length(t.j), nz, length(win))
        u = similar(rho_t)
        v = similar(rho_t)
        ncom_time = ncom_time_all[win]
        nbad = Dict("temp" => 0, "salt" => 0, "u" => 0, "v" => 0)
        for k in eachindex(ts_files)
            r = intersect(t_end[k]-nt_each[k]+1:t_end[k], win)   # global records of this file that fall in the window
            isempty(r) && continue
            rf = keep_each[k][r .- (t_end[k] - nt_each[k])]   # same records, numbered within the file
            it = r .- (first(win) - 1)                         # ... and within the window
            temp, salt = NCDataset(ts_files[k]) do ds
                ds["layer_temperature"][t.i, t.j, :, rf], ds["layer_salinity"][t.i, t.j, :, rf]
            end
            nbad["temp"] += clean_dry!(temp, wet)
            nbad["salt"] += clean_dry!(salt, wet)
            density!(view(rho_t, :, :, :, it), temp, salt, p_lev, lon_t, lat_t, wet)
            temp = salt = nothing
            NCDataset(uv_files[k]) do ds
                u[:, :, :, it] = ds["u_velocity"][t.i, t.j, :, rf]
                v[:, :, :, it] = ds["v_velocity"][t.i, t.j, :, rf]
            end
            k % 5 == 0 && println(tag, "read day $k / $(length(ts_files))")
        end
        nbad["u"] = clean_dry!(u, wet)
        nbad["v"] = clean_dry!(v, wet)
        for (name, n) in nbad
            n > 0 && @warn "$(tag)$name has $n missing / fill values at wet points — set to 0"
        end

        # ---- rho' = rho - time mean at the same level (a fixed depth in NCOM)   [xi, eta, z, time]
        # time mean at each (xi, eta, z); summed in Float64 so 480 Float32 values don't lose precision
        rho_t .-= Float32.(sum(Float64, rho_t, dims = 4) ./ size(rho_t, 4))
        rho_p = rho_t   # overwritten in place, so this is rho' from here on
        rho_t = nothing
        println(tag, "rho': range ", extrema(rho_p), " kg m-3")

        # ---- hydrostatic pressure perturbation from rho', integrated down from the surface (p = 0 at z = 0):
        #   p_w at the bottom of layer k = p_w at its top + g * rho'[k] * Hz[k]
        #   p_hyd[k] = average of the two interfaces of layer k -> pressure at the level centre
        #   p_hyd[xi, eta, z, time]   [Pa], on the same points as rho_p
        #   NCOM's level 1 is the surface, so the loop runs k = 1 -> nz (the ROMS version runs nz -> 1)
        #   The free surface is left out: the levels are fixed, so the top layer does not stretch with ssh.
        g = 9.81f0
        p_hyd = similar(rho_p)
        p_top = zeros(Float32, size(rho_p, 1), size(rho_p, 2), size(rho_p, 4))   # p_w at the top of the current layer; 0 at the surface
        for k in 1:nz
            @views dp = g .* rho_p[:, :, k, :] .* Hz[:, :, k]   # pressure added across layer k
            @views p_hyd[:, :, k, :] .= p_top .+ 0.5f0 .* dp
            p_top .+= dp                                       # now p_w at the bottom of layer k
        end
        println(tag, "p_hyd: ", size(p_hyd), "  range ", extrema(p_hyd), " Pa")
        rho_p = p_top = nothing   # free memory
        GC.gc()

        # ---- baroclinic parts: remove the thickness-weighted depth mean from each column at each time, in place.
        #   p_bc, u_bc, v_bc = p', u', v'   [xi, eta, z, time]
        # check: the depth integral of each baroclinic field should be zero. Printed as the worst column's
        # |∫ f' dz| relative to that field's largest ∫ |f| dz, so "zero" means Float32 rounding (~1e-6), whatever the units
        for (name, f) in (("p'", p_hyd), ("u'", u), ("v'", v))
            scale = maximum(sum(abs.(f) .* Hz, dims = 3))
            f .-= depth_mean(f, Hz)
            resid = maximum(abs, sum(f .* Hz, dims = 3))
            println(tag, "depth integral of $name: max |∫ f' dz| = $resid   relative to max ∫ |f| dz: $(resid / scale)")
            scale == 0 || resid / scale < 1f-4 || @warn "$(tag)depth integral of $name is not close to zero"
        end
        p_bc, u_bc, v_bc = p_hyd, u, v   # overwritten in place above
        p_hyd = u = v = nothing

        # ---- per filter band: time filter p', u', v' -> p'u', p'v' -> depth integral.
        #      One band at a time, so memory does not grow with the number of rows in `filters`.
        #   Fx["M2"], Fy["hp24"], ...   depth-integrated energy flux [W m-1], [xi, eta, time_keep], NaN over land
        #   Fx is eastward, Fy northward (NCOM's u, v are east / north)
        dt_hr = dt / 3600
        t_keep = keeps[iw] .- (first(win) - 1)   # this window's kept records (keeps[iw]), numbered within the window
        flux_time = ncom_time[t_keep]
        land = dropdims(sum(Hz, dims = 3), dims = 3) .== 0
        Fx = Dict{String, Array{Float32, 3}}()
        Fy = Dict{String, Array{Float32, 3}}()
        for flt in filters
            # each field is filtered over the full window, then only the kept records are used
            p_f = filter_time(p_bc, flt, dt_hr)[:, :, :, t_keep]
            u_f = filter_time(u_bc, flt, dt_hr)[:, :, :, t_keep]
            Fx[flt.name] = dropdims(sum(p_f .* u_f .* Hz, dims = 3), dims = 3)
            println(tag, flt.name, ":  p ", extrema(p_f), " Pa   u ", extrema(u_f), " m/s")
            u_f = nothing
            v_f = filter_time(v_bc, flt, dt_hr)[:, :, :, t_keep]
            Fy[flt.name] = dropdims(sum(p_f .* v_f .* Hz, dims = 3), dims = 3)
            Fx[flt.name][land, :] .= NaN32
            Fy[flt.name][land, :] .= NaN32
            println(tag, flt.name, ":  v ", extrema(v_f), " m/s   time-mean flux over the tile's water  Fx ",
                    mean(filter(isfinite, Fx[flt.name])), "   Fy ", mean(filter(isfinite, Fy[flt.name])), " W m-1")
            p_f = v_f = nothing
            GC.gc()
        end
        p_bc = u_bc = v_bc = nothing   # free memory

        # ---- save this piece (one tile x the kept part of one window) so all pieces can be joined afterwards.
        #   xi, eta   where the piece sits in the full NCOM grid (global indices, 1-based)
        #   record    where it sits in the run (index into the whole run's records), MT the matching time
        # Written under a temporary name and renamed at the end, so a killed run never leaves a half-written piece.
        NCDataset(nc_fname * ".tmp", "c") do ds
            defDim(ds, "xi", length(t.i))
            defDim(ds, "eta", length(t.j))
            defDim(ds, "time", length(flux_time))
            defVar(ds, "xi", Int32.(t.i), ("xi",), attrib = ["long_name" => "global NCOM xi index (1-based)"])
            defVar(ds, "eta", Int32.(t.j), ("eta",), attrib = ["long_name" => "global NCOM eta index (1-based)"])
            defVar(ds, "record", Int32.(win[t_keep]), ("time",), attrib = ["long_name" => "index of this time in the whole run's records (1-based)"])
            defVar(ds, "MT", flux_time ./ 86400, ("time",), attrib = ["long_name" => "time", "units" => "days since 1900-12-31 00:00:00"])
            for flt in filters
                band = flt.kind == "band" ? "band-pass $(flt.T[1])-$(flt.T[2]) hr" : "$(flt.kind)-pass $(flt.T) hr"
                defVar(ds, "Fx_$(flt.name)", Fx[flt.name], ("xi", "eta", "time"), attrib = [
                    "long_name" => "depth-integrated energy flux p'u' (eastward), $band, Butterworth order $(flt.N)", "units" => "W m-1"])
                defVar(ds, "Fy_$(flt.name)", Fy[flt.name], ("xi", "eta", "time"), attrib = [
                    "long_name" => "depth-integrated energy flux p'v' (northward), $band, Butterworth order $(flt.N)", "units" => "W m-1"])
            end
            ds.attrib["tile"] = "ix = $ix of $(cfg.ndomx), iy = $iy of $(cfg.ndomy)"
            ds.attrib["window"] = "window $iw of $(length(windows)): $(ncom_date_all[first(win)]) -> $(ncom_date_all[last(win)]); kept records $(first(keeps[iw]))-$(last(keeps[iw])) of the run"
            ds.attrib["rho_reference"] = "time mean at each fixed NCOM level (fixed depth); in-situ density from TEOS-10"
            ds.attrib["direction"] = "Fx eastward, Fy northward"
            ds.attrib["source"] = cfg.datadir
        end
        mv(nc_fname * ".tmp", nc_fname, force = true)
        println(tag, "saved ", nc_fname, "   (this worker's peak memory so far: ", round(Sys.maxrss() / 2^30, digits = 1), " GB)")
        return nc_fname
    end

end   # @everywhere

##
# check: what each filter does to a unit-amplitude tide over one window (amplitude left inside window 1's kept records;
# 1 = untouched, 0 = removed)
tides = (M2 = 12.42, S2 = 12.0, K1 = 23.93, O1 = 25.82)   # periods in hours
for flt in filters
    gains = [maximum(abs, filter_time(reshape(sin.(2π .* (0:nwin-1) .* (dt / 3600) ./ T), 1, :), flt, dt / 3600)[1, keeps[1]]) for T in tides]
    println("filter ", rpad(flt.name, 6), join(["$(k) $(round(gn, digits = 3))" for (k, gn) in zip(keys(tides), gains)], "   "))
end

##
# run the pieces in parallel: pmap hands them out to whichever worker is free, one piece per worker at a time, and
# blocks here until all are done. Pieces already saved are skipped, so an interrupted run can simply be started again.
pieces = vec([(ix, iy, iw) for ix in 1:ndomx, iy in 1:ndomy, iw in eachindex(windows)])
# pieces = first(pieces, nworkers())   # uncomment for a quick trial: one piece per worker instead of the whole domain
println("running $(length(pieces)) pieces on $(nworkers()) workers")
@time pmap(pc -> energyflux_piece(pc..., cfg), pieces; on_error = ex -> println("a piece failed: ", ex))

# # one piece on this process, without the workers (handy for testing a change):
# # energyflux_piece(1, 1, 1, cfg; overwrite = true)
# pieces = vec([(ix, iy, iw) for ix in 1:ndomx, iy in 1:ndomy, iw in eachindex(windows)])
# println("running $(length(pieces)) pieces on $(nworkers()) workers")
# energyflux_piece(1, 1, 1, cfg; overwrite = true)
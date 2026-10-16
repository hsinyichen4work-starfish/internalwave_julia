using Distributed

# worker processes for the parallel run at the bottom. Added only once, so re-running this block in the same session is safe
n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")) - 1)   # -1 reserves a CPU for this main process
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # several processes read the same his files; workers inherit this
# each worker is told how much memory it may use before it must clean up. Without this Julia sizes its cleanup to the
# whole node's memory, not the job's share, and leftovers from earlier pieces pile up until Slurm kills the job.
# share = (job memory - 4 GB for this main process) * 0.85 / workers;  SLURM_MEM_PER_NODE is in MB
job_mem_mb = parse(Int, get(ENV, "SLURM_MEM_PER_NODE", "64000"))
heap_mb = floor(Int, (job_mem_mb - 4000) * 0.85 / n_workers)
nprocs() == 1 && addprocs(n_workers; exeflags = "--heap-size-hint=$(heap_mb)M")
println("job memory $(job_mem_mb) MB -> heap size hint $(heap_mb) MB per worker")
println("running with $(nprocs() - 1) worker processes")

# packages and functions every process needs (plotting stays on the main process only)
@everywhere begin
    using NCDatasets, Dates, Statistics, DSP
    include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
    include("/home/hchen54/internalwave_julia/function/butter_filters.jl")   # bandpass_butter, lowhighpass_butter (need DSP)
end
using CairoMakie
CairoMakie.activate!()
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon_1008_2/after_joint"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/expanse/lustre/projects/uso101/hchen54/figure/amazon_900m_3mon_1008/energy_flux"   # separate from his_rejoint so the two scripts' skip checks don't interfere
nc_dum_save = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon_1008_2/energyflux_dum"
mkpath(figure_path)
mkpath(nc_dum_save)

##
# fname = joinpath(datadir, "roms_dia.20220829010000.nc")
# ocean_time, up,vp = NCDataset(fname) do ds
#     show(ds)
#     ds["ocean_time"][:], ds["up"][:,:,:], ds["vp"][:,:,:]
# end

mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
    ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
end

# Split the rho grid into ndomx × ndomy tiles, each padded by `overlap` points (clipped at the domain edge).
# Per tile:
#   i, j           rho-point ranges to read from file (core + halo)
#   iu, jv         matching xi_u / eta_v ranges, so the chunk is itself a staggered mini-grid
#                  (u -> [iu, j], v -> [i, jv])
#   icore, jcore   global rho ranges this tile owns (no halo; tiles' cores cover the grid exactly once)
#   iloc, jloc     where the core sits inside the chunk that was read: chunk[iloc, jloc] -> global[icore, jcore]
function tile_ranges(nx, ny, ndomx, ndomy; overlap = 3)
    function split1d(n, ndom)
        edges = round.(Int, range(0, n, length = ndom + 1))
        map(1:ndom) do k
            core = edges[k]+1:edges[k+1]
            halo = max(1, first(core) - overlap):min(n, last(core) + overlap)
            loc = (first(core)-first(halo)+1):(last(core)-first(halo)+1)
            (; core, halo, loc)
        end
    end
    xs, ys = split1d(nx, ndomx), split1d(ny, ndomy)
    [(i = x.halo, j = y.halo,
      iu = first(x.halo):last(x.halo)-1, jv = first(y.halo):last(y.halo)-1,
      icore = x.core, jcore = y.core, iloc = x.loc, jloc = y.loc)
     for x in xs, y in ys]
end

ndomx, ndomy = 8, 16
tiles = tile_ranges(size(mask_rho)..., ndomx, ndomy; overlap = 2)   # ndomx × ndomy matrix of NamedTuples

his_files = sort(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(datadir)))   # joined his files only, in time order
his_files = joinpath.(datadir, his_files)

##
# (1) time axis of the whole run — only ocean_time is read here (seconds since 1994-01-01), no 3D fields
his_time_each = [NCDataset(f) do ds; Float64.(ds["ocean_time"][:]) end for f in his_files]
nt_each = length.(his_time_each)   # records per file
t_end = cumsum(nt_each)
nt = t_end[end]
his_time_all = reduce(vcat, his_time_each)
his_date_all = DateTime(1994, 1, 1) .+ Second.(round.(Int, his_time_all))
dt = median(diff(his_time_all))   # output interval [s]
all(d -> isapprox(d, dt, rtol = 1e-6), diff(his_time_all)) ||
    @warn "ocean_time is not evenly spaced (gap or duplicated record) — the filters assume a uniform dt" extrema(diff(his_time_all))
println("his: $nt records in $(length(his_files)) files, dt = $(dt / 3600) hr, ",
        his_date_all[1], " -> ", his_date_all[end], "  (", round((his_time_all[end] - his_time_all[1]) / 86400, digits = 2), " days)")

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
    println("window $w: records $win  ", his_date_all[first(win)], " -> ", his_date_all[last(win)],
            w == 1 ? "" : "  (overlap with previous: $(round(ov * dt / 86400, digits = 2)) days)")
    println("          keeps $(keeps[w])  ", his_date_all[first(keeps[w])], " -> ", his_date_all[last(keeps[w])],
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

# which mean density rho' is measured from. The s levels ride up and down with zeta, so the two choices differ:
#   "s_level"   rho' = rho - time mean on the same s level.
#               A level's own up-and-down motion with the surface tide gives no rho'.
#   "fixed_z"   rho' = rho - mean density profile evaluated at the depth the level is at right now.
#               Measured at a fixed depth, so stratification heaved past that depth by the surface tide does give rho'.
rho_reference = "fixed_z"

# everything energyflux_piece needs, in one bundle, so the function uses no global variables
# (that is what lets it be sent to other processes later)
cfg = (; datadir, nc_dum_save, his_files, nt_each, t_end, his_time_all, his_date_all, dt,
       windows, keeps, tiles, ndomx, ndomy, h, filters, rho_reference)

##
# helper functions and the flux calculation itself — defined on every process
@everywhere begin

    # mean profile rho_mean (on the mean level depths z_mean, both [xi, eta, s]) looked up at the depths z[xi, eta, s, time],
    # by linear interpolation in each column. Outside the mean profile (e.g. above the mean position of the top level at
    # high water) the end value is held.
    function rho_at_depth(z_mean, rho_mean, z)
        nx, ny, nz, nt = size(z)
        out = similar(z)
        @inbounds for n in 1:nt, j in 1:ny, i in 1:nx
            m = 1                                   # z_mean[m] .. z_mean[m+1] brackets the current depth; only ever moves up
            for k in 1:nz
                zz = z[i, j, k, n]
                while m < nz - 1 && z_mean[i, j, m+1] < zz
                    m += 1
                end
                w = clamp((zz - z_mean[i, j, m]) / (z_mean[i, j, m+1] - z_mean[i, j, m]), 0f0, 1f0)
                out[i, j, k, n] = rho_mean[i, j, m] + w * (rho_mean[i, j, m+1] - rho_mean[i, j, m])
            end
        end
        out
    end

    # thickness-weighted depth mean of a 4D field: sum_s(f * Hz) / sum_s(Hz)
    #   f, Hz [x, y, s, time] on the same points  ->  [x, y, 1, time]
    #   the s dimension is kept (length 1) so the result subtracts straight from f:  f .- depth_mean(f, Hz)
    function depth_mean(f, Hz)
        size(f) == size(Hz) || error("depth_mean: f is $(size(f)) but Hz is $(size(Hz))")
        sum(f .* Hz, dims = 3) ./ sum(Hz, dims = 3)
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

    # file one piece is saved to
    piece_fname(ix, iy, iw, cfg) =
        joinpath(cfg.nc_dum_save, "energyflux_w$(lpad(iw, 2, '0'))_tile_$(lpad(ix, 2, '0'))_$(lpad(iy, 2, '0')).nc")

    ##
    # the flux calculation for one piece: tile (ix, iy) over window iw.
    # Reads rho, u, v, zeta -> rho' -> hydrostatic p -> removes depth means -> time filters -> p'u', p'v' -> depth integral,
    # freeing or overwriting each big array as soon as it is no longer needed (peak ≈ 7 arrays the size of rho_t),
    # and saves the tile core over the window's kept records to one nc file in cfg.nc_dum_save. Returns the file name.
    # A piece whose file already exists is skipped unless overwrite = true, so an interrupted batch can simply be rerun.
    function energyflux_piece(ix, iy, iw, cfg; overwrite = false)
        (; his_files, nt_each, t_end, his_time_all, his_date_all, dt, windows, keeps, tiles, filters, rho_reference) = cfg
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

        # ---- read rho, u, v, zeta for this tile over this window
        #   rho_t, u, v [xi, eta, s, time]   zeta[xi, eta, time]   — all on rho points
        #   u and v are moved from their own points onto rho points as they are read (u2rho / v2rho: average of the two
        #   neighbours; the tile's outermost row/column is a copy of the nearest u/v, which lands in the halo, not the core)
        nz = NCDataset(his_files[1]) do ds; ds.dim["s_rho"] end
        println(tag, "rho_t: ", (length(t.i), length(t.j), nz, length(win)), "  ≈ ",
                round(length(t.i) * length(t.j) * nz * length(win) * 4 / 2^30, digits = 1), " GB  (u, v about the same each)")
        rho_t = Array{Float32}(undef, length(t.i), length(t.j), nz, length(win))
        u = Array{Float32}(undef, length(t.i), length(t.j), nz, length(win))
        v = Array{Float32}(undef, length(t.i), length(t.j), nz, length(win))
        zeta = Array{Float32}(undef, length(t.i), length(t.j), length(win))
        his_time = his_time_all[win]
        for (k, f) in enumerate(his_files)
            r = intersect(t_end[k]-nt_each[k]+1:t_end[k], win)   # global records of this file that fall in the window
            isempty(r) && continue
            rf = r .- (t_end[k] - nt_each[k])   # same records, numbered within the file
            it = r .- (first(win) - 1)          # ... and within the window
            NCDataset(f) do ds
                rho_t[:, :, :, it] = ds["rho"][t.i, t.j, :, rf]
                u[:, :, :, it] = u2rho(ds["u"][t.iu, t.j, :, rf])
                v[:, :, :, it] = v2rho(ds["v"][t.i, t.jv, :, rf])
                zeta[:, :, it] = ds["zeta"][t.i, t.j, rf]
            end
            k % 20 == 0 && println(tag, "read $k / $(length(his_files))")
        end

        # ---- z grid for this tile and window, same formula the model uses (set_depth.F):
        #   z = zeta + (zeta + h) * (hc*sc + Cs*h) / (h + hc)
        # Cs_r / Cs_w are the model's own stretching curves, stored as attributes in every his file.
        #   z_r[xi, eta, s, time]     depth of rho levels (where rho, u, v live)   k = 1 bottom ... nz surface
        #   z_w[xi, eta, s_w, time]   depth of layer interfaces, nz+1 of them: z_w[:,:,1,:] = -h, z_w[:,:,end,:] = zeta
        hc, Cs_r, Cs_w = NCDataset(his_files[1]) do ds
            ds.attrib["hc"], Float64.(ds.attrib["Cs_r"]), Float64.(ds.attrib["Cs_w"])
        end
        sc_r = ((1:nz) .- nz .- 0.5) ./ nz
        sc_w = ((0:nz) .- nz) ./ nz
        h_t = Float64.(cfg.h[t.i, t.j])
        # time-independent part: z0 = (hc*sc + Cs*h) / (h + hc), so that z = zeta + (zeta + h) * z0
        z0_r = (hc .* reshape(sc_r, 1, 1, :) .+ reshape(Cs_r, 1, 1, :) .* h_t) ./ (h_t .+ hc)   # [xi, eta, s]
        z0_w = (hc .* reshape(sc_w, 1, 1, :) .+ reshape(Cs_w, 1, 1, :) .* h_t) ./ (h_t .+ hc)   # [xi, eta, s_w]
        zeta4 = reshape(zeta, size(zeta, 1), size(zeta, 2), 1, :)   # [xi, eta, 1, time] so it broadcasts over s
        z_r = Float32.(zeta4 .+ (zeta4 .+ h_t) .* z0_r)
        z_w = Float32.(zeta4 .+ (zeta4 .+ h_t) .* z0_w)
        zeta4 = nothing

        # sanity checks: interfaces run from the sea bed to the free surface, and every rho level sits inside its layer
        @assert z_w[:, :, 1, :] ≈ repeat(-Float32.(h_t), 1, 1, size(z_w, 4))
        @assert z_w[:, :, end, :] ≈ zeta
        @assert all(@views z_w[:, :, 1:end-1, :] .< z_r .< z_w[:, :, 2:end, :])
        println(tag, "z_r: ", size(z_r), "  z_w: ", size(z_w), "  depth range in tile: ", extrema(h_t), " m")

        # ---- rho' relative to the chosen mean (see rho_reference in the settings)   [xi, eta, s, time]
        # time mean at each (xi, eta, s) -> rho_mean[xi, eta, s]; summed in Float64 so 700+ Float32 values don't lose precision
        rho_mean = Float32.(dropdims(sum(Float64, rho_t, dims = 4), dims = 4) ./ size(rho_t, 4))
        if rho_reference == "s_level"
            rho_t .-= rho_mean
        elseif rho_reference == "fixed_z"
            z_r_mean = Float32.(dropdims(sum(Float64, z_r, dims = 4), dims = 4) ./ size(z_r, 4))   # mean depth of each level [xi, eta, s]
            rho_t .-= rho_at_depth(z_r_mean, rho_mean, z_r)
            z_r_mean = nothing
        else
            error("rho_reference must be \"s_level\" or \"fixed_z\", got \"$rho_reference\"")
        end
        rho_p = rho_t   # rho_t was overwritten in place, so this is rho' from here on
        println(tag, "rho' ($rho_reference): range ", extrema(rho_p), " kg m-3   zeta range ", extrema(zeta), " m")
        rho_t = rho_mean = z_r = nothing   # free memory: not used below

        # ---- hydrostatic pressure perturbation from rho', integrated down from the free surface (p = 0 at z = zeta):
        #   p_w at the bottom of layer k = p_w at its top + g * rho'[k] * Hz[k]
        #   p_hyd[k] = average of the two interfaces of layer k -> pressure at the rho level
        #   Hz[xi, eta, s, time]      layer thickness [m], sums over s to h + zeta
        #   p_hyd[xi, eta, s, time]   [Pa], on the same points as rho_p
        g = 9.81f0
        Hz = diff(z_w, dims = 3)
        z_w = nothing   # free memory: only Hz is used below
        p_hyd = similar(rho_p)
        p_top = zeros(Float32, size(rho_p, 1), size(rho_p, 2), size(rho_p, 4))   # p_w at the top of the current layer; 0 at the surface
        for k in nz:-1:1
            @views dp = g .* rho_p[:, :, k, :] .* Hz[:, :, k, :]   # pressure added across layer k
            @views p_hyd[:, :, k, :] .= p_top .+ 0.5f0 .* dp
            p_top .+= dp                                           # now p_w at the bottom of layer k
        end
        println(tag, "p_hyd: ", size(p_hyd), "  range ", extrema(p_hyd), " Pa")
        rho_p = p_top = nothing   # free memory
        GC.gc()

        # ---- baroclinic parts: remove the thickness-weighted depth mean from each column at each time, in place.
        #   p_bc, u_bc, v_bc = p', u', v'   [xi, eta, s, time]   (p, u, v are all on rho points, so they share Hz)
        # check: the depth integral of each baroclinic field should be zero. Printed as the worst column's
        # |∫ f' dz| relative to that field's largest ∫ |f| dz, so "zero" means Float32 rounding (~1e-6), whatever the units
        for (name, f) in (("p'", p_hyd), ("u'", u), ("v'", v))
            scale = maximum(sum(abs.(f) .* Hz, dims = 3))
            f .-= depth_mean(f, Hz)
            resid = maximum(abs, sum(f .* Hz, dims = 3))
            println(tag, "depth integral of $name: max |∫ f' dz| = $resid   relative to max ∫ |f| dz: $(resid / scale)")
            resid / scale < 1f-4 || @warn "$(tag)depth integral of $name is not close to zero"
        end
        p_bc, u_bc, v_bc = p_hyd, u, v   # overwritten in place above
        p_hyd = u = v = nothing

        # ---- per filter band: time filter p', u', v' -> p'u', p'v' -> depth integral.
        #      One band at a time, so memory does not grow with the number of rows in `filters`.
        #   Fx["M2"], Fy["hp24"], ...   depth-integrated energy flux [W m-1], [xi, eta, time_keep], tile core only (no halo)
        #   u, v are along the grid's xi / eta directions, not yet rotated to east / north (grid_angle)
        dt_hr = dt / 3600
        t_keep = keeps[iw] .- (first(win) - 1)   # this window's kept records (keeps[iw]), numbered within the window
        flux_time = his_time[t_keep]
        Hz_keep = Hz[:, :, :, t_keep]
        Hz = nothing
        GC.gc()
        Fx = Dict{String, Array{Float32, 3}}()
        Fy = Dict{String, Array{Float32, 3}}()
        for flt in filters
            # each field is filtered over the full window, then only the kept records are used
            p_f = filter_time(p_bc, flt, dt_hr)[:, :, :, t_keep]
            u_f = filter_time(u_bc, flt, dt_hr)[:, :, :, t_keep]
            Fx[flt.name] = dropdims(sum(p_f .* u_f .* Hz_keep, dims = 3), dims = 3)[t.iloc, t.jloc, :]
            println(tag, flt.name, ":  p ", extrema(p_f), " Pa   u ", extrema(u_f), " m/s")
            u_f = nothing
            v_f = filter_time(v_bc, flt, dt_hr)[:, :, :, t_keep]
            Fy[flt.name] = dropdims(sum(p_f .* v_f .* Hz_keep, dims = 3), dims = 3)[t.iloc, t.jloc, :]
            println(tag, flt.name, ":  v ", extrema(v_f), " m/s   time-mean flux over the tile core  Fx ", mean(Fx[flt.name]),
                    "   Fy ", mean(Fy[flt.name]), " W m-1")
            p_f = v_f = nothing
            GC.gc()
        end
        p_bc = u_bc = v_bc = Hz_keep = nothing   # free memory

        # ---- save this piece (one tile core x the kept part of one window) so all pieces can be joined afterwards.
        #   xi_rho, eta_rho   where the piece sits in the full grid (global rho indices, 1-based)
        #   record            where it sits in the run (index into the whole run's his records), ocean_time the matching time
        # Written under a temporary name and renamed at the end, so a killed run never leaves a half-written piece.
        NCDataset(nc_fname * ".tmp", "c") do ds
            defDim(ds, "xi_rho", length(t.icore))
            defDim(ds, "eta_rho", length(t.jcore))
            defDim(ds, "time", length(flux_time))
            defVar(ds, "xi_rho", Int32.(t.icore), ("xi_rho",), attrib = ["long_name" => "global xi_rho index (1-based)"])
            defVar(ds, "eta_rho", Int32.(t.jcore), ("eta_rho",), attrib = ["long_name" => "global eta_rho index (1-based)"])
            defVar(ds, "record", Int32.(win[t_keep]), ("time",), attrib = ["long_name" => "index of this time in the whole run's his records (1-based)"])
            defVar(ds, "ocean_time", flux_time, ("time",), attrib = ["long_name" => "Time since 1994/01/01", "units" => "second"])
            for flt in filters
                band = flt.kind == "band" ? "band-pass $(flt.T[1])-$(flt.T[2]) hr" : "$(flt.kind)-pass $(flt.T) hr"
                defVar(ds, "Fx_$(flt.name)", Fx[flt.name], ("xi_rho", "eta_rho", "time"), attrib = [
                    "long_name" => "depth-integrated energy flux p'u' along xi, $band, Butterworth order $(flt.N)", "units" => "W m-1"])
                defVar(ds, "Fy_$(flt.name)", Fy[flt.name], ("xi_rho", "eta_rho", "time"), attrib = [
                    "long_name" => "depth-integrated energy flux p'v' along eta, $band, Butterworth order $(flt.N)", "units" => "W m-1"])
            end
            ds.attrib["tile"] = "ix = $ix of $(cfg.ndomx), iy = $iy of $(cfg.ndomy)"
            ds.attrib["window"] = "window $iw of $(length(windows)): $(his_date_all[first(win)]) -> $(his_date_all[last(win)]); kept records $(first(keeps[iw]))-$(last(keeps[iw])) of the run"
            ds.attrib["rho_reference"] = rho_reference
            ds.attrib["direction"] = "Fx, Fy are along the grid's xi / eta directions, not rotated to east / north"
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

# one piece on this process, without the workers (handy for testing a change):
# energyflux_piece(1, 1, 1, cfg; overwrite = true)

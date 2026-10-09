using NCDatasets, CairoMakie, Dates, Statistics, DSP
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")

grid_fname = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
datadir = "/home/hsinyi/roms_data/output_test/diatest"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hsinyi/figure/20260929_output/dia"   # separate from his_rejoint so the two scripts' skip checks don't interfere
mkpath(figure_path)

##
fname = joinpath(datadir, "roms_dia.20220902100000.nc")
ocean_time, up,vp = NCDataset(fname) do ds
    show(ds)
    ds["ocean_time"][:], ds["up"][:,:,:], ds["vp"][:,:,:]
end

mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
    ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
end

# time range to compute the flux for: [t_start, t_end). pad is extra time read on both sides for the filters, then discarded
t_start = DateTime(2022, 9, 2, 0)
t_end   = t_start + Day(1)
pad     = Hour(0)      # test value for diatest (only one day of files); use Day(6) on the real folder, like PAD_DAYS in dbry_make.m
file_dt = Hour(4)      # each his file holds 4 hourly records
g = 9.81f0             # m s-2
ntile = (4, 4)         # the domain is processed in ntile[1] x ntile[2] horizontal tiles, one at a time, to limit memory

# time filters applied to p', u', v' before the flux. T = cutoff period(s) in hours, N = Butterworth order.
# add or edit rows here: kind "high" / "low" takes one period, "band" takes (short, long)
filters = [
    (name = "hp28", kind = "high", T = 28,      N = 5),
    (name = "M2",   kind = "band", T = (9, 15), N = 4),
]

# filter x along its last dimension (time) with one row of `filters`; dt in hours
function filter_time(x, flt, dt)
    nd = ndims(x)
    y = reshape(permutedims(x, (nd, 1:nd-1...)), size(x, nd), :)   # butter_filters work along dim 1
    yf = flt.kind == "band" ? bandpass_butter(y, flt.T[1], flt.T[2], dt, flt.N) :
                              lowhighpass_butter(y, flt.T, dt, flt.N, flt.kind)
    return permutedims(reshape(yf, size(x, nd), size(x)[1:nd-1]...), (2:nd..., 1))
end

# split 1:n into k consecutive index ranges
tile_ranges(n, k) = [round(Int, (i - 1) * n / k) + 1:round(Int, i * n / k) for i in 1:k]

his_all   = filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(datadir))
his_start = [DateTime(match(r"\d{14}", f).match, dateformat"yyyymmddHHMMSS") for f in his_all]
his_file  = Dict(zip(his_start, his_all))            # file start time -> file name
his_files = collect(t_start - pad:file_dt:t_end + pad - file_dt)   # start times of the files needed, padding included
missing_start = filter(t -> !haskey(his_file, t), his_files)
isempty(missing_start) || error("his files missing in the padded time range (the filters need an unbroken record): $missing_start")
println("his files to read: ", length(his_files))

fname = joinpath(datadir, his_file[his_files[1]])
theta_s, theta_b, hc, nx, ny, N = NCDataset(fname) do ds
    ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"], ds.dim["xi_rho"], ds.dim["eta_rho"], ds.dim["s_rho"]
end

his_time = reduce(vcat, [NCDataset(ds -> ds["ocean_time"][:], joinpath(datadir, his_file[t])) for t in his_files])   # seconds
nt = length(his_time)
dt = (his_time[2] - his_time[1]) / 3600   # hours
all(diff(his_time) .≈ dt * 3600) || error("his records are not evenly spaced in time")
rec_time = his_files[1] .+ Second.(round.(Int, his_time .- his_time[1]))   # DateTime of each record (first record = file name time)
t_keep = findall(t -> t_start <= t < t_end, rec_time)   # records kept after filtering, i.e. without the padding
flux_time = rec_time[t_keep]

# depth-integrated energy flux (W/m) at rho points for each filter: Fx[name], Fy[name] are (xi, eta, time), time = flux_time
Fx = Dict(flt.name => zeros(Float32, nx, ny, length(t_keep)) for flt in filters)
Fy = Dict(flt.name => zeros(Float32, nx, ny, length(t_keep)) for flt in filters)

## tile by tile: read the whole padded record, rho' -> p', u' v', filter, flux
# to try a single tile in the REPL, set ix, iy = tile_ranges(nx, ntile[1])[1], tile_ranges(ny, ntile[2])[1] and run the lines of the loop body
for iy in tile_ranges(ny, ntile[2]), ix in tile_ranges(nx, ntile[1])
    println("tile xi = $ix, eta = $iy")
    # u / v points needed to bring u, v onto this tile's rho points (u point i sits between rho points i and i+1)
    ixu = max(first(ix) - 1, 1):min(last(ix), nx - 1)
    iyv = max(first(iy) - 1, 1):min(last(iy), ny - 1)

    zeta = zeros(Float32, length(ix), length(iy), nt)
    rho  = zeros(Float32, length(ix), length(iy), N, nt)
    u_bc = zeros(Float32, length(ixu), length(iy), N, nt)
    v_bc = zeros(Float32, length(ix), length(iyv), N, nt)
    for (i, t) in enumerate(his_files)
        NCDataset(joinpath(datadir, his_file[t])) do ds
            it = (1:ds.dim["time"]) .+ (i - 1) * ds.dim["time"]
            zeta[:, :, it]   = ds["zeta"][ix, iy, :]
            rho[:, :, :, it] = ds["rho"][ix, iy, :, :]
            # u (xi_u, eta_u, s_rho, time) - ubar (xi_u, eta_u, time) -> baroclinic velocity, still on the u/v grids
            # named u_bc/v_bc because up/vp are already the pressure flux read from the dia file
            u_bc[:, :, :, it] = ds["u"][ixu, iy, :, :] .- reshape(ds["ubar"][ixu, iy, :], length(ixu), length(iy), 1, :)
            v_bc[:, :, :, it] = ds["v"][ix, iyv, :, :] .- reshape(ds["vbar"][ix, iyv, :], length(ix), length(iyv), 1, :)
        end
    end

    # u' v' on rho points, same rule as u2rho / v2rho: average the two neighbours, copy the nearest one at the domain edge
    u_bc = 0.5f0 .* (u_bc[clamp.(ix .- 1, 1, nx - 1) .- first(ixu) .+ 1, :, :, :] .+ u_bc[clamp.(ix, 1, nx - 1) .- first(ixu) .+ 1, :, :, :])
    v_bc = 0.5f0 .* (v_bc[:, clamp.(iy .- 1, 1, ny - 1) .- first(iyv) .+ 1, :, :] .+ v_bc[:, clamp.(iy, 1, ny - 1) .- first(iyv) .+ 1, :, :])

    # zgrid in roms
    z_r = zeros(Float32, length(ix), length(iy), N, nt)       # depth of rho levels (m, negative down)
    z_w = zeros(Float32, length(ix), length(iy), N + 1, nt)   # depth of w levels: z_w[:,:,1,t] = -h, z_w[:,:,end,t] = zeta
    for t in 1:nt
        # zlevs3 returns (N, xi, eta) -> permute to (xi, eta, N)
        z_dum, _ = zlevs3(h[ix, iy], zeta[:, :, t], theta_s, theta_b, hc, N, "r", "new2008")
        z_r[:, :, :, t] = permutedims(z_dum, (2, 3, 1))
        z_dum, _ = zlevs3(h[ix, iy], zeta[:, :, t], theta_s, theta_b, hc, N, "w", "new2008")
        z_w[:, :, :, t] = permutedims(z_dum, (2, 3, 1))
    end
    Hz = diff(z_w, dims=3)   # layer thickness (xi, eta, s_rho, time), sum over s = h + zeta

    # rho' = rho - time mean over the whole padded record on each s-level (same as rho_bar in dbry_fun.jl)
    rhop = rho .- mean(rho, dims=4)

    # hydrostatic pressure perturbation (Pa) at rho levels: g * trapezoid integral of rho' from the top rho level down
    # p_hyd = 0 at the top rho level (k = N); a constant offset per column drops out once the depth mean of p is removed
    p_hyd = zeros(Float32, size(rhop))
    for k in N-1:-1:1
        @views p_hyd[:, :, k, :] .= p_hyd[:, :, k+1, :] .+
            g .* 0.5f0 .* (rhop[:, :, k, :] .+ rhop[:, :, k+1, :]) .* (z_r[:, :, k+1, :] .- z_r[:, :, k, :])
    end

    # baroclinic pressure: remove the thickness-weighted depth mean of p_hyd in each column and time
    p_bc = p_hyd .- sum(p_hyd .* Hz, dims=3) ./ sum(Hz, dims=3)

    # energy flux: filter p', u', v' in time first, then integrate the product over depth (same order as dbry_fun.jl).
    # done one s-level at a time so the filtered 4D fields are never held in memory
    for flt in filters
        Fx_tile = zeros(Float32, length(ix), length(iy), length(t_keep))
        Fy_tile = zeros(Float32, length(ix), length(iy), length(t_keep))
        lk = ReentrantLock()
        Threads.@threads for k in 1:N
            p_f = filter_time(p_bc[:, :, k, :], flt, dt)[:, :, t_keep]
            fx = p_f .* filter_time(u_bc[:, :, k, :], flt, dt)[:, :, t_keep] .* Hz[:, :, k, t_keep]
            fy = p_f .* filter_time(v_bc[:, :, k, :], flt, dt)[:, :, t_keep] .* Hz[:, :, k, t_keep]
            lock(lk) do
                Fx_tile .+= fx
                Fy_tile .+= fy
            end
        end
        Fx[flt.name][ix, iy, :] = Fx_tile
        Fy[flt.name][ix, iy, :] = Fy_tile
    end
end

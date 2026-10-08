using NCDatasets, CairoMakie, Dates, Statistics
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

# his files to compute the flux for: start time inside [t_start, t_end)
t_start = DateTime(2022, 9, 2, 8)
t_end   = t_start + Hour(8)
half_window = Hour(8)  # test value for diatest (only one day of files); use Day(1) on the real folder. rhobar for each file = mean of rho over the files within +- half_window of it
file_dt = Hour(4)      # each his file holds 4 hourly records
g = 9.81f0             # m s-2

his_all   = filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(datadir))
his_start = [DateTime(match(r"\d{14}", f).match, dateformat"yyyymmddHHMMSS") for f in his_all]
his_file  = Dict(zip(his_start, his_all))            # file start time -> file name
his_files = sort(filter(t -> t_start <= t < t_end, his_start))   # start times of the files to process
println("his files to process: ", length(his_files))

fname = joinpath(datadir, his_file[his_files[1]])
theta_s, theta_b, hc, nx, ny, N = NCDataset(fname) do ds
    ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"], ds.dim["xi_rho"], ds.dim["eta_rho"], ds.dim["s_rho"]
end

## file by file: moving-mean rhobar, z grid, u' v', rho'
# rho_sums keeps, for each file currently inside the moving window, (sum of rho over its records, number of records),
# so every file's rho is read once for the mean and the window slides without holding the full 4D rho
rho_sums = Dict{DateTime, Tuple{Array{Float32, 3}, Int}}()

# to try a single file in the REPL, set t0 = his_files[1] and run the lines of the loop body
for t0 in his_files
    # rhobar: mean of rho over the files starting within [t0 - half_window, t0 + half_window], on each s-level
    win = t0 - half_window:file_dt:t0 + half_window
    missing_start = filter(t -> !haskey(his_file, t), win)
    if !isempty(missing_start)
        @warn "skip $(his_file[t0]): his files missing in its averaging window" missing_start
        continue
    end
    foreach(t -> delete!(rho_sums, t), filter(t -> t < first(win), collect(keys(rho_sums))))
    for t in win
        haskey(rho_sums, t) && continue
        rho_sums[t] = NCDataset(joinpath(datadir, his_file[t])) do ds
            rho = ds["rho"][:,:,:,:]
            dropdims(sum(rho, dims=4), dims=4), size(rho, 4)
        end
    end
    rhobar = sum(first(rho_sums[t]) for t in win) ./ sum(last(rho_sums[t]) for t in win)   # (xi, eta, s_rho)

    his_time, zeta, u, v, rho, ubar, vbar = NCDataset(joinpath(datadir, his_file[t0])) do ds
        ds["ocean_time"][:], ds["zeta"][:,:,:], ds["u"][:,:,:,:], ds["v"][:,:,:,:], ds["rho"][:,:,:,:],
        ds["ubar"][:,:,:], ds["vbar"][:,:,:]
    end
    nt = length(his_time)

    # zgrid in roms
    z_r = zeros(Float32, nx, ny, N, nt)       # depth of rho levels (m, negative down)
    z_w = zeros(Float32, nx, ny, N + 1, nt)   # depth of w levels: z_w[:,:,1,t] = -h, z_w[:,:,end,t] = zeta
    for t in 1:nt
        # zlevs3 returns (N, xi, eta) -> permute to (xi, eta, N)
        z_dum, _ = zlevs3(h, zeta[:, :, t], theta_s, theta_b, hc, N, "r", "new2008")
        z_r[:, :, :, t] = permutedims(z_dum, (2, 3, 1))
        z_dum, _ = zlevs3(h, zeta[:, :, t], theta_s, theta_b, hc, N, "w", "new2008")
        z_w[:, :, :, t] = permutedims(z_dum, (2, 3, 1))
    end
    Hz = diff(z_w, dims=3)   # layer thickness (xi, eta, s_rho, time), sum over s = h + zeta

    # u (xi_u, eta_u, s_rho, time) - ubar (xi_u, eta_u, time) -> baroclinic velocity, still on the u/v grids
    # named u_bc/v_bc because up/vp are already the pressure flux read from the dia file
    u_bc = u .- reshape(ubar, size(ubar, 1), size(ubar, 2), 1, size(ubar, 3))
    v_bc = v .- reshape(vbar, size(vbar, 1), size(vbar, 2), 1, size(vbar, 3))

    rhop = rho .- rhobar

    # hydrostatic pressure perturbation (Pa) at rho levels: g * trapezoid integral of rho' from the top rho level down
    # p_hyd = 0 at the top rho level (k = N); a constant offset per column drops out once the depth mean of p is removed
    p_hyd = zeros(Float32, size(rhop))
    for k in N-1:-1:1
        @views p_hyd[:, :, k, :] .= p_hyd[:, :, k+1, :] .+
            g .* 0.5f0 .* (rhop[:, :, k, :] .+ rhop[:, :, k+1, :]) .* (z_r[:, :, k+1, :] .- z_r[:, :, k, :])
    end

    # baroclinic pressure: remove the thickness-weighted depth mean of p_hyd in each column and time
    p_bc = p_hyd .- sum(p_hyd .* Hz, dims=3) ./ sum(Hz, dims=3)

    ## depth integrate check
    # depth integrals of u', v', p' should be ~0; printed next to the integral of |f| for scale
    for (name, f, dz) in (("u_bc", u_bc, rho2u(Hz)), ("v_bc", v_bc, rho2v(Hz)), ("p_bc", p_bc, Hz))
        int_f = sum(f .* dz, dims=3)
        int_abs = sum(abs.(f) .* dz, dims=3)
        println(name, ": max |int f dz| = ", maximum(abs.(int_f)), "   max int |f| dz = ", maximum(int_abs),
                "   ratio = ", maximum(abs.(int_f)) / maximum(int_abs))
    end

    # the energy flux for this file goes here
end

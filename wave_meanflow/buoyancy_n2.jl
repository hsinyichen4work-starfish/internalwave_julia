#= buoyancy_n2.jl
Buoyancy B and buoyancy frequency N² of the window means (step 3 of wave_mean_flow.jl).
Needs `using NCDatasets, Dates, Printf, GibbsSeaWater`, function/zlevs3.jl (load_all_hpc.jl) and meanflow_window.jl
(window_file, window_done, write_window_mean).

ROMS rho of this run = potential density at surface pressure - rho0 (rho0 = 1027.5, global attribute of the his
files): rho + rho0 matches GSW sigma-0 to < 0.01 kg m-3 at 0-4800 m (in-situ density at 4000 m is ~1046). So
  B      = -g rho / rho0                                      [m s-2], rho points (xi_rho, eta_rho, s_rho)
  N2     = gsw_nsquared(SA, CT, p, lat)                       [s-2],   w points  (xi_rho, eta_rho, s_w)
           SA from salt (practical) and p, CT from temp (potential temperature), p from z_r (gsw_p_from_z);
           local reference pressure, so it is right at depth too.
  N2rho  = -g / rho0 (rho[k+1] - rho[k]) / (z_r[k+1] - z_r[k])  [s-2],   w points, from ROMS rho; surface-referenced
           density, so it drifts from N2 at depth (thermobaric); kept as a check.
N2 / N2rho are found on the N-1 interior w levels (between rho levels k and k+1 -> w level k+1); the bottom (w 1) and
surface (w N+1) levels have no rho level on one side and take the value of the nearest interior level.
z_r from the window mean zeta and h (zlevs3, "new2008" = Vtransform 2 / Vstretching 4, checked against Cs_w of the
his files). k = 1 is the bottom, as in ROMS.
=#

const G_ACC = 9.81   # [m s-2], as in ROMS

buoyancy(rho; rho0, g = G_ACC) = Float32.(-g .* rho ./ rho0)

# depths of the rho (or w) levels in Julia order (xi, eta, k)
z_levels(h, zeta, sp, type) = permutedims(zlevs3(h, zeta, sp.theta_s, sp.theta_b, sp.hc, sp.N, type, sp.scoord)[1], (2, 3, 1))

# N² from ROMS rho on the w levels (xi, eta, N + 1)
function n2_rho(rho, zr; rho0, g = G_ACC)
    nx, ny, N = size(rho)
    n2 = Array{Float32}(undef, nx, ny, N + 1)
    @views n2[:, :, 2:N] .= -g / rho0 .* (rho[:, :, 2:N] .- rho[:, :, 1:N-1]) ./ (zr[:, :, 2:N] .- zr[:, :, 1:N-1])
    @views n2[:, :, 1] .= n2[:, :, 2]; @views n2[:, :, N+1] .= n2[:, :, N]
    return n2
end

# N² from TEOS-10 (gsw_nsquared) on the w levels (xi, eta, N + 1), column by column
function n2_gsw(temp, salt, zr, lon, lat)
    nx, ny, N = size(temp)
    n2 = Array{Float32}(undef, nx, ny, N + 1)
    p = zeros(N); SA = zeros(N); CT = zeros(N); la = zeros(N)
    for j in 1:ny, i in 1:nx
        for k in 1:N
            p[k]  = gsw_p_from_z(zr[i, j, k], lat[i, j])
            SA[k] = gsw_sa_from_sp(salt[i, j, k], p[k], lon[i, j], lat[i, j])
            CT[k] = gsw_ct_from_pt(SA[k], temp[i, j, k])
        end
        la .= lat[i, j]
        c, _ = gsw_nsquared(SA, CT, p, la)
        @views n2[i, j, 2:N] .= c
        n2[i, j, 1] = c[1]; n2[i, j, N+1] = c[end]
    end
    return n2
end

#= ---------------------------------------------------------------------------------------------------------------
B, N2, N2rho of every window, from the window means of rho, temp, salt, zeta in mean_root/<var>/ (window_means).
  files = window_bn2(win, mean_root; windows = 1:win.nwin, grid_file, his_file)
  files["B"][w], files["N2"][w], files["N2rho"][w]
Output: mean_root/B/, mean_root/N2/, mean_root/N2rho/, one file per window, named and checked like the means
(existing complete files are skipped). his_file = any his file of the run (rho0 and the s-coordinate are read there).
--------------------------------------------------------------------------------------------------------------- =#
function window_bn2(win, mean_root; windows = 1:win.nwin, grid_file, his_file, overwrite = false,
                    t_ref = DateTime(1994, 1, 1), verbose = true)
    outs = ("B", "N2", "N2rho")
    fn(v, w) = window_file(joinpath(mean_root, v), v, w, win.t0[w])
    for v in outs; mkpath(joinpath(mean_root, v)); end
    todo = [w for w in windows if overwrite || !all(v -> window_done(fn(v, w), v, win, w, t_ref), outs)]
    verbose && @printf("B / N2: %d of %d windows to do\n", length(todo), length(windows))
    isempty(todo) && return Dict(v => [fn(v, w) for w in windows] for v in outs)

    rho0, sp = NCDataset(his_file) do ds
        a = ds.attrib
        Float64(a["rho0"]), (theta_s = Float64(a["theta_s"]), theta_b = Float64(a["theta_b"]), hc = Float64(a["hc"]),
                             N = ds.dim["s_rho"], scoord = "new2008", Cs_w = Float64.(a["Cs_w"]))
    end
    _, Cs = zlevs3(fill(1000.0, 2, 2), zeros(2, 2), sp.theta_s, sp.theta_b, sp.hc, sp.N, "w", sp.scoord)
    maximum(abs.(Cs .- sp.Cs_w)) < 1e-10 || error("s-coordinate is not $(sp.scoord): Cs differs from Cs_w of $his_file")
    h, lon, lat = NCDataset(g -> (Float64.(g["h"].var[:, :]), Float64.(g["lon_rho"].var[:, :]), Float64.(g["lat_rho"].var[:, :])), grid_file)

    rd(v, w) = NCDataset(ds -> (a = ds[mean_name(v)]; a.var[ntuple(_ -> Colon(), ndims(a) - 1)..., 1]), fn_in(v, w))
    fn_in(v, w) = window_file(joinpath(mean_root, mean_name(v)), mean_name(v), w, win.t0[w])   # window mean of his var v
    for w in todo, v in ("rho", "temp", "salt", "zeta")
        window_done(fn_in(v, w), mean_name(v), win, w, t_ref) || error("window $w: no mean of $v in $(joinpath(mean_root, v)); run window_means first")
    end

    rdims, wdims = ("xi_rho", "eta_rho", "s_rho", "time"), ("xi_rho", "eta_rho", "s_w", "time")
    att = Dict("B"     => ["long_name" => "buoyancy -g rho / rho0 of the window mean rho", "units" => "meter second-2"],
               "N2"    => ["long_name" => "buoyancy frequency squared, TEOS-10 gsw_nsquared of the window mean temp, salt",
                           "units" => "second-2"],
               "N2rho" => ["long_name" => "buoyancy frequency squared, -g/rho0 drho/dz of the window mean ROMS rho",
                           "units" => "second-2"])
    t0 = time()
    for w in todo
        rho, temp, salt, zeta = rd("rho", w), rd("temp", w), rd("salt", w), Float64.(rd("zeta", w))
        nrec = NCDataset(ds -> Int(ds["nrec"][1]), fn_in("rho", w))
        zr = z_levels(h, zeta, sp, "r")
        write_window_mean(fn("B", w), "B", buoyancy(rho; rho0), rdims, att["B"], win, w, nrec, t_ref)
        write_window_mean(fn("N2rho", w), "N2rho", n2_rho(rho, zr; rho0), wdims, att["N2rho"], win, w, nrec, t_ref)
        write_window_mean(fn("N2", w), "N2", n2_gsw(temp, salt, zr, lon, lat), wdims, att["N2"], win, w, nrec, t_ref)
        verbose && @printf("  window %d (%s): B, N2, N2rho written, %.1f min\n", w, win.t0[w], (time() - t0) / 60)
        GC.gc()
    end
    return Dict(v => [fn(v, w) for w in windows] for v in outs)
end

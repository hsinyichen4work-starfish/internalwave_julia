#= getting_modes.jl
Vertical modes of every window from its background N² (step 4 of wave_mean_flow.jl).
Needs `using NCDatasets, Dates, Printf, LinearAlgebra`, function/zlevs3.jl (load_all_hpc.jl), meanflow_window.jl
(window_file, window_done) and buoyancy_n2.jl (z_levels).

Same eigenvalue problem as pass 1 of DFF/bcef_modes_puv.jl, but N² is the TEOS-10 one of the window (N2/, step 3,
gsw_nsquared) instead of -g/rho0 drho/dz of ROMS rho (that one is too small at depth: ROMS rho is surface-referenced):
  W'' + (N² / c²) W = 0,   W = 0 at the bottom and the surface (rigid lid)
  (Siyanbola et al. 2024 Eq. 10; Gerkema Ch. 5.1, hydrostatic). Levels: z_r, z_w of the window mean zeta.
  W on the w levels, second differences over the layers, N² on the interior w levels 2..N (held >= N2min, so a mixed
  or slightly unstable layer does not break the problem). A W = λ B W with B = diag(N² Δz) is made symmetric by
  y = B^½ W; the nmode smallest eigenvalues λn = 1 / cn².
Output per mode n:
  cn      eigenspeed [m s-1] (no rotation)                                          (xi_rho, eta_rho, mode)
  phi     φn = dΦn/dz on the rho levels: structure of p', u', v' (Eq. 11);
          (1/H) ∫ φm φn dz = δmn (Eq. 12), φn(surface) > 0                         (xi_rho, eta_rho, s_rho, mode)
  Phi     Φn on the w levels: structure of w' and of the vertical displacement; Φn = 0 at bottom and surface,
          scaled so that φn = dΦn/dz exactly [m]; then (1/H) ∫ N² Φm Φn dz = cn² δmn   (xi_rho, eta_rho, s_w, mode)
The discrete φn are exactly orthogonal (Σ φm φn Hz = Wmᵀ A Wn); checked on sampled points each window.
Checked against constant N on a stretched grid in bcef_modes_puv.jl: cn = N H / (nπ), φn = √2 cos(nπ z / H).
=#

using LinearAlgebra

# modes of one column: zr [N], zw [N+1], N2 on the interior w levels [N-1] (w levels 2..N)
function column_modes(zr, zw, n2w; nmode = 3, N2min = 1e-7)
    nz = length(zr)
    Hz = diff(zw)                                   # layer thickness [N]
    dzc = diff(zr)                                  # distance between rho levels = around interior w levels [N-1]
    B = max.(n2w, N2min) .* dzc
    d = (1 ./ Hz[2:nz] .+ 1 ./ Hz[1:nz-1]) ./ B
    e = (-1 ./ Hz[2:nz-1]) ./ sqrt.(B[1:end-1] .* B[2:end])
    F = eigen(SymTridiagonal(d, e), 1:nmode)        # the nmode smallest eigenvalues, increasing
    c = 1 ./ sqrt.(F.values)                        # c1 > c2 > c3 ...
    φ = zeros(nz, nmode); Φ = zeros(nz + 1, nmode)
    H = sum(Hz)
    for n in 1:nmode
        W = [0; F.vectors[:, n] ./ sqrt.(B); 0]     # Wn on the w levels, 0 at bottom and surface
        φn = diff(W) ./ Hz                          # Eq. (11)
        a = sqrt(sum(φn .^ 2 .* Hz) / H)            # Eq. (12) normalization
        s = φn[end] < 0 ? -1 / a : 1 / a            # and positive at the surface
        φ[:, n] .= s .* φn
        Φ[:, n] .= s .* W                           # same factor: φn = dΦn/dz stays exact
    end
    return c, φ, Φ
end

# modes of every column of one window
function all_modes(zr, zw, n2; nmode = 3, N2min = 1e-7)
    nx, ny, nz = size(zr)
    φ = zeros(Float32, nx, ny, nz, nmode); Φ = zeros(Float32, nx, ny, nz + 1, nmode); c = zeros(Float32, nx, ny, nmode)
    for j in 1:ny, i in 1:nx
        cc, ph, Ph = column_modes(Float64.(zr[i, j, :]), Float64.(zw[i, j, :]), Float64.(n2[i, j, 2:nz]); nmode, N2min)
        c[i, j, :] .= cc; φ[i, j, :, :] .= ph; Φ[i, j, :, :] .= Ph
    end
    return c, φ, Φ
end

# worst |(1/H) ∫ φm φn dz - δmn| over a sample of columns deeper than hmin
function mode_orthogonality(φ, zw, h; step = 25, hmin = 50)
    worst = 0.0
    for j in 1:step:size(φ, 2), i in 1:step:size(φ, 1)
        h[i, j] > hmin || continue
        hz = diff(Float64.(zw[i, j, :])); P = Float64.(φ[i, j, :, :])
        worst = max(worst, maximum(abs.((P .* hz)' * P ./ sum(hz) - I)))
    end
    return worst
end

#= ---------------------------------------------------------------------------------------------------------------
Modes of every window, from N2/ (step 3) and the window mean zeta.
  files = window_modes(win, mean_root; windows = 1:win.nwin, grid_file, his_file, nmode = 3)
  files[w] = mean_root/modes<nmode>/modes<nmode>_win<NNN>_<t0>.nc with phi, Phi, c (+ nrec, t_start, t_end, ocean_time)
Existing complete files are skipped (window_done on "c"). his_file = any his file of the run (s-coordinate).
--------------------------------------------------------------------------------------------------------------- =#
function window_modes(win, mean_root; windows = 1:win.nwin, grid_file, his_file, nmode = 3, N2min = 1e-7,
                      overwrite = false, t_ref = DateTime(1994, 1, 1), verbose = true)
    tag = "modes$nmode"
    out = joinpath(mean_root, tag); mkpath(out)
    fn(w) = window_file(out, tag, w, win.t0[w])
    fn_in(v, w) = window_file(joinpath(mean_root, mean_name(v)), mean_name(v), w, win.t0[w])   # window mean of his var v
    for w in windows; rm(fn(w) * ".tmp"; force = true); end
    todo = [w for w in windows if overwrite || !window_done(fn(w), "c", win, w, t_ref)]
    verbose && @printf("modes (%d): %d of %d windows to do\n", nmode, length(todo), length(windows))
    isempty(todo) && return [fn(w) for w in windows]
    for w in todo, v in ("N2", "zeta")
        window_done(fn_in(v, w), mean_name(v), win, w, t_ref) || error("window $w: no $(mean_name(v)) in $(joinpath(mean_root, mean_name(v))); run steps 2-3 first")
    end

    sp = NCDataset(his_file) do ds
        a = ds.attrib
        (theta_s = Float64(a["theta_s"]), theta_b = Float64(a["theta_b"]), hc = Float64(a["hc"]), N = ds.dim["s_rho"],
         scoord = "new2008")
    end
    nmode <= sp.N - 1 || error("nmode = $nmode but only $(sp.N - 1) interior w levels")
    h = NCDataset(g -> Float64.(g["h"].var[:, :]), grid_file)
    rd(v, w) = NCDataset(ds -> (a = ds[mean_name(v)]; a.var[ntuple(_ -> Colon(), ndims(a) - 1)..., 1]), fn_in(v, w))
    sec(t) = Dates.value(Millisecond(t - t_ref)) / 1000
    tunits = "seconds since " * Dates.format(t_ref, "yyyy-mm-dd HH:MM:SS")

    t0 = time()
    for w in todo
        zeta = Float64.(rd("zeta", w)); n2 = rd("N2", w)
        nrec = NCDataset(ds -> Int(ds["nrec"][1]), fn_in("N2", w))
        zr, zw = z_levels(h, zeta, sp, "r"), z_levels(h, zeta, sp, "w")
        c, φ, Φ = all_modes(zr, zw, n2; nmode, N2min)
        orth = mode_orthogonality(φ, zw, h)
        zr = zw = n2 = nothing; GC.gc()                 # free before writing (login node: 8 GB virtual memory)

        NCDataset(fn(w) * ".tmp", "c"; attrib = ["title" => "vertical modes of the window mean N2 (TEOS-10)",
                                                "window" => w, "window_length_h" => win.window_length,
                                                "window_overlap" => win.window_overlap, "nmode" => nmode,
                                                "N2min" => N2min, "orthogonality_error" => orth,
                                                "created" => string(now())]) do ds
            defDim(ds, "xi_rho", size(c, 1)); defDim(ds, "eta_rho", size(c, 2))
            defDim(ds, "s_rho", sp.N); defDim(ds, "s_w", sp.N + 1); defDim(ds, "mode", nmode); defDim(ds, "time", 1)
            vc = defVar(ds, "c", Float32, ("xi_rho", "eta_rho", "mode");
                   attrib = ["long_name" => "mode eigenspeed (no rotation)", "units" => "meter second-1"])
            vp = defVar(ds, "phi", Float32, ("xi_rho", "eta_rho", "s_rho", "mode");
                   attrib = ["long_name" => "p, u, v mode structure phi_n = dPhi_n/dz, (1/H) int phi_m phi_n dz = delta_mn, positive at the surface"])
            vP = defVar(ds, "Phi", Float32, ("xi_rho", "eta_rho", "s_w", "mode");
                   attrib = ["long_name" => "w / displacement mode structure Phi_n, 0 at bottom and surface, (1/H) int N2 Phi_m Phi_n dz = c_n^2 delta_mn",
                             "units" => "meter"])
            for n in 1:nmode                            # one mode at a time: no full-size copies
                vc[:, :, n] = c[:, :, n]; vp[:, :, :, n] = φ[:, :, :, n]; vP[:, :, :, n] = Φ[:, :, :, n]
            end
            defVar(ds, "ocean_time", [sec(win.tmid[w])], ("time",); attrib = ["units" => tunits, "long_name" => "window centre"])
            defVar(ds, "t_start", [sec(win.t0[w])], ("time",); attrib = ["units" => tunits, "long_name" => "first record of the window"])
            defVar(ds, "t_end", [sec(win.t1[w])], ("time",); attrib = ["units" => tunits, "long_name" => "last record of the window"])
            defVar(ds, "nrec", Int32[nrec], ("time",); attrib = ["long_name" => "his records averaged"])
        end
        mv(fn(w) * ".tmp", fn(w); force = true)
        if verbose
            deep = h .> 1000
            @printf("  window %d (%s): c1 median %.2f m/s (h > 1000 m: %.2f), c%d median %.2f; orthogonality %.1e, %.1f min\n",
                    w, win.t0[w], median(c[:, :, 1]), median(c[:, :, 1][deep]), nmode, median(c[:, :, nmode]), orth,
                    (time() - t0) / 60)
        end
        GC.gc()
    end
    return [fn(w) for w in windows]
end

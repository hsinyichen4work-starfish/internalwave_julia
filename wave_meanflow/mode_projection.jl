#= mode_projection.jl
Wave parts ρ′, u′, v′ at a fixed depth, p′, and their projection on the window's vertical modes, computed on the fly
from the his files at every hourly record (step 5 of wave_mean_flow.jl). Nothing 3-D is saved, only the modal
amplitudes (2-D per mode and record).
Needs `using NCDatasets, Dates, Printf`, records_by_file (time_dir.jl), u2rho / v2rho (uv2rho.jl), meanflow_window.jl
(window_file, window_done) and wave_perturb.jl (wave_owner). Inputs: window means of rho, u, v, zeta (section 2) and
the modes (section 4) of the window.

For a record at time t in window w (with overlap: the window with the nearest centre, wave_owner):
  levels   z_r(t) = ζ(t) + (ζ(t) + h) z0_r,  Hz(t) = (ζ(t) + h) Δz0_w   (Vtransform 2, z0 fixed in time), H = h + ζ(t)
  fixed z  ρ′ = ρ(t) − ρ̄_w(z_r(t)),  u′ = u(t) − ū_w(z_r(t)),  v′ = v(t) − v̄_w(z_r(t)):
           the window means, on the levels of the window mean ζ̄_w, interpolated linearly in each column to the
           record's depths (outside the mean profile the end value is held), so the tidal heave of the s-levels
           through the mean profile is not counted as a wave. u, v moved to rho points first (u2rho, v2rho);
           along the grid xi / eta (not rotated).
  p′       hydrostatic from ρ′, integrated down from the free surface (p′ = 0 at z = ζ):  p′_k = Σ_{above} g ρ′ Hz + ½ g ρ′_k Hz_k
  baroclinic: the thickness-weighted depth mean removed from p′, u′, v′
  modes    pn = (1/H) ∫ p′ φn dz,  un, vn alike (Siyanbola et al. 2024 Eq. 15), φn of window w (on its mean levels;
           taken per s-level, as in DFF/bcef_modes_puv.jl). Depth-integrated flux = Σn H pn un, no cross-mode terms.
As DFF/bcef_modes_puv.jl pass 2, except: per-window means and modes (not one record mean), N² from TEOS-10, and u′, v′
also at a fixed depth (there u, v had only the depth mean removed).

Output: mean_root/proj<nmode>/proj<nmode>_win<NNN>_<t0>.nc, one file per window, all its records:
  p_1..p_N [Pa], u_1..u_N, v_1..v_N [m s-1]  (xi_rho, eta_rho, time), ocean_time (every record), t_start, t_end, nrec
  pvar_mode  window mean of H pn²       (xi_rho, eta_rho, mode)   [Pa² m]   -> share of p′ variance per mode
  pvar_total window mean of ∫ p′² dz    (xi_rho, eta_rho)         [Pa² m]
~23 MB per record for 10 modes (~1.7 GB per 3-day window). Written under .tmp, renamed when complete; skipped on rerun.
=#

# stretching z0 of the rho and w levels (Vtransform 2): z = ζ + (ζ + h) z0, fixed in time
function s_stretch(his_file, h)
    hc, Cs_r, Cs_w, N = NCDataset(his_file) do ds
        Float64(ds.attrib["hc"]), Float64.(ds.attrib["Cs_r"]), Float64.(ds.attrib["Cs_w"]), ds.dim["s_rho"]
    end
    sc_r = ((1:N) .- N .- 0.5) ./ N; sc_w = ((0:N) .- N) ./ N
    z0_r = Float32.((hc .* reshape(sc_r, 1, 1, :) .+ reshape(Cs_r, 1, 1, :) .* h) ./ (h .+ hc))
    z0_w = Float32.((hc .* reshape(sc_w, 1, 1, :) .+ reshape(Cs_w, 1, 1, :) .* h) ./ (h .+ hc))
    return z0_r, diff(z0_w, dims = 3)                # z0_r (xi, eta, N), Δz0_w (xi, eta, N)
end

# u (xi_u) / v (eta_v) moved to rho points in place, as u2rho / v2rho (edges copied from the nearest u / v point)
function u2rho!(ur, ub)
    nx = size(ur, 1)
    @inbounds for k in axes(ur, 3), j in axes(ur, 2)
        ur[1, j, k] = ub[1, j, k]; ur[nx, j, k] = ub[nx-1, j, k]
        for i in 2:nx-1; ur[i, j, k] = 0.5f0 * (ub[i-1, j, k] + ub[i, j, k]); end
    end
    return ur
end
function v2rho!(vr, vb)
    ny = size(vr, 2)
    @inbounds for k in axes(vr, 3), i in axes(vr, 1)
        vr[i, 1, k] = vb[i, 1, k]; vr[i, ny, k] = vb[i, ny-1, k]
        for j in 2:ny-1; vr[i, j, k] = 0.5f0 * (vb[i, j-1, k] + vb[i, j, k]); end
    end
    return vr
end

# one record: modal amplitudes P, U, V (xi, eta, mode) and ∫p′² dz (xi, eta), column by column
function project_record!(P, U, V, ptot, ρ, u, v, ζ, h, z0_r, dz0, zbar, ρbar, ubar, vbar, φ; g = 9.81f0)
    nx, ny, N = size(ρ); nm = size(φ, 4)
    ρp = zeros(Float32, N); up = zeros(Float32, N); vp = zeros(Float32, N); p = zeros(Float32, N); hz = zeros(Float32, N)
    @inbounds for j in 1:ny, i in 1:nx
        H = ζ[i, j] + h[i, j]
        m = 1
        for k in 1:N                                  # fixed-depth wave parts (one set of weights for ρ, u, v)
            zz = ζ[i, j] + H * z0_r[i, j, k]
            while m < N - 1 && zbar[i, j, m+1] < zz; m += 1; end
            w = clamp((zz - zbar[i, j, m]) / (zbar[i, j, m+1] - zbar[i, j, m]), 0f0, 1f0)
            ρp[k] = ρ[i, j, k] - (ρbar[i, j, m] + w * (ρbar[i, j, m+1] - ρbar[i, j, m]))
            up[k] = u[i, j, k] - (ubar[i, j, m] + w * (ubar[i, j, m+1] - ubar[i, j, m]))
            vp[k] = v[i, j, k] - (vbar[i, j, m] + w * (vbar[i, j, m+1] - vbar[i, j, m]))
            hz[k] = H * dz0[i, j, k]
        end
        ptop = 0f0                                    # hydrostatic p′ from the surface down
        for k in N:-1:1
            dp = g * ρp[k] * hz[k]; p[k] = ptop + 0.5f0 * dp; ptop += dp
        end
        sp = 0f0; su = 0f0; sv = 0f0                  # depth means removed
        for k in 1:N; sp += p[k] * hz[k]; su += up[k] * hz[k]; sv += vp[k] * hz[k]; end
        sp /= H; su /= H; sv /= H
        t = 0f0
        for k in 1:N; p[k] -= sp; up[k] -= su; vp[k] -= sv; t += p[k]^2 * hz[k]; end
        ptot[i, j] = t
        for n in 1:nm                                 # projections (1/H) ∫ f′ φn dz
            ap = 0f0; au = 0f0; av = 0f0
            for k in 1:N; f = φ[i, j, k, n] * hz[k]; ap += p[k] * f; au += up[k] * f; av += vp[k] * f; end
            P[i, j, n] = ap / H; U[i, j, n] = au / H; V[i, j, n] = av / H
        end
    end
    return nothing
end

function window_project(win, data_folder, mean_root; windows = 1:win.nwin, grid_file, nmode = 10, overwrite = false,
                        t_ref = DateTime(1994, 1, 1), verbose = true)
    td = win.td
    tag = "proj$nmode"
    out = joinpath(mean_root, tag); mkpath(out)
    own = wave_owner(win)
    recs_of = Dict(w => findall(==(w), own) for w in windows)
    fn(w) = window_file(out, tag, w, win.t0[w])
    fn_in(v, w) = window_file(joinpath(mean_root, mean_name(v)), mean_name(v), w, win.t0[w])   # window mean of his var v
    for w in windows; rm(fn(w) * ".tmp"; force = true); end
    todo = sort([w for w in windows if !isempty(recs_of[w]) &&
                 (overwrite || !window_done(fn(w), "p_1", win, w, t_ref; nrec = length(recs_of[w]), t0 = td.tdate[recs_of[w][1]]))])
    verbose && @printf("%s: %d of %d windows to do\n", tag, length(todo), length(windows))
    isempty(todo) && return [fn(w) for w in windows]
    mfile(w) = window_file(joinpath(mean_root, "modes$nmode"), "modes$nmode", w, win.t0[w])
    for w in todo
        for v in ("rho", "u", "v", "zeta")
            window_done(fn_in(v, w), mean_name(v), win, w, t_ref) || error("window $w: no mean of $v; run section 2 first")
        end
        window_done(mfile(w), "c", win, w, t_ref) || error("window $w: no $(basename(mfile(w))); run section 4 with nmode = $nmode")
    end

    h = NCDataset(g -> Float32.(g["h"].var[:, :]), grid_file)
    nx, ny = size(h)
    his1 = joinpath(data_folder, td.files[1])
    z0_r, dz0 = s_stretch(his1, Float64.(h))
    N = size(z0_r, 3)
    tunits = "seconds since " * Dates.format(t_ref, "yyyy-mm-dd HH:MM:SS")
    sec(t) = Dates.value(Millisecond(t - t_ref)) / 1000
    rd(v, w) = NCDataset(d -> (a = d[mean_name(v)]; a.var[ntuple(_ -> Colon(), ndims(a) - 1)..., 1]), fn_in(v, w))

    t0 = time()
    for w in todo
        rs = recs_of[w]; slot = Dict(n => i for (i, n) in enumerate(rs))
        # window background: means on the levels of the mean ζ, u, v at rho points; the window's modes
        ζbar = rd("zeta", w)
        zbar = ζbar .+ (ζbar .+ h) .* z0_r
        ρbar = rd("rho", w)
        ubar = u2rho!(Array{Float32}(undef, nx, ny, N), rd("u", w)); vbar = v2rho!(Array{Float32}(undef, nx, ny, N), rd("v", w))
        φ = NCDataset(d -> d["phi"].var[:, :, :, :], mfile(w))
        size(φ, 4) == nmode || error("$(mfile(w)) has $(size(φ, 4)) modes, nmode = $nmode")

        P = zeros(Float32, nx, ny, nmode); U = similar(P); V = similar(P); ptot = zeros(Float32, nx, ny)
        pvm = zeros(Float64, nx, ny, nmode); pvt = zeros(Float64, nx, ny)
        ρ = Array{Float32}(undef, nx, ny, N); ζ = Array{Float32}(undef, nx, ny)
        ub = Array{Float32}(undef, nx - 1, ny, N); vb = Array{Float32}(undef, nx, ny - 1, N)
        ur = Array{Float32}(undef, nx, ny, N); vr = Array{Float32}(undef, nx, ny, N)

        ds = NCDataset(fn(w) * ".tmp", "c"; attrib = ["title" => "mode projections of the fixed-depth wave parts p', u', v'",
                                                     "window" => w, "window_length_h" => win.window_length,
                                                     "window_overlap" => win.window_overlap, "nmode" => nmode,
                                                     "modes_file" => mfile(w), "created" => string(now())])
        try
            defDim(ds, "xi_rho", nx); defDim(ds, "eta_rho", ny); defDim(ds, "mode", nmode)
            defDim(ds, "time", length(rs)); defDim(ds, "one", 1)
            for n in 1:nmode, (v, ln, un) in (("p", "baroclinic pressure p'", "Pa"), ("u", "baroclinic velocity u' along xi", "meter second-1"),
                                              ("v", "baroclinic velocity v' along eta", "meter second-1"))
                defVar(ds, "$(v)_$n", Float32, ("xi_rho", "eta_rho", "time"); chunksizes = [nx, ny, 1],
                       attrib = ["long_name" => "mode-$n amplitude of the $ln (fixed-depth wave part)", "units" => un])
            end
            defVar(ds, "ocean_time", sec.(td.tdate[rs]), ("time",); attrib = ["units" => tunits, "long_name" => "record time"])
            defVar(ds, "t_start", [sec(td.tdate[rs[1]])], ("one",); attrib = ["units" => tunits, "long_name" => "first record"])
            defVar(ds, "t_end", [sec(td.tdate[rs[end]])], ("one",); attrib = ["units" => tunits, "long_name" => "last record"])

            need = falses(td.nt); need[rs] .= true
            for (fi, recs) in records_by_file(td)
                recs = filter(r -> need[r[2]], recs)
                isempty(recs) && continue
                NCDataset(joinpath(data_folder, td.files[fi])) do his
                    for (k, n) in recs
                        NCDatasets.load!(his["rho"].var, ρ, :, :, :, k); NCDatasets.load!(his["zeta"].var, ζ, :, :, k)
                        NCDatasets.load!(his["u"].var, ub, :, :, :, k); NCDatasets.load!(his["v"].var, vb, :, :, :, k)
                        project_record!(P, U, V, ptot, ρ, u2rho!(ur, ub), v2rho!(vr, vb), ζ, h, z0_r, dz0, zbar, ρbar, ubar, vbar, φ)
                        i = slot[n]
                        for m in 1:nmode
                            ds["p_$m"][:, :, i] = P[:, :, m]; ds["u_$m"][:, :, i] = U[:, :, m]; ds["v_$m"][:, :, i] = V[:, :, m]
                            @views pvm[:, :, m] .+= (ζ .+ h) .* P[:, :, m] .^ 2
                        end
                        pvt .+= ptot
                    end
                end
            end
            nr = length(rs)
            defVar(ds, "pvar_mode", Float32.(pvm ./ nr), ("xi_rho", "eta_rho", "mode");
                   attrib = ["long_name" => "window mean of H pn^2 (mode-n baroclinic pressure variance, depth-integrated)", "units" => "Pa2 m"])
            defVar(ds, "pvar_total", Float32.(pvt ./ nr), ("xi_rho", "eta_rho");
                   attrib = ["long_name" => "window mean of int p'^2 dz (all baroclinic pressure variance, depth-integrated)", "units" => "Pa2 m"])
            defVar(ds, "nrec", Int32[nr], ("one",); attrib = ["long_name" => "records in this file"])
        finally
            close(ds)
        end
        mv(fn(w) * ".tmp", fn(w); force = true)
        if verbose
            share = sum(pvm) / sum(pvt)
            @printf("  %s window %d: %d records, modes 1-%d hold %.1f%% of ∫p′² dz (mode 1 %.1f%%) -> %s, %.1f min\n",
                    tag, w, length(rs), nmode, 100share, 100sum(pvm[:, :, 1]) / sum(pvt), basename(fn(w)), (time() - t0) / 60)
        end
        φ = ρbar = ubar = vbar = zbar = nothing; GC.gc()
    end
    return [fn(w) for w in windows]
end

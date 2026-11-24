#= Ans.jl
Step 10 of wave_mean_flow.jl: wave - mean flow energy terms An (advection by the background flow) between modes,
one window and one band at a time. Modes m, n = 0..NMODE: mode 0 = barotropic (u_bt, v_bt; φ0 = 1, gradients.jl
mode0), modes 1..NMODE = u_n, v_n of step 9 with φn of step 4. Port of O. Siyanbola's anrn_matlab/An*.m.
Needs gradients.jl (rho_spacing, grad2d), mode_projection.jl (s_stretch, u2rho!, v2rho!), meanflow_window.jl,
wave_perturb.jl (wave_owner), wave_bands.jl (BANDS).

An1: advection of the wave kinetic energy by the background horizontal flow (U, V), anrn_matlab/An1.m:
  An1[m, n] = −ρ0 ( ⟨un ∂x um⟩ ∫U φm φn dz + ⟨un ∂y um⟩ ∫V φm φn dz + ⟨vn ∂x vm⟩ ∫U φm φn dz + ⟨vn ∂y vm⟩ ∫V φm φn dz )
            = −ρ0 ⟨ ∫ un φn (U·∇)(um φm) dz ⟩   with ∇φm dropped (as An1.m)            [W m-2]
  ⟨⟩ = mean over the window's hourly records; U, V the window mean (step 2) at rho points; φn and dz = Hz of the window
  mean ζ (zeta_window), the levels the modes were computed on (step 4); ∂x, ∂y = grad2d along the grid ξ, η.
  Index order as An1.m: An1[:, :, m+1, n+1] for the gradient mode m and the velocity mode n. Σ_m Σ_n An1 is the
  whole term: −ρ0 ⟨∫ u'·(U·∇)u' dz⟩ with u' = Σ un φn.

An2: vertical advection of the wave kinetic energy by the background W, anrn_matlab/An2.m:
  An2[m, n] = ρ0 ⟨um un + vm vn⟩ (1 / cm²) ∫ N² W Φm φn dz                                    [W m-2]
            = −ρ0 ⟨ ∫ un φn W ∂z(um φm) dz ⟩ + (v alike),   since ∂φm/∂z = −(N² / cm²) Φm (W'' + N²/c² W = 0)
  W the window mean (on rho levels), N² (TEOS-10, step 3) and Φm (step 4) averaged from the w levels to the rho levels,
  cm the eigenspeed. Mode 0: Φ0 = z + H, c0 = √(g H) (gradients.jl mode0), as the mode 0 of An2.m's eig file; its
  ∂φ0/∂z = 0 holds only to O(N² H / g), so the m = 0 row of An2 is small but not exactly the W ∂z of φ0 = 1.
  No final minus (as An2.m): the minus of −u'·W ∂z u' and the one of ∂φ/∂z cancel.
  Sign of dz: An1.m has a final minus and An2.m none; both match the formulas here with dz > 0, so An*.m also used
  dz > 0 (its zw from the surface down) and the signs here are the same as theirs.

An3: advection of the wave available potential energy by the background horizontal flow (U, V), anrn_matlab/An3.m:
  An3[m, n] = −1 / (ρ0 cm² cn²) ( ⟨pn ∂x pm⟩ ∫U N² Φm Φn dz + ⟨pn ∂y pm⟩ ∫V N² Φm Φn dz )        [W m-2]
            = −(ρ0 / N²) ⟨ ∫ bn (U·∇) bm dz ⟩   with the modal buoyancy b' = −Σ pn N² Φn / (ρ0 cn²), ∇Φm dropped
  pn = p_n of step 9, mode 0 = p_bt (depth mean of p_wave + ρ0 g zeta_wave, as An3.m's pbar_zeta); ∂x, ∂y = grad2d;
  N², Φ, c, U, V as in An1 / An2. Index order: gradient mode m (dim 3), pressure mode n (dim 4).
  1 / (cm² cn²): on the nearly mixed Amazon shelf (h < 50 m, N² at the 1e-7 floor of step 4, c ~ 1e-3 m/s) An3 is
  ~1e4 times An1 (window 1, bp9_15); where h > 200 m it is the size of An1. Mask shallow water before area sums.

An4: vertical advection of the wave available potential energy by the background W, anrn_matlab/An4.m:
  An4[m, n] = −1 / (ρ0 cm² cn²) ⟨pm pn⟩ ∫ W ∂z(N² Φm) Φn dz                                    [W m-2]
            = −(ρ0 / N²) ⟨ ∫ bn W ∂z bm dz ⟩   with bm = −pm N² Φm / (ρ0 cm²)
  ∂z(N² Φm) = ddz_w2r (gradients.jl): N² Φm on the w levels (both live there), differenced over each layer to the rho
  levels (zw of the window mean ζ). Mode 0: Φ0 = z + H on the w levels. W, pn, Φn (rho levels), c as above.
  Index order: mode m of ∂z(N² Φm), cm (dim 3), mode n of Φn (dim 4). Same 1/(cm² cn²) as An3 (shelf blow-up).

An5: advection of the wave APE through the horizontal change of the eigenspeed, anrn_matlab/An5.m:
  An5[m, n] = 2 ⟨pm pn⟩ / (ρ0 cm³ cn²) ( ∂x cm ∫U N² Φm Φn dz + ∂y cm ∫V N² Φm Φn dz )            [W m-2]
  the part of −(ρ0/N²) ⟨∫ bn (U·∇) bm dz⟩ that An3 drops: ∇ acting on the 1/cm² of bm = −pm N² Φm / (ρ0 cm²),
  ∇(1/cm²) = −2 ∇cm / cm³ (∇Φm itself still dropped). No final minus (as An5.m). ∂x cm, ∂y cm = grad2d of c (step 4;
  mode 0: c0 = √(g H)), edges NaN. Index order: mode m of cm, ∇cm (dim 3), mode n (dim 4). 1/(cm³ cn²): shelf blow-up.

Output, per term (An1, An2, ...): mean_root/<term>_<band>/<term>_<band>_win<NNN>_<t0>.nc: <term>_<band> (xi_rho,
eta_rho, mode_m, mode_n), Float32, ~93 MB per window with NMODE = 10 (An1: edges NaN from grad2d). The inputs of a window
are read once for all its terms. Written under .tmp, renamed when complete; done windows are skipped.
=#

# inputs of the An / Rn terms for window w and band: 2-D modal amplitudes (modes 0..nmode, records), their
# horizontal gradients, the background U, V at rho points, φ (modes 0..nmode) and dz of the window-mean levels
function an_window_inputs(win, mean_root, w, band; grid_file, nmode, his_file, t_ref = DateTime(1994, 1, 1))
    bn = band.name
    fw(nm) = window_file(joinpath(mean_root, nm), nm, w, win.t0[w])
    h, pm, pn = NCDataset(g -> (Float64.(g["h"].var[:, :]), g["pm"].var[:, :], g["pn"].var[:, :]), grid_file)
    dx, dy = Float32.(rho_spacing(pm, pn)[1]), Float32.(rho_spacing(pm, pn)[2])
    nx, ny = size(h)
    # velocity amplitudes, mode 0 = u_bt, v_bt moved to rho points: (nx, ny, nmode+1, nt)
    un = NCDataset(d -> d["u_n"].var[:, :, 1:nmode, :], fw("u_n_$bn")); vn = NCDataset(d -> d["v_n"].var[:, :, 1:nmode, :], fw("v_n_$bn"))
    nt = size(un, 4)
    ub = NCDataset(d -> d["u_bt_$bn"].var[:, :, :], fw("u_bt_$bn")); vb = NCDataset(d -> d["v_bt_$bn"].var[:, :, :], fw("v_bt_$bn"))
    size(ub, 3) == nt || error("window $w: u_bt has $(size(ub, 3)) records, u_n $nt")
    u = Array{Float32}(undef, nx, ny, nmode + 1, nt); v = similar(u)
    u2rho!(view(u, :, :, 1, :), ub); v2rho!(view(v, :, :, 1, :), vb)
    u[:, :, 2:end, :] .= un; v[:, :, 2:end, :] .= vn
    un = vn = ub = vb = nothing
    ux, uy = grad2d(u, dx, dy); vx, vy = grad2d(v, dx, dy)
    # pressure amplitudes (An3), mode 0 = p_bt (rho points): (nx, ny, nmode+1, nt), and their gradients
    p = Array{Float32}(undef, nx, ny, nmode + 1, nt)
    p[:, :, 1, :] .= NCDataset(d -> d["p_bt_$bn"].var[:, :, :], fw("p_bt_$bn"))
    p[:, :, 2:end, :] .= NCDataset(d -> d["p_n"].var[:, :, 1:nmode, :], fw("p_n_$bn"))
    px, py = grad2d(p, dx, dy)
    # background flow at rho points
    U = NCDataset(d -> d["U"].var[:, :, :, 1], fw("U")); V = NCDataset(d -> d["V"].var[:, :, :, 1], fw("V"))
    N = size(U, 3)
    Ur = Array{Float32}(undef, nx, ny, N); Vr = similar(Ur); u2rho!(Ur, U); v2rho!(Vr, V)
    # φ, mode 0 = 1; dz = Hz of the window-mean ζ
    φ = Array{Float32}(undef, nx, ny, N, nmode + 1); φ[:, :, :, 1] .= 1
    φ[:, :, :, 2:end] .= NCDataset(d -> d["phi"].var[:, :, :, 1:nmode], fw("modes$nmode"))
    ζbar = NCDataset(d -> Float64.(d["zeta_window"].var[:, :, 1]), fw("zeta_window"))
    z0r, dz0 = s_stretch(his_file, h)
    dz = Float32.((ζbar .+ h) .* dz0)
    # An2: W (rho levels), N² and Φ moved from the w levels to the rho levels, c; mode 0 from mode0 (Φ0 = z + H, c0 = √(gH))
    W = NCDataset(d -> d["W"].var[:, :, :, 1], fw("W"))
    n2 = NCDataset(d -> d["N2"].var[:, :, :, 1], fw("N2")); N2r = @views 0.5f0 .* (n2[:, :, 1:N] .+ n2[:, :, 2:N+1])
    Φr = Array{Float32}(undef, nx, ny, N, nmode + 1); c = Array{Float32}(undef, nx, ny, nmode + 1)
    zr = ζbar .+ (ζbar .+ h) .* z0r
    Φr[:, :, :, 1] .= zr .+ h; c[:, :, 1] .= sqrt.(G_ACC .* (h .+ ζbar))
    Φw = NCDataset(d -> d["Phi"].var[:, :, :, 1:nmode], fw("modes$nmode"))
    @views Φr[:, :, :, 2:end] .= 0.5f0 .* (Φw[:, :, 1:N, :] .+ Φw[:, :, 2:N+1, :])
    # An4: ∂z(N² Φm) on the rho levels from N², Φm on the w levels (ddz_w2r); mode 0: Φ0 = zw + h
    zw = cat(-h, ζbar .+ (ζbar .+ h) .* (-1 .+ cumsum(dz0, dims = 3)); dims = 3)       # w levels 1..N+1
    N2Φz = Array{Float32}(undef, nx, ny, N, nmode + 1)
    N2Φz[:, :, :, 1] .= ddz_w2r(n2 .* (zw .+ h), zw)
    for m in 1:nmode; N2Φz[:, :, :, m+1] .= ddz_w2r(n2 .* view(Φw, :, :, :, m), zw); end
    Φw = n2 = nothing
    c[:, :, 2:end] .= NCDataset(d -> d["c"].var[:, :, 1:nmode], fw("modes$nmode"))
    cx, cy = grad2d(c, dx, dy)                                         # An5: ∂x cm, ∂y cm (nx, ny, nmode+1)
    return (; u, v, ux, uy, vx, vy, p, px, py, U = Ur, V = Vr, W, N2r, φ, Φr, N2Φz, c, cx, cy, dz, nt, N, nx, ny)
end

# ∫ X φm φn dz for every m, n (symmetric), X = U or V: (nx, ny, nm, nm)
function int_phi_mn(X, φ, dz)
    nx, ny, N, nm = size(φ)
    I = zeros(Float32, nx, ny, nm, nm); wk = zeros(Float64, N)
    @inbounds for j in 1:ny, i in 1:nx
        for k in 1:N; wk[k] = X[i, j, k] * dz[i, j, k]; end
        for n in 1:nm, m in 1:n
            s = 0.0
            for k in 1:N; s += wk[k] * φ[i, j, k, m] * φ[i, j, k, n]; end
            I[i, j, m, n] = s; I[i, j, n, m] = s
        end
    end
    return I
end

# An1 of one window from an_window_inputs: (nx, ny, nm, nm), [m, n]
function an1(A; rho0)
    nx, ny, nm, nt = size(A.u)
    IU = int_phi_mn(A.U, A.φ, A.dz); IV = int_phi_mn(A.V, A.φ, A.dz) # ∫U φₘφₙ dz & ∫V φₘφₙ dz
    out = Array{Float32}(undef, nx, ny, nm, nm)
    @inbounds for n in 1:nm, m in 1:nm, j in 1:ny, i in 1:nx
        sx = 0.0; sy = 0.0                                 # ⟨un ∂x um + vn ∂x vm⟩, ⟨un ∂y um + vn ∂y vm⟩
        for t in 1:nt # time mean process ⟨⟩
            sx += A.u[i, j, n, t] * A.ux[i, j, m, t] + A.v[i, j, n, t] * A.vx[i, j, m, t]
            sy += A.u[i, j, n, t] * A.uy[i, j, m, t] + A.v[i, j, n, t] * A.vy[i, j, m, t]
        end
        out[i, j, m, n] = -rho0 * (sx / nt * IU[i, j, m, n] + sy / nt * IV[i, j, m, n])
    end #  `sx / nt` = ⟨uₙ∂ₓuₘ + vₙ∂ₓvₘ⟩  `sy / nt` = ⟨uₙ∂ᵧuₘ + vₙ∂ᵧvₘ⟩
    return out
end
# sx/nt · ∫U φₘφₙ dz   →  the U ∂ₓ part of (U·∇), sy/nt · ∫V φₘφₙ dz   →  the V ∂ᵧ part of (U·∇)

# An2 of one window from an_window_inputs: (nx, ny, nm, nm), [m, n]
# An2[m, n] = ρ₀ ⟨uₘuₙ + vₘvₙ⟩ · (1/cₘ²) ∫ N² W Φₘ φₙ dz   [W/m²]
function an2(A; rho0)
    nx, ny, nm, nt = size(A.u); N = A.N
    out = Array{Float32}(undef, nx, ny, nm, nm); wk = zeros(Float64, N)
    @inbounds for j in 1:ny, i in 1:nx
        for k in 1:N; wk[k] = A.N2r[i, j, k] * A.W[i, j, k] * A.dz[i, j, k]; end # N² · W · dz
        for n in 1:nm, m in 1:nm
            s = 0.0                                        # ∫ N² W Φm φn dz
            for k in 1:N; s += wk[k] * A.Φr[i, j, k, m] * A.φ[i, j, k, n]; end
            p = 0.0                                        # ⟨um un + vm vn⟩
            for t in 1:nt; p += A.u[i, j, m, t] * A.u[i, j, n, t] + A.v[i, j, m, t] * A.v[i, j, n, t]; end
            out[i, j, m, n] = rho0 * p / nt * s / Float64(A.c[i, j, m])^2
        end
    end
    return out
end

# An3 of one window from an_window_inputs: (nx, ny, nm, nm), [m, n]
# An3[m, n] = −1/(ρ₀ cₘ² cₙ²) · ( ⟨pₙ∂ₓpₘ⟩ ∫U N² ΦₘΦₙ dz + ⟨pₙ∂ᵧpₘ⟩ ∫V N² ΦₘΦₙ dz )   [W/m²]
function an3(A; rho0)
    nx, ny, nm, nt = size(A.p); N = A.N
    out = Array{Float32}(undef, nx, ny, nm, nm)
    wu = zeros(Float64, N); wv = zeros(Float64, N); IU = zeros(Float64, nm, nm); IV = zeros(Float64, nm, nm)
    @inbounds for j in 1:ny, i in 1:nx
        for k in 1:N; q = A.N2r[i, j, k] * A.dz[i, j, k]; wu[k] = A.U[i, j, k] * q; wv[k] = A.V[i, j, k] * q; end
        for n in 1:nm, m in 1:n                            # ∫ U N² Φm Φn dz, ∫ V N² Φm Φn dz (symmetric)
            su = 0.0; sv = 0.0
            for k in 1:N; ΦΦ = A.Φr[i, j, k, m] * A.Φr[i, j, k, n]; su += wu[k] * ΦΦ; sv += wv[k] * ΦΦ; end
            IU[m, n] = IU[n, m] = su; IV[m, n] = IV[n, m] = sv
        end
        for n in 1:nm, m in 1:nm
            sx = 0.0; sy = 0.0                             # ⟨pn ∂x pm⟩, ⟨pn ∂y pm⟩ (sums over records)
            for t in 1:nt; sx += A.p[i, j, n, t] * A.px[i, j, m, t]; sy += A.p[i, j, n, t] * A.py[i, j, m, t]; end
            cc = Float64(A.c[i, j, m])^2 * Float64(A.c[i, j, n])^2
            out[i, j, m, n] = -(sx / nt * IU[m, n] + sy / nt * IV[m, n]) / (rho0 * cc)
        end
    end
    return out
end

# An4 of one window from an_window_inputs: (nx, ny, nm, nm), [m, n]
# An4[m, n] = −1/(ρ₀ cₘ² cₙ²) · ⟨pₘpₙ⟩ · ∫ W ∂z(N²Φₘ) Φₙ dz   [W/m²]
function an4(A; rho0)
    nx, ny, nm, nt = size(A.p); N = A.N
    out = Array{Float32}(undef, nx, ny, nm, nm); wk = zeros(Float64, N)
    @inbounds for j in 1:ny, i in 1:nx
        for k in 1:N; wk[k] = A.W[i, j, k] * A.dz[i, j, k]; end
        for n in 1:nm, m in 1:nm
            s = 0.0                                        # ∫ W ∂z(N² Φm) Φn dz
            for k in 1:N; s += wk[k] * A.N2Φz[i, j, k, m] * A.Φr[i, j, k, n]; end
            q = 0.0                                        # ⟨pm pn⟩ (sum over records)
            for t in 1:nt; q += A.p[i, j, m, t] * A.p[i, j, n, t]; end
            cc = Float64(A.c[i, j, m])^2 * Float64(A.c[i, j, n])^2
            out[i, j, m, n] = -(q / nt) * s / (rho0 * cc)
        end
    end
    return out
end

# An5 of one window from an_window_inputs: (nx, ny, nm, nm), [m, n]
# An5[m, n] = 2⟨pₘpₙ⟩ / (ρ₀ cₘ³ cₙ²) · ( ∂ₓcₘ ∫U N² ΦₘΦₙ dz + ∂ᵧcₘ ∫V N² ΦₘΦₙ dz ) 
# leftover term from An3 from horizontal advection on APE
function an5(A; rho0)
    nx, ny, nm, nt = size(A.p); N = A.N
    out = Array{Float32}(undef, nx, ny, nm, nm)
    wu = zeros(Float64, N); wv = zeros(Float64, N)
    @inbounds for j in 1:ny, i in 1:nx
        for k in 1:N; q = A.N2r[i, j, k] * A.dz[i, j, k]; wu[k] = A.U[i, j, k] * q; wv[k] = A.V[i, j, k] * q; end
        for n in 1:nm, m in 1:nm
            su = 0.0; sv = 0.0                             # ∫ U N² Φm Φn dz, ∫ V N² Φm Φn dz
            for k in 1:N; ΦΦ = A.Φr[i, j, k, m] * A.Φr[i, j, k, n]; su += wu[k] * ΦΦ; sv += wv[k] * ΦΦ; end
            q = 0.0                                        # ⟨pm pn⟩ (sum over records)
            for t in 1:nt; q += A.p[i, j, m, t] * A.p[i, j, n, t]; end
            cm = Float64(A.c[i, j, m]); cn = Float64(A.c[i, j, n])
            out[i, j, m, n] = 2 * (q / nt) * (A.cx[i, j, m] * su + A.cy[i, j, m] * sv) / (rho0 * cm^3 * cn^2)
        end
    end
    return out
end

# the An terms: name -> (function, description, formula)
const AN_TERMS = Dict(
    "An1" => (an1, "advection of wave KE by the background horizontal flow (U, V) between modes (An1.m)",
              "An1[m,n] = -rho0 ( <un dx um + vn dx vm> int U phi_m phi_n dz + <un dy um + vn dy vm> int V phi_m phi_n dz )",
              "gradient mode m (dim 3) and velocity mode n (dim 4)"),
    "An2" => (an2, "vertical advection of wave KE by the background W between modes (An2.m)",
              "An2[m,n] = rho0 <um un + vm vn> (1/cm^2) int N2 W Phi_m phi_n dz",
              "mode m of Phi_m, cm (dim 3) and mode n of phi_n (dim 4)"),
    "An3" => (an3, "advection of wave APE by the background horizontal flow (U, V) between modes (An3.m)",
              "An3[m,n] = -1/(rho0 cm^2 cn^2) ( <pn dx pm> int U N2 Phi_m Phi_n dz + <pn dy pm> int V N2 Phi_m Phi_n dz )",
              "gradient mode m (dim 3) and pressure mode n (dim 4)"),
    "An4" => (an4, "vertical advection of wave APE by the background W between modes (An4.m)",
              "An4[m,n] = -1/(rho0 cm^2 cn^2) <pm pn> int W dz(N2 Phi_m) Phi_n dz",
              "mode m of dz(N2 Phi_m), cm (dim 3) and mode n of Phi_n (dim 4)"),
    "An5" => (an5, "advection of wave APE by the background horizontal flow (U, V) through the horizontal change of cm (An5.m)",
              "An5[m,n] = 2 <pm pn> / (rho0 cm^3 cn^2) ( dx cm int U N2 Phi_m Phi_n dz + dy cm int V N2 Phi_m Phi_n dz )",
              "mode m of cm, grad cm (dim 3) and mode n (dim 4)"))

# step 10 driver: the terms of window w, band; the inputs read once per window for all its terms not yet done
function window_An(win, data_folder, mean_root; band, terms = ["An1", "An2", "An3", "An4", "An5"], grid_file, nmode, windows = 1:win.nwin,
                   t_ref = DateTime(1994, 1, 1), verbose = true)
    td = win.td; his1 = joinpath(data_folder, td.files[1])
    own = wave_owner(win); recs_of = [findall(==(w), own) for w in 1:win.nwin]
    for t in terms; haskey(AN_TERMS, t) || error("term $t: not one of $(sort(collect(keys(AN_TERMS))))"); end
    nm(t) = "$(t)_$(band.name)"
    fo(t, w) = window_file(joinpath(mean_root, nm(t)), nm(t), w, win.t0[w])
    done(t, w) = window_done(fo(t, w), nm(t), win, w, t_ref; nrec = length(recs_of[w]), t0 = td.tdate[recs_of[w][1]])
    todo = Dict(w => [t for t in terms if !done(t, w)] for w in windows if !isempty(recs_of[w]))
    filter!(p -> !isempty(p.second), todo)
    verbose && @printf("%s, band %s: %d of %d windows to do\n", join(terms, ", "), band.name, length(todo), length(windows))
    rho0 = NCDataset(d -> Float64(d.attrib["rho0"]), his1)
    tunits = "seconds since " * Dates.format(t_ref, "yyyy-mm-dd HH:MM:SS")
    sec(t) = Dates.value(Millisecond(t - t_ref)) / 1000
    for t in terms; mkpath(joinpath(mean_root, nm(t))); end
    t0 = time()
    for w in sort(collect(keys(todo)))
        A = an_window_inputs(win, mean_root, w, band; grid_file, nmode, his_file = his1, t_ref)
        rs = recs_of[w]; length(rs) == A.nt || error("window $w: $(A.nt) records in u_n, $(length(rs)) expected")
        for t in todo[w]
            f, what, formula, dims = AN_TERMS[t]
            a = f(A; rho0)
            NCDataset(fo(t, w) * ".tmp", "c"; attrib = ["title" => "$(nm(t)): $what, Ans.jl", "formula" => formula,
                                                       "modes" => "0 = barotropic (u_bt, v_bt, phi0 = 1, Phi0 = z + H, c0 = sqrt(g H)), 1..$nmode = u_n, v_n of modes$nmode",
                                                       "rho0" => rho0, "window" => w, "created" => string(now())]) do ds
                defDim(ds, "xi_rho", size(a, 1)); defDim(ds, "eta_rho", size(a, 2))
                defDim(ds, "mode_m", nmode + 1); defDim(ds, "mode_n", nmode + 1); defDim(ds, "one", 1)
                defVar(ds, nm(t), a, ("xi_rho", "eta_rho", "mode_m", "mode_n"); attrib = ["units" => "W m-2",
                       "long_name" => "$t for $dims, window mean"])
                defVar(ds, "mode_m", Int32.(0:nmode), ("mode_m",)); defVar(ds, "mode_n", Int32.(0:nmode), ("mode_n",))
                defVar(ds, "t_start", [sec(td.tdate[rs[1]])], ("one",); attrib = ["units" => tunits])
                defVar(ds, "t_end", [sec(td.tdate[rs[end]])], ("one",); attrib = ["units" => tunits])
                defVar(ds, "nrec", Int32[length(rs)], ("one",); attrib = ["long_name" => "records averaged"])
            end
            mv(fo(t, w) * ".tmp", fo(t, w); force = true)
            tot = dropdims(sum(a, dims = (3, 4)), dims = (3, 4)); ok = .!isnan.(tot)
            verbose && @printf("  %s window %d: area mean Σmn %.3e, Σ diag %.3e, mode 0 row/col %.3e W/m2, %.1f min\n", nm(t), w,
                               mean(tot[ok]), mean(sum(a[:, :, k, k] for k in 1:nmode+1)[ok]),
                               mean((dropdims(sum(a[:, :, 1, :], dims = 3), dims = 3) .+ dropdims(sum(a[:, :, 2:end, 1], dims = 3), dims = 3))[ok]),
                               (time() - t0) / 60)
            a = nothing
        end
        A = nothing; GC.gc()
    end
    return Dict(nm(t) => [fo(t, w) for w in windows] for t in terms)
end

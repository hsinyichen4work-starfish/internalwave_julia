#= mode_check.jl
Step 9.5: checks of the baroclinic split (step 8) and the mode projections (step 9), per band and window.
  (1) depth mean of the baroclinic part is zero:  for f = p, u, v at every column and record
        r = |(1/H) ∫ f_bc dz| / sqrt((1/H) ∫ f_bc² dz)      (on the field's own grid, Hz and H of the record)
      should be ~1e-6 or less (Float32). Columns with a tiny f_bc (rms below a floor) are left out.
      f_bc as in step 8: u_bc = u_wave − u_bt, v_bc = v_wave − v_bt, p_bc = p_wave + ρ0 g zeta_wave − p_bt.
  (2) energy flux, full vs modal (window means, rho grid, u_bc / v_bc moved to rho points as in step 9):
        F_full  = ⟨ ∫ p_bc u_bc dz ⟩               (and v)   [W m-1]
        F_modal = Σn ⟨ H p_n u_n ⟩,  F_mode[n] = ⟨ H p_n u_n ⟩
      With Hz(t) = Hz_window (h + ζ(t)) / (h + ζ_window) (one factor per column), the φn stay orthogonal at every
      record, so F_full − F_modal = flux in modes > NMODE only (no cross-mode terms).
Needs barotropic.jl / baroclinic_modes.jl / wave_bands.jl helpers (wave_grid, u2rho!, v2rho!), the files of steps 5-9.

Output per band: mean_root/modecheck_<band>/modecheck_<band>_win<NNN>_<t0>.nc with
  Fx_full, Fy_full, Fx_modal, Fy_modal (xi_rho, eta_rho), Fx_mode, Fy_mode (xi_rho, eta_rho, mode)
  and a text summary mean_root/modecheck_<band>/modecheck_<band>_summary.txt (rewritten from all window files).
=#

# rms of f_bc below this: column left out of (1). p is 1 Pa: p_bc = p_wave + ρ0 g zeta_wave − p_bt cancels a surface
# term of ~1e4 Pa in Float32 (~1e-3 Pa rounding), which would dominate the ratio where p_bc itself is ~0.01 Pa
const BC_FLOOR = Dict("p" => 1f0, "u" => 1f-5, "v" => 1f-5)

# (1) for one record: |depth mean| / rms of x in each column -> (max, sum, count) over columns with rms > floor
function bc_mean_stats(x, ζ, h, dz0, floor)
    nx, ny, N = size(x); mx = 0.0; s = 0.0; c = 0
    @inbounds for j in 1:ny, i in 1:nx
        H = ζ[i, j] + h[i, j]; m = 0.0; q = 0.0
        for k in 1:N; hz = H * dz0[i, j, k]; m += x[i, j, k] * hz; q += Float64(x[i, j, k])^2 * hz; end
        rms = sqrt(q / H); rms > floor || continue
        r = abs(m / H) / rms; mx = max(mx, r); s += r; c += 1
    end
    return mx, s, c
end

# (2) for one record: add ∫ p u dz, ∫ p v dz (rho grid) to Fx, Fy
function add_flux!(Fx, Fy, p, ur, vr, ζ, h, dz0)
    nx, ny, N = size(p)
    @inbounds for j in 1:ny, i in 1:nx
        H = ζ[i, j] + h[i, j]; ax = 0.0; ay = 0.0
        for k in 1:N; hz = H * dz0[i, j, k]; ax += p[i, j, k] * ur[i, j, k] * hz; ay += p[i, j, k] * vr[i, j, k] * hz; end
        Fx[i, j] += ax; Fy[i, j] += ay
    end
    return nothing
end

function window_mode_check(win, data_folder, mean_root; band, grid_file, nmode = 10, windows = 1:win.nwin, sponge = 55,
                           t_ref = DateTime(1994, 1, 1), verbose = true)
    td = win.td
    own = wave_owner(win)
    recs_of = [findall(==(w), own) for w in 1:win.nwin]
    ws = [w for w in windows if !isempty(recs_of[w])]
    bn = band.name; tag = "modecheck_$bn"
    fwin(nm, w) = window_file(joinpath(mean_root, nm), nm, w, win.t0[w])
    for w in ws, nm in ("p_wave_$bn", "u_wave_$bn", "v_wave_$bn", "zeta_wave_$bn", "p_bt_$bn", "u_bt_$bn", "v_bt_$bn",
                        "p_n_$bn", "u_n_$bn", "v_n_$bn")
        isfile(fwin(nm, w)) || error("no $nm for window $w: run steps 5-9 first")
    end
    Gr = wave_grid(win, data_folder, mean_root, "rho", grid_file)
    Gu = wave_grid(win, data_folder, mean_root, "u", grid_file)
    Gv = wave_grid(win, data_folder, mean_root, "v", grid_file)
    rho0 = NCDataset(ds -> Float32(ds.attrib["rho0"]), joinpath(data_folder, td.files[1])); g = 9.81f0
    nx, ny, N = size(Gr.dz0)
    p = Array{Float32}(undef, nx, ny, N); ub = Array{Float32}(undef, nx - 1, ny, N); vb = Array{Float32}(undef, nx, ny - 1, N)
    ur = Array{Float32}(undef, nx, ny, N); vr = similar(ur)
    pbt = Array{Float32}(undef, nx, ny); zw = similar(pbt); ubt = Array{Float32}(undef, nx - 1, ny); vbt = Array{Float32}(undef, nx, ny - 1)
    Pn = Array{Float32}(undef, nx, ny, nmode); Un = similar(Pn); Vn = similar(Pn)
    mkpath(joinpath(mean_root, tag))

    t0 = time()
    for w in ws
        rs = recs_of[w]; fo = fwin(tag, w)
        isfile(fo) && (verbose && println("  $tag window $w: done, skipped"); continue)
        st = Dict(f => [0.0, 0.0, 0] for f in ("p", "u", "v"))     # max, sum, count of the ratio of check (1)
        Fx = zeros(nx, ny); Fy = zeros(nx, ny); Fxn = zeros(nx, ny, nmode); Fyn = zeros(nx, ny, nmode)
        ds = Dict(nm => NCDataset(fwin(nm, w)) for nm in ("p_wave_$bn", "u_wave_$bn", "v_wave_$bn", "zeta_wave_$bn",
                                                           "p_bt_$bn", "u_bt_$bn", "v_bt_$bn", "p_n_$bn", "u_n_$bn", "v_n_$bn"))
        try
            for (i, n) in enumerate(rs)
                NCDatasets.load!(ds["p_wave_$bn"]["p_wave_$bn"].var, p, :, :, :, i)
                NCDatasets.load!(ds["u_wave_$bn"]["u_wave_$bn"].var, ub, :, :, :, i)
                NCDatasets.load!(ds["v_wave_$bn"]["v_wave_$bn"].var, vb, :, :, :, i)
                NCDatasets.load!(ds["p_bt_$bn"]["p_bt_$bn"].var, pbt, :, :, i); NCDatasets.load!(ds["zeta_wave_$bn"]["zeta_wave_$bn"].var, zw, :, :, i)
                NCDatasets.load!(ds["u_bt_$bn"]["u_bt_$bn"].var, ubt, :, :, i); NCDatasets.load!(ds["v_bt_$bn"]["v_bt_$bn"].var, vbt, :, :, i)
                p .-= pbt .- rho0 * g .* zw                                     # p_bc (step 8)
                ub .-= ubt; vb .-= vbt                                          # u_bc, v_bc on their grids
                ζr, ζu, ζv = view(Gr.ζ, :, :, n), view(Gu.ζ, :, :, n), view(Gv.ζ, :, :, n)
                for (f, x, G, ζ) in (("p", p, Gr, ζr), ("u", ub, Gu, ζu), ("v", vb, Gv, ζv))   # check (1)
                    mx, s, c = bc_mean_stats(x, ζ, G.h, G.dz0, BC_FLOOR[f])
                    st[f][1] = max(st[f][1], mx); st[f][2] += s; st[f][3] += c
                end
                u2rho!(ur, ub); v2rho!(vr, vb)
                add_flux!(Fx, Fy, p, ur, vr, ζr, Gr.h, Gr.dz0)                   # check (2): full flux
                NCDatasets.load!(ds["p_n_$bn"]["p_n"].var, Pn, :, :, :, i)
                NCDatasets.load!(ds["u_n_$bn"]["u_n"].var, Un, :, :, :, i); NCDatasets.load!(ds["v_n_$bn"]["v_n"].var, Vn, :, :, :, i)
                H = ζr .+ Gr.h
                for m in 1:nmode                                                 # modal flux
                    @views Fxn[:, :, m] .+= H .* Pn[:, :, m] .* Un[:, :, m]
                    @views Fyn[:, :, m] .+= H .* Pn[:, :, m] .* Vn[:, :, m]
                end
            end
        finally
            foreach(close, values(ds))
        end
        nr = length(rs)
        Fx ./= nr; Fy ./= nr; Fxn ./= nr; Fyn ./= nr
        rm(fo * ".tmp"; force = true)
        NCDataset(fo * ".tmp", "c"; attrib = ["title" => "step 9.5 mode checks, band $bn, window $w",
                 "check1_p_max" => st["p"][1], "check1_p_mean" => st["p"][2] / max(st["p"][3], 1),
                 "check1_u_max" => st["u"][1], "check1_u_mean" => st["u"][2] / max(st["u"][3], 1),
                 "check1_v_max" => st["v"][1], "check1_v_mean" => st["v"][2] / max(st["v"][3], 1),
                 "check1_note" => "|depth mean of f_bc| / rms of f_bc over columns and records (rms floors p $(BC_FLOOR["p"]) Pa, u, v $(BC_FLOOR["u"]) m/s)",
                 "nmode" => nmode, "nrec" => nr, "created" => string(now())]) do d
            defDim(d, "xi_rho", nx); defDim(d, "eta_rho", ny); defDim(d, "mode", nmode)
            for (nm, a, ln) in (("Fx_full", Fx, "window mean of int p_bc u_bc dz (xi)"), ("Fy_full", Fy, "window mean of int p_bc v_bc dz (eta)"),
                                ("Fx_modal", dropdims(sum(Fxn, dims = 3), dims = 3), "sum_n window mean of H p_n u_n (xi)"),
                                ("Fy_modal", dropdims(sum(Fyn, dims = 3), dims = 3), "sum_n window mean of H p_n v_n (eta)"))
                defVar(d, nm, Float32.(a), ("xi_rho", "eta_rho"); attrib = ["units" => "W m-1", "long_name" => ln])
            end
            defVar(d, "Fx_mode", Float32.(Fxn), ("xi_rho", "eta_rho", "mode"); attrib = ["units" => "W m-1", "long_name" => "window mean of H p_n u_n (xi)"])
            defVar(d, "Fy_mode", Float32.(Fyn), ("xi_rho", "eta_rho", "mode"); attrib = ["units" => "W m-1", "long_name" => "window mean of H p_n v_n (eta)"])
        end
        mv(fo * ".tmp", fo; force = true)
        verbose && @printf("  %s window %d: (1) max |mean|/rms p %.1e u %.1e v %.1e | (2) Σ|F_modal| / Σ|F_full| %.3f, %.1f min\n",
                           tag, w, st["p"][1], st["u"][1], st["v"][1],
                           sum(hypot.(sum(Fxn, dims = 3), sum(Fyn, dims = 3))) / sum(hypot.(Fx, Fy)), (time() - t0) / 60)
    end
    mode_check_summary(win, mean_root, band, grid_file; sponge)
end

# text summary of all window files of a band (also rerun on its own after windows were added)
function mode_check_summary(win, mean_root, band, grid_file; sponge = 55)
    bn = band.name; tag = "modecheck_$bn"
    pm, pn, h = NCDataset(gf -> (Float64.(gf["pm"].var[:, :]), Float64.(gf["pn"].var[:, :]), Float64.(gf["h"].var[:, :])), grid_file)
    dA = 1 ./ (pm .* pn); nx, ny = size(h)
    inner = falses(nx, ny); inner[sponge+1:nx-sponge, sponge+1:ny-sponge] .= true
    out = joinpath(mean_root, tag, "$(tag)_summary.txt")
    open(out, "w") do io
        println(io, "# Step 9.5 mode checks, band $bn (mode_check.jl), $(now())")
        println(io, "# (1) |(1/H) int f_bc dz| / rms(f_bc) over all columns and records of the window: max and mean (should be ~1e-6)")
        println(io, "# (2) energy flux F = (Fx, Fy) [W/m], window means: full = <int p_bc u_bc dz>, modal = sum_n <H p_n u_n>, n = 1..NMODE")
        println(io, "#     ratio = sum|F_modal| dA / sum|F_full| dA;  corr = pattern correlation of (Fx, Fy) full vs modal;")
        println(io, "#     rel.diff = rms|F_modal - F_full| / rms|F_full|;  'interior' = without the $sponge-point sponge;  mean|F| = area mean of |F_full| [kW/m]")
        @printf(io, "\n# %-4s %-13s | %-17s %-17s %-17s | %-30s | %-30s | %s\n", "win", "start", "(1) p max/mean", "(1) u max/mean",
                "(1) v max/mean", "(2) all: ratio corr rel.diff", "(2) interior: ratio corr rel.diff", "mean|F_full| all / interior [kW/m]")
        for w in 1:win.nwin
            f = window_file(joinpath(mean_root, tag), tag, w, win.t0[w]); isfile(f) || continue
            NCDataset(f) do d
                a = d.attrib
                Fx, Fy, Mx, My = (Float64.(d[v].var[:, :]) for v in ("Fx_full", "Fy_full", "Fx_modal", "Fy_modal"))
                stats(m) = (sum(hypot.(Mx[m], My[m]) .* dA[m]) / sum(hypot.(Fx[m], Fy[m]) .* dA[m]),
                            cor(vcat(Fx[m], Fy[m]), vcat(Mx[m], My[m])),
                            sqrt(mean(abs2, hypot.(Mx[m] .- Fx[m], My[m] .- Fy[m]))) / sqrt(mean(abs2, hypot.(Fx[m], Fy[m]))))
                al = trues(nx, ny)
                @printf(io, "  %-4d %-13s | %.1e / %.1e  %.1e / %.1e  %.1e / %.1e | %8.4f %8.4f %8.4f    | %8.4f %8.4f %8.4f    | %7.3f / %7.3f\n",
                        w, Dates.format(win.t0[w], "yyyy-mm-ddTHH"), a["check1_p_max"], a["check1_p_mean"], a["check1_u_max"], a["check1_u_mean"],
                        a["check1_v_max"], a["check1_v_mean"], stats(al)..., stats(inner)..., sum(hypot.(Fx, Fy) .* dA) / sum(dA) / 1e3,
                        sum((hypot.(Fx, Fy) .* dA)[inner]) / sum(dA[inner]) / 1e3)
            end
        end
    end
    println("wrote ", out)
    return out
end

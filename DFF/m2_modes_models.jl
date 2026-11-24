#= m2_modes_models.jl
Hsin-Yi Chen, USM DMS
Internal-tide strength in several models: vertical-mode decomposition of the baroclinic pressure and velocity and a
tidal harmonic fit of the modal amplitudes (M2 is the target; S2, N2, K1, O1 are fitted with it so they do not leak
into M2), then the energy and energy flux of each constituent and mode. Same method as DFF/bcef_modes_puv.jl +
DFF/mode_bcef_harmonic.jl, written once for every model (ROMS his, HYCOM, NCOM, GLORYS); see m2_modes_fun.jl.

Pass 1  mean state from records every P1_DT = 6 h: in-situ rho, SA, CT, level depths -> N^2 (GSW) -> modes phi_n, c_n
Pass 2  every record: rho' at the actual depth, hydrostatic p', depth means removed, projections
            pn = (1/H) int p' phi_n dz,  un, vn likewise (east, north)
        accumulated straight into the normal equations of the fit (no time series stored), workers in parallel
Fit     y = mean + trend + sum_k A'_k cos(w_k (t - tc) - g'_k)  (as function/harm_fit.jl), every point, field, mode
Energetics per constituent and mode, from the fit (time means over a tidal cycle):
        HKE = 1/4 rho0 H (A_u^2 + A_v^2)          [J m-2]
        APE = 1/4 H A_p^2 / (rho0 c_n^2)          [J m-2]   (hydrostatic mode: int N^2 xi^2 = H p^2 / (rho0 c^2)^2 ...)
        F   = 1/2 H A_p A_u cos(g_p - g_u), 1/2 H A_p A_v cos(g_p - g_v)   [W m-1], east / north
Output: M2_OUT/<SRC>_modes_harmonic[_s<n>].nc on the model's own grid (cut to the 2700 m ROMS box), and a text report.
Figures and the common-area comparison: DFF/m2_modes_models_plot.jl.

Settings: see m2_modes_fun.jl (M2_HOST, M2_SRC, M2_T0, M2_T1, M2_NMODE, M2_CONS, M2_STRIDE, M2_OUT);
  M2_NWORK worker processes (default SLURM_CPUS_PER_TASK - 1, else 4), M2_HEAP heap hint per worker (default 6G).

Usage:
  expanse:  sbatch sbatch_files/julia_m2_modes.sh                               # ROMS_HYCOMparent, ROMS_GLORYSparent, GLORYS
            M2_SRC=GLORYS M2_NWORK=4 julia --project=. DFF/m2_modes_models.jl
  aurelia:  cd ~/Documents/Julia
            M2_SRC=HYCOM,NCOM2700 M2_NWORK=24 nohup julia --project=. -t 8 DFF/m2_modes_models.jl > m2modes.log 2>&1 &
  test:     M2_STRIDE=10 M2_T1=2022-09-10 M2_CONS=M2,K1 ...
=#

using Distributed
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
const NWORK = parse(Int, get(ENV, "M2_NWORK", string(max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "5")) - 1))))
nprocs() == 1 && NWORK > 1 && addprocs(NWORK; exeflags = ["--project=$(Base.active_project())", "--heap-size-hint=$(get(ENV, "M2_HEAP", "6G"))"])
const FUN = joinpath(@__DIR__, "m2_modes_fun.jl")
@everywhere include($FUN)
println("host $HOST, $(nworkers()) workers, window $T0 .. $T1, $NMODE modes, stride $STRIDE, output $OUT")

function run_source(name)
    t_start = time(); sname = srcname(name)
    println("===== $sname")
    @everywhere (global ST = setup($name))
    st = Main.ST; nx, ny = size(st.kb); nt = length(st.tsec)
    dth = median(diff(st.tsec)) / 3600
    rep = String[]; say(s) = (println(s); push!(rep, s))
    say(@sprintf("%s (%s): %d records %s .. %s, dt %.2f h; grid %d x %d x %d, %d columns used; constituents %s",
                 sname, st.src.label, nt, st.times[1], st.times[end], dth, nx, ny, st.nk, count(>(0), st.kb), join(st.cons, " ")))
    nt > 2 + 2length(st.cons) || error("too few records")

    # pass 1: mean state, records every P1_DT hours
    step = max(1, round(Int, P1_DT / dth))
    units1 = filter(u -> !isempty(u.recs), [(file = u.file, recs = filter(r -> (r[2] - 1) % step == 0, u.recs)) for u in st.units])
    t1 = time()
    S = @distributed (addsums) for u in units1
        pass1_unit(Main.ST, u)
    end
    ms = mean_state(st, S); S = nothing; GC.gc()
    say(@sprintf("pass 1: %d records (every %d), %.1f min; modes in %d columns", sum(length(u.recs) for u in units1), step, (time() - t1) / 60, count(>(0), ms.kb)))
    for q in 1:NMODE
        cq = filter(isfinite, ms.c[:, :, q]); deep = filter(isfinite, ms.c[:, :, q][ms.H .> 1000])
        say(@sprintf("  mode %d: c median %.2f m/s (h > 1000 m: %.2f m/s)", q, median(cq), isempty(deep) ? NaN : median(deep)))
    end
    @everywhere (global MS = $ms)

    # pass 2: modal amplitudes into the normal equations
    tc = (st.tsec[1] + st.tsec[end]) / 2; span = (st.tsec[end] - st.tsec[1]) / 2
    t2 = time()
    R = @distributed (addsums) for u in Main.ST.units
        pass2_unit(Main.ST, Main.MS, u, tc, span)
    end
    @everywhere (global MS = nothing; GC.gc())
    say(@sprintf("pass 2: %d records, %.1f min; %d columns with a missing record (dropped)", nt, (time() - t2) / 60, count(>(0), R.NB)))

    # fit: C = (A'A) \ (A'y) for every field and point
    K = length(st.cons); np = 2 + 2K; nf = 3NMODE
    GA = zeros(np, np); for t in st.tsec; a = design_row(t, tc, span, st.cons); GA .+= a * a'; end
    F = cholesky(Symmetric(GA))
    Bm = reshape(R.B, nf, np, nx * ny)
    amp = fill(NaN, nx, ny, nf, K); pha = fill(NaN, nx, ny, nf, K); resid = fill(NaN, nx, ny, nf); sdev = fill(NaN, nx, ny, nf)
    good = (ms.kb .> 0) .& (R.NB .== 0)
    for p in findall(vec(good)), f in 1:nf
        b = Bm[f, :, p]; cvec = F \ b
        i, j = Tuple(CartesianIndices((nx, ny))[p])
        for k in 1:K
            a_, b_ = cvec[2k + 1], cvec[2k + 2]
            amp[i, j, f, k] = hypot(a_, b_); pha[i, j, f, k] = mod(rad2deg(atan(b_, a_)), 360)
        end
        ss = R.SS[f, i, j]
        resid[i, j, f] = sqrt(max(ss - dot(cvec, b), 0) / (nt - np))
        sdev[i, j, f] = sqrt(max(ss / nt - (b[1] / nt)^2, 0))
    end

    # energetics per constituent and mode
    H = ms.H; c = ms.c
    HKE = fill(NaN, nx, ny, NMODE, K); APE = similar(HKE); Fx = similar(HKE); Fy = similar(HKE)
    for k in 1:K, q in 1:NMODE
        Ap, gp = amp[:, :, 3q - 2, k], pha[:, :, 3q - 2, k]
        Au, gu = amp[:, :, 3q - 1, k], pha[:, :, 3q - 1, k]
        Av, gv = amp[:, :, 3q, k], pha[:, :, 3q, k]
        HKE[:, :, q, k] .= 0.25 .* RHO0 .* H .* (Au .^ 2 .+ Av .^ 2)
        APE[:, :, q, k] .= 0.25 .* H .* Ap .^ 2 ./ (RHO0 .* c[:, :, q] .^ 2)
        Fx[:, :, q, k] .= 0.5 .* H .* Ap .* Au .* cos.(deg2rad.(gp .- gu))
        Fy[:, :, q, k] .= 0.5 .* H .* Ap .* Av .* cos.(deg2rad.(gp .- gv))
    end
    km = findfirst(==("M2"), st.cons)
    if km !== nothing
        for q in 1:NMODE
            e = filter(isfinite, (HKE[:, :, q, km] .+ APE[:, :, q, km])[H .> 200]); f = filter(isfinite, hypot.(Fx[:, :, q, km], Fy[:, :, q, km])[H .> 200])
            say(@sprintf("  M2 mode %d (h > 200 m, own grid): median E %.1f J/m2 (max %.0f), median |F| %.1f W/m (max %.0f)", q,
                         median(e), maximum(e), median(f), maximum(f)))
        end
    end

    # save
    mkpath(OUT); fout = OUT * "$(sname)_modes_harmonic$(SFX).nc"; isfile(fout) && rm(fout)
    fname_f = vcat([["p$q", "u$q", "v$q"] for q in 1:NMODE]...)
    NCDataset(fout * ".tmp", "c") do ds
        defDim(ds, "xi", nx); defDim(ds, "eta", ny); defDim(ds, "mode", NMODE); defDim(ds, "con", K); defDim(ds, "field", nf)
        ds.attrib["title"] = "Vertical-mode harmonic fit of baroclinic p, u, v: $(st.src.label), $(Date(st.times[1])) to $(Date(st.times[end]))"
        ds.attrib["source"] = sname; ds.attrib["label"] = st.src.label; ds.attrib["data"] = st.src.data
        ds.attrib["method"] = "DFF/m2_modes_models.jl: TEOS-10 in-situ rho' at the actual depth, hydrostatic p', depth means removed, " *
                              "projection on phi_n ((1/H) int phi_m phi_n dz = delta_mn), plain sinusoid fit (harm_fit form), u, v east / north"
        ds.attrib["energetics"] = "HKE = rho0 H (A_u^2 + A_v^2) / 4, APE = H A_p^2 / (4 rho0 c^2), F = H A_p A_u cos(g_p - g_u) / 2; rho0 = $RHO0"
        ds.attrib["tc"] = string(T_REF + Millisecond(round(Int, 1000tc))); ds.attrib["records"] = nt; ds.attrib["dt_hours"] = dth
        ds.attrib["span_hours"] = 2span / 3600; ds.attrib["stride"] = STRIDE; ds.attrib["nmode"] = NMODE
        defVar(ds, "con", st.cons, ("con",)); defVar(ds, "field", fname_f, ("field",))
        defVar(ds, "lon", st.lon, ("xi", "eta"); attrib = ["units" => "degrees_east"])
        defVar(ds, "lat", st.lat, ("xi", "eta"); attrib = ["units" => "degrees_north"])
        defVar(ds, "H", Float32.(H), ("xi", "eta"); attrib = ["units" => "m", "long_name" => "mean water depth of the model column"])
        defVar(ds, "used", Int8.(good), ("xi", "eta"); attrib = ["long_name" => "1 = column analysed"])
        defVar(ds, "c", Float32.(c), ("xi", "eta", "mode"); attrib = ["units" => "m s-1", "long_name" => "mode eigenspeed (no rotation)"])
        defVar(ds, "amp", Float32.(amp), ("xi", "eta", "field", "con"); attrib = ["units" => "Pa (p), m s-1 (u, v)", "long_name" => "amplitude A' of the modal amplitude"], deflatelevel = 3)
        defVar(ds, "pha", Float32.(pha), ("xi", "eta", "field", "con"); attrib = ["units" => "deg", "long_name" => "phase g' relative to tc"], deflatelevel = 3)
        defVar(ds, "resid_std", Float32.(resid), ("xi", "eta", "field"); attrib = ["long_name" => "residual std of the fit"], deflatelevel = 3)
        defVar(ds, "std", Float32.(sdev), ("xi", "eta", "field"); attrib = ["long_name" => "std of the modal amplitude"], deflatelevel = 3)
        for (nm, v, u, ln) in (("HKE", HKE, "J m-2", "depth-integrated horizontal kinetic energy"), ("APE", APE, "J m-2", "depth-integrated available potential energy"),
                               ("Fx", Fx, "W m-1", "depth-integrated energy flux, east"), ("Fy", Fy, "W m-1", "depth-integrated energy flux, north"))
            defVar(ds, nm, Float32.(v), ("xi", "eta", "mode", "con"); attrib = ["units" => u, "long_name" => ln], deflatelevel = 3)
        end
    end
    mv(fout * ".tmp", fout; force = true)
    say(@sprintf("wrote %s (%.1f min)", fout, (time() - t_start) / 60))
    write(OUT * "$(sname)_modes_harmonic$(SFX).txt", join(rep, "\n") * "\n")
    @everywhere (global ST = nothing; GC.gc())
end

for name in split(get(ENV, "M2_SRC", P.default), ",")
    run_source(String(name))
end

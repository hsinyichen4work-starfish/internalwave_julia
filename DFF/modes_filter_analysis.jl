#= modes_filter_analysis.jl
Hsin-Yi Chen, USM DMS
Energy of the modal internal tide from band-passed time series (no harmonic fit, no DFF): the modal pressure and
velocity pn, un, vn (bcef_modes_puv.jl, hourly, the vertical-mode projections of p', u', v') are time filtered with a
Butterworth filter, the same bands and order as the flux pieces of energy_flux/model_check_energyflux.jl:
    M2   band-pass 9-15 h      hp28  high-pass 28 h      hp9   high-pass 9 h       (order 4, filtfilt: zero phase,
                                                                                     twice the roll-off)
The harmonic fit keeps only the energy phase-locked to the tidal constituents; a band keeps everything in it.
The filter runs over the whole record (from 2022-08-24); the 3-day spin-up (as the harmonic fit) is dropped after
filtering, so the start-up transient of the filter falls in the dropped part.

Steps (MF_STEPS, ":" also works in sbatch --export):
  1  read: grid, structure (H, c_n, φ_n at the bottom), the puv record and the analysis window; read pn, un, vn of
     each mode in MF_MODES (default 1,2), one at a time, and print size, memory and values (ready for the filter)
     ubar, vbar (bt_uvz.nc, same hourly record) -> hourly W_BT = -(ubar dh/dx + vbar dh/dy) (bt_gradh.nc), the
     barotropic forcing of the conversion (filtered as one field later: the slope is fixed in time)
  2  Butterworth filter of pn, un, vn (MF_MODES) and W_BT for every band -> <bcef_dir>/model_and_filter/filt_<band>_<field>.nc
     (analysis window only; batch job: sbatch_files/julia_modes_filter.sh). MF_TEST_ROWS=255: a few rows, nothing saved.
  3  surface velocity of the modes, u_n(x, y, 0, t) = u_n(t) φ_n(0): maps of u, v (east / north) and speed with arrows,
     mode 1 / mode 2 / modes 1+2, for the raw record ("all") and each band (the step-2 files): MF_SNAP=2022-09-06T12:00
     one snapshot per set -> ~/figure_check (test); MF_SNAP= (empty) -> movies (frames removed after) -> MF_MOVIE_DIR
  4  energy budget, KE_n = ½ρ0H<un²+vn²> (and its u, v parts) per band and mode, time means (full window, Aug, Sep,
     Oct, Nov) -> model_and_filter/filt_KE.nc; strip average (±200 km) and figures -> MF_E_FIG (figure/<run>/mode_filter)
  5  energy budget, APE_n = H<pn²>/(2ρ0c_n²), the same way -> model_and_filter/filt_APE.nc and figures (with KE for comparison)
  6  energy flux F_n = H<pn un>, H<pn vn> (grid xi / eta), the same way -> model_and_filter/filt_F.nc and figures
  7  flux divergence ∇·F_n = pm pn [∂ξ(Fx/pn) + ∂η(Fy/pm)] of the step-6 maps -> model_and_filter/filt_divF.nc and figures
  8  conversion conv_n = φ_n(-H) <W_BT pn> (barotropic to mode n) -> model_and_filter/filt_conv.nc and figures (with ∇·F)
  9  residual D_n = conv_n - ∇·F_n (steps 7, 8) -> model_and_filter/filt_D.nc, figures and a summary of the strip integrals
 10  RMS surface speed √<u_s² + v_s²>, u_s = Σ u_n φ_n(0), mode 1 / mode 2 / modes 1+2 (with the cross term), raw record and
     each band, time means by period -> model_and_filter/filt_surfvel.nc, heatmaps and strip lines (batch: MF_STEPS=10)
Usage:
  julia DFF/modes_filter_analysis.jl                                          # amazon_2700m_cmems_0801 (ddf_bcef_0824)
  DFF_RUN=amazon_900m_3mon_1008_2 julia DFF/modes_filter_analysis.jl          # 900 m (a field is 5.5 GB: batch job)
=#
using NCDatasets, Statistics, Dates, Printf, DSP
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
haskey(ENV, "DFF_RUN") || (ENV["DFF_RUN"] = "amazon_2700m_cmems_0801")
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")               # dff_run, run_dir, run_grid_fname, fig_dir
include("/home/hchen54/internalwave_julia/function/butter_filters.jl")   # bandpass_butter, lowhighpass_butter

# ============================================================================
# (1) settings, paths, grid, structure, the puv record
# ============================================================================
NMODE = parse(Int, get(ENV, "NMODE", "3"))                     # modes in the decomposition (file names: bcef_modes3_*)
# modes analysed here (default 1 and 2: most of the energy in this region; MF_MODES=1:2:3 for all three)
modes = parse.(Int, split(get(ENV, "MF_MODES", "1,2"), r"[,:; ]+"; keepempty = false))
all(1 .<= modes .<= NMODE) || error("MF_MODES = $modes: the decomposition has modes 1-$NMODE")
default_bcef = Dict("amazon_2700m_cmems_0801" => "ddf_bcef_0824")
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, get(default_bcef, dff_run, "ddf_bcef")))
puv_fname    = joinpath(bcef_dir, "bcef_modes$(NMODE)_puv.nc")
struct_fname = joinpath(bcef_dir, "bcef_modes$(NMODE)_structure.nc")
btuv_fname   = joinpath(bcef_dir, "bt_uvz.nc")       # hourly ubar, vbar on the rho points (W_BT_make.jl step 1)
gradh_fname  = joinpath(bcef_dir, "bt_gradh.nc")     # dh/dx, dh/dy along grid xi / eta (W_BT_make.jl step 3)
for f in (puv_fname, struct_fname)
    isfile(f) || error("missing $f: run bcef_modes_puv.jl first")
end
for f in (btuv_fname, gradh_fname)
    isfile(f) || error("missing $f: run W_BT_make.jl (BT_STEPS=1:3) first")
end
mf_steps = parse.(Int, split(get(ENV, "MF_STEPS", "1"), r"[,:; ]+"; keepempty = false))
mf_fig_dir = joinpath(dirname(fig_dir), "mode_filter")   # figures / movies of steps 3-9: figure/<run>/mode_filter
SPINUP = parse(Float64, get(ENV, "SPINUP", "3.0"))   # [days] dropped at the start, as mode_bcef_harmonic.jl

# time filters: the table of energy_flux/model_check_energyflux.jl (edit here to change a band or add one)
#   kind "band": T = (short, long) periods kept in between; "high": periods shorter than T; "low": longer than T
#   T in hours; N = Butterworth order (filtfilt: forward and backward, zero phase, twice the roll-off)
filters = [
    (name = "M2",   kind = "band", T = (9, 15), N = 4),
    (name = "hp28", kind = "high", T = 28,      N = 4),
    (name = "hp9",  kind = "high", T = 9,       N = 4),
]

# --- grid (rho points) and the vertical-mode structure
lon, lat, pm, pn, angle, mask = NCDataset(run_grid_fname) do g
    [Float64.(g[v][:, :]) for v in ("lon_rho", "lat_rho", "pm", "pn", "angle", "mask_rho")]
end
h, zeta_mean, c_mode, phi_bot, phi_top = NCDataset(struct_fname) do s
    ns = s.dim["s_rho"]
    Float64.(s["h"][:, :]), Float64.(s["zeta_mean"][:, :]), Float64.(s["c"][:, :, :]),
    Float64.(s["phi"][:, :, 1, :]),                    # φ_n at the bottom rho level (s_rho = 1), (xi, eta, mode)
    Float64.(s["phi"][:, :, ns, :])                    # φ_n at the top rho level (surface), (xi, eta, mode)
end
H = h .+ zeta_mean
nx, ny = size(h)

# --- the puv record: times, sampling, the analysis window
ds_puv = NCDataset(puv_fname)
size(ds_puv["p1"]) == (nx, ny, ds_puv.dim["time"]) || error("puv $(size(ds_puv["p1"])) does not match the grid $nx x $ny")
nt = ds_puv.dim["time"]
date = ds_puv["ocean_time"][:]                                   # DateTime
dt_hr = Dates.value(date[2] - date[1]) / 3.6e6                   # [h]
all(diff(Dates.value.(date)) .== Dates.value(date[2] - date[1])) || error("puv records are not evenly spaced")
t_keep = findfirst(>=(date[1] + Millisecond(round(Int, SPINUP * 86400e3))), date):nt   # analysis window (after filtering)
println("$dff_run: $puv_fname")
println("  grid $nx x $ny, $(count(mask .> 0)) wet points; modes $(join(modes, ",")) of $NMODE; h $(round.(extrema(h[mask .> 0]), digits = 0)) m")
println("  puv: $nt records every $dt_hr h, $(date[1]) -> $(date[end])")
println("  analysis window (after the filter, $(SPINUP) days dropped): $(length(t_keep)) records, $(date[t_keep[1]]) -> $(date[t_keep[end]])")
println("  filters: ", join(["$(f.name) ($(f.kind) $(f.T) h, order $(f.N))" for f in filters], ", "))

# --- barotropic velocity (hourly, rho points, grid xi / eta) and the bottom slope: W_BT = -(ubar dh/dx + vbar dh/dy).
# The slope is fixed in time, so the filter (linear) of W_BT equals W_BT of the filtered ubar, vbar: W_BT is formed
# hourly first and filtered as one field. The bt record must be the puv record (same start, step and length).
ds_bt = NCDataset(btuv_fname)
size(ds_bt["ubar"]) == (nx, ny, nt) || error("bt_uvz $(size(ds_bt["ubar"])) does not match puv $((nx, ny, nt))")
date_bt = ds_bt["ocean_time"][:]
date_bt == date || error("bt_uvz times differ from puv: $(date_bt[1]) -> $(date_bt[end]) vs $(date[1]) -> $(date[end])")
dhdx, dhdy = NCDataset(g -> (Float64.(g["dhdx"][:, :]), Float64.(g["dhdy"][:, :])), gradh_fname)
slope_ok = isfinite.(dhdx) .& isfinite.(dhdy)                  # NaN on and next to land (W_BT_make.jl)
dhdx[.!slope_ok] .= 0; dhdy[.!slope_ok] .= 0
println("  barotropic: $btuv_fname, same $nt records; slope from $gradh_fname ($(count(.!slope_ok)) points without a slope: W_BT = 0)")

# one modal field (name "p1", "u2", ...) as Float32 (xi, eta, time); land, fill and non-finite values 0 (the filter
# needs finite numbers; land is masked again afterwards). Read raw (.var: no Union{Missing} copy) into one
# preallocated array, nblk records at a time, so the peak is the field itself (1.8 GB on the 2.7 km grid).
# Retries for Lustre timeouts (not for OutOfMemoryError). read_mode!(a, name) refills an existing array: reuse one
# buffer for field after field (freed arrays are not returned to the system at once, and the login node caps at 8 GB).
read_mode(name; kw...) = read_mode!(Array{Float32}(undef, nx, ny, nt), name; kw...)
read_mode!(a, name; kw...) = read_var!(a, ds_puv, name; kw...)
function read_var!(a, ds, name; nblk = 100, tries = 5)
    v = ds[name].var
    fv = haskey(v.attrib, "_FillValue") ? Float32(v.attrib["_FillValue"]) : nothing
    size(a) == (nx, ny, nt) || error("read_mode!: buffer $(size(a)), field $((nx, ny, nt))")
    for k1 in 1:nblk:nt
        k2 = min(k1 + nblk - 1, nt)
        for t in 1:tries
            try
                NCDatasets.load!(v, view(a, :, :, k1:k2), :, :, k1:k2)
                break
            catch e
                (e isa OutOfMemoryError || t == tries) && rethrow()
                @warn "read $name records $k1-$k2 failed ($(sprint(showerror, e)[1:min(end, 80)])), retry $t"
                sleep(10t)
            end
        end
    end
    @inbounds for i in eachindex(a)
        x = a[i]
        (!isfinite(x) || (fv !== nothing && x == fv)) && (a[i] = 0f0)
    end
    a
end

# hourly W_BT = -(ubar dh/dx + vbar dh/dy) [m s-1] into w (xi, eta, time), built nblk records at a time from ubar,
# vbar (bt_uvz.nc), so only w is field-sized (the login node cannot hold two 1.7 GB fields). Points without a slope: 0.
# nblk: 24 records on the login node (8 GB virtual memory cap), 100 in a batch job (4x fewer, faster reads)
function read_wbt!(w; nblk = haskey(ENV, "SLURM_JOB_ID") ? 100 : 24, tries = 5)
    size(w) == (nx, ny, nt) || error("read_wbt!: buffer $(size(w)), field $((nx, ny, nt))")
    vu, vv = ds_bt["ubar"].var, ds_bt["vbar"].var
    gx, gy = Float32.(dhdx), Float32.(dhdy)
    ub, vb = Array{Float32}(undef, nx, ny, nblk), Array{Float32}(undef, nx, ny, nblk)
    for k1 in 1:nblk:nt
        k2 = min(k1 + nblk - 1, nt); m = k2 - k1 + 1
        for t in 1:tries
            try
                NCDatasets.load!(vu, view(ub, :, :, 1:m), :, :, k1:k2)
                NCDatasets.load!(vv, view(vb, :, :, 1:m), :, :, k1:k2)
                break
            catch e
                (e isa OutOfMemoryError || t == tries) && rethrow()
                @warn "read ubar / vbar records $k1-$k2 failed ($(sprint(showerror, e)[1:min(end, 80)])), retry $t"
                sleep(10t)
            end
        end
        @inbounds for k in 1:m, j in 1:ny, i in 1:nx
            u, v = ub[i, j, k], vb[i, j, k]
            x = -(u * gx[i, j] + v * gy[i, j])
            w[i, j, k1 + k - 1] = isfinite(x) ? x : 0f0
        end
    end
    w
end

if 1 in mf_steps
    wet = mask .> 0
    deep = wet .& (H .> 1000)
    ic, jc = cld(nx, 2), cld(ny, 2)
    println("\nstep 1: read the modal fields of modes $(join(modes, ",")) (one at a time); deep = H > 1000 m, centre ($ic, $jc), H = $(round(Int, H[ic, jc])) m")
    a = Array{Float32}(undef, nx, ny, nt)                      # one buffer for every field
    for n in modes, name in ("p$n", "u$n", "v$n")
        local t0 = time()
        read_mode!(a, name)
        s = vec(std(view(a, :, :, t_keep); dims = 3))[vec(deep)]
        @printf("  %s: %s, %.2f GB, %.0f s; non-finite %d, land all 0 %s; std over the window (deep) median %.3g, max %.3g %s; centre std %.3g\n",
                name, string(size(a)), sizeof(a) / 2^30, time() - t0, count(!isfinite, a),
                all(i -> all(==(0f0), view(a, i, :)), findall(.!wet)), median(s), maximum(s), ds_puv[name].attrib["units"],
                std(view(a, ic, jc, t_keep)))
        get(ENV, "MF_MEMLOG", "0") == "1" && @printf("    memory: live %.2f GB, max RSS %.2f GB, %s\n", Base.gc_live_bytes() / 2^30, Sys.maxrss() / 2^30, strip(filter(l -> startswith(l, "VmSize"), readlines("/proc/self/status"))[1]))
    end
    # barotropic: ubar, vbar -> hourly W_BT, into the same buffer
    local t0 = time()
    read_wbt!(a)
    sd = vec(std(view(a, :, :, t_keep); dims = 3))
    k = argmax(sd .* vec(deep))
    @printf("  W_BT = -(ubar dh/dx + vbar dh/dy): %.0f s; non-finite %d; std over the window (deep) median %.3g, 99th pct %.3g mm/s;\n",
            time() - t0, count(!isfinite, a), 1e3 * median(sd[vec(deep)]), 1e3 * quantile(sd[vec(deep)], 0.99))
    @printf("    largest (deep) %.3g mm/s at (xi, eta) = %s, h = %.0f m, |grad h| = %.3f; centre std %.3g mm/s\n",
            1e3 * sd[k], string(Tuple(CartesianIndices((nx, ny))[k])), h[k], hypot(dhdx[k], dhdy[k]), 1e3 * std(view(a, ic, jc, t_keep)))
    a = nothing; GC.gc()
    @printf("  c_n and bottom φ_n over the deep points: %s\n",
            join(["mode $n: c $(round.(extrema(c_mode[:, :, n][deep]); digits = 2)) m/s, φ(-H) $(round.(extrema(phi_bot[:, :, n][deep]); digits = 2))" for n in modes], "; "))
end

# ============================================================================
# (2) Butterworth filter of pn, un, vn (MF_MODES) and W_BT, every band -> <bcef_dir>/model_and_filter/
# ============================================================================
# Each field is read once (whole record from 2022-08-24), filtered along time for every band (filtfilt of the table
# `filters`, rows of the grid in parallel threads, Float64 inside the filter) and the analysis window t_keep (spin-up
# dropped) is saved: model_and_filter/filt_<band>_<field>.nc, <field> = p1, u1, v1, p2, u2, v2, wbt; variable named as the
# field, (xi_rho, eta_rho, time), Float32. A file already there and marked complete is skipped (rerun to finish a job).
# Peak memory: the field and its filtered window (3.3 GB on the 2.7 km grid, 11 GB on 900 m): batch job
# (sbatch_files/julia_modes_filter.sh). Size: 7 fields x 3 bands x 1.6 GB = 34 GB (2.7 km), ~110 GB (900 m).
# MF_TEST_ROWS=255 (or 200:300 for a range): filter only those grid rows, print the band variances, save nothing
# (runs on the login node, one thread: W_BT needs julia -t 1 there). MF_FIELDS=p1:wbt: only those fields. Before the fields: the gain of each band at tidal periods (synthetic sines, record length nt).
filt_dir = joinpath(bcef_dir, "model_and_filter")
apply_filter(y, flt) = flt.kind == "band" ? bandpass_butter(y, flt.T[1], flt.T[2], dt_hr, flt.N) :
                                            lowhighpass_butter(y, flt.T, dt_hr, flt.N, flt.kind)
# filter rows js of a (xi, eta, time) -> o[:, j, :] over the window t_keep (o: (xi, eta or rows, window))
function filter_rows!(o, a, flt, js; out_row = j -> j)
    Threads.@threads for j in js
        y = Float64.(permutedims(view(a, :, j, :)))           # (time, xi): the filter works along dim 1
        yf = apply_filter(y, flt)
        @views o[:, out_row(j), :] .= Float32.(permutedims(yf[t_keep, :]))
    end
    o
end
filt_fname(flt, fld) = joinpath(filt_dir, "filt_$(flt.name)_$(fld).nc")
fld_units = Dict('p' => "Pa", 'u' => "m s-1", 'v' => "m s-1", 'w' => "m s-1")
fld_desc(fld) = fld == "wbt" ? "barotropic bottom vertical velocity W_BT = -(ubar dh/dx + vbar dh/dy)" :
    "mode-$(fld[2:end]) amplitude of the baroclinic " * Dict('p' => "pressure p'", 'u' => "velocity u' along grid xi", 'v' => "velocity v' along grid eta")[fld[1]]
flt_desc(flt) = flt.kind == "band" ? "Butterworth band-pass $(flt.T[1])-$(flt.T[2]) h, order $(flt.N), filtfilt" :
                                     "Butterworth $(flt.kind)-pass $(flt.T) h, order $(flt.N), filtfilt"
is_done(f) = isfile(f) && NCDataset(d -> get(d.attrib, "complete", 0) == 1, f)

if 2 in mf_steps
    test_env = get(ENV, "MF_TEST_ROWS", "")
    test_rows = isempty(test_env) ? Int[] : (occursin(":", test_env) ?
        (r = parse.(Int, split(test_env, ":")); collect(r[1]:r[end])) : parse.(Int, split(test_env, r"[, ]+")))
    is_test = !isempty(test_rows)
    println("\nstep 2: Butterworth filter, $(Threads.nthreads()) threads; ", is_test ? "TEST rows $(extrema(test_rows)), nothing saved" : "-> $filt_dir")
    # gain of each band (amplitude, filtfilt) at tidal periods: synthetic sines over the record, middle half
    let tt = 0:dt_hr:(nt - 1) * dt_hr, mid = (nt ÷ 4):(3nt ÷ 4)
        pers = [("M4", 6.21), ("hp9 cut", 9.0), ("S2", 12.0), ("M2", 12.42), ("N2", 12.66), ("K1", 23.93), ("O1", 25.82), ("2 days", 48.0)]
        println("  gain (amplitude) at:  ", join([@sprintf("%8s", p) for (p, _) in pers]))
        for flt in filters
            g = [(y = sin.(2π .* tt ./ T); yf = apply_filter(collect(y), flt); std(yf[mid]) / std(y[mid])) for (_, T) in pers]
            println(@sprintf("  %-6s", flt.name), "              ", join([@sprintf("%8.3f", x) for x in g]))
        end
    end
    fields = [[f * string(n) for n in modes for f in ("p", "u", "v")]; "wbt"]
    haskey(ENV, "MF_FIELDS") && (fields = String.(split(ENV["MF_FIELDS"], r"[,:; ]+"; keepempty = false)))   # e.g. MF_FIELDS=wbt
    is_test || mkpath(filt_dir)
    wet = mask .> 0; deep = wet .& (H .> 1000)
    a = Array{Float32}(undef, nx, ny, nt)
    o = is_test ? Array{Float32}(undef, nx, length(test_rows), length(t_keep)) : Array{Float32}(undef, nx, ny, length(t_keep))
    for fld in fields
        if !is_test && all(flt -> is_done(filt_fname(flt, fld)), filters)
            println("  $fld: every band done, skipped"); continue
        end
        t0 = time()
        fld == "wbt" ? read_wbt!(a) : read_mode!(a, fld)
        raw_var = is_test ? [var(view(a, :, j, t_keep); dims = 2) for j in test_rows] : nothing
        @printf("  %s: read %.0f s\n", fld, time() - t0)
        for flt in filters
            fname = filt_fname(flt, fld)
            (!is_test && is_done(fname)) && (println("    $(flt.name): done, skipped"); continue)
            t0 = time()
            if is_test
                filter_rows!(o, a, flt, test_rows; out_row = j -> j - test_rows[1] + 1)
                fr = [mean(vec(var(view(o, :, k, :); dims = 2))[deep[:, j]] ./ raw_var[k][deep[:, j]]) for (k, j) in enumerate(test_rows) if any(deep[:, j])]
                @printf("    %-5s %.0f s; share of the variance in the band (deep points of the test rows): mean %.3f (%.3f - %.3f)\n",
                        flt.name, time() - t0, mean(fr), extrema(fr)...)
                continue
            end
            filter_rows!(o, a, flt, 1:ny)
            tf = time() - t0
            tmp = fname * ".tmp"; isfile(tmp) && rm(tmp)
            NCDataset(tmp, "c") do d
                defDim(d, "xi_rho", nx); defDim(d, "eta_rho", ny); defDim(d, "time", length(t_keep))
                d.attrib["title"] = "$(fld_desc(fld)), $(flt_desc(flt)), $dff_run"
                d.attrib["filter"] = flt_desc(flt)
                d.attrib["band"] = flt.name
                d.attrib["window"] = "$(date[t_keep[1]]) to $(date[t_keep[end]]) ($(length(t_keep)) records; filtered over $(date[1]) to $(date[end]), $(SPINUP) days dropped after)"
                d.attrib["source"] = fld == "wbt" ? "$btuv_fname, $gradh_fname" : puv_fname
                d.attrib["script"] = "internalwave_julia/DFF/modes_filter_analysis.jl (step 2)"
                d.attrib["complete"] = 0
                defVar(d, "ocean_time", date[t_keep], ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
                v = defVar(d, fld, Float32, ("xi_rho", "eta_rho", "time"); chunksizes = [nx, ny, 1],
                           attrib = ["units" => fld_units[fld[1]], "long_name" => "$(fld_desc(fld)), $(flt.name) band"])
                for k1 in 1:100:length(t_keep)
                    k2 = min(k1 + 99, length(t_keep))
                    v.var[:, :, k1:k2] = view(o, :, :, k1:k2)
                end
                d.attrib["complete"] = 1
            end
            mv(tmp, fname; force = true)
            sd = vec(std(o; dims = 3))[vec(deep)]
            @printf("    %-5s filter %.0f s, write %.0f s; std (deep) median %.3g, max %.3g %s -> %s\n", flt.name, tf, time() - t0 - tf,
                    median(sd), maximum(sd), fld_units[fld[1]], basename(fname))
        end
        GC.gc()
    end
    a = o = nothing; GC.gc()
end

# ============================================================================
# (3) surface velocity of the modes: snapshots / movies, raw record and each band
# ============================================================================
# u_n(x, y, 0, t) = u_n(t) φ_n(z = 0) (φ_n at the top rho level, (1/H) ∫ φ² dz = 1), the same for v; modes 1, 2 and
# their sum. Sets: band "all" (the raw puv record, no filter: includes the subtidal flow) and every band of `filters`
# (model_and_filter/filt_<band>_<u|v><n>.nc, step 2). u, v rotated from grid xi / eta to east / north (angle).
# Figure: 1 row x 3 columns, u | v | speed with arrows; moorings (yellow), 200 m isobath; shelf h < 200 m grey.
# Colour ranges fixed per set (99th percentile of |u|, |v|, speed over deep water in 24 records across the window),
# so a movie's frames compare.
#   MF_SNAP=2022-09-06T12:00 (default): one figure per set -> MF_SNAP_DIR (default ~/figure_check)
#   MF_SNAP= (empty): one movie per set, MF_MOVIE_STRIDE hours apart (default 1), MF_FPS (default 12)
#     -> MF_MOVIE_DIR (default figure/<run>/mode_filter); the png frames are deleted after.
#   MF_BANDS (default all:M2:hp28:hp9), MF_SETS (default 1:2:12; "12" = modes 1+2)
if 3 in mf_steps
    using CairoMakie
    include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")       # topdown_axis, plot_curvilinear!, quiver_curvilinear!
    include("/home/hchen54/internalwave_julia/function/mooring_transect.jl")   # read_moorings
    snap_env = get(ENV, "MF_SNAP", "2022-09-06T12:00")
    bands3 = String.(split(get(ENV, "MF_BANDS", "all:M2:hp28:hp9"), r"[,:; ]+"; keepempty = false))
    sets3 = String.(split(get(ENV, "MF_SETS", "1:2:12"), r"[,:; ]+"; keepempty = false))
    set_modes = Dict("1" => [1], "2" => [2], "12" => [1, 2], "3" => [3], "123" => [1, 2, 3])
    set_lab = Dict("1" => "mode 1", "2" => "mode 2", "12" => "modes 1+2", "3" => "mode 3", "123" => "modes 1+2+3")
    band_lab = Dict("all" => "raw (no filter)", "M2" => "M2 band 9-15 h", "hp28" => "high-pass 28 h", "hp9" => "high-pass 9 h")
    for st in sets3
        all(in(modes), set_modes[st]) || error("set $st needs modes $(set_modes[st]); MF_MODES = $modes")
    end
    # moorings, map frame
    run_mooring = Dict("amazon_2700m_cmems_0801" => "roms_cmems_grd_2700m_mor_edata.nc")
    mooring_file = "/expanse/lustre/projects/uso102/hchen54/input/grid/$(get(run_mooring, dff_run, "roms_grd_900m_mor_edata.nc"))"
    moor_name, moor_lon, moor_lat = read_moorings(mooring_file)
    lon180 = ifelse.(lon .> 180, lon .- 360, lon)
    moor_lon180 = ifelse.(moor_lon .> 180, moor_lon .- 360, moor_lon)
    show = (mask .> 0) .& (h .>= 200)
    masked(x) = ifelse.(show, x, NaN)
    deep3 = show .& (H .> 1000)
    ca, sa = cos.(angle), sin.(angle)
    dx_km = mean(1 ./ pm) / 1e3
    qskip = max(1, round(Int, 36 / dx_km))                       # one arrow every ~36 km
    map_lims = (extrema(lon180) .+ (-0.05, 0.05), extrema(lat) .+ (-0.05, 0.05))
    map_w = 420
    map_h = round(Int, map_w * (map_lims[2][2] - map_lims[2][1]) / ((map_lims[1][2] - map_lims[1][1]) * cosd(mean(map_lims[2]))))
    # one record k of the window (1..length(t_keep)) of field fld ("u1", ...) in band b, Float64 (xi, eta)
    filt_ds = Dict{String, NCDataset}()
    function rec(b, fld, k)
        if b == "all"
            x = Float64.(ds_puv[fld].var[:, :, t_keep[k]])
        else
            f = filt_fname((; name = b), fld)
            isfile(f) || error("no $f: run step 2 first")
            d = get!(() -> NCDataset(f), filt_ds, f)
            x = Float64.(d[fld].var[:, :, k])
        end
        ifelse.(isfinite.(x), x, 0.0)
    end
    # surface u, v (east / north) of a set at record k
    function surf_uv(b, st, k)
        U, V = zeros(nx, ny), zeros(nx, ny)
        for n in set_modes[st]
            U .+= rec(b, "u$n", k) .* phi_top[:, :, n]
            V .+= rec(b, "v$n", k) .* phi_top[:, :, n]
        end
        U .* ca .- V .* sa, U .* sa .+ V .* ca
    end
    # fixed colour ranges per set: 24 records across the window
    function set_ranges(b, st)
        ks = round.(Int, range(1, length(t_keep), length = 24))
        au, as_ = Float64[], Float64[]
        for k in ks
            ue, vn = surf_uv(b, st, k)
            append!(au, abs.(ue[deep3])); append!(au, abs.(vn[deep3])); append!(as_, hypot.(ue, vn)[deep3])
        end
        round(quantile(au, 0.99); sigdigits = 2), round(quantile(as_, 0.99); sigdigits = 2)
    end
    function frame(b, st, k, rng)
        ue, vn = surf_uv(b, st, k)
        sp = hypot.(ue, vn)
        umax, smax = rng
        fig = Figure(size = (3map_w + 180, map_h + 230), fontsize = 15)
        Label(fig[0, 1:3], "Surface velocity u(x, y, 0, t), $(set_lab[st]), $(band_lab[b]); $dff_run   $(Dates.format(date[t_keep[k]], "yyyy-mm-dd HH:MM"))";
              font = :bold, fontsize = 17)
        for (c, (x, ttl, cmap, cr, lab)) in enumerate(((ue, "u (east)", :balance, (-umax, umax), "u [m s⁻¹]"),
                                                      (vn, "v (north)", :balance, (-umax, umax), "v [m s⁻¹]"),
                                                      (sp, "speed with u, v", :viridis, (0, smax), "speed [m s⁻¹]")))
            ax = topdown_axis(fig[1, c]; title = ttl, backgroundcolor = :gray80, width = map_w, height = map_h, aspect = nothing,
                              ylabel = c == 1 ? "Latitude" : "", yticklabelsvisible = c == 1)
            hm = plot_curvilinear!(ax, lon180, lat, masked(x); colormap = cmap, colorrange = cr)
            contour!(ax, lon180, lat, h; levels = [200], color = :gray30, linewidth = 0.7)
            c == 3 && quiver_curvilinear!(ax, lon180, lat, masked(ue), masked(vn); skip = qskip, lengthscale = 0.9 / smax,
                                          color = (:white, 0.85), shaftwidth = 1, tipwidth = 5, tiplength = 5)
            scatter!(ax, moor_lon180, moor_lat; marker = :circle, markersize = 7, color = :yellow, strokecolor = :black, strokewidth = 0.6)
            limits!(ax, map_lims...)
            Colorbar(fig[2, c], hm; vertical = false, flipaxis = false, label = lab, width = map_w - 40)
        end
        Label(fig[3, 1:3], "u_n(x, y, 0, t) = u_n(t) φ_n(0) (bcef_modes_puv.jl, (1/H)∫φ²dz = 1), summed over the modes; rotated to east / north; " *
              "shelf (h < 200 m) grey; yellow: moorings; colour range fixed per movie", fontsize = 12)
        resize_to_layout!(fig)
        fig
    end
    if !isempty(snap_env)   # test: one snapshot per set
        snap_dir = expanduser(get(ENV, "MF_SNAP_DIR", "~/figure_check")); mkpath(snap_dir)
        k = findfirst(==(DateTime(snap_env)), date[t_keep])
        k === nothing && error("MF_SNAP = $snap_env is not a record of the window $(date[t_keep[1]]) .. $(date[t_keep[end]])")
        println("\nstep 3: snapshots at $(date[t_keep[k]]) (window record $k) -> $snap_dir")
        for b in bands3, st in sets3
            rng = set_ranges(b, st)
            fig = frame(b, st, k, rng)
            fname = joinpath(snap_dir, "mode_surf_uv_$(b)_mode$(st)_$(Dates.format(date[t_keep[k]], "yyyymmddHH")).png")
            save(fname, fig; px_per_unit = 1.5); fig = nothing; GC.gc()
            @printf("  %-4s %-9s colour range |u|,|v| %.3g, speed %.3g m/s -> %s\n", b, set_lab[st], rng..., basename(fname))
        end
    else                    # movies
        movie_dir = get(ENV, "MF_MOVIE_DIR", mf_fig_dir)
        mkpath(movie_dir)
        stride = parse(Int, get(ENV, "MF_MOVIE_STRIDE", "2")); fps = parse(Int, get(ENV, "MF_FPS", "12"))
        ks = 1:stride:length(t_keep)
        println("\nstep 3: movies, $(length(ks)) frames each ($(stride) h apart, $fps fps) -> $movie_dir")
        for b in bands3, st in sets3
            rng = set_ranges(b, st)
            frame_dir = mktempdir(movie_dir; prefix = "frames_$(b)_mode$(st)_")
            t0 = time()
            for (i, k) in enumerate(ks)
                fig = frame(b, st, k, rng)
                save(joinpath(frame_dir, @sprintf("f%05d.png", i)), fig; px_per_unit = 1); fig = nothing
                i % 50 == 0 && (GC.gc(); @printf("    %d / %d frames (%.0f s)\n", i, length(ks), time() - t0))
            end
            mname = joinpath(movie_dir, "mode_surf_uv_$(b)_mode$(st).mp4")
            run(`$(Makie.FFMPEG_jll.ffmpeg()) -y -loglevel error -framerate $fps -i $(joinpath(frame_dir, "f%05d.png"))
                 -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" $mname`)
            rm(frame_dir; recursive = true)                           # the png frames are not kept
            @printf("  %-4s %-9s %d frames, %.0f s -> %s (frames deleted)\n", b, set_lab[st], length(ks), time() - t0, mname)
        end
    end
    foreach(close, values(filt_ds))
end

# ============================================================================
# (4), (5) energy budget: modal KE and APE of each band, time means and the ±200 km strip average
# ============================================================================
# From the step-2 files (band "all" = the raw puv record), per band (MF_E_BANDS, default M2:hp28:hp9) and mode (MF_MODES):
#   (4) KE_n  = ½ ρ0 H < un² + vn² >  [J m-2], kept with its parts KEu = ½ρ0H<un²> (grid xi, across the mooring line)
#       and KEv = ½ρ0H<vn²> (grid eta, along it)                      -> model_and_filter/filt_KE.nc
#   (5) APE_n = H < pn² > / (2 ρ0 c_n²)  [J m-2] (c_n: the mode speed at each point, structure file)
#                                                                     -> model_and_filter/filt_APE.nc
# Time means over 5 periods: full = the analysis window 2022-08-27 -> 11-30, aug (08-27 -> 08-31, 5 days only), sep, oct,
# nov; variables <name>_<band>_mode<n> (xi, eta, period). Read 100 records at a time (runs on the login node). A saved
# file is reused (only the figures are redrawn); MF_E_RECOMPUTE=1 computes it again.
# Strip average as mode_dff_binplot.jl: per grid row the mean over xi = ix_line ± 200 km (fixed distance), shelf
# (h < 200 m) and the 54 km edge bands left out, rows with < 50 % deep cover not drawn, the plots end 250 km past CPIES7.
# Figures -> MF_E_FIG (default figure/<run>/mode_filter), per band, <T> = KE or APE:
#   filt_<T>_<band>.png               per mode: map (full window) and the strip mean ± 1 std (KE: with its u, v parts;
#                                     APE: with the strip-mean KE for comparison, as the harmonic APE plot)
#   filt_<T>_<band>_months.png        per mode: strip mean of the 5 periods
#   filt_<T>_<band>_heatmap_month.png maps, rows modes, columns Aug / Sep / Oct / Nov / full (one colour range per row)
e_periods = [("full", m -> true), ("aug", m -> m == 8), ("sep", m -> m == 9), ("oct", m -> m == 10), ("nov", m -> m == 11)]
ke_fname = joinpath(filt_dir, "filt_KE.nc")
ape_fname = joinpath(filt_dir, "filt_APE.nc")
his0 = first(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(joinpath(run_dir, "after_joint"))))
rho0 = NCDataset(d -> Float64(d.attrib["rho0"]), joinpath(run_dir, "after_joint", his0))   # [kg m-3]

# time means over the periods of w * x² (x: field fld of band b, read nblk records at a time; w: weight (xi, eta))
# -> (xi, eta, period), records per period. The sum runs in a typed kernel; plist[k] = the periods of record k.
function acc_sq!(acc::Array{Float64, 3}, buf::Array{Float32, 3}, m::Int, plist::Vector{Vector{Int}}, k0::Int)
    nxb, nyb = size(buf, 1), size(buf, 2)
    for kk in 1:m, p in plist[k0 + kk]
        @inbounds for j in 1:nyb, i in 1:nxb
            x = buf[i, j, kk]
            acc[i, j, p] += isfinite(x) ? Float64(x)^2 : 0.0
        end
    end
    acc
end
function mean_sq_periods(b, fld, w; nblk = 100)
    mon = month.(date[t_keep])
    plist = [findall(f -> f(mon[k]), last.(e_periods)) for k in eachindex(t_keep)]
    nper = [count(k -> p in plist[k], eachindex(t_keep)) for p in eachindex(e_periods)]
    acc = zeros(nx, ny, length(e_periods))
    buf = Array{Float32}(undef, nx, ny, nblk)
    if b == "all"
        v, ks = ds_puv[fld].var, t_keep           # raw record: the window's records of the puv file
    else
        f = filt_fname((; name = b), fld); isfile(f) || error("no $f: run step 2 first")
        d = NCDataset(f); v, ks = d[fld].var, 1:length(t_keep)
    end
    for k1 in 1:nblk:length(t_keep)
        k2 = min(k1 + nblk - 1, length(t_keep)); m = k2 - k1 + 1
        NCDatasets.load!(v, view(buf, :, :, 1:m), :, :, ks[k1]:ks[k2])
        acc_sq!(acc, buf, m, plist, k1 - 1)
    end
    b == "all" || close(d)
    for p in eachindex(e_periods)
        acc[:, :, p] .*= w ./ nper[p]
    end
    acc, nper
end
# the energy terms of a step: compute(b, n) -> Dict(name => (xi, eta, period)), saved to fname (or the saved file reused)
function energy_file(fname, keys_needed, compute, label, note; unit = startswith(label, "F") ? "kW/m" : "kJ/m²", units_nc = "J m-2", psc = 1e-3)
    need = !isfile(fname) || get(ENV, "MF_E_RECOMPUTE", "0") == "1" || !NCDataset(d -> all(haskey(d, k) for k in keys_needed), fname)
    if !need
        println("  $label from $fname (MF_E_RECOMPUTE=1 to recompute)")
        return
    end
    println("  $label -> $fname")
    E = Dict{String, Array{Float64, 3}}(); nper = Int[]
    dm = (mask .> 0) .& (H .> 1000)
    for b in e_bands, n in modes
        t0 = time()
        Eb, nper = compute(b, n)
        merge!(E, Eb)
        main = first(k for k in keys(Eb) if startswith(k, split(label)[1] * "_"))
        @printf("    %-4s mode %d: %.0f s; deep-mean %s (%s) %s\n", b, n, time() - t0, split(label)[1], unit,
                join(["$(e_periods[p][1]) $(round(mean(Eb[main][:, :, p][dm]) * psc, digits = 3))" for p in eachindex(e_periods)], ", "))
    end
    tmp = fname * ".tmp"; isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do d
        defDim(d, "xi_rho", nx); defDim(d, "eta_rho", ny); defDim(d, "period", length(e_periods))
        d.attrib["title"] = "Modal $label of the Butterworth bands (time means), $dff_run"
        d.attrib["periods"] = join(first.(e_periods), ",")
        d.attrib["window"] = "$(date[t_keep[1]]) to $(date[t_keep[end]]); aug from $(date[t_keep[1]])"
        d.attrib["rho0"] = rho0
        d.attrib["note"] = note
        d.attrib["script"] = "internalwave_julia/DFF/modes_filter_analysis.jl"
        defVar(d, "nrec", Int32.(nper), ("period",); attrib = ["long_name" => "hourly records in the period"])
        for (k, a) in E
            defVar(d, k, Float32.(a), ("xi_rho", "eta_rho", "period"); attrib = ["units" => units_nc, "long_name" => k])
        end
    end
    mv(tmp, fname; force = true)
    println("  wrote $fname")
end

if any(in(mf_steps), (4, 5, 6, 7, 8, 9, 10))
    e_bands = String.(split(get(ENV, "MF_E_BANDS", "M2:hp28:hp9"), r"[,:; ]+"; keepempty = false))
    # --- strip, moorings, map frame (as mode_dff_binplot.jl)
    using CairoMakie
    include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")       # topdown_axis, plot_curvilinear!
    include("/home/hchen54/internalwave_julia/function/mooring_transect.jl")   # read_moorings
    fig_e = expanduser(get(ENV, "MF_E_FIG", mf_fig_dir))
    mkpath(fig_e)
    run_mooring4 = Dict("amazon_2700m_cmems_0801" => "roms_cmems_grd_2700m_mor_edata.nc")
    mfile = "/expanse/lustre/projects/uso102/hchen54/input/grid/$(get(run_mooring4, dff_run, "roms_grd_900m_mor_edata.nc"))"
    mname, mlon, mlat = read_moorings(mfile)
    mij = NCDataset(g -> [(round(Int, g["$(m)_mor"][1] + 1), round(Int, g["$(m)_mor"][2] + 1)) for m in mname], mfile)
    lon180 = ifelse.(lon .> 180, lon .- 360, lon); mlon180 = ifelse.(mlon .> 180, mlon .- 360, mlon)
    nxc = isodd(nx) ? nx : nx - 1                                    # the DFF crop (odd), as the bin plots
    dxk = mean(1 ./ pm) / 1e3
    edge = round(Int, 54 / dxk)
    inside = (edge .< (1:nxc) .<= nxc - edge) .& (edge .< (1:ny)' .<= ny - edge)
    showE = falses(nx, ny); showE[1:nxc, :] .= (mask[1:nxc, :] .> 0) .& (h[1:nxc, :] .>= 200) .& inside
    maskedE(a) = ifelse.(showE, a, NaN)
    ix_line = round(Int, 0.5 + (nxc + 1) / 2)
    ix_side = round(Int, 200 / dxk); strip = (ix_line - ix_side):(ix_line + ix_side)
    line_km = [0; cumsum(0.5 .* (1 ./ pn[ix_line, 1:end-1] .+ 1 ./ pn[ix_line, 2:end]))] ./ 1e3
    j_node = argmin(abs.(line_km .- 200))
    j_crop = min(argmin(abs.(line_km .- (line_km[mij[findfirst(==("CPIES7"), mname)][2]] + 250))), ny - edge)
    moor_in = [(nm, j, abs(i - ix_line) <= 1) for (nm, (i, j)) in zip(mname, mij) if abs(i - ix_line) <= ix_side]
    function strip_stats(a)
        m, s = fill(NaN, ny), fill(NaN, ny)
        for j in 1:ny
            v = [a[i, j] for i in strip if showE[i, j] && isfinite(a[i, j])]
            length(v) / length(strip) >= 0.5 || continue
            m[j], s[j] = mean(v), std(v)
        end
        m, s
    end
    rows_ok(m) = isfinite.(m) .& ((1:ny) .<= j_crop)
    map_w = 400
    lims = (extrema(lon180[showE]) .+ (-0.1, 0.1), extrema(lat[showE]) .+ (-0.1, 0.1))
    map_h = round(Int, map_w * (lims[2][2] - lims[2][1]) / ((lims[1][2] - lims[1][1]) * cosd(mean(lims[2]))))
    pt = 1300 / (18.0 / 2.54 * 72)
    strip_note = "strip: ξ = $(first(strip))–$(last(strip)) (green on the maps), one mean per grid row (η); shelf (h < 200 m) and 54 km edge bands left out;\n" *
                 "rows with < 50% deep cover not drawn; cyan: north end of the line plots, 250 km past CPIES7; red star / dashed: 200 km node; grey lines: moorings"
    blab = Dict("all" => "raw record (no filter)", "M2" => "M2 band 9-15 h", "hp28" => "high-pass 28 h", "hp9" => "high-pass 9 h")
    win_str = "$(Dates.format(date[t_keep[1]], "yyyy-mm-dd")) – $(Dates.format(date[t_keep[end]], "yyyy-mm-dd"))"
    println("\nenergy: strip xi = $(first(strip))-$(last(strip)) (±200 km around $ix_line); line plots end at eta $j_crop ($(round(Int, line_km[j_crop])) km); figures -> $fig_e")

    # --- figure pieces
    function deco_line!(al, ytop)
        hlines!(al, [0]; color = :black, linewidth = 0.6)
        for (nm, j, on) in moor_in
            vlines!(al, [line_km[j]]; color = on ? :gray60 : :gray85, linewidth = on ? 0.8 : 0.5)
            on && text!(al, line_km[j], ytop; text = nm * " ", rotation = π / 2, align = (:right, :center), fontsize = 7pt, color = :gray35)
        end
        vlines!(al, [line_km[j_node]]; color = :red, linestyle = :dash, linewidth = 1.2)
    end
    function deco_map!(ax; small = false)
        contour!(ax, lon180, lat, h; levels = [1000, 2000, 3000], color = :gray40, linewidth = small ? 0.5 : 0.6)
        for (ix, col) in ((first(strip), :green), (ix_line, :black), (last(strip), :green))
            lines!(ax, lon180[ix, :], lat[ix, :]; color = col, linestyle = :dash, linewidth = small ? 1 : 1.2)
        end
        lines!(ax, lon180[strip, j_crop], lat[strip, j_crop]; color = :cyan, linestyle = :dash, linewidth = small ? 1.5 : 2)
        scatter!(ax, mlon180, mlat; marker = :circle, markersize = small ? 4 : 5, color = :yellow, strokewidth = small ? 0.4 : 0.5)
        scatter!(ax, [lon180[ix_line, j_node]], [lat[ix_line, j_node]]; marker = :star5, markersize = small ? 11 : 14, color = :red,
                 strokecolor = :black, strokewidth = small ? 0.5 : 0.6)
        limits!(ax, lims...)
    end
    rdE(D, k, p; sc = 1e-3) = Float64.(D[k][:, :, p]) .* sc          # default J m-2 -> kJ m-2
    # full window: map of T and its strip mean ± 1 std, plus extra lines (extra(b, n) -> [(curve, colour, style, label)])
    # sym: signed field (symmetric colour range, y range not from 0); unit / sc: label and factor of the plotted values
    function fig_full(D, T, b, maptitle, linelabel, extra; sym = false, unit = "kJ m⁻²", sc = 1e-3)
        fig = Figure(size = (1300, 800length(modes)), fontsize = 10pt)
        axl = Axis[]
        for (row, n) in enumerate(modes)
            A = rdE(D, "$(T)_$(b)_mode$n", 1; sc)
            lim = round(quantile(abs.(filter(isfinite, maskedE(A))), 0.99); sigdigits = 2)
            ax = topdown_axis(fig[row, 1]; title = maptitle(n), xlabel = row == length(modes) ? "Longitude" : "",
                              backgroundcolor = :gray80, width = map_w, height = map_h, aspect = nothing)
            hm = plot_curvilinear!(ax, lon180, lat, maskedE(A); colormap = sym ? :balance : :thermal, colorrange = sym ? (-lim, lim) : (0, lim))
            deco_map!(ax)
            Colorbar(fig[row, 2], hm; height = map_h)
            Am, As = strip_stats(A)
            ok = rows_ok(Am)
            ex = extra(b, n)
            al = Axis(fig[row, 3]; title = "$(T)_$n averaged across the strip (±200 km, ξ = $(first(strip))–$(last(strip)))", ylabel = "$(T)_$n [$unit]",
                      xlabel = row == length(modes) ? "distance along the mooring line from the south boundary [km]" : "",
                      xticks = 0:100:ceil(Int, line_km[end]), xminorticks = IntervalsBetween(2), xminorticksvisible = true,
                      xminorgridvisible = true, width = 2map_w + 60, height = map_h)
            v = filter(isfinite, vcat((Am .+ As)[ok], sym ? (Am .- As)[ok] : Float64[], [e[1][ok] for e in ex]..., 0.0))
            # signed fields: 2nd-98th percentile (the large values right at the slope edge run off the axis)
            ylo, yhi = sym ? (min(quantile(v, 0.02), 0.0), max(quantile(v, 0.98), 0.0)) : (0.0, maximum(v)); ytop = yhi + 0.4 * (yhi - ylo)
            ylims!(al, sym ? ylo - 0.05 * (yhi - ylo) : 0.0, ytop); deco_line!(al, ytop)
            band!(al, line_km, ifelse.(ok, sym ? Am .- As : max.(Am .- As, 0), NaN), ifelse.(ok, Am .+ As, NaN); color = (:royalblue, 0.15), label = "$T ± 1 std across the strip")
            for (cv, col, ls, lab) in ex
                lines!(al, line_km, ifelse.(ok, cv, NaN); color = col, linestyle = ls, linewidth = 2, label = lab)
            end
            lines!(al, line_km, ifelse.(ok, Am, NaN); color = :royalblue, linewidth = 2.5, label = linelabel)
            xlims!(al, line_km[findfirst(ok)], line_km[j_crop])
            push!(axl, al)
            @printf("    %-4s mode %d full: strip-mean %s %.3f %s%s; at the node %.3f\n", b, n, T, mean(Am[ok]), unit,
                    join([@sprintf(", %s %.3f", split(e[4])[1], mean(e[1][ok])) for e in ex]), Am[j_node])
        end
        Label(fig[0, :], "$(blab[b]) modal $T (Butterworth), mean across ±200 km of the mooring line\n$dff_run, $win_str"; fontsize = 11pt, font = :bold)
        Legend(fig[length(modes) + 1, :], axl[end]; orientation = :horizontal, framevisible = false, labelsize = 9pt, nbanks = 2)
        Label(fig[length(modes) + 2, :], strip_note; fontsize = 8pt)
        resize_to_layout!(fig)
        fname = joinpath(fig_e, "filt_$(T)_$(b).png"); save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()   # full
        println("    saved $fname")
    end
    # the strip mean of T for the 5 periods
    function fig_months(D, T, b; sym = false, unit = "kJ m⁻²", sc = 1e-3, mlist = modes, mlab = n -> "mode $n", cmap = nothing)
        nrecE = D["nrec"][:]
        pcol = [:black, Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3], Makie.wong_colors()[6]]
        plab = ["full ($(nrecE[1] ÷ 24) d)", "Aug ($(nrecE[2] ÷ 24) d)", "Sep", "Oct", "Nov"]
        fig = Figure(size = (1300, 450length(mlist)), fontsize = 10pt)
        for (row, n) in enumerate(mlist)
            ms = [strip_stats(rdE(D, "$(T)_$(b)_mode$n", p; sc))[1] for p in eachindex(e_periods)]
            ok = rows_ok(ms[1])
            al = Axis(fig[row, 1]; title = "$(T)_$n ($(mlab(n))) averaged across the strip, by period", ylabel = "$(T)_$n [$unit]",
                      xlabel = row == length(mlist) ? "distance along the mooring line from the south boundary [km]" : "",
                      xticks = 0:100:ceil(Int, line_km[end]), xminorticks = IntervalsBetween(2), xminorticksvisible = true,
                      xminorgridvisible = true, width = 1000, height = 330)
            v = filter(isfinite, vcat([m[ok] for m in ms]..., 0.0))
            ylo, yhi = sym ? (min(quantile(v, 0.02), 0.0), max(quantile(v, 0.98), 0.0)) : (0.0, maximum(v)); ytop = yhi + 0.35 * (yhi - ylo)
            ylims!(al, sym ? ylo - 0.05 * (yhi - ylo) : 0.0, ytop); deco_line!(al, ytop)
            for p in eachindex(e_periods)
                lines!(al, line_km, ifelse.(ok, ms[p], NaN); color = pcol[p], linewidth = p == 1 ? 2.8 : 1.6, label = plab[p])
            end
            xlims!(al, line_km[findfirst(ok)], line_km[j_crop])
            row == 1 && Legend(fig[1:length(mlist), 2], al; framevisible = false, labelsize = 9pt)
            @printf("    %-4s mode %s strip-mean %s by period: %s %s\n", b, n, T,
                    join(["$(e_periods[p][1]) $(round(mean(ms[p][ok]), digits = 3))" for p in eachindex(e_periods)], ", "), unit)
        end
        Label(fig[0, :], "$(blab[b]) modal $T by period, mean across ±200 km of the mooring line\n$dff_run"; fontsize = 11pt, font = :bold)
        Label(fig[length(mlist) + 1, :], "Aug: $(nrecE[2] ÷ 24) days only (from 08-27, after the spin-up); red dashed: 200 km node; grey lines: moorings"; fontsize = 8pt)
        resize_to_layout!(fig)
        fname = joinpath(fig_e, "filt_$(T)_$(b)_months.png"); save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        println("    saved $fname")
    end
    # maps of T for the 5 periods: rows modes, columns Aug, Sep, Oct, Nov, full; one colour range per row
    function fig_heatmap(D, T, b, formula; sym = false, unit = "kJ m⁻²", sc = 1e-3, mlist = modes, mlab = n -> "mode $n", cmap = nothing)
        nrecE = D["nrec"][:]
        hm_w = 260; hm_h = round(Int, hm_w * map_h / map_w)
        pord = [2, 3, 4, 5, 1]
        ptitle = Dict(1 => "full ($(nrecE[1] ÷ 24) d)", 2 => "Aug ($(nrecE[2] ÷ 24) d)", 3 => "Sep", 4 => "Oct", 5 => "Nov")
        fig = Figure(fontsize = 10pt)
        for (row, n) in enumerate(mlist)
            As = [rdE(D, "$(T)_$(b)_mode$n", p; sc) for p in eachindex(e_periods)]
            lim = round(quantile(abs.(filter(isfinite, vcat([vec(maskedE(A)) for A in As]...))), 0.99); sigdigits = 2)
            hm = nothing
            for (c, p) in enumerate(pord)
                ax = topdown_axis(fig[row, c]; title = "$(mlab(n)), $(ptitle[p])", backgroundcolor = :gray80, width = hm_w, height = hm_h,
                                  aspect = nothing, xlabel = row == length(mlist) ? "Longitude" : "", ylabel = c == 1 ? "Latitude" : "",
                                  yticklabelsvisible = c == 1, xticklabelsvisible = row == length(mlist), xticks = WilkinsonTicks(3))
                hm = plot_curvilinear!(ax, lon180, lat, maskedE(As[p]); colormap = cmap === nothing ? (sym ? :balance : :thermal) : cmap, colorrange = sym ? (-lim, lim) : (0, lim))
                deco_map!(ax; small = true)
            end
            Colorbar(fig[row, 6], hm; height = hm_h, label = "$(T)_$n [$unit]")
        end
        Label(fig[0, :], "$(blab[b]) modal $T = $formula by period\n$dff_run"; fontsize = 11pt, font = :bold)
        Label(fig[length(mlist) + 1, :], "one colour range per mode (all periods); Aug: $(nrecE[2] ÷ 24) days only (from 08-27, after the spin-up);\n" *
              "green: ±200 km strip; cyan: north end of the line plots; red star: 200 km node; shelf (h < 200 m) and 54 km edge bands grey"; fontsize = 8pt)
        colgap!(fig.layout, 8); rowgap!(fig.layout, 8)
        resize_to_layout!(fig)
        fname = joinpath(fig_e, "filt_$(T)_$(b)_heatmap_month.png"); save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        println("    saved $fname")
    end
end

# time means over the periods of w * x * y (two fields of band b) -> (xi, eta, period), records per period
function acc_prod!(acc::Array{Float64, 3}, b1::Array{Float32, 3}, b2::Array{Float32, 3}, m::Int, plist::Vector{Vector{Int}}, k0::Int)
    nxb, nyb = size(b1, 1), size(b1, 2)
    for kk in 1:m, p in plist[k0 + kk]
        @inbounds for j in 1:nyb, i in 1:nxb
            x = Float64(b1[i, j, kk]) * Float64(b2[i, j, kk])
            acc[i, j, p] += isfinite(x) ? x : 0.0
        end
    end
    acc
end
function mean_prod_periods(b, f1, f2, w; nblk = 100)
    mon = month.(date[t_keep])
    plist = [findall(f -> f(mon[k]), last.(e_periods)) for k in eachindex(t_keep)]
    nper = [count(k -> p in plist[k], eachindex(t_keep)) for p in eachindex(e_periods)]
    acc = zeros(nx, ny, length(e_periods))
    buf1, buf2 = Array{Float32}(undef, nx, ny, nblk), Array{Float32}(undef, nx, ny, nblk)
    ds_ = Any[]
    b == "all" && any(==("wbt"), (f1, f2)) && error("W_BT has no raw ('all') record: use a filtered band")
    vs = map((f1, f2)) do f
        b == "all" && return ds_puv[f].var
        fn = filt_fname((; name = b), f); isfile(fn) || error("no $fn: run step 2 first")
        d = NCDataset(fn); push!(ds_, d); d[f].var
    end
    ks = b == "all" ? t_keep : (1:length(t_keep))
    for k1 in 1:nblk:length(t_keep)
        k2 = min(k1 + nblk - 1, length(t_keep)); m = k2 - k1 + 1
        NCDatasets.load!(vs[1], view(buf1, :, :, 1:m), :, :, ks[k1]:ks[k2])
        NCDatasets.load!(vs[2], view(buf2, :, :, 1:m), :, :, ks[k1]:ks[k2])
        acc_prod!(acc, buf1, buf2, m, plist, k1 - 1)
    end
    foreach(close, ds_)
    for p in eachindex(e_periods)
        acc[:, :, p] .*= w ./ nper[p]
    end
    acc, nper
end

# ---------------------------------------------------------------------------- (4) KE
if 4 in mf_steps
    println("\nstep 4: KE = ½ρ0H<u²+v²>, rho0 = $rho0, bands $(join(e_bands, ",")), modes $(join(modes, ","))")
    wKE = 0.5 .* rho0 .* H
    energy_file(ke_fname, ["KE_$(b)_mode$n" for b in e_bands for n in modes],
                (b, n) -> begin
                    ku, np_ = mean_sq_periods(b, "u$n", wKE); kv, _ = mean_sq_periods(b, "v$n", wKE)
                    Dict("KE_$(b)_mode$n" => ku .+ kv, "KEu_$(b)_mode$n" => ku, "KEv_$(b)_mode$n" => kv), np_
                end, "KE",
                "KE = ½ rho0 H <un² + vn²>, KEu = ½ rho0 H <un²> (grid xi), KEv = ½ rho0 H <vn²> (grid eta); H = h + mean zeta; " *
                "un, vn: model_and_filter/filt_<band>_<u|v><n>.nc (band 'all': the raw puv record)")
    D = NCDataset(ke_fname)
    for b in e_bands
        fig_full(D, "KE", b, n -> "KE_$n = ½ρ₀H⟨u² + v²⟩, mode $n", "KE = ½ρ₀H⟨u² + v²⟩, strip mean",
                 (b, n) -> [(strip_stats(rdE(D, "KEu_$(b)_mode$n", 1))[1], :darkorange, :dot, "½ρ₀H⟨u²⟩ (across the line)"),
                            (strip_stats(rdE(D, "KEv_$(b)_mode$n", 1))[1], :forestgreen, :dash, "½ρ₀H⟨v²⟩ (along the line)")])
        fig_months(D, "KE", b)
        fig_heatmap(D, "KE", b, "½ρ₀H⟨u² + v²⟩")
    end
    close(D)
end

# ---------------------------------------------------------------------------- (5) APE
if 5 in mf_steps
    println("\nstep 5: APE = H<p²>/(2ρ0c²), rho0 = $rho0, c_n from $struct_fname, bands $(join(e_bands, ",")), modes $(join(modes, ","))")
    energy_file(ape_fname, ["APE_$(b)_mode$n" for b in e_bands for n in modes],
                (b, n) -> begin
                    a, np_ = mean_sq_periods(b, "p$n", H ./ (2 .* rho0 .* c_mode[:, :, n] .^ 2))
                    Dict("APE_$(b)_mode$n" => a), np_
                end, "APE",
                "APE = H <pn²> / (2 rho0 c_n²); c_n: mode-n eigenspeed at each point ($(basename(struct_fname))); H = h + mean zeta; " *
                "pn: model_and_filter/filt_<band>_p<n>.nc (band 'all': the raw puv record)")
    D = NCDataset(ape_fname)
    DK = isfile(ke_fname) ? NCDataset(ke_fname) : nothing           # KE (step 4) for comparison on the line plots
    for b in e_bands
        hasKE(n) = DK !== nothing && haskey(DK, "KE_$(b)_mode$n")
        fig_full(D, "APE", b, n -> "APE_$n = H⟨p²⟩ / (2ρ₀c_$(n)²), mode $n", "APE = H⟨p²⟩ / (2ρ₀c²), strip mean",
                 (b, n) -> hasKE(n) ? [(strip_stats(rdE(DK, "KE_$(b)_mode$n", 1))[1], :crimson, :dash, "KE = ½ρ₀H⟨u² + v²⟩ (step 4), strip mean")] : [])
        fig_months(D, "APE", b)
        fig_heatmap(D, "APE", b, "H⟨p²⟩ / (2ρ₀c²)")
    end
    DK === nothing || close(DK)
    close(D)
end

# ---------------------------------------------------------------------------- (6) energy flux
# F_n = (H <pn un>, H <pn vn>) [W m-1] along grid xi (across the mooring line) and eta (along it), the mode-n depth-
# integrated flux of the band (no cross-mode terms: (1/H) ∫ φm φn dz = δmn). -> model_and_filter/filt_F.nc: Fx_, Fy_<band>_mode<n>
# Figures (kW m-1): filt_F_<band>_mode<n>.png (rows Fx, Fy: map and strip mean ± 1 std), filt_F_<band>_months.png (strip
# means of the 5 periods, columns Fx, Fy, rows modes), filt_F_<band>_heatmap_month.png (|F| with arrows east / north,
# rows modes, columns Aug / Sep / Oct / Nov / full, one colour range per row).
if 6 in mf_steps
    println("\nstep 6: F = H<p u>, H<p v>, bands $(join(e_bands, ",")), modes $(join(modes, ","))")
    f_fname = joinpath(filt_dir, "filt_F.nc")
    energy_file(f_fname, ["Fy_$(b)_mode$n" for b in e_bands for n in modes],
                (b, n) -> begin
                    fx, np_ = mean_prod_periods(b, "p$n", "u$n", H); fy, _ = mean_prod_periods(b, "p$n", "v$n", H)
                    Dict("Fy_$(b)_mode$n" => fy, "Fx_$(b)_mode$n" => fx), np_
                end, "Fy flux",
                "Fx = H <pn un> (grid xi, across the mooring line), Fy = H <pn vn> (grid eta, along it) [W m-1]; H = h + mean zeta; " *
                "pn, un, vn: model_and_filter/filt_<band>_<p|u|v><n>.nc (band 'all': the raw puv record)")
    NCDataset(f_fname, "a") do d                                    # units: W m-1 (energy_file writes J m-2)
        for k in keys(d); k == "nrec" || (d[k].attrib["units"] = "W m-1"); end
    end
    D = NCDataset(f_fname)
    nrecF = D["nrec"][:]
    rdF(k, p) = Float64.(D[k][:, :, p]) ./ 1e3                     # [kW m-1]
    ca_, sa_ = cos.(angle), sin.(angle)
    qskip = max(1, round(Int, 36 / dxk))
    comp_lab = Dict("Fx" => "F_x = H⟨p u⟩ (across the line)", "Fy" => "F_y = H⟨p v⟩ (along the line)")
    for b in e_bands
        # per mode: rows Fx, Fy (map + strip line)
        for n in modes
            fig = Figure(size = (1300, 1600), fontsize = 10pt)
            axl = Axis[]
            for (row, c) in enumerate(("Fx", "Fy"))
                A = rdF("$(c)_$(b)_mode$n", 1)
                lim = round(quantile(abs.(filter(isfinite, maskedE(A))), 0.99); sigdigits = 2)
                ax = topdown_axis(fig[row, 1]; title = "$(comp_lab[c]), mode $n", xlabel = row == 2 ? "Longitude" : "",
                                  backgroundcolor = :gray80, width = map_w, height = map_h, aspect = nothing)
                hm = plot_curvilinear!(ax, lon180, lat, maskedE(A); colormap = :balance, colorrange = (-lim, lim))
                deco_map!(ax)
                Colorbar(fig[row, 2], hm; height = map_h)
                Am, As = strip_stats(A)
                ok = rows_ok(Am)
                al = Axis(fig[row, 3]; title = "$(c)_$n averaged across the strip (±200 km, ξ = $(first(strip))–$(last(strip)))", ylabel = "$(c)_$n [kW m⁻¹]",
                          xlabel = row == 2 ? "distance along the mooring line from the south boundary [km]" : "",
                          xticks = 0:100:ceil(Int, line_km[end]), xminorticks = IntervalsBetween(2), xminorticksvisible = true,
                          xminorgridvisible = true, width = 2map_w + 60, height = map_h)
                v = filter(isfinite, vcat((Am .+ As)[ok], (Am .- As)[ok], 0.0))
                ylo, yhi = extrema(v); ytop = yhi + 0.35 * (yhi - ylo)
                ylims!(al, ylo - 0.05 * (yhi - ylo), ytop); deco_line!(al, ytop)
                band!(al, line_km, ifelse.(ok, Am .- As, NaN), ifelse.(ok, Am .+ As, NaN); color = (:royalblue, 0.15), label = "F ± 1 std across the strip")
                lines!(al, line_km, ifelse.(ok, Am, NaN); color = :royalblue, linewidth = 2.5, label = "F, strip mean")
                xlims!(al, line_km[findfirst(ok)], line_km[j_crop])
                push!(axl, al)
                @printf("    %-4s mode %d %s full: strip mean %.3f kW/m (max %.3f at %.0f km); at the node %.3f\n", b, n, c,
                        mean(Am[ok]), maximum(Am[ok]), line_km[ok][argmax(Am[ok])], Am[j_node])
            end
            Label(fig[0, :], "$(blab[b]) mode-$n energy flux (Butterworth), mean across ±200 km of the mooring line\n$dff_run, $win_str"; fontsize = 11pt, font = :bold)
            Legend(fig[3, :], axl[end]; orientation = :horizontal, framevisible = false, labelsize = 9pt)
            Label(fig[4, :], strip_note * "\nF_x, F_y along grid ξ / η (η: along the mooring line, + toward its north end); maps: one symmetric colour range per panel"; fontsize = 8pt)
            resize_to_layout!(fig)
            fname = joinpath(fig_e, "filt_F_$(b)_mode$(n).png"); save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
            println("    saved $fname")
        end
        # strip means of the 5 periods: columns Fx, Fy; rows modes
        pcol = [:black, Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3], Makie.wong_colors()[6]]
        plab = ["full ($(nrecF[1] ÷ 24) d)", "Aug ($(nrecF[2] ÷ 24) d)", "Sep", "Oct", "Nov"]
        fig = Figure(fontsize = 10pt)
        local al
        for (row, n) in enumerate(modes), (col, c) in enumerate(("Fx", "Fy"))
            ms = [strip_stats(rdF("$(c)_$(b)_mode$n", p))[1] for p in eachindex(e_periods)]
            ok = rows_ok(ms[1])
            al = Axis(fig[row, col]; title = "$(c)_$n (mode $n) across the strip, by period", ylabel = "$(c)_$n [kW m⁻¹]",
                      xlabel = row == length(modes) ? "distance along the mooring line [km]" : "",
                      xticks = 0:100:ceil(Int, line_km[end]), width = 560, height = 300)
            v = filter(isfinite, vcat([m[ok] for m in ms]..., 0.0)); ylo, yhi = extrema(v); ytop = yhi + 0.35 * (yhi - ylo)
            ylims!(al, ylo - 0.05 * (yhi - ylo), ytop); deco_line!(al, ytop)
            for p in eachindex(e_periods)
                lines!(al, line_km, ifelse.(ok, ms[p], NaN); color = pcol[p], linewidth = p == 1 ? 2.8 : 1.6, label = plab[p])
            end
            xlims!(al, line_km[findfirst(ok)], line_km[j_crop])
            @printf("    %-4s mode %d %s strip mean by period: %s kW/m\n", b, n, c,
                    join(["$(e_periods[p][1]) $(round(mean(ms[p][ok]), digits = 3))" for p in eachindex(e_periods)], ", "))
        end
        Legend(fig[1:length(modes), 3], al; framevisible = false, labelsize = 9pt)
        Label(fig[0, :], "$(blab[b]) modal energy flux by period, mean across ±200 km of the mooring line\n$dff_run"; fontsize = 11pt, font = :bold)
        Label(fig[length(modes) + 1, :], "F_x across, F_y along the mooring line (grid ξ / η); Aug: $(nrecF[2] ÷ 24) days only (from 08-27); red dashed: 200 km node; grey: moorings"; fontsize = 8pt)
        resize_to_layout!(fig)
        fname = joinpath(fig_e, "filt_F_$(b)_months.png"); save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        println("    saved $fname")
        # |F| maps with arrows: rows modes, columns Aug, Sep, Oct, Nov, full
        hm_w = 260; hm_h = round(Int, hm_w * map_h / map_w)
        pord = [2, 3, 4, 5, 1]
        ptitle = Dict(1 => "full ($(nrecF[1] ÷ 24) d)", 2 => "Aug ($(nrecF[2] ÷ 24) d)", 3 => "Sep", 4 => "Oct", 5 => "Nov")
        fig = Figure(fontsize = 10pt)
        for (row, n) in enumerate(modes)
            Fx = [rdF("Fx_$(b)_mode$n", p) for p in eachindex(e_periods)]
            Fy = [rdF("Fy_$(b)_mode$n", p) for p in eachindex(e_periods)]
            M = [hypot.(Fx[p], Fy[p]) for p in eachindex(e_periods)]
            lim = round(quantile(filter(isfinite, vcat([vec(maskedE(A)) for A in M]...)), 0.99); sigdigits = 2)
            hm = nothing
            for (c, p) in enumerate(pord)
                ax = topdown_axis(fig[row, c]; title = "mode $n, $(ptitle[p])", backgroundcolor = :gray80, width = hm_w, height = hm_h,
                                  aspect = nothing, xlabel = row == length(modes) ? "Longitude" : "", ylabel = c == 1 ? "Latitude" : "",
                                  yticklabelsvisible = c == 1, xticklabelsvisible = row == length(modes), xticks = WilkinsonTicks(3))
                hm = plot_curvilinear!(ax, lon180, lat, maskedE(M[p]); colormap = :plasma, colorrange = (0, lim))
                Fe = Fx[p] .* ca_ .- Fy[p] .* sa_; Fn = Fx[p] .* sa_ .+ Fy[p] .* ca_
                quiver_curvilinear!(ax, lon180, lat, maskedE(Fe), maskedE(Fn); skip = 2qskip, lengthscale = 0.9 / lim,
                                    color = (:white, 0.8), shaftwidth = 0.6, tipwidth = 3, tiplength = 3)
                deco_map!(ax; small = true)
            end
            Colorbar(fig[row, 6], hm; height = hm_h, label = "|F_$n| [kW m⁻¹]")
        end
        Label(fig[0, :], "$(blab[b]) modal energy flux |F| = |H⟨p u⟩, H⟨p v⟩| by period\n$dff_run"; fontsize = 11pt, font = :bold)
        Label(fig[length(modes) + 1, :], "one colour range per mode (all periods); arrows: F east / north; Aug: $(nrecF[2] ÷ 24) days only (from 08-27, after the spin-up);\n" *
              "green: ±200 km strip; cyan: north end of the line plots; red star: 200 km node; shelf (h < 200 m) and 54 km edge bands grey"; fontsize = 8pt)
        colgap!(fig.layout, 8); rowgap!(fig.layout, 8)
        resize_to_layout!(fig)
        fname = joinpath(fig_e, "filt_F_$(b)_heatmap_month.png"); save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
        println("    saved $fname")
    end
    close(D)
end

# ---------------------------------------------------------------------------- (7) flux divergence
# ∇·F_n = pm pn [∂ξ(Fx / pn) + ∂η(Fy / pm)] [W m-2] of the time-mean flux maps of step 6 (each band, mode, period), as
# mode_dff_binplot.jl step 8: centred differences on the curvilinear grid, NaN off the deep (h >= 200 m), inside-the-
# edges region. > 0: the flux carries energy out (a sink of mode n there, filled by conversion or the other bands /
# modes); < 0: energy arrives and is lost or passed on. -> model_and_filter/filt_divF.nc: divF_<band>_mode<n>
# Figures (mW m-2): filt_divF_<band>.png (map, strip mean ± 1 std), _months.png, _heatmap_month.png. Printed: the strip
# integral up to the crop (Σ ∇·F dA over the deep strip points, MW) and its north / south parts of the node.
function cdiff(a, dim)                                             # centred difference per grid step (NaN at the ends)
    d = fill(NaN, size(a))
    if dim == 1
        d[2:end-1, :] .= (a[3:end, :] .- a[1:end-2, :]) ./ 2
    else
        d[:, 2:end-1] .= (a[:, 3:end] .- a[:, 1:end-2]) ./ 2
    end
    d
end
flux_div(Fx, Fy) = pm .* pn .* (cdiff(Fx ./ pn, 1) .+ cdiff(Fy ./ pm, 2))
if 7 in mf_steps
    f_fname = joinpath(filt_dir, "filt_F.nc"); isfile(f_fname) || error("no $f_fname: run step 6 first")
    dv_fname = joinpath(filt_dir, "filt_divF.nc")
    println("\nstep 7: ∇·F from $f_fname -> $dv_fname")
    DF = NCDataset(f_fname)
    nrecF = DF["nrec"][:]
    tmp = dv_fname * ".tmp"; isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do d
        defDim(d, "xi_rho", nx); defDim(d, "eta_rho", ny); defDim(d, "period", length(e_periods))
        d.attrib["title"] = "Divergence of the modal energy flux of the Butterworth bands (time means), $dff_run"
        d.attrib["periods"] = join(first.(e_periods), ",")
        d.attrib["note"] = "divF = pm pn [d/dxi(Fx / pn) + d/deta(Fy / pm)], centred differences, of model_and_filter/filt_F.nc; " *
                           "NaN off the deep (h >= 200 m), inside-the-54-km-edges region"
        d.attrib["script"] = "internalwave_julia/DFF/modes_filter_analysis.jl (step 7)"
        defVar(d, "nrec", Int32.(nrecF), ("period",); attrib = ["long_name" => "hourly records in the period"])
        for b in unique([split(k, "_")[2] for k in keys(DF) if startswith(k, "Fy_")]), n in modes   # every band of filt_F.nc
            haskey(DF, "Fy_$(b)_mode$n") || continue
            dv = zeros(nx, ny, length(e_periods))
            for p in eachindex(e_periods)
                dv[:, :, p] .= flux_div(Float64.(DF["Fx_$(b)_mode$n"][:, :, p]), Float64.(DF["Fy_$(b)_mode$n"][:, :, p]))
                dv[.!showE, p] .= NaN
            end
            defVar(d, "divF_$(b)_mode$n", Float32.(dv), ("xi_rho", "eta_rho", "period"); attrib = ["units" => "W m-2", "long_name" => "divF_$(b)_mode$n"])
        end
    end
    mv(tmp, dv_fname; force = true)
    close(DF)
    println("  wrote $dv_fname")
    # strip integrals [MW] of the full window: whole strip up to the crop, south and north of the node
    area = 1 ./ (pm .* pn)
    D = NCDataset(dv_fname)
    for b in e_bands, n in modes
        dv = Float64.(D["divF_$(b)_mode$n"][:, :, 1])
        sint(js) = sum(dv[i, j] * area[i, j] for i in strip, j in js if showE[i, j] && isfinite(dv[i, j]); init = 0.0) / 1e6
        @printf("    %-4s mode %d: strip integral of ∇·F up to the crop %.1f MW (south of the node %.1f, node -> crop %.1f)\n",
                b, n, sint(1:j_crop), sint(1:j_node - 1), sint(j_node:j_crop))
    end
    for b in e_bands
        haskey(D, "divF_$(b)_mode$(modes[1])") || (println("    no divF_$b: run step 6 for band $b"); continue)
        fig_full(D, "divF", b, n -> "∇·F_$n, mode $n", "∇·F, strip mean", (b, n) -> []; sym = true, unit = "mW m⁻²", sc = 1e3)
        fig_months(D, "divF", b; sym = true, unit = "mW m⁻²", sc = 1e3)
        fig_heatmap(D, "divF", b, "∇·(H⟨p u⟩, H⟨p v⟩)"; sym = true, unit = "mW m⁻²", sc = 1e3)
    end
    close(D)
end

# ---------------------------------------------------------------------------- (8) conversion
# conv_n = φ_n(-H) < W_BT pn > [W m-2]: the work of the barotropic bottom vertical velocity W_BT = -(ubar dh/dx +
# vbar dh/dy) against the mode-n bottom pressure pn φ_n(-H) (φ_n at the bottom rho level), both band-passed (step-2
# files filt_<band>_wbt.nc, filt_<band>_p<n>.nc): the band version of the harmonic ½ A_W A_p φ_n(-H) cos(g_W - g_p)
# (W_BT_make.jl step 5). > 0: barotropic -> mode n. -> model_and_filter/filt_conv.nc: conv_<band>_mode<n>
# Figures (mW m-2): filt_conv_<band>.png (map, strip mean ± 1 std, with the strip-mean ∇·F of step 7), _months.png,
# _heatmap_month.png. Printed: strip integrals up to the crop [MW], south of the node and node -> crop.
if 8 in mf_steps
    cv_fname = joinpath(filt_dir, "filt_conv.nc")
    println("\nstep 8: conv = φ(-H)<W_BT p>, bands $(join(e_bands, ",")), modes $(join(modes, ","))")
    energy_file(cv_fname, ["conv_$(b)_mode$n" for b in e_bands for n in modes],
                (b, n) -> begin
                    c, np_ = mean_prod_periods(b, "wbt", "p$n", phi_bot[:, :, n])
                    Dict("conv_$(b)_mode$n" => c), np_
                end, "conv",
                "conv = phi_n(-H) <W_BT pn> [W m-2]; W_BT = -(ubar dh/dx + vbar dh/dy); phi_n at the bottom rho level " *
                "($(basename(struct_fname))); W_BT, pn: model_and_filter/filt_<band>_<wbt|p<n>>.nc"; unit = "mW/m²", units_nc = "W m-2", psc = 1e3)
    area = 1 ./ (pm .* pn)
    D = NCDataset(cv_fname)
    dv_fname = joinpath(filt_dir, "filt_divF.nc")
    DV = isfile(dv_fname) ? NCDataset(dv_fname) : nothing
    sint(a, js) = sum(a[i, j] * area[i, j] for i in strip, j in js if showE[i, j] && isfinite(a[i, j]); init = 0.0) / 1e6
    for b in e_bands, n in modes
        haskey(D, "conv_$(b)_mode$n") || continue
        c = Float64.(D["conv_$(b)_mode$n"][:, :, 1])
        @printf("    %-4s mode %d: strip integral of conv up to the crop %.1f MW (south of the node %.1f, node -> crop %.1f)",
                b, n, sint(c, 1:j_crop), sint(c, 1:j_node - 1), sint(c, j_node:j_crop))
        if DV !== nothing && haskey(DV, "divF_$(b)_mode$n")
            dv = Float64.(DV["divF_$(b)_mode$n"][:, :, 1])
            @printf("; conv - ∇·F %.1f MW (south %.1f, north %.1f)", sint(c, 1:j_crop) - sint(dv, 1:j_crop),
                    sint(c, 1:j_node - 1) - sint(dv, 1:j_node - 1), sint(c, j_node:j_crop) - sint(dv, j_node:j_crop))
        end
        println()
    end
    for b in e_bands
        haskey(D, "conv_$(b)_mode$(modes[1])") || continue
        hasDV(n) = DV !== nothing && haskey(DV, "divF_$(b)_mode$n")
        fig_full(D, "conv", b, n -> "conv_$n = φ_$(n)(-H)⟨W_BT p_$n⟩, mode $n", "conv, strip mean",
                 (b, n) -> hasDV(n) ? [(1e3 .* strip_stats(Float64.(DV["divF_$(b)_mode$n"][:, :, 1]))[1], :crimson, :dash, "∇·F (step 7), strip mean")] : [];
                 sym = true, unit = "mW m⁻²", sc = 1e3)
        fig_months(D, "conv", b; sym = true, unit = "mW m⁻²", sc = 1e3)
        fig_heatmap(D, "conv", b, "φ(-H)⟨W_BT p⟩"; sym = true, unit = "mW m⁻²", sc = 1e3)
    end
    DV === nothing || close(DV)
    close(D)
end

# ---------------------------------------------------------------------------- (9) residual D = conv - ∇·F, budget summary
# Steady energy balance of mode n in a band: conv_n = ∇·F_n + D_n, so D_n = conv_n - ∇·F_n [W m-2] is what the band /
# mode loses locally (> 0: dissipation, transfer to other modes or frequencies; < 0: gains from them), from the time-mean
# maps of steps 7 (filt_divF.nc) and 8 (filt_conv.nc); NaN where ∇·F is (off the deep, inside-the-edges region).
#   -> model_and_filter/filt_D.nc: D_<band>_mode<n>
# Figures (mW m-2): filt_D_<band>.png (map, strip mean ± 1 std, with the strip means of conv and ∇·F), _months.png,
# _heatmap_month.png; filt_budget_summary.png: strip integrals [MW] of conv, ∇·F and D for each band and mode (and
# modes together), south of the 200 km node and from the node to the crop (full window).
if 9 in mf_steps
    dv_fname = joinpath(filt_dir, "filt_divF.nc"); cv_fname = joinpath(filt_dir, "filt_conv.nc")
    for f in (dv_fname, cv_fname); isfile(f) || error("no $f: run steps 7 and 8 first"); end
    d_fname = joinpath(filt_dir, "filt_D.nc")
    DV, CV = NCDataset(dv_fname), NCDataset(cv_fname)
    bands_D = [f.name for f in filters if haskey(CV, "conv_$(f.name)_mode$(modes[1])") && all(n -> haskey(DV, "divF_$(f.name)_mode$n"), modes)]   # table order
    println("\nstep 9: D = conv - ∇·F, bands $(join(bands_D, ",")) -> $d_fname")
    tmp = d_fname * ".tmp"; isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do d
        defDim(d, "xi_rho", nx); defDim(d, "eta_rho", ny); defDim(d, "period", length(e_periods))
        d.attrib["title"] = "Residual of the modal energy balance D = conv - divF of the Butterworth bands (time means), $dff_run"
        d.attrib["periods"] = join(first.(e_periods), ",")
        d.attrib["note"] = "D = conv - divF [W m-2] from model_and_filter/filt_conv.nc and filt_divF.nc; > 0: local loss (dissipation, " *
                           "transfer to other modes / frequencies), < 0: local gain; NaN where divF is NaN"
        d.attrib["script"] = "internalwave_julia/DFF/modes_filter_analysis.jl (step 9)"
        defVar(d, "nrec", CV["nrec"][:], ("period",); attrib = ["long_name" => "hourly records in the period"])
        for b in bands_D, n in modes
            Dm = Float64.(CV["conv_$(b)_mode$n"][:, :, :]) .- Float64.(DV["divF_$(b)_mode$n"][:, :, :])
            defVar(d, "D_$(b)_mode$n", Float32.(Dm), ("xi_rho", "eta_rho", "period"); attrib = ["units" => "W m-2", "long_name" => "D_$(b)_mode$n"])
        end
    end
    mv(tmp, d_fname; force = true)
    println("  wrote $d_fname")
    DD = NCDataset(d_fname)
    # line-plot figures
    for b in e_bands
        b in bands_D || (println("    no D for band $b"); continue)
        fig_full(DD, "D", b, n -> "D_$n = conv_$n - ∇·F_$n, mode $n", "D = conv - ∇·F, strip mean",
                 (b, n) -> [(1e3 .* strip_stats(Float64.(CV["conv_$(b)_mode$n"][:, :, 1]))[1], :darkorange, :dot, "conv (step 8), strip mean"),
                            (1e3 .* strip_stats(Float64.(DV["divF_$(b)_mode$n"][:, :, 1]))[1], :crimson, :dash, "∇·F (step 7), strip mean")];
                 sym = true, unit = "mW m⁻²", sc = 1e3)
        fig_months(DD, "D", b; sym = true, unit = "mW m⁻²", sc = 1e3)
        fig_heatmap(DD, "D", b, "conv - ∇·F"; sym = true, unit = "mW m⁻²", sc = 1e3)
    end
    # summary: strip integrals [MW], full window, south of the node / node -> crop
    area = 1 ./ (pm .* pn)
    sint(a, js) = sum(a[i, j] * area[i, j] for i in strip, j in js if showE[i, j] && isfinite(a[i, j]); init = 0.0) / 1e6
    regions = [("south of the node (slope, < 200 km)", 1:j_node - 1), ("node -> crop (200 - $(round(Int, line_km[j_crop])) km)", j_node:j_crop)]
    groups = String[]; vals = Dict{Tuple{String, Int}, Vector{Float64}}()   # (group, region) => [conv, divF, D]
    for b in bands_D
        tot = [zeros(3) for _ in regions]
        for n in modes
            g = "$b\nmode $n"; push!(groups, g)
            c = Float64.(CV["conv_$(b)_mode$n"][:, :, 1]); dv = Float64.(DV["divF_$(b)_mode$n"][:, :, 1]); dd = Float64.(DD["D_$(b)_mode$n"][:, :, 1])
            for (r, (_, js)) in enumerate(regions)
                vals[(g, r)] = [sint(c, js), sint(dv, js), sint(dd, js)]; tot[r] .+= vals[(g, r)]
            end
        end
        g = "$b\nmodes $(join(modes, "+"))"; push!(groups, g)
        for r in eachindex(regions); vals[(g, r)] = tot[r]; end
    end
    println("  strip integrals [MW], full window: conv, ∇·F, D")
    for g in groups, (r, (rn, _)) in enumerate(regions)
        @printf("    %-16s %-36s conv %7.1f  ∇·F %7.1f  D %7.1f\n", replace(g, "\n" => " "), rn, vals[(g, r)]...)
    end
    fig = Figure(fontsize = 10pt)
    tcol = [:darkorange, :crimson, :royalblue]; tlab = ["conv (barotropic → mode)", "∇·F (flux divergence)", "D = conv − ∇·F (local loss)"]
    local axb
    for (r, (rn, _)) in enumerate(regions)
        axb = Axis(fig[r, 1]; title = rn, ylabel = "strip integral [MW]", xticks = (1:length(groups), groups), width = 1100, height = 330,
                   xticklabelsize = 8pt)
        xs = repeat(1:length(groups), inner = 3); dg = repeat(1:3, outer = length(groups))
        ys = vcat([vals[(g, r)] for g in groups]...)
        barplot!(axb, xs, ys; dodge = dg, color = tcol[dg], strokecolor = :black, strokewidth = 0.4)
        hlines!(axb, [0]; color = :black, linewidth = 0.8)
        for k in 1:length(bands_D) - 1                              # separators between the bands
            vlines!(axb, [k * (length(modes) + 1) + 0.5]; color = :gray50, linestyle = :dash, linewidth = 0.8)
        end
    end
    Legend(fig[length(regions) + 1, 1], [PolyElement(color = c, strokecolor = :black, strokewidth = 0.4) for c in tcol], tlab;
           orientation = :horizontal, framevisible = false, labelsize = 9pt)
    Label(fig[0, 1], "Modal energy budget of the Butterworth bands, integrated over the ±200 km strip\n$dff_run, $win_str"; fontsize = 11pt, font = :bold)
    Label(fig[length(regions) + 2, 1], "conv = φ_n(-H)⟨W_BT p_n⟩ (step 8), ∇·F of F = (H⟨p u⟩, H⟨p v⟩) (steps 6, 7), D = conv − ∇·F (step 9); Σ over the strip's deep points " *
          "(h ≥ 200 m, outside the 54 km edge bands)\nD > 0: lost locally (dissipation, transfer to other modes / frequencies); D < 0: gained from them; " *
          "hp28 ≈ M2 + hp9 + diurnal", fontsize = 8pt)
    resize_to_layout!(fig)
    fname = joinpath(fig_e, "filt_budget_summary.png"); save(fname, fig; px_per_unit = 2); fig = nothing; GC.gc()
    println("    saved $fname")
    close(DD); close(DV); close(CV)
end

# ---------------------------------------------------------------------------- (10) RMS surface speed
# U_rms = √< u_s² + v_s² > [m s-1], u_s = Σ_n u_n(t) φ_n(0) (φ_n at the top rho level), for mode 1, mode 2 and modes 1+2
# (the sum includes the 1×2 cross term, so it is not the sum of the two). The time mean of a band-passed u, v is ~0
# (it oscillates); the RMS is the wave amplitude (√2 × the amplitude for a single sinusoid). Grid xi / eta or east /
# north gives the same speed. Bands MF_SV_BANDS (default all:M2:hp28:hp9; "all": the raw puv record, mean flow of
# the modes included). -> model_and_filter/filt_surfvel.nc: Urms_<band>_mode<1|2|12> (xi, eta, period); reused unless
# MF_E_RECOMPUTE=1. Figures -> MF_E_FIG: filt_Urms_<band>_heatmap_month.png (rows the 3 sets, columns Aug .. full,
# cmocean speed) and filt_Urms_<band>_months.png (strip means by period).
function acc_surf!(a1::Array{Float64, 3}, a2::Array{Float64, 3}, a12::Array{Float64, 3}, u1::Array{Float32, 3}, v1::Array{Float32, 3},
                   u2::Array{Float32, 3}, v2::Array{Float32, 3}, f1::Matrix{Float64}, f2::Matrix{Float64},
                   m::Int, plist::Vector{Vector{Int}}, k0::Int)
    nxb, nyb = size(u1, 1), size(u1, 2)
    fin(x) = isfinite(x) ? Float64(x) : 0.0
    for kk in 1:m, p in plist[k0 + kk]
        @inbounds for j in 1:nyb, i in 1:nxb
            x1 = fin(u1[i, j, kk]) * f1[i, j]; y1 = fin(v1[i, j, kk]) * f1[i, j]
            x2 = fin(u2[i, j, kk]) * f2[i, j]; y2 = fin(v2[i, j, kk]) * f2[i, j]
            a1[i, j, p] += x1^2 + y1^2
            a2[i, j, p] += x2^2 + y2^2
            a12[i, j, p] += (x1 + x2)^2 + (y1 + y2)^2
        end
    end
end
if 10 in mf_steps
    all(in(modes), (1, 2)) || error("step 10 needs modes 1 and 2 (MF_MODES)")
    sv_bands = String.(split(get(ENV, "MF_SV_BANDS", "all:M2:hp28:hp9"), r"[,:; ]+"; keepempty = false))
    sv_fname = joinpath(filt_dir, "filt_surfvel.nc")
    sv_sets = ["1", "2", "12"]
    sv_keys = ["Urms_$(b)_mode$s" for b in sv_bands for s in sv_sets]
    println("\nstep 10: RMS surface speed √<u_s² + v_s²>, bands $(join(sv_bands, ",")), sets mode 1 / 2 / 1+2")
    need = !isfile(sv_fname) || get(ENV, "MF_E_RECOMPUTE", "0") == "1" || !NCDataset(d -> all(haskey(d, k) for k in sv_keys), sv_fname)
    if need
        mon = month.(date[t_keep])
        plist = [findall(f -> f(mon[k]), last.(e_periods)) for k in eachindex(t_keep)]
        nper = [count(k -> p in plist[k], eachindex(t_keep)) for p in eachindex(e_periods)]
        f1, f2 = Float64.(phi_top[:, :, 1]), Float64.(phi_top[:, :, 2])
        f1[.!isfinite.(f1)] .= 0; f2[.!isfinite.(f2)] .= 0
        nblk = 100
        bufs = [Array{Float32}(undef, nx, ny, nblk) for _ in 1:4]
        S = Dict{String, Array{Float64, 3}}()
        dm = (mask .> 0) .& (H .> 1000)
        for b in sv_bands
            t0 = time()
            a1, a2, a12 = (zeros(nx, ny, length(e_periods)) for _ in 1:3)
            flds = ("u1", "v1", "u2", "v2")
            ds_ = NCDataset[]
            vs = map(flds) do f
                b == "all" && return ds_puv[f].var
                fn = filt_fname((; name = b), f); isfile(fn) || error("no $fn: run step 2 first")
                d = NCDataset(fn); push!(ds_, d); d[f].var
            end
            ks = b == "all" ? t_keep : (1:length(t_keep))
            for k1 in 1:nblk:length(t_keep)
                k2 = min(k1 + nblk - 1, length(t_keep)); m = k2 - k1 + 1
                for q in 1:4
                    NCDatasets.load!(vs[q], view(bufs[q], :, :, 1:m), :, :, ks[k1]:ks[k2])
                end
                acc_surf!(a1, a2, a12, bufs..., f1, f2, m, plist, k1 - 1)
            end
            foreach(close, ds_)
            for (s, a) in zip(sv_sets, (a1, a2, a12))
                for p in eachindex(e_periods)
                    a[:, :, p] .= sqrt.(a[:, :, p] ./ nper[p])
                end
                S["Urms_$(b)_mode$s"] = a
            end
            @printf("  %-4s %.0f s; deep-mean U_rms (m/s) full: mode 1 %.3f, mode 2 %.3f, modes 1+2 %.3f\n", b, time() - t0,
                    (mean(S["Urms_$(b)_mode$s"][:, :, 1][dm]) for s in sv_sets)...)
            GC.gc()
        end
        tmp = sv_fname * ".tmp"; isfile(tmp) && rm(tmp)
        NCDataset(tmp, "c") do d
            defDim(d, "xi_rho", nx); defDim(d, "eta_rho", ny); defDim(d, "period", length(e_periods))
            d.attrib["title"] = "RMS surface speed of the modes, Butterworth bands and the raw record, $dff_run"
            d.attrib["periods"] = join(first.(e_periods), ",")
            d.attrib["window"] = "$(date[t_keep[1]]) to $(date[t_keep[end]]); aug from $(date[t_keep[1]])"
            d.attrib["note"] = "Urms = sqrt(<u_s^2 + v_s^2>), u_s = sum_n u_n phi_n(0) over the set (mode1, mode2, mode12 = 1+2 with " *
                               "the cross term); band 'all' = raw puv record (no filter)"
            d.attrib["script"] = "internalwave_julia/DFF/modes_filter_analysis.jl"
            defVar(d, "nrec", Int32.(nper), ("period",); attrib = ["long_name" => "hourly records in the period"])
            for (k, a) in S
                defVar(d, k, Float32.(a), ("xi_rho", "eta_rho", "period"); attrib = ["units" => "m s-1", "long_name" => k])
            end
        end
        mv(tmp, sv_fname; force = true)
        println("  wrote $sv_fname")
    else
        println("  from $sv_fname (MF_E_RECOMPUTE=1 to recompute)")
    end
    D = NCDataset(sv_fname)
    slab = Dict("1" => "mode 1", "2" => "mode 2", "12" => "modes 1+2")
    for b in sv_bands
        haskey(blab, b) || (blab[b] = b)
        fig_heatmap(D, "Urms", b, "√⟨u_s² + v_s²⟩ at the surface"; unit = "m s⁻¹", sc = 1.0, mlist = sv_sets, mlab = s -> slab[s], cmap = :speed)
        fig_months(D, "Urms", b; unit = "m s⁻¹", sc = 1.0, mlist = sv_sets, mlab = s -> slab[s])
    end
    close(D)
end

#= W_BT_make.jl
Hsin-Yi Chen, USM DMS
Barotropic vertical velocity at the bottom, W_BT = -U_BT · ∇H (U_BT = (ubar, vbar), H the water depth), the forcing of
the barotropic-to-baroclinic conversion C_n of each vertical mode (with the modal bottom pressure p_n φ_n(-H)).

Step 1 (this part): read ubar, vbar (and zeta) from the his files of the analysis window, move them from the u / v
points to the rho points, and write them, one his file at a time, to
    <run>/<BCEF_OUT>/bt_uvz.nc      ubar, vbar [m s-1] along grid xi / eta, zeta [m]  (xi_rho, eta_rho, time)
The same window and output folder as the modal decomposition (bcef_modes_puv.jl, START_DATE): for
amazon_2700m_cmems_0801 the his files from 2022-08-24 on, output in ddf_bcef_0824.
~7 GB for the 2700 m run (3 fields x 378 x 509 x 2353 records x 4 bytes): only one his file is in memory at a time.

Usage:
  julia DFF/W_BT_make.jl                        # amazon_2700m_cmems_0801, 2022-08-24 on (all 589 his files)
  BT_NFILES=8 julia DFF/W_BT_make.jl            # first 8 files only (32 h), quick test -> bt_uvz_test.nc
  DFF_RUN=... START_DATE=... BCEF_OUT=... julia DFF/W_BT_make.jl
=#

using NCDatasets, Dates, Statistics, Printf, LinearAlgebra
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
haskey(ENV, "DFF_RUN") || (ENV["DFF_RUN"] = "amazon_2700m_cmems_0801")   # default run of this script
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")         # dff_run, run_dir, run_grid_fname
include("/home/hchen54/internalwave_julia/function/uv2rho.jl")     # u2rho, v2rho
include("/home/hchen54/internalwave_julia/function/harm_fit.jl")   # harm_fit, TIDE_FREQ

# ============================================================================
# (1) settings, his files, grid; ubar, vbar, zeta -> rho points -> bt_uvz.nc
# ============================================================================
default_bcef  = Dict("amazon_2700m_cmems_0801" => "ddf_bcef_0824")   # output folder per run
default_start = Dict("amazon_2700m_cmems_0801" => "2022-08-24")      # first his file per run (after the spin-up)
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, get(default_bcef, dff_run, "ddf_bcef")))
mkpath(bcef_dir)
datadir = joinpath(run_dir, "after_joint")
nfiles_max = parse(Int, get(ENV, "BT_NFILES", "0"))                  # 0: all files of the window
# steps: BT_STEPS=1 (read his -> bt_uvz.nc), 2 (harmonic fit of ubar, vbar -> bt_harmonic.nc),
# 3 (bottom slope grad h -> bt_gradh.nc, check map in ~/figure_check),
# 4 (W_BT of each constituent from 2 and 3 -> bt_wbt_harmonic.nc, check map),
# 5 (conversion conv_n of each constituent and mode -> bt_conv_harmonic.nc); default 1,2,3,4,5.
# Step 1 is skipped when bt_uvz.nc already exists (delete it to re-read).
bt_steps = parse.(Int, split(get(ENV, "BT_STEPS", "1,2,3,4,5"), r"[,:; ]+"; keepempty = false))
out_fname = joinpath(bcef_dir, nfiles_max > 0 ? "bt_uvz_test.nc" : "bt_uvz.nc")

# his files of the window: the file name is the time of its first record
his_files = joinpath.(datadir, sort(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(datadir))))
file_time(f) = DateTime(match(r"(\d{14})\.nc$", f)[1], DateFormat("yyyymmddHHMMSS"))
start_str = get(ENV, "START_DATE", get(default_start, dff_run, ""))
if !isempty(start_str)
    t_start = DateTime(start_str)
    his_files = filter(f -> file_time(f) >= t_start, his_files)
end
nfiles_max > 0 && (his_files = his_files[1:min(nfiles_max, end)])
isempty(his_files) && error("no his files in $datadir from $start_str")

# grid: depth, metrics, rotation, land mask (rho points)
h, pm, pn, angle, mask, lon, lat, f_cor = NCDataset(run_grid_fname) do g
    Float64.(g["h"][:, :]), Float64.(g["pm"][:, :]), Float64.(g["pn"][:, :]), Float64.(g["angle"][:, :]),
    Float64.(g["mask_rho"][:, :]), Float64.(g["lon_rho"][:, :]), Float64.(g["lat_rho"][:, :]), Float64.(g["f"][:, :])
end
nx, ny = size(h)

# record count and times (each file's ocean_time, seconds since 1994-01-01)
t_ref = DateTime(1994, 1, 1)
rec_per_file = [NCDataset(d -> d.dim["time"], f) for f in his_files]
nt = sum(rec_per_file)
println("$dff_run: $(length(his_files)) his files from $(basename(first(his_files))) to $(basename(last(his_files))), $nt records")
println("  grid $nx x $ny (rho); h $(round.(extrema(h[mask .> 0]), digits = 0)) m; output $out_fname")

# read one his file with retries (Lustre timeouts show up as HDF / IO errors): ubar, vbar on rho points, zeta, time
function read_his(f; tries = 5)
    for t in 1:tries
        try
            return NCDataset(f) do d
                ub = u2rho(Float32.(nomissing(d["ubar"][:, :, :], 0f0)))   # (xi_u, eta_rho, t) -> (xi_rho, eta_rho, t)
                vb = v2rho(Float32.(nomissing(d["vbar"][:, :, :], 0f0)))   # (xi_rho, eta_v, t) -> (xi_rho, eta_rho, t)
                zt = Float32.(nomissing(d["zeta"][:, :, :], 0f0))
                tt = Float64.(d["ocean_time"].var[:])                     # raw seconds since 1994-01-01
                ub, vb, zt, tt
            end
        catch e
            t == tries && rethrow()
            println("  read $(basename(f)) failed ($(first(sprint(showerror, e), 80))), retry $t in $(2^t) min"); flush(stdout)
            sleep(60 * 2^t)
        end
    end
end

if 1 in bt_steps && isfile(out_fname)
    println("step 1: $out_fname exists, not re-read (delete it to re-read the his files)")
elseif 1 in bt_steps
tmp_fname = out_fname * ".tmp"
isfile(tmp_fname) && rm(tmp_fname)
t0 = time()
NCDataset(tmp_fname, "c") do o
    defDim(o, "xi_rho", nx); defDim(o, "eta_rho", ny); defDim(o, "time", nt)
    o.attrib["title"] = "Barotropic velocity and sea level on the rho points, $dff_run (input of W_BT = -U_BT · grad H)"
    o.attrib["source"] = "$datadir: $(basename(first(his_files))) to $(basename(last(his_files)))"
    o.attrib["note"] = "ubar, vbar along grid xi / eta (not east / north), moved from the u / v points with u2rho / v2rho (edge rows copied)"
    defVar(o, "lon_rho", lon, ("xi_rho", "eta_rho"); attrib = ["units" => "degree_east"])
    defVar(o, "lat_rho", lat, ("xi_rho", "eta_rho"); attrib = ["units" => "degree_north"])
    defVar(o, "h", h, ("xi_rho", "eta_rho"); attrib = ["units" => "m"])
    defVar(o, "mask_rho", mask, ("xi_rho", "eta_rho"))
    vt = defVar(o, "ocean_time", Float64, ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00", "calendar" => "proleptic_gregorian"])
    vu = defVar(o, "ubar", Float32, ("xi_rho", "eta_rho", "time"); attrib = ["units" => "m s-1", "long_name" => "barotropic velocity along grid xi"],
                chunksizes = [nx, ny, 1], deflatelevel = 1)
    vv = defVar(o, "vbar", Float32, ("xi_rho", "eta_rho", "time"); attrib = ["units" => "m s-1", "long_name" => "barotropic velocity along grid eta"],
                chunksizes = [nx, ny, 1], deflatelevel = 1)
    vz = defVar(o, "zeta", Float32, ("xi_rho", "eta_rho", "time"); attrib = ["units" => "m", "long_name" => "free surface"],
                chunksizes = [nx, ny, 1], deflatelevel = 1)
    r = 0
    for (k, f) in enumerate(his_files)
        ub, vb, zt, tt = read_his(f)
        rr = r+1:r+length(tt)
        vu[:, :, rr] = ub; vv[:, :, rr] = vb; vz[:, :, rr] = zt; vt[rr] = tt
        r = last(rr)
        ub = vb = zt = nothing
        k % 10 == 0 && (NCDatasets.sync(o); GC.gc())   # flush to disk and free memory: the login node allows 8 GB
        (k % 50 == 0 || k == length(his_files)) &&
            (println("  $k / $(length(his_files)) files, $r records ($(round(Int, time() - t0)) s)"); flush(stdout))
    end
    r == nt || error("wrote $r records, expected $nt")
end
mv(tmp_fname, out_fname; force = true)
println("wrote $out_fname")

# quick check: time-mean |U_BT| and the records' time step
NCDataset(out_fname) do d
    t = t_ref .+ Millisecond.(round.(Int, 1000 .* d["ocean_time"].var[:]))
    ub, vb = d["ubar"][:, :, 1], d["vbar"][:, :, 1]
    sp = hypot.(ub, vb)[mask .> 0]
    @printf("  %s -> %s, dt = %s; first record |U_BT|: median %.3f, max %.3f m/s\n", t[1], t[end], t[2] - t[1], median(sp), maximum(sp))
end
end   # step 1

# ============================================================================
# (2) harmonic fit of ubar, vbar (harm_fit, as mode_bcef_harmonic.jl) -> bt_harmonic.nc
# ============================================================================
# The same record window and constituents as the modal fit (mode_bcef_harmonic.jl: SPINUP days after the first record,
# M2, S2, N2, K1, O1, M4, MS4, MN4, M6), so both have the same record centre tc and their phases g' compare directly:
# the conversion C_n needs the phase difference between W_BT (from ubar, vbar) and the modal bottom pressure.
# Checked against tc of bcef_modes<N>_harmonic.nc when that file is there.
# Output: <bcef_dir>/bt_harmonic.nc (bt_harmonic_test.nc for BT_NFILES): ubar_amp, ubar_pha, ..._amp_err, ..._pha_err,
# ..._mean, ..._trend, ..._resid_std, ..._std for ubar and vbar (xi_rho, eta_rho[, con]).
if 2 in bt_steps
    CONS   = String.(split(get(ENV, "HARM_CONS", "M2,S2,N2,K1,O1,M4,MS4,MN4,M6"), r"[,:; ]+"; keepempty = false))
    SPINUP = parse(Float64, get(ENV, "SPINUP", "3.0"))                # days dropped at the start, as the modal fit
    isfile(out_fname) || error("no $out_fname: run step 1 first")
    harm_out = joinpath(bcef_dir, nfiles_max > 0 ? "bt_harmonic_test.nc" : "bt_harmonic.nc")
    tsec_all = NCDataset(d -> Float64.(d["ocean_time"].var[:]), out_fname)   # seconds since 1994-01-01
    t0 = findfirst(>=(tsec_all[1] + SPINUP * 86400 - 1), tsec_all)
    tidx = t0:length(tsec_all)
    tsec = tsec_all[tidx]
    tc = (tsec[1] + tsec[end]) / 2
    tcdate = t_ref + Millisecond(round(Int, 1000tc))
    @printf("\nstep 2: harmonic fit of ubar, vbar: %d records %s -> %s (%.1f days spin-up dropped), tc = %s; cons %s\n",
            length(tidx), t_ref + Millisecond(round(Int, 1000tsec[1])), t_ref + Millisecond(round(Int, 1000tsec[end])),
            SPINUP, tcdate, join(CONS, ","))
    modal_harm = joinpath(bcef_dir, "bcef_modes3_harmonic.nc")
    if isfile(modal_harm) && nfiles_max == 0
        tc_modal = NCDataset(d -> Float64(d.attrib["tc_seconds_since_1994"]), modal_harm)
        abs(tc - tc_modal) < 1 ? println("  same tc as $(basename(modal_harm)): phases compare directly") :
            @warn "tc differs from $(basename(modal_harm)) by $((tc - tc_modal) / 3600) h: phases do not compare directly"
    end
    # one field at a time, blocks of 100 records with retries (Lustre)
    function read_bt(name)
        a = Array{Float32}(undef, nx, ny, length(tidx))
        t = @elapsed for b in Iterators.partition(eachindex(tidx), 100)
            for attempt in 1:6
                try
                    NCDataset(d -> (a[:, :, b] .= nomissing(d[name][:, :, tidx[b]], NaN32)), out_fname)
                    break
                catch e
                    attempt == 6 && rethrow()
                    @printf("  %s records %d-%d failed (%s); retry %d in %d min\n", name, tidx[first(b)], tidx[last(b)],
                            first(sprint(showerror, e), 80), attempt, 2^attempt); flush(stdout)
                    sleep(60 * 2^attempt)
                end
            end
        end
        @printf("  read %s: %.2f GB in %.0f s\n", name, sizeof(a) / 1e9, t)
        a
    end
    sea = (mask .> 0) .& (h .>= 200)
    fits = Dict{String, Any}()
    for name in ("ubar", "vbar")
        println("== $name")
        a = read_bt(name)
        fits[name] = harm_fit(a, tsec, CONS)
        a = nothing; GC.gc()
        hf = fits[name]
        for (k, c) in enumerate(CONS)
            A = hf.amp[:, :, k][sea]
            @printf("   %-4s median A' %.3g m/s (90th pct %.3g), median error %.2g\n", c, median(A), quantile(A, 0.9),
                    median(hf.aerr[:, :, k][sea]))
        end
        @printf("   tides explain %.1f%% of the variance (median over h >= 200 m)\n",
                100 * (1 - median(hf.resid[sea] .^ 2 ./ hf.std[sea] .^ 2)))
    end
    tmp = harm_out * ".tmp"
    isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do o
        defDim(o, "xi_rho", nx); defDim(o, "eta_rho", ny); defDim(o, "con", length(CONS))
        o.attrib["title"] = "Plain sinusoid harmonic fit of the barotropic velocity ubar, vbar (grid xi / eta), $dff_run"
        o.attrib["model"] = "y = mean + trend (t - tc)/span + sum A' cos(w (t - tc) - g'); no nodal corrections, no inference"
        o.attrib["tc"] = string(tcdate); o.attrib["tc_seconds_since_1994"] = tc
        o.attrib["records"] = length(tidx); o.attrib["spinup_days"] = SPINUP
        o.attrib["source"] = out_fname; o.attrib["script"] = "internalwave_julia/DFF/W_BT_make.jl"
        defVar(o, "con", CONS, ("con",))
        defVar(o, "freq", [TIDE_FREQ[c] for c in CONS], ("con",); attrib = ["units" => "deg/hour"])
        defVar(o, "lon_rho", lon, ("xi_rho", "eta_rho")); defVar(o, "lat_rho", lat, ("xi_rho", "eta_rho"))
        defVar(o, "h", h, ("xi_rho", "eta_rho"); attrib = ["units" => "m"])
        defVar(o, "mask_rho", mask, ("xi_rho", "eta_rho"))
        for name in ("ubar", "vbar")
            hf = fits[name]
            for (nm, x, u, ln) in (("amp", hf.amp, "m s-1", "amplitude A'"), ("pha", hf.pha, "deg", "phase g' at tc"),
                                   ("amp_err", hf.aerr, "m s-1", "amplitude error"), ("pha_err", hf.perr, "deg", "phase error"))
                defVar(o, "$(name)_$nm", Float32.(x), ("xi_rho", "eta_rho", "con"); attrib = ["units" => u, "long_name" => "$name $ln"], deflatelevel = 3)
            end
            for (nm, x, ln) in (("mean", hf.mean, "time mean"), ("trend", hf.trend, "change over the record"),
                                ("resid_std", hf.resid, "residual std"), ("std", hf.std, "std of the data"))
                defVar(o, "$(name)_$nm", Float32.(x), ("xi_rho", "eta_rho"); attrib = ["units" => "m s-1", "long_name" => "$name $ln"], deflatelevel = 3)
            end
        end
    end
    mv(tmp, harm_out; force = true)
    println("wrote $harm_out")
end

# ============================================================================
# (3) bottom slope ∇h on the rho points, along grid xi / eta (the axes of ubar, vbar)
# ============================================================================
# dh/dx = pm ∂h/∂ξ, dh/dy = pn ∂h/∂η (pm, pn = 1/dx, 1/dy at the rho points): centred differences inside,
# one-sided at the edge rows. h from the grid file (the bathymetry the model uses); ∇ζ left out (~1e-6 vs ~1e-2
# on the slope), so W_BT = -(ubar dh/dx + vbar dh/dy). Land points (mask_rho = 0) are set to NaN.
# Output: <bcef_dir>/bt_gradh.nc (dhdx, dhdy [m/m], slope = |∇h|); check map ~/figure_check/W_BT_gradh.png
if 3 in bt_steps
    # centred difference along dimension dim, one-sided at both ends; per grid step (index units)
    function dindex(a, dim)
        d = similar(a)
        n = size(a, dim)
        sl(k) = dim == 1 ? (k, :) : (:, k)
        d[sl(2:n-1)...] .= (a[sl(3:n)...] .- a[sl(1:n-2)...]) ./ 2
        d[sl(1)...] .= a[sl(2)...] .- a[sl(1)...]
        d[sl(n)...] .= a[sl(n)...] .- a[sl(n-1)...]
        d
    end
    land = mask .== 0
    dhdx = pm .* dindex(h, 1)                     # [m/m] along grid xi
    dhdy = pn .* dindex(h, 2)                     # [m/m] along grid eta
    # a difference that reaches over land uses the land h (the minimum depth): not a real slope there
    near_land = land .| circshift(land, (1, 0)) .| circshift(land, (-1, 0)) .| circshift(land, (0, 1)) .| circshift(land, (0, -1))
    dhdx[near_land] .= NaN; dhdy[near_land] .= NaN
    slope = hypot.(dhdx, dhdy)
    deep = .!near_land .& (h .>= 200)
    @printf("\nstep 3: grad h: |∇h| median %.4f, 90th pct %.4f, max %.3f (h >= 200 m); over the slope 200-3000 m: median %.4f\n",
            median(slope[deep]), quantile(slope[deep], 0.9), maximum(slope[deep]),
            median(slope[deep .& (h .<= 3000)]))
    grad_out = joinpath(bcef_dir, "bt_gradh.nc")
    tmp = grad_out * ".tmp"
    isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do o
        defDim(o, "xi_rho", nx); defDim(o, "eta_rho", ny)
        o.attrib["title"] = "Bottom slope on the rho points along grid xi / eta, $dff_run grid ($(basename(run_grid_fname)))"
        o.attrib["note"] = "dhdx = pm dh/dxi, dhdy = pn dh/deta, centred differences (one-sided at the edges); NaN on land and next to it"
        defVar(o, "lon_rho", lon, ("xi_rho", "eta_rho")); defVar(o, "lat_rho", lat, ("xi_rho", "eta_rho"))
        defVar(o, "h", h, ("xi_rho", "eta_rho"); attrib = ["units" => "m"])
        defVar(o, "dhdx", dhdx, ("xi_rho", "eta_rho"); attrib = ["units" => "1", "long_name" => "dh/dx along grid xi"])
        defVar(o, "dhdy", dhdy, ("xi_rho", "eta_rho"); attrib = ["units" => "1", "long_name" => "dh/dy along grid eta"])
        defVar(o, "slope", slope, ("xi_rho", "eta_rho"); attrib = ["units" => "1", "long_name" => "|grad h|"])
    end
    mv(tmp, grad_out; force = true)
    println("wrote $grad_out")

    # check map: |∇h|, dh/dx, dh/dy (the shelf h < 200 m greyed)
    @eval using CairoMakie
    include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")
    lon180 = ifelse.(lon .> 180, lon .- 360, lon)
    shw(a) = ifelse.(deep, a, NaN)
    smax = round(quantile(slope[deep], 0.99); sigdigits = 2)
    fig = Figure(size = (1500, 650), fontsize = 14)
    for (k, (a, ttl, cm, cr)) in enumerate([(slope, "|∇h|", :thermal, (0, smax)),
                                            (dhdx, "∂h/∂x (grid ξ, across the mooring line)", :balance, (-smax, smax)),
                                            (dhdy, "∂h/∂y (grid η, along the mooring line)", :balance, (-smax, smax))])
        ax = topdown_axis(fig[1, k]; title = ttl, ylabel = k == 1 ? "Latitude" : "", backgroundcolor = :gray80)
        hm = plot_curvilinear!(ax, lon180, lat, shw(a); colormap = cm, colorrange = cr)
        contour!(ax, lon180, lat, h; levels = [200, 1000, 2000, 3000], color = :gray40, linewidth = 0.6)
        Colorbar(fig[2, k], hm; vertical = false, flipaxis = false, label = "[m/m]")
    end
    Label(fig[0, :], "bottom slope of the $dff_run grid; h < 200 m and points next to land not drawn; isobaths 200 / 1000 / 2000 / 3000 m"; fontsize = 15)
    mkpath(expanduser("~/figure_check"))
    fname = expanduser("~/figure_check/W_BT_gradh.png")
    save(fname, fig; px_per_unit = 2)
    println("saved $fname")
end

# ============================================================================
# (4) W_BT = -(ubar dh/dx + vbar dh/dy) of each constituent, from the fit (2) and the slope (3)
# ============================================================================
# Linear in ubar, vbar, so each constituent comes from complex amplitudes: with A cos(ωt' - g) = Re(A e^{-ig} e^{iωt'})
# (t' = t - tc), U = A_u e^{-i g_u}, V = A_v e^{-i g_v}, Ŵ = -(dh/dx U + dh/dy V): A_W = |Ŵ|, g_W = -arg Ŵ (0-360).
# Same tc as the modal fit, so g_W compares directly with the phase of the modal bottom pressure for C_n.
# Output: <bcef_dir>/bt_wbt_harmonic.nc: W_amp [m s-1], W_pha [deg] (xi_rho, eta_rho, con); NaN on and next to land.
# Check map: ~/figure_check/W_BT_<con>.png (HARM_W_CHECK, default M2)
if 4 in bt_steps
    harm_in = joinpath(bcef_dir, nfiles_max > 0 ? "bt_harmonic_test.nc" : "bt_harmonic.nc")
    grad_in = joinpath(bcef_dir, "bt_gradh.nc")
    (isfile(harm_in) && isfile(grad_in)) || error("need $harm_in (step 2) and $grad_in (step 3)")
    hw = NCDataset(harm_in)
    wcons = String.(hw["con"][:])
    tc_w = Float64(hw.attrib["tc_seconds_since_1994"])
    gx, gy = NCDataset(g -> (Float64.(g["dhdx"][:, :]), Float64.(g["dhdy"][:, :])), grad_in)
    camp(name, k) = Float64.(hw["$(name)_amp"][:, :, k]) .* cis.(-deg2rad.(Float64.(hw["$(name)_pha"][:, :, k])))
    W_amp = fill(NaN32, nx, ny, length(wcons)); W_pha = fill(NaN32, nx, ny, length(wcons))
    wsea = isfinite.(gx) .& (h .>= 200)
    println("\nstep 4: W_BT = -(ubar dh/dx + vbar dh/dy) per constituent (tc = $(hw.attrib["tc"]))")
    for (k, c) in enumerate(wcons)
        W = .-(gx .* camp("ubar", k) .+ gy .* camp("vbar", k))
        W_amp[:, :, k] .= abs.(W)
        W_pha[:, :, k] .= mod.(-rad2deg.(Base.angle.(W)), 360)   # Base.angle: `angle` is the grid rotation here
        A = Float64.(W_amp[:, :, k][wsea])
        @printf("   %-4s A_W median %.2e m/s, 90th pct %.2e, 99th pct %.2e (h >= 200 m)\n", c, median(A), quantile(A, 0.9), quantile(A, 0.99))
    end
    close(hw)
    wout = joinpath(bcef_dir, nfiles_max > 0 ? "bt_wbt_harmonic_test.nc" : "bt_wbt_harmonic.nc")
    tmp = wout * ".tmp"
    isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do o
        defDim(o, "xi_rho", nx); defDim(o, "eta_rho", ny); defDim(o, "con", length(wcons))
        o.attrib["title"] = "Barotropic bottom vertical velocity W_BT = -(ubar dh/dx + vbar dh/dy) per constituent, $dff_run"
        o.attrib["model"] = "W_BT(t) = W_amp cos(w (t - tc) - W_pha), from the harmonic fit of ubar, vbar ($(basename(harm_in))) and grad h ($(basename(grad_in)))"
        o.attrib["tc_seconds_since_1994"] = tc_w
        o.attrib["script"] = "internalwave_julia/DFF/W_BT_make.jl"
        defVar(o, "con", wcons, ("con",))
        defVar(o, "lon_rho", lon, ("xi_rho", "eta_rho")); defVar(o, "lat_rho", lat, ("xi_rho", "eta_rho"))
        defVar(o, "h", h, ("xi_rho", "eta_rho"); attrib = ["units" => "m"])
        defVar(o, "W_amp", W_amp, ("xi_rho", "eta_rho", "con"); attrib = ["units" => "m s-1", "long_name" => "W_BT amplitude"], deflatelevel = 3)
        defVar(o, "W_pha", W_pha, ("xi_rho", "eta_rho", "con"); attrib = ["units" => "deg", "long_name" => "W_BT phase g' at tc"], deflatelevel = 3)
    end
    mv(tmp, wout; force = true)
    println("wrote $wout")

    # check map of one constituent: A_W and g_W
    @eval using CairoMakie
    include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")
    lon180 = ifelse.(lon .> 180, lon .- 360, lon)
    for c in String.(split(get(ENV, "HARM_W_CHECK", "M2"), r"[,:; ]+"; keepempty = false))
        k = findfirst(==(c), wcons)
        k === nothing && continue
        A = ifelse.(wsea, Float64.(W_amp[:, :, k]) .* 1e3, NaN)       # [mm/s]
        g = ifelse.(wsea, Float64.(W_pha[:, :, k]), NaN)
        amax = round(quantile(filter(isfinite, A), 0.99); sigdigits = 2)
        fig = Figure(size = (1100, 650), fontsize = 14)
        ax1 = topdown_axis(fig[1, 1]; title = "A_W of $c (W_BT amplitude)", backgroundcolor = :gray80)
        hm1 = plot_curvilinear!(ax1, lon180, lat, A; colormap = :thermal, colorrange = (0, amax))
        ax2 = topdown_axis(fig[1, 2]; title = "g_W of $c (phase at tc)", ylabel = "", backgroundcolor = :gray80)
        hm2 = plot_curvilinear!(ax2, lon180, lat, g; colormap = :romaO, colorrange = (0, 360))
        for ax in (ax1, ax2)
            contour!(ax, lon180, lat, h; levels = [200, 1000, 2000, 3000], color = :gray40, linewidth = 0.6)
        end
        Colorbar(fig[2, 1], hm1; vertical = false, flipaxis = false, label = "A_W [mm/s]")
        Colorbar(fig[2, 2], hm2; vertical = false, flipaxis = false, label = "g_W [°]", ticks = 0:90:360)
        Label(fig[0, :], "$c barotropic bottom vertical velocity W_BT = -(ubar ∂h/∂x + vbar ∂h/∂y), $dff_run; h < 200 m not drawn"; fontsize = 15)
        fname = expanduser("~/figure_check/W_BT_$(c).png")
        save(fname, fig; px_per_unit = 2)
        println("saved $fname")
    end
end

# ============================================================================
# (5) barotropic-to-baroclinic conversion conv_n = <W_BT p_n,bot> of each constituent and mode
# ============================================================================
# The modal pressure at the bottom: p_n,bot(t) = p_n(t) φ_n(-H) (φ_n of bcef_modes<N>_structure.nc at the bottom rho
# level, sign kept: (1/H)∫φ² dz = 1, φ > 0 at the surface). With both fits at the same tc, for one constituent
#     conv_n = ½ A_W A_p,n φ_n(-H) cos(g_W - g_p,n)    [W m-2]
# positive: energy from the barotropic tide into mode n (named conv_n, not C_n: c_n is the mode speed).
# Output: <bcef_dir>/bt_conv_harmonic.nc: conv_<con>_mode<n> (xi_rho, eta_rho), phi_bot (xi_rho, eta_rho, mode);
# printed: the area integral over h >= 200 m inside the edge bands [GW].
if 5 in bt_steps
    NMODE_S = parse(Int, get(ENV, "NMODE", "3"))
    w_in = joinpath(bcef_dir, nfiles_max > 0 ? "bt_wbt_harmonic_test.nc" : "bt_wbt_harmonic.nc")
    p_in = joinpath(bcef_dir, "bcef_modes$(NMODE_S)_harmonic.nc")
    s_in = joinpath(bcef_dir, "bcef_modes$(NMODE_S)_structure.nc")
    for f in (w_in, p_in, s_in)
        isfile(f) || error("missing $f")
    end
    dw, dp = NCDataset(w_in), NCDataset(p_in)
    abs(Float64(dw.attrib["tc_seconds_since_1994"]) - Float64(dp.attrib["tc_seconds_since_1994"])) < 1 ||
        error("W_BT and the modal fit have different tc: phases do not compare")
    wc, pc = String.(dw["con"][:]), String.(dp["con"][:])
    cmodes = parse.(Int, split(dp.attrib["modes"], ","))
    phi_bot = NCDataset(d -> Float64.(d["phi"][:, :, 1, :]), s_in)     # bottom rho level (k = 1), (xi, eta, mode)
    edge_s = round(Int, parse(Float64, get(ENV, "EDGE_KM", "54")) / (mean(1 ./ pm) / 1e3))
    inside_s = (edge_s .< (1:nx) .<= nx - edge_s) .& (edge_s .< (1:ny)' .<= ny - edge_s)
    area = 1 ./ (pm .* pn)                                               # [m2] per cell
    creg = inside_s .& (h .>= 200) .& (mask .> 0)
    println("\nstep 5: conversion conv_n = ½ A_W A_p φ_n(-H) cos(g_W - g_p), modes $(join(cmodes, ",")); integral over h >= 200 m, $(edge_s)-point edges left out")
    conv = Dict{String, Matrix{Float64}}()
    for c in wc
        kp = findfirst(==(c), pc)
        kp === nothing && continue
        kw = findfirst(==(c), wc)
        AW, gW = Float64.(dw["W_amp"][:, :, kw]), Float64.(dw["W_pha"][:, :, kw])
        tot = 0.0
        line = @sprintf("   %-4s", c)
        for n in cmodes
            Ap, gp = Float64.(dp["p$(n)_amp"][:, :, kp]), Float64.(dp["p$(n)_pha"][:, :, kp])
            cv = 0.5 .* AW .* Ap .* phi_bot[:, :, n] .* cosd.(gW .- gp)
            conv["conv_$(c)_mode$n"] = cv
            ok = creg .& isfinite.(cv)
            I = sum(cv[ok] .* area[ok]) / 1e9
            pos = sum(max.(cv[ok], 0) .* area[ok]) / 1e9
            tot += I
            line *= @sprintf("  mode %d: %.3f GW (positive part %.3f)", n, I, pos)
        end
        println(line * @sprintf("  | sum %.3f GW", tot))
    end
    close(dw); close(dp)
    cout = joinpath(bcef_dir, nfiles_max > 0 ? "bt_conv_harmonic_test.nc" : "bt_conv_harmonic.nc")
    tmp = cout * ".tmp"
    isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do o
        defDim(o, "xi_rho", nx); defDim(o, "eta_rho", ny); defDim(o, "mode", size(phi_bot, 3))
        o.attrib["title"] = "Barotropic-to-baroclinic conversion conv_n = <W_BT p_n,bot> per constituent and mode, $dff_run"
        o.attrib["formula"] = "conv_n = 1/2 A_W A_p,n phi_n(-H) cos(g_W - g_p,n); W_BT = -(ubar dh/dx + vbar dh/dy); positive: barotropic -> mode n"
        o.attrib["sources"] = "$(basename(w_in)), $(basename(p_in)), $(basename(s_in))"
        o.attrib["modes"] = join(cmodes, ","); o.attrib["script"] = "internalwave_julia/DFF/W_BT_make.jl"
        defVar(o, "lon_rho", lon, ("xi_rho", "eta_rho")); defVar(o, "lat_rho", lat, ("xi_rho", "eta_rho"))
        defVar(o, "h", h, ("xi_rho", "eta_rho"); attrib = ["units" => "m"])
        defVar(o, "phi_bot", phi_bot, ("xi_rho", "eta_rho", "mode"); attrib = ["long_name" => "mode structure at the bottom rho level"])
        for k in sort(collect(keys(conv)))
            defVar(o, k, Float32.(conv[k]), ("xi_rho", "eta_rho"); attrib = ["units" => "W m-2", "long_name" => "conversion into $(split(k, "_")[3]) at $(split(k, "_")[2])"], deflatelevel = 3)
        end
    end
    mv(tmp, cout; force = true)
    println("wrote $cout")
end

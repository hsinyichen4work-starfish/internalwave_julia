#= wave_mean_flow.jl
Hsin-Yi Chen, USM DMS
Wave - mean flow interaction in the ROMS runs.
Step 1 (this file so far): directory setup; next: read the variables and separate the background (mean / sub-tidal)
flow from the waves (tidal / internal-wave band).

Run picked with WMF_RUN (keys of RUNS below); default amazon_2700m_cmems_0801:
  amazon_2700m_cmems_0801   2700 m, CMEMS ini/bry + ERA5 + TPXO10 tides, 55-point sponge, 2022-08-01 -> 11-30
                            378 x 509 rho points, 128 s-levels (theta_s 6, theta_b 3, hc 250)

Input  (run_dir/after_joint, joined files; the per-tile files sit in run_dir itself):
  roms_his.*.nc   hourly snapshots (dt 3600 s), 4 records per file: zeta, ubar, vbar, u, v, w, omega, temp, salt,
                  rho, Akv, Akt, hbls   <- the only input used here
  (roms_avg.*.nc daily averages and roms_ext.*.done markers are not used)
Background / wave split in time windows:
  SPAN_START, SPAN_END   analysis span [start, end) (default 2022-09-15 -> 10-15, 30 days)
  VARS            his variables averaged (default zeta (ssh), ubar, vbar, u, v, w, temp, salt, rho)
  WINDOW_LENGTH   window length [h] (default 24*3 = 3 days)
  WINDOW_OVERLAP  fraction of a window shared with the next one, 0 <= overlap < 1:
                  0 = no overlap (step = window length), 0.5 = half (2-day window moves by 1 day);
                  1 = full overlap, taken as a running window that moves by one his record (1 h)
Output (out_dir):  run_dir/meanflowdata/L<len>h_ov<overlap>/<var>/<var>_win<NNN>_<t0>.nc   one window mean per file
Figures (fig_dir): figure/<run>/wave_meanflow
Small helper functions for this analysis live in this folder (wave_meanflow/), e.g. meanflow_window.jl.

Usage:
  julia --project=/home/hchen54/internalwave_julia wave_meanflow/wave_mean_flow.jl
  WMF_RUN=amazon_2700m_cmems_0801 julia --project=. --heap-size-hint=3G wave_meanflow/wave_mean_flow.jl
=#

using NCDatasets, Printf, Statistics, Dates, GibbsSeaWater, DSP

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
include("/home/hchen54/internalwave_julia/function/butter_filters.jl")   # bandpass_butter, lowhighpass_butter (Butterworth filtfilt)
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")   # time_dir, records_by_file, data_load, zlevs3, uvp_masks, rho2u/v, ...
include("/home/hchen54/internalwave_julia/wave_meanflow/meanflow_window.jl")   # helpers of this analysis (background / wave split)
include("/home/hchen54/internalwave_julia/wave_meanflow/buoyancy_n2.jl")       # buoyancy, n2_rho, n2_gsw, window_bn2
include("/home/hchen54/internalwave_julia/wave_meanflow/getting_modes.jl")     # column_modes, all_modes, window_modes
include("/home/hchen54/internalwave_julia/wave_meanflow/wave_perturb.jl")      # wave_owner; window_wave (3-D wave parts, checks only)
include("/home/hchen54/internalwave_julia/wave_meanflow/mode_projection.jl")   # s_stretch; window_project (later step: mode projections)
include("/home/hchen54/internalwave_julia/wave_meanflow/wave_bands.jl")        # steps 5-6: rho/u/v/zeta_wave -> hp28, bp9_15, hp9 bands
include("/home/hchen54/internalwave_julia/wave_meanflow/barotropic.jl")        # step 7: u_bt, v_bt, p_bt per band

# ============================================================================
# (0) directories
# ============================================================================
const BASE = "/expanse/lustre/projects/uso102/hchen54/"
const RUNS = Dict(
    "amazon_2700m_cmems_0801" => (fold = "amazon_2700m_cmems_0801",
                                  grid = BASE * "input/grid/roms_cmems_grd_2700m.nc",
                                  moor = BASE * "input/grid/roms_cmems_grd_2700m_mor_edata.nc"),
)

const RUN = get(ENV, "WMF_RUN", "amazon_2700m_cmems_0801")
haskey(RUNS, RUN) || error("WMF_RUN = $RUN, not one of $(sort(collect(keys(RUNS))))")
const cfg = RUNS[RUN]

const run_dir     = BASE * cfg.fold                        # run folder (per-tile output + logs)
const data_folder = joinpath(run_dir, "after_joint") * "/"  # joined roms_his / roms_avg files (input)
const grid_file   = cfg.grid                               # model grid (grid_file attribute of the output)
const moor_file   = cfg.moor                               # mooring locations
const out_dir     = joinpath(run_dir, "meanflowdata")       # window means (background) of this analysis
const fig_dir     = BASE * "figure/$RUN/wave_meanflow"     # figures
const CHECK_DIR   = expanduser("~/figure_check")           # copy of figures for checking; nothing = no copy

# input files and time axis (his only)
const HIS_PREFIX = "roms_his."   # hourly snapshots: waves + background
const DT_HIS     = 3600          # [s]
const t_ref      = DateTime(1994, 1, 1)   # ocean_time reference ("Time since 1994/01/01")
const SPAN_START = DateTime(2022, 9, 15)    # analysis span [SPAN_START, SPAN_END): 30 days
const SPAN_END   = DateTime(2022, 10, 15)

# background / wave split window
# his variables to average, in one run (the windows are found once); WMF_VARS="u,v" runs only those
const VARS = String.(split(get(ENV, "WMF_VARS", "zeta,ubar,vbar,u,v,w,temp,salt,rho"), ","))
const WINDOW_LENGTH  = 24 * 3     # [h] window length (3 days)
const WINDOW_OVERLAP = 0.0        # 0 = no overlap, 0.5 = half (2-day window moves 1 day), 1 = moves by one record

for (name, p) in (("run_dir", run_dir), ("data_folder", data_folder))
    isdir(p) || error("$name: no directory $p")
end
for (name, p) in (("grid_file", grid_file), ("moor_file", moor_file))
    isfile(p) || error("$name: no file $p")
end
mkpath(out_dir); mkpath(fig_dir)
CHECK_DIR === nothing || mkpath(CHECK_DIR)

println("run      ", RUN)
println("input    ", data_folder)
println("grid     ", grid_file)
println("output   ", out_dir)
println("figures  ", fig_dir)

# ============================================================================
# (1) windows in the analysis span
# ============================================================================
win = window_setup(data_folder, VARS[1]; tstart = SPAN_START, tend = SPAN_END, window_length = WINDOW_LENGTH,
                   window_overlap = WINDOW_OVERLAP, prefix = HIS_PREFIX, dt = DT_HIS, t_ref)
println("$(win.nwin) windows")

# ============================================================================
# (2) window means (background), one file per window
# ============================================================================
# WMF_WINDOWS = "1:3" or "5" runs only those windows (e.g. split between jobs); default all
const WINDOWS  = haskey(ENV, "WMF_WINDOWS") ? vec(collect(eval(Meta.parse(ENV["WMF_WINDOWS"])))) : collect(1:win.nwin)
const mean_root = joinpath(out_dir, @sprintf("L%gh_ov%.2f", WINDOW_LENGTH, WINDOW_OVERLAP))   # one folder per variable
# WMF_STEPS = "2,5" runs only those sections (2 means, 3 B/N², 4 modes, 5-6 wave parts + bands, 7 barotropic part);
# default all.
# Batch: the per-variable array runs "2" (each task its own variable), then "3,4" and "5" (per variable) after it.
const STEPS = parse.(Int, split(get(ENV, "WMF_STEPS", "2,3,4,5,7"), ","))
2 in STEPS && (mean_files = window_means(win, data_folder, mean_root, VARS; windows = WINDOWS, grid_file))   # done windows skipped

# ============================================================================
# (3) buoyancy B = -g rho / rho0 and N² of the window means (buoyancy_n2.jl)
# ============================================================================
# from the means of rho, temp, salt, zeta (section 2); B on rho points, N2 (TEOS-10) and N2rho (ROMS rho) on w points
# -> mean_root/B/, mean_root/N2/, mean_root/N2rho/, one file per window; done windows skipped
3 in STEPS && (bn2_files = window_bn2(win, mean_root; windows = WINDOWS, grid_file, his_file = joinpath(data_folder, win.td.files[1])))

# ============================================================================
# (4) vertical modes of each window from its N² (getting_modes.jl)
# ============================================================================
# W'' + (N²/c²) W = 0, W = 0 at bottom and surface, N² = TEOS-10 N2 of section 3, levels of the window mean zeta
# -> mean_root/modes<NMODE>/: c (cn), phi (φn: p, u, v on rho levels), Phi (Φn: w, displacement on w levels)
const NMODE = parse(Int, get(ENV, "WMF_NMODE", "10"))  # number of modes kept (WMF_NMODE=3 for the 3-mode set)
4 in STEPS && (mode_files = window_modes(win, mean_root; windows = WINDOWS, grid_file, his_file = joinpath(data_folder, win.td.files[1]),
                          nmode = NMODE))

# ============================================================================
# (5) wave parts at a fixed depth  +  (6) three frequency bands of them (wave_bands.jl)
# ============================================================================
# (5) var_wave = var − background at the record's depths (fixed z), every hourly record: rho_wave, u_wave, v_wave,
#     zeta_wave; no depth mean removed yet. Not saved: kept in memory one horizontal tile at a time.
# (6) each var_wave through a 4th-order Butterworth filtfilt over the whole span: hp28 (28 h high-pass),
#     bp9_15 (9-15 h band-pass), hp9 (9 h high-pass). Saved: mean_root/<var>_wave_<band>/, one file per window.
# Always the whole span (the filter needs the whole time series): WMF_WINDOWS does not apply here.
# WMF_BACKGROUND = "interp" (default: linear in time between the window means at the window centres, no steps at window
# edges) or "window" (the plain window mean U / V / rho_window / zeta_window; its steps every 3 days leak into the bands:
# tile test, within 12 h of a window edge the error was ~35% of the hp9 rms, ~21% hp28, ~10% bp9_15)
const BAND_VARS  = haskey(ENV, "WMF_BAND_VARS") ? String.(split(ENV["WMF_BAND_VARS"], ",")) : filter(in(("zeta", "rho", "u", "v")), VARS)
const BACKGROUND = get(ENV, "WMF_BACKGROUND", "interp")
const FILT_ORDER = 4
if 5 in STEPS || 6 in STEPS
    band_files = Dict(v => window_wave_bands(win, data_folder, mean_root; var = v, grid_file, order = FILT_ORDER,
                                             background = BACKGROUND) for v in BAND_VARS)
end

# ============================================================================
# (7) barotropic part (depth mean, mode 0) of each band (barotropic.jl)
# ============================================================================
# u_bt = (1/H) ∫ u_wave dz, v_bt alike, p_bt = ρ0 g zeta_wave + (1/H) ∫ p_wave dz, with Hz, H of each record, per band
# -> mean_root/<field>_bt_<band>/ (2-D per record). Baroclinic part not saved: u_bc = u_wave − u_bt when needed.
# WMF_BT_FIELDS = "u,v,p" (default) or a subset (batch: one field per task)
const BT_FIELDS = String.(split(get(ENV, "WMF_BT_FIELDS", "u,v,p"), ","))
if 7 in STEPS
    bt_files = Dict(f => window_bt(win, data_folder, mean_root; field = f, grid_file) for f in BT_FIELDS)
end

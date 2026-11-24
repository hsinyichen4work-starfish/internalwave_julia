# files_detide.jl
#
# Removes the tides from the ROMS boundary files built from HYCOM
# (roms_bry_2700m_1104_hycom_<cycle>.nc, made by input_make.jl), so that the
# ROMS_HYCOMparent run can take its tides from TPXO instead (tpxo_tide_make_1104.py),
# as the CMEMS-parent run does.
#
# Method (harmonic analysis with Python UTide, called through PyCall):
#   1. The daily bry files (9 records 12Z..12Z, 3-hourly, last record of one file =
#      first record of the next) are joined into one series per boundary point:
#      2022-08-21 12Z .. 2022-12-01 12Z, ~98 days.
#   2. UTide picks the constituents (Rayleigh criterion, constit = "auto") and gives the
#      nodal-corrected basis F*cos(U+V), F*sin(U+V) for each one, via utide.reconstruct.
#      The script drops two groups from that list:
#        - frequencies >= FMAX (S4, M6, M8, ...). They sit at or above the 4 cpd Nyquist
#          frequency of 3-hourly data, so they alias and blow up.
#        - frequencies < FMIN (MM, MSF, ...). These are subtidal; removing them would remove
#          real HYCOM low-frequency variability.
#   3. The fit [mean, trend, tides] is ordinary least squares, the same as
#      utide.solve(method = "ols"). It runs as one matrix solve over every point of a
#      boundary variable (about 600k 3D series in total, too many for one utide.solve call
#      per point). Before running the files, the script checks the result against
#      utide.solve / utide.reconstruct at one point.
#   4. detided = original - tidal part (mean and trend are kept). Each record goes back to
#      its own file, so the shared 12Z record is identical in both files that hold it.
#
# Default variables: zeta, ubar, vbar, u, v on all four boundaries. temp / salt carry
# internal tides as well; set DT_TS=1 to detide them too.
# u is detided level by level and ubar on its own. ubar therefore ends up only nearly
# equal to the depth mean of u (the depth weights move with zeta). ROMS reads the two
# fields separately, so this is fine.
#
# Python: utide is installed in ~/.venvs/utide (venv on /usr/bin/python3, the Python
# PyCall is built with). PYCALL_JL_RUNTIME_PYTHON below points PyCall at that venv.
# To rebuild the venv:
#     python3 -m venv --without-pip ~/.venvs/utide
#     python3 -m pip --python ~/.venvs/utide/bin/python install utide
#
# Run (aurelia):
#     cd ~/Documents/Julia
#     nohup julia --project=. -t 8 input_file_make/files_detide.jl > detide.log 2>&1 &
# Options (environment variables):
#     DT_DRYRUN=1   fit and print diagnostics only; nothing is written
#     DT_TS=1       also detide temp and salt
#     DT_VARS=zeta,ubar,vbar   detide only these variables
#
# Output: OUT_DIR/roms_bry_2700m_1104_hycom_detide_<cycle>.nc, a copy of each input file
# with the detided variables written over it. The dbry_flux files are not touched.

ENV["PYCALL_JL_RUNTIME_PYTHON"] = get(ENV, "PYCALL_JL_RUNTIME_PYTHON", expanduser("~/.venvs/utide/bin/python"))
using PyCall, NCDatasets, LinearAlgebra, Statistics, Printf, Dates

##
const BRY_DIR  = "/home/hsinyi/roms_data/new_design_fix/bry/"
const OUT_DIR  = "/home/hsinyi/roms_data/new_design_fix/bry_detide/"
const GRD_FILE = "/home/hsinyi/roms_data/new_design_fix/roms_grd_2700m_1104.nc"
const RUN_TAG  = "2700m_1104_hycom"
const OUT_TAG  = "2700m_1104_hycom_detide"
const EPOCH    = "1994-01-01"         # bry_time = days since 1994-01-01
const FMIN     = 0.5                  # cpd; drop long-period constituents (MM, MSF, ...)
const FMAX     = 3.95                 # cpd; drop S4 and higher (Nyquist of 3-hourly = 4 cpd)
const DRYRUN   = get(ENV, "DT_DRYRUN", "0") == "1"

const SIDES = ("south", "east", "north", "west")
basevars = get(ENV, "DT_TS", "0") == "1" ? ["zeta", "ubar", "vbar", "u", "v", "temp", "salt"] :
                                           ["zeta", "ubar", "vbar", "u", "v"]
haskey(ENV, "DT_VARS") && (basevars = String.(split(ENV["DT_VARS"], ",")))
const VARS = ["$(v)_$(s)" for v in basevars for s in SIDES]

bry_files = sort(filter(f -> occursin(Regex("^roms_bry_$(RUN_TAG)_\\d{10}\\.nc\$"), f), readdir(BRY_DIR)))
isempty(bry_files) && error("no roms_bry_$(RUN_TAG)_*.nc in $BRY_DIR")
_cycle(f) = match(r"_(\d{10})\.nc$", f)[1]
out_name(f) = "roms_bry_$(OUT_TAG)_$(_cycle(f)).nc"
println("$(length(bry_files)) bry files: $(first(bry_files)) .. $(last(bry_files))")
println("variables: ", join(basevars, ", "), DRYRUN ? "   (DRY RUN, nothing written)" : "")

## ---- global time axis ---------------------------------------------------------------
# hours since EPOCH, as Int, so the overlapping 12Z records match exactly
file_hours = [NCDataset(joinpath(BRY_DIR, f)) do ds
                  round.(Int, ds["bry_time"][:] .* 24)
              end for f in bry_files]
hours = sort(unique(reduce(vcat, file_hours)))
all(diff(hours) .== hours[2] - hours[1]) || error("bry_time is not evenly spaced across files")
t = hours ./ 24                                   # days since EPOCH
file_idx = [searchsortedfirst.(Ref(hours), h) for h in file_hours]
t0 = DateTime(EPOCH)
@printf("time: %s .. %s, %d records, dt = %g h, span %.1f days\n",
        t0 + Hour(hours[1]), t0 + Hour(hours[end]), length(t), hours[2] - hours[1], t[end] - t[1])

lat = NCDataset(GRD_FILE) do ds; mean(ds["lat_rho"][:, :]); end

## ---- constituents and basis from UTide ----------------------------------------------
py"""
import numpy as np, copy, utide

def _solve(t, x, lat, epoch, constit):
    return utide.solve(t, x, lat=lat, epoch=epoch, constit=constit, method='ols',
                       conf_int='none', trend=True, nodal=True, verbose=False)

def utide_basis(t, lat, epoch, fmin, fmax):
    t = np.asarray(t, dtype=float)
    # Constituent choice depends only on the time axis (Rayleigh), not on the data.
    x = np.cos(2*np.pi*1.9322736*t)
    c = _solve(t, x, lat, epoch, 'auto')
    frq = np.asarray(c['aux']['frq']) * 24.0                 # cph -> cpd
    keep = [str(n) for n, f in zip(c['name'], frq) if fmin <= f < fmax]
    drop = [str(n) for n, f in zip(c['name'], frq) if not (fmin <= f < fmax)]
    c = _solve(t, x, lat, epoch, keep)
    names = [str(n) for n in c['name']]
    nc = len(names)
    B = np.zeros((t.size, 2*nc))
    for j in range(nc):
        for k, g in enumerate((0.0, 90.0)):            # g = 0 -> F cos(U+V), g = 90 -> F sin(U+V)
            cc = copy.deepcopy(c)
            cc['A'] = np.zeros(nc); cc['A'][j] = 1.0
            cc['g'] = np.zeros(nc); cc['g'][j] = g
            cc['mean'] = 0.0; cc['slope'] = 0.0
            B[:, 2*j+k] = utide.reconstruct(t, cc, epoch=epoch, verbose=False, min_SNR=0, min_PE=0).h
    return names, np.asarray(c['aux']['frq'])*24.0, drop, B

def utide_tide(t, x, lat, epoch, constit):
    # utide.solve + reconstruct at one point; returns the tidal part (no mean, no trend)
    t = np.asarray(t, dtype=float); x = np.asarray(x, dtype=float)
    c = _solve(t, x, lat, epoch, list(constit))
    h = utide.reconstruct(t, c, epoch=epoch, verbose=False, min_SNR=0, min_PE=0).h
    return h - c['mean'] - c['slope']*(t - t.mean())
"""

names, frq, dropped, B = py"utide_basis"(t, lat, EPOCH, FMIN, FMAX)
println("UTide: $(length(names)) constituents kept (lat = $(round(lat, digits = 2))):")
println("  ", join(["$(n)($(round(f, digits = 3)))" for (n, f) in sort(collect(zip(names, frq)), by = last)], " "))
println("  dropped: ", join(dropped, " "))
const NAMES = names
const iM2 = findfirst(==("M2"), NAMES)

# least squares design: [1, t - mean(t), basis]; trend uses the same centring as utide
const A  = hcat(ones(length(t)), t .- mean(t), B)
const QRA = qr(A)
@printf("design matrix %d x %d, cond = %.1f\n", size(A)..., cond(A))

"Tidal part of the columns of X (nt x npts). Columns with NaN get zero tide."
function tidal_part(X::Matrix{Float64})
    good = vec(all(isfinite, X; dims = 1))
    T = zeros(size(X))
    any(good) || return T, zeros(size(A, 2), size(X, 2))
    C = zeros(size(A, 2), size(X, 2))
    C[:, good] = QRA \ X[:, good]
    T[:, good] = B * C[3:end, good]
    return T, C
end

## ---- check against utide.solve at one point -----------------------------------------
let v = "zeta_west"
    X = zeros(length(t), 0)
    for (k, f) in enumerate(bry_files)
        a = NCDataset(ds -> Float64.(ds[v][:, :]), joinpath(BRY_DIR, f))   # (eta_rho, rec)
        k == 1 && (X = fill(NaN, length(t), size(a, 1)))
        X[file_idx[k], :] = permutedims(a)
    end
    j = argmax(vec(std(X; dims = 1)))
    T, _ = tidal_part(X[:, j:j])
    Tu = py"utide_tide"(t, X[:, j], lat, EPOCH, NAMES)
    @printf("check %s[%d]: max |Julia - utide.solve| = %.2e m (tide std %.3f m)\n",
            v, j, maximum(abs.(T[:, 1] .- Tu)), std(Tu))
    maximum(abs.(T[:, 1] .- Tu)) < 1e-6 * max(std(Tu), 1e-3) + 1e-8 ||
        error("matrix fit does not reproduce utide.solve; stopping")
end

## ---- output files ------------------------------------------------------------------
if !DRYRUN
    mkpath(OUT_DIR)
    for f in bry_files
        cp(joinpath(BRY_DIR, f), joinpath(OUT_DIR, out_name(f)); force = true)
    end
    println("copied $(length(bry_files)) files to $OUT_DIR")
end

## ---- detide, one variable at a time ---------------------------------------------------
# NCDatasets order: 2D (pts, rec), 3D (pts, s_rho, rec)
summary = Tuple{String, Float64, Float64, Float64}[]
for v in VARS
    tv = time()
    a1 = NCDataset(ds -> Array(ds[v]), joinpath(BRY_DIR, bry_files[1]))
    sz = size(a1)[1:end-1]
    npts = prod(sz)
    X = fill(NaN, length(t), npts)
    for (k, f) in enumerate(bry_files)
        a = NCDataset(ds -> Array(ds[v]), joinpath(BRY_DIR, f))
        X[file_idx[k], :] = permutedims(reshape(Float64.(a), npts, :))
    end
    any(isnan, X) && @warn "$v: $(count(isnan, X)) missing/NaN values; those columns are left as is"

    T, C = tidal_part(X)
    tide_rms = sqrt(mean(abs2, T))
    var_red  = 1 - sum(abs2, X .- T .- mean(X; dims = 1)) / max(sum(abs2, X .- mean(X; dims = 1)), eps())
    m2amp = iM2 === nothing ? NaN : maximum(hypot.(C[2 + 2iM2 - 1, :], C[2 + 2iM2, :]))
    push!(summary, (v, tide_rms, var_red, m2amp))
    @printf("%-12s %7d pts  tide rms %.4f  max M2 amp %.4f  variance removed %5.1f%%  (%.1f s)\n",
            v, npts, tide_rms, m2amp, 100var_red, time() - tv)

    if !DRYRUN
        Xd = X .- T
        for (k, f) in enumerate(bry_files)
            out = Float32.(reshape(permutedims(Xd[file_idx[k], :]), sz..., :))
            NCDataset(joinpath(OUT_DIR, out_name(f)), "a") do ds
                ds[v][:] = out
            end
        end
    end
    X = T = C = nothing; GC.gc()
end

if !DRYRUN
    note = "$(Dates.format(now(), "yyyy-mm-dd HH:MM")) files_detide.jl: tides removed from " *
           join(basevars, ",") * " by UTide OLS harmonic fit over $(t0 + Hour(hours[1])) .. " *
           "$(t0 + Hour(hours[end])) ($(length(NAMES)) constituents: $(join(NAMES, " "))); source $(BRY_DIR)"
    for f in bry_files
        NCDataset(joinpath(OUT_DIR, out_name(f)), "a") do ds
            ds.attrib["detide_history"] = note
        end
    end
    println("done: $(length(bry_files)) detided files in $OUT_DIR")
end

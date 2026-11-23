#= mode_table.jl
Text table of the mode eigenspeeds cn and wavelengths of step 4 (modes<NMODE>/), per window and mode:
  c_n          eigenspeed [m s-1] (no rotation), from the modes files
  lambda_eig   eigenvalue of W'' + (N²/c²) W = 0:  λn = 1 / cn²  [s² m-2]
  L_M2         M2 wavelength with rotation  L = 2π / k,  k = sqrt(ω² − f²) / cn  [km],  ω = 2π / 12.4206 h, f of the grid
Statistics over the whole domain and over h > 1000 m (median, 10th and 90th percentile), then the values at the
moorings (nearest rho point, read_moorings). Written next to the modes files: modes<NMODE>/modes<NMODE>_c_lambda.txt

  julia --project=. wave_meanflow/mode_table.jl            # WMF_NMODE=3 for the 3-mode set
=#
using NCDatasets, Dates, Printf, Statistics
include("/home/hchen54/internalwave_julia/function/mooring_transect.jl")   # read_moorings

const BASE  = "/expanse/lustre/projects/uso102/hchen54/"
const RUN   = "amazon_2700m_cmems_0801"
const ROOT  = BASE * RUN * "/meanflowdata/L72h_ov0.00/"
const GRID  = BASE * "input/grid/roms_cmems_grd_2700m.nc"
const MOOR  = BASE * "input/grid/roms_cmems_grd_2700m_mor_edata.nc"
const NMODE = parse(Int, get(ENV, "WMF_NMODE", "10"))
const ω_M2  = 2π / (12.4206 * 3600)

tag = "modes$NMODE"
files = sort(filter(f -> startswith(f, tag * "_win") && endswith(f, ".nc"), readdir(ROOT * tag)))
h, f, lon, lat = NCDataset(g -> (g["h"].var[:, :], g["f"].var[:, :], g["lon_rho"].var[:, :], g["lat_rho"].var[:, :]), GRID)
lon = ifelse.(lon .> 180, lon .- 360, lon)
deep = h .> 1000
kfac = sqrt.(ω_M2^2 .- f .^ 2)                                   # k = kfac / c
mnames, mlon, mlat = read_moorings(MOOR)
mij = [Tuple(argmin((lon .- mlon[m]) .^ 2 .+ (lat .- mlat[m]) .^ 2)) for m in eachindex(mnames)]

out = ROOT * tag * "/" * tag * "_c_lambda.txt"
open(out, "w") do io
    println(io, "# Mode eigenspeeds and wavelengths, $RUN, window means L72h_ov0.00, N² TEOS-10 (getting_modes.jl), $(now())")
    println(io, "# c_n [m/s] eigenspeed (no rotation); lambda_eig = 1/c_n^2 [s^2/m^2] eigenvalue of W'' + (N^2/c^2) W = 0;")
    println(io, "# L_M2 [km] = 2 pi / k, k = sqrt(w_M2^2 - f^2) / c_n, M2 period 12.4206 h, f of the grid")
    println(io, "# domain: all $(length(h)) rho points; deep: h > 1000 m ($(count(deep)) points). p10 / p90 = 10th / 90th percentile")
    @printf(io, "\n# %-6s %-13s %4s | %-29s | %-29s | %-29s | %-22s\n", "window", "start", "mode",
            "c_n domain: median p10 p90", "c_n deep: median p10 p90", "L_M2 deep [km]: med p10 p90", "lambda_eig deep: median")
    tab = Dict{Tuple{Int, Int}, Vector{Float64}}()                # (window, mode) -> c at the moorings
    for (w, fn) in enumerate(files)
        c, t0 = NCDataset(ROOT * tag * "/" * fn) do ds
            ds["c"].var[:, :, :], ds["t_start"][1]
        end
        for n in 1:NMODE
            cn = Float64.(c[:, :, n]); cd = cn[deep]; L = (2π .* cn ./ kfac)[deep] ./ 1e3
            q(a) = (median(vec(a)), quantile(vec(a), 0.1), quantile(vec(a), 0.9))
            @printf(io, "  %6d %-13s %4d | %8.3f %8.3f %8.3f    | %8.3f %8.3f %8.3f    | %8.1f %8.1f %8.1f    | %12.4e\n",
                    w, Dates.format(t0, "yyyy-mm-ddTHH"), n, q(cn)..., q(cd)..., q(L)..., median(1 ./ cd .^ 2))
            tab[(w, n)] = [cn[i, j] for (i, j) in mij]
        end
    end
    println(io, "\n# At the moorings (nearest rho point): c_n [m/s] / L_M2 [km], per window and mode")
    for (m, nm) in enumerate(mnames)
        i, j = mij[m]
        @printf(io, "\n# %-8s lon %8.3f lat %6.3f -> xi %3d eta %3d, h %7.1f m, f %.2e s-1\n", nm, mlon[m], mlat[m], i, j, h[i, j], f[i, j])
        @printf(io, "# %-6s %s\n", "window", join([@sprintf("%-15s", "mode $n c / L") for n in 1:NMODE], " "))
        for w in eachindex(files)
            @printf(io, "  %6d %s\n", w, join([@sprintf("%6.3f / %6.1f ", tab[(w, n)][m], 2π * tab[(w, n)][m] / kfac[i, j] / 1e3) for n in 1:NMODE], " "))
        end
    end
end
println("wrote ", out)

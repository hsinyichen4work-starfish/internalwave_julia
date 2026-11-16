#= DFF_energy_on_bcef.jl
Hsin-Yi Chen, USM DMS
Modal KE and APE of each direction from the time-series DFF: the companion of DFF_on_bcef.jl (which keeps only the
flux). The same DFF (dff_spectrum / dff_sector, half-plane sectors north 0-180° / south 180-360° in grid coordinates,
bands M2 9-15 h, hp24, hp9) of pn, un, vn (bcef_modes_puv.jl); from each direction's fields:
    KE_n  = ½ ρ0 H < un² + vn² >                 [J m-2]
    APE_n = H < pn² > / (2 ρ0 c_n²)              [J m-2]   (c_n: eigenspeed at each point, structure file)
    cross = all - north - south  (the N-S interference terms: 2 ρ0 H <u_N u_S + v_N v_S> / 2, ...)
The same as the harmonic-fit KE / APE of mode_dff_binplot.jl (¼ ρ0 H A², H A_p² / (4 ρ0 c²)) for one sinusoid.
Outputs (the flux files of DFF_on_bcef.jl are not touched):
  <bcef_dir>/bcef_dff_energy_modes<N>.nc                       <KE|APE>_<band>_<sector>_<mode> (xi, eta, period) time means
  <bcef_dir>/bcef_dff_energy_modes<N>_strip_xi<a>-<b>.nc       the same, mean across the ±STRIP_KM strip, every hour (eta, time)
Switches: ENERGY_MODES (default all, e.g. 1), ENERGY_BANDS (default M2:hp24:hp9). With a subset, the outputs are
named ..._test.nc and modesum is left out. Memory: one mode at a time, 3 spectra + 3 fields (2700 m ~14 GB,
900 m ~40 GB): batch job (sbatch_files/julia_dff_energy.sh), not the login node.
Usage:
  sbatch sbatch_files/julia_dff_energy.sh                                                  # 2700 m, everything
  sbatch --export=ALL,ENERGY_MODES=1,ENERGY_BANDS=M2 sbatch_files/julia_dff_energy.sh      # test: mode 1, M2
=#
using NCDatasets, Dates, Statistics, FFTW, Printf
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
haskey(ENV, "DFF_RUN") || (ENV["DFF_RUN"] = "amazon_2700m_cmems_0801")
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")   # dff_run, run_dir, run_grid_fname
include("/home/hchen54/internalwave_julia/DFF/dff_fun.jl")   # waxis, wdir, dff_sector, dff_spectrum
FFTW.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")))

# ============================================================================
# (1) settings, input, grid
# ============================================================================
NMODE = parse(Int, get(ENV, "NMODE", "3"))
default_bcef = Dict("amazon_2700m_cmems_0801" => "ddf_bcef_0824")
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, get(default_bcef, dff_run, "ddf_bcef")))
puv_fname = joinpath(bcef_dir, "bcef_modes$(NMODE)_puv.nc")
struct_fname = joinpath(bcef_dir, "bcef_modes$(NMODE)_structure.nc")
modes_env, bands_env = get(ENV, "ENERGY_MODES", "all"), get(ENV, "ENERGY_BANDS", "all")
modes = modes_env == "all" ? collect(1:NMODE) : parse.(Int, split(modes_env, r"[,:; ]+"; keepempty = false))
band_sel = bands_env == "all" ? ["M2", "hp24", "hp9"] : String.(split(bands_env, r"[,:; ]+"; keepempty = false))
is_test = modes != collect(1:NMODE) || Set(band_sel) != Set(["M2", "hp24", "hp9"])
tag = is_test ? "_test" : ""
out_fname = joinpath(bcef_dir, "bcef_dff_energy_modes$(NMODE)$(tag).nc")

ds_puv = NCDataset(puv_fname)
nx, ny, nt = ds_puv.dim["xi_rho"], ds_puv.dim["eta_rho"], ds_puv.dim["time"]
nx_f, ny_f = isodd(nx) ? nx : nx - 1, isodd(ny) ? ny : ny - 1   # odd sizes, as DFF_on_bcef.jl
date = ds_puv["ocean_time"][:]
H, cn = NCDataset(struct_fname) do ds
    Float32.(ds["h"][1:nx_f, 1:ny_f] .+ ds["zeta_mean"][1:nx_f, 1:ny_f]), Float32.(ds["c"][1:nx_f, 1:ny_f, :])
end
land_mask, pm = NCDataset(run_grid_fname) do g
    g["mask_rho"][1:nx_f, 1:ny_f], Float64.(g["pm"][1:nx_f, 1:ny_f])
end
his0 = first(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(joinpath(run_dir, "after_joint"))))
rho0 = NCDataset(d -> Float32(d.attrib["rho0"]), joinpath(run_dir, "after_joint", his0))
println("$dff_run: modes $(join(modes, ",")), bands $(join(band_sel, ",")); pn, un, vn $nx x $ny x $nt -> $nx_f x $ny_f; rho0 $rho0")
println("  $(date[1]) -> $(date[end]); output $out_fname")

# directions, bands, periods: as DFF_on_bcef.jl
θ = wdir(waxis(nx_f), waxis(ny_f))
sectors = [("all", 0, 360), ("north", 0, 180), ("south", 180, 360)]
nf = cld(nt, 2)
freq_cph = (0:nf-1) ./ (nt * Dates.value(date[2] - date[1]) / 3.6e6)
bands_all = [("M2", (1 / 15 .<= freq_cph .<= 1 / 9), "M2 band, periods 9-15 h"),
             ("hp24", (freq_cph .> 1 / 24), "periods shorter than 24 h"),
             ("hp9", (freq_cph .> 1 / 9), "periods shorter than 9 h")]
bands = filter(b -> b[1] in band_sel, bands_all)
chunk(d1, d2) = (a = findfirst(>=(DateTime(d1)), date); b = findlast(<(DateTime(d2)), date);
                 a === nothing || b === nothing || b <= a ? nothing : a:b)
periods = filter(x -> x[2] !== nothing,
                 [("full", 1:nt), ("augsep", chunk(Date(2022, 8, 1), Date(2022, 10, 1))),
                  ("oct", chunk(Date(2022, 10, 1), Date(2022, 11, 1))), ("nov", chunk(Date(2022, 11, 1), Date(2022, 12, 1)))])

# the strip (as DFF_on_bcef.jl / mode_dff_binplot.jl)
strip_km = parse(Float64, get(ENV, "STRIP_KM", "200"))
strip_hmin = 200.0f0
ix_strip_c = round(Int, 0.5 + (nx_f + 1) / 2)
nside = round(Int, strip_km / (mean(1 ./ pm) / 1e3))
strip_cols = max(1, ix_strip_c - nside):min(nx_f, ix_strip_c + nside)
strip_ok = (land_mask[strip_cols, :] .> 0) .& (H[strip_cols, :] .>= strip_hmin)
strip_n = vec(sum(strip_ok; dims = 1))
strip_fname = joinpath(bcef_dir, "bcef_dff_energy_modes$(NMODE)_strip_xi$(first(strip_cols))-$(last(strip_cols))$(tag).nc")
println("  strip xi = $(first(strip_cols))-$(last(strip_cols)); periods: ", join(["$p $(length(t)) h" for (p, t) in periods], ", "))

# strip mean of w a b per row and hour (w: weight per point, (x, y)); typed loop, contiguous along xi
function strip_mean_k(a, b, w, cols, ok, n)
    ny_, nt_ = size(a, 2), size(a, 3)
    acc = zeros(Float32, ny_, nt_)
    @inbounds for t in 1:nt_, j in 1:ny_
        s = 0f0
        for (k, i) in enumerate(cols)
            ok[k, j] && (s += w[i, j] * a[i, j, t] * b[i, j, t])
        end
        acc[j, t] = s / max(n[j], 1)
    end
    acc
end
# time means of w a b per period, (x, y, period)
function wmean_periods(a, b, w)
    out = zeros(Float32, nx_f, ny_f, length(periods))
    for (ip, (_, tidx)) in enumerate(periods)
        for t in tidx
            @views out[:, :, ip] .+= w .* a[:, :, t] .* b[:, :, t]
        end
        out[:, :, ip] ./= length(tidx)
    end
    out
end

# ============================================================================
# (2) per mode, band, sector: KE and APE
# ============================================================================
Emap = Dict{Tuple{String, String, String, String}, Array{Float32, 3}}()   # (mode, band, sector, KE/APE) => (x, y, period)
Estrip = Dict{Tuple{String, String, String, String}, Matrix{Float32}}()   # => (eta, time)
for n in modes
    m = "mode$n"
    println("\n==== mode $n ====")
    wKE = 0.5f0 * rho0 .* H                                   # ½ ρ0 H
    wAPE = H ./ (2f0 * rho0 .* cn[:, :, n] .^ 2)               # H / (2 ρ0 c_n²)
    spec = Dict{String, Array{ComplexF32, 3}}()
    for name in ("p$n", "u$n", "v$n")
        a = nomissing(ds_puv[name][1:nx_f, 1:ny_f, :], NaN32)
        any(!isfinite, a) && error("$name has NaN/Inf")
        spec[name] = dff_spectrum(a); a = nothing; GC.gc()
    end
    for (b, fkeep, _) in bands, (sec, a1, a2) in sectors
        mask = a1 .<= θ .< a2
        t0 = time()
        p, u, v = (dff_sector(spec[f], mask, nt, land_mask; fkeep = fkeep) for f in ("p$n", "u$n", "v$n"))
        for a in (p, u, v)   # land NaN -> 0 (outside the means anyway; keeps sums finite)
            a[.!isfinite.(a)] .= 0f0
        end
        Emap[(m, b, sec, "KE")] = wmean_periods(u, u, wKE) .+ wmean_periods(v, v, wKE)
        Emap[(m, b, sec, "APE")] = wmean_periods(p, p, wAPE)
        Estrip[(m, b, sec, "KE")] = strip_mean_k(u, u, wKE, strip_cols, strip_ok, strip_n) .+ strip_mean_k(v, v, wKE, strip_cols, strip_ok, strip_n)
        Estrip[(m, b, sec, "APE")] = strip_mean_k(p, p, wAPE, strip_cols, strip_ok, strip_n)
        deep = H .> 1000
        @printf("mode %d %-4s %-5s: deep-mean KE %.3f, APE %.3f kJ/m² (%.0f s)\n", n, b, sec,
                mean(Emap[(m, b, sec, "KE")][:, :, 1][deep]) / 1e3, mean(Emap[(m, b, sec, "APE")][:, :, 1][deep]) / 1e3, time() - t0)
        p = u = v = nothing; GC.gc()
    end
    for (b, _, _) in bands, e in ("KE", "APE")
        Emap[(m, b, "cross", e)] = Emap[(m, b, "all", e)] .- Emap[(m, b, "north", e)] .- Emap[(m, b, "south", e)]
        Estrip[(m, b, "cross", e)] = Estrip[(m, b, "all", e)] .- Estrip[(m, b, "north", e)] .- Estrip[(m, b, "south", e)]
    end
    spec = nothing; GC.gc()
end
close(ds_puv)
mode_tags = ["mode$n" for n in modes]
if !is_test   # modes together: no cross-mode terms ((1/H) ∫ φm φn dz = δmn)
    for (b, _, _) in bands, sec in ("all", "north", "south", "cross"), e in ("KE", "APE")
        Emap[("modesum", b, sec, e)] = sum(Emap[("mode$n", b, sec, e)] for n in modes)
        Estrip[("modesum", b, sec, e)] = sum(Estrip[("mode$n", b, sec, e)] for n in modes)
    end
    push!(mode_tags, "modesum")
end

# ============================================================================
# (3) save
# ============================================================================
edesc = Dict("KE" => "kinetic energy ½ ρ0 H <un² + vn²>", "APE" => "available potential energy H <pn²> / (2 ρ0 c_n²)")
for (fname, D, dims) in ((out_fname, Emap, ("xi_rho", "eta_rho", "period")), (strip_fname, Estrip, ("eta_rho", "time")))
    tmp = fname * ".tmp"; isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do ds
        defDim(ds, "xi_rho", nx_f); defDim(ds, "eta_rho", ny_f); defDim(ds, "period", length(periods)); defDim(ds, "time", nt)
        defVar(ds, "ocean_time", date, ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
        defVar(ds, "strip_n", Int32.(strip_n), ("eta_rho",); attrib = ["long_name" => "deep (H >= $(Int(strip_hmin)) m) points per row in the strip mean"])
        ds.attrib["periods"] = join(["$p = $(date[t[1]]) - $(date[t[end]])" for (p, t) in periods], "; ")
        ds.attrib["rho0"] = rho0
        ds.attrib["strip"] = "xi = $(first(strip_cols))-$(last(strip_cols)) (±$(strip_km) km around xi = $ix_strip_c), points with H >= $(Int(strip_hmin)) m"
        ds.attrib["note"] = "DFF of pn, un, vn (bcef_modes_puv.jl) as DFF_on_bcef.jl; sectors all / north 0-180 / south 180-360 (grid); " *
                            "cross = all - north - south; <KE|APE>_<band>_<sector>_<mode>"
        ds.attrib["script"] = "internalwave_julia/DFF/DFF_energy_on_bcef.jl"
        for (k, a) in D
            m, b, sec, e = k
            defVar(ds, "$(e)_$(b)_$(sec)_$(m)", a, dims; attrib = ["units" => "J m-2", "long_name" => "$(edesc[e]), $m, $b, $sec" *
                   (dims[2] == "time" ? ", mean across the strip" : ", time mean")])
        end
    end
    mv(tmp, fname; force = true)
    println("wrote $fname")
end

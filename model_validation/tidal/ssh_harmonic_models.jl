#= ssh_harmonic_models.jl
Hsin-Yi Chen, USM DMS
Plain sinusoid harmonic fit of sea-surface height for several models (same method as ssh_harmonic.jl,
function/harm_fit.jl), on each model's native grid, over one common time window.
Settings, machine and sources: tidal_models_setup.jl (TC_HOST, TC_SRC, TC_T0, TC_T1, TC_STRIDE).

Reading:
  ROMS     roms_his.*.nc zeta (hourly snapshots), whole grid, land -> NaN from mask_rho
  HYCOM    hycom_<tag>_ssh.nc (3-hourly), Longitude / Latitude from hycom_<tag>_grd.nc
  NCOM     <tag>_ssh.nc (hourly), Longitude / Latitude from <tag>_lthick.nc (curvilinear)
  HYCOM / NCOM: daily files overlap by one record, the newer file's record is kept; only the index box
  around the 2700 m ROMS grid (+ MARGIN deg) is read.
Fit:  eta = mean + trend + sum_k A'_k cos(w_k (t - tc) - g'_k), CONS = M2 S2 N2 K1 O1, t in seconds since 1994-01-01
  for every model, so the comparison converts all of them with the same V, u, f at tc.
Output: P.out/<SRC>_sshfit[_s<STRIDE>].nc  (lon, lat 2-D native grid, -180..180; amp, pha, errors, mean, ...)

Usage:
  expanse:  sbatch sbatch_files/julia_tidal_models.sh                       # both ROMS runs, fit + FES comparison
            TC_SRC=ROMS_HYCOMparent TC_STRIDE=10 julia --project=. model_validation/tidal/ssh_harmonic_models.jl   # quick test
  aurelia:  cd ~/Documents/Julia
            TC_SRC=HYCOM,NCOM2700 nohup julia --project=. -t 8 model_validation/tidal/ssh_harmonic_models.jl > tidefit.log 2>&1 &
=#

using NCDatasets, Printf, Statistics, Dates, LinearAlgebra
include(joinpath(@__DIR__, "tidal_models_setup.jl"))
for f in ("time_dir.jl", "data_load.jl", "harm_fit.jl"); include(P.func * f); end   # time_dir, data_load, harm_fit

const MARGIN = 0.3                                       # deg around the target grid read from HYCOM / NCOM
println("host $HOST, sources $(join(SRC_RUN, ", ")), window $T0 .. $T1, stride $STRIDE")

## target box: lon (-180..180) / lat extent of the 2700 m ROMS grid
const BOX = NCDataset(P.grid) do g
    lo = lon180.(g["lon_rho"][:, :]); la = g["lat_rho"][:, :]
    (lon = (minimum(lo) - MARGIN, maximum(lo) + MARGIN), lat = (minimum(la) - MARGIN, maximum(la) + MARGIN))
end
nanbig(x) = (isfinite(x) && abs(x) < 1e10) ? Float32(x) : NaN32

## --- readers: (; lon, lat, h, ssh [m] (x, y, t), tsec [s since T_REF], note)
function read_roms(src)
    lon, lat, h = NCDataset(src.grid) do g
        nx, ny = size(g["lon_rho"])
        ix = 1:STRIDE:nx; jy = 1:STRIDE:ny
        lon180.(g["lon_rho"][ix, jy]), g["lat_rho"][ix, jy], g["h"][ix, jy]
    end
    nx, ny = NCDataset(g -> size(g["lon_rho"]), src.grid)
    td = time_dir(src.data; t_ref = T_REF, tstart = T0, tend = T1)
    ssh = data_load(src.data, "zeta", td; ix = 1:STRIDE:nx, jy = 1:STRIDE:ny, grid_file = src.grid)
    return (; lon, lat, h, ssh, tsec = td.tsec, note = "roms_his zeta, $(src.data)")
end

function read_hn(src)
    re = Regex("^" * src.prefix * raw"(\d{10})_ssh\.nc$")
    tags = sort([m[1] for m in (match(re, n) for n in readdir(src.data)) if m !== nothing])
    isempty(tags) && error("no $(src.prefix)<tag>_ssh.nc in $(src.data)")
    fname(tag, kind) = joinpath(src.data, "$(src.prefix)$(tag)_$kind.nc")
    lon, lat = NCDataset(fname(tags[1], src.gkind)) do ds
        lon180.(Float64.(ds["Longitude"].var[:, :])), Float64.(ds["Latitude"].var[:, :])
    end
    inb = (BOX.lon[1] .<= lon .<= BOX.lon[2]) .& (BOX.lat[1] .<= lat .<= BOX.lat[2])
    p = findall(inb); I = minimum(first.(Tuple.(p))):maximum(first.(Tuple.(p))); J = minimum(last.(Tuple.(p))):maximum(last.(Tuple.(p)))
    Is = first(I):STRIDE:last(I); Js = first(J):STRIDE:last(J)
    # times: MT = days since 1900-12-31; a later tag overwrites a shared time
    mt2dt(mt) = round(DateTime(1900, 12, 31) + Millisecond(round(Int, mt * 86_400_000)), Minute(1))
    rec = Dict{DateTime, Tuple{String, Int}}()
    for tag in tags, (k, t) in enumerate(NCDataset(ds -> mt2dt.(ds["MT"].var[:]), fname(tag, "ssh")))
        T0 <= t <= T1 && (rec[t] = (tag, k))
    end
    times = sort(collect(keys(rec))); nt = length(times)
    nt == 0 && error("no records of $(src.data) in $T0 .. $T1")
    tidx = Dict(t => n for (n, t) in enumerate(times))
    @printf("%d tags %s .. %s, %d records in the window, box %s x %s of %s\n", length(tags), tags[1], tags[end], nt, I, J, size(lon))
    ssh = Array{Float32}(undef, length(Is), length(Js), nt)
    bytag = Dict{String, Vector{Tuple{Int, Int}}}()       # tag -> [(record in file, time index)]
    for t in times; (tag, k) = rec[t]; push!(get!(bytag, tag, Tuple{Int, Int}[]), (k, tidx[t])); end
    t0 = time()
    for (q, tag) in enumerate(sort(collect(keys(bytag))))
        NCDataset(fname(tag, "ssh")) do ds
            ks = first.(bytag[tag])
            blk = ds["ssh"].var[I, J, minimum(ks):maximum(ks)]
            for (k, n) in bytag[tag]
                ssh[:, :, n] .= nanbig.(view(blk, 1:STRIDE:length(I), 1:STRIDE:length(J), k - minimum(ks) + 1))
            end
        end
        q % 20 == 0 && @printf("  %d / %d files, %.1f min\n", q, length(bytag), (time() - t0) / 60)
    end
    tsec = [Dates.value(Millisecond(t - T_REF)) / 1000 for t in times]
    return (; lon = lon[Is, Js], lat = lat[Is, Js], h = fill(NaN, length(Is), length(Js)), ssh, tsec,
              note = "$(src.prefix)<tag>_ssh.nc, $(src.data), index box $I x $J")
end

## --- one source
function run_fit(sname)
    src = SOURCES[sname]; t_start = time()
    println("===== $sname: $(src.label)")
    d = src.kind == :roms ? read_roms(src) : read_hn(src)
    nt = length(d.tsec); dt = diff(d.tsec) ./ 3600
    tdate = T_REF .+ Millisecond.(round.(Int, 1000 .* d.tsec))
    @printf("%d records %s .. %s, dt %.2f..%.2f h, %d points (%d wet)\n", nt, tdate[1], tdate[end], extrema(dt)...,
            prod(size(d.lon)), count(isfinite, d.ssh[:, :, 1]))
    # a point is used only when every record is finite (ROMS land is NaN, HYCOM / NCOM land is NaN or fill)
    bad = vec(any(!isfinite, reshape(d.ssh, :, nt); dims = 2))
    ssh = reshape(d.ssh, :, nt); ssh[bad, :] .= 0f0      # fitted, then set to NaN below

    hf = harm_fit(reshape(ssh, size(d.ssh)), d.tsec, CONS; scale = 100)     # m -> cm
    for F in (hf.amp, hf.pha, hf.aerr, hf.perr); for k in axes(F, 3); view(F, :, :, k)[reshape(bad, size(d.lon))] .= NaN; end; end
    for F in (hf.mean, hf.trend, hf.resid, hf.std); F[reshape(bad, size(d.lon))] .= NaN; end
    tcdate = T_REF + Millisecond(round(Int, 1000hf.tc))
    println("record centre tc = $tcdate")
    for (k, c) in enumerate(CONS)
        @printf("  %-3s median A' %6.2f cm, median err %.3f cm\n", c, median(filter(!isnan, hf.amp[:, :, k])), median(filter(!isnan, hf.aerr[:, :, k])))
    end

    mkpath(P.out); fout = P.out * fitfile(sname); isfile(fout) && rm(fout)
    NX, NY = size(d.lon)
    NCDataset(fout, "c") do ds
        defDim(ds, "xi", NX); defDim(ds, "eta", NY); defDim(ds, "con", length(CONS))
        ds.attrib["title"] = "Plain sinusoid harmonic fit of SSH, $(src.label), $(Date(tdate[1])) to $(Date(tdate[end]))"
        ds.attrib["source"] = sname; ds.attrib["label"] = src.label; ds.attrib["data"] = d.note
        ds.attrib["model"] = "eta = mean + trend (t - tc)/span + sum A' cos(w (t - tc) - g'); no nodal corrections, no inference"
        ds.attrib["to_greenwich"] = "A = A' / f(tc), G = g' + V(tc) + u(tc)"
        ds.attrib["tc"] = string(tcdate); ds.attrib["tc_seconds_since_1994"] = hf.tc
        ds.attrib["records"] = nt; ds.attrib["span_hours"] = (d.tsec[end] - d.tsec[1]) / 3600
        ds.attrib["dt_hours"] = median(dt); ds.attrib["stride"] = STRIDE
        ds.attrib["script"] = "internalwave_julia/model_validation/tidal/ssh_harmonic_models.jl"
        defVar(ds, "con", CONS, ("con",))
        defVar(ds, "freq", [TIDE_FREQ[c] for c in CONS], ("con",); attrib = ["units" => "deg/hour"])
        defVar(ds, "lon", d.lon, ("xi", "eta"); attrib = ["units" => "degrees_east"])
        defVar(ds, "lat", d.lat, ("xi", "eta"); attrib = ["units" => "degrees_north"])
        defVar(ds, "h", d.h, ("xi", "eta"); attrib = ["units" => "m", "long_name" => "model depth (NaN when not read)"])
        for (nm, v, u, ln) in (("amp", hf.amp, "cm", "amplitude A' (no nodal correction)"), ("pha", hf.pha, "deg", "phase g' relative to tc"),
                               ("amp_err", hf.aerr, "cm", "white-noise std error of A'"), ("pha_err", hf.perr, "deg", "white-noise std error of g'"))
            defVar(ds, nm, Float32.(v), ("xi", "eta", "con"); attrib = ["units" => u, "long_name" => ln], deflatelevel = 3)
        end
        for (nm, v, u, ln) in (("mean", hf.mean, "cm", "mean SSH"), ("trend", hf.trend, "cm", "linear SSH change over the record"),
                               ("resid_std", hf.resid, "cm", "residual std"), ("ssh_std", hf.std, "cm", "SSH std"))
            defVar(ds, nm, Float32.(v), ("xi", "eta"); attrib = ["units" => u, "long_name" => ln], deflatelevel = 3)
        end
    end
    @printf("wrote %s (%.1f min)\n", fout, (time() - t_start) / 60)
    d = nothing; GC.gc()
end

for s in SRC_RUN; run_fit(s); end

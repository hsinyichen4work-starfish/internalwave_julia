#= mooring_halfmonth_avg.jl
Hsin-Yi Chen, USM DMS
Half-monthly (days 1-15, 16-end), monthly and total means of the extract_mooring_all.jl output,
the same windows as meanflow/mean_flow_avg_loop.jl.

Input : <MX_OUTDIR>/<SRC>/<SRC>_<name>.nc, <SRC>_M1_CPIES7.nc, <SRC>_M1_CPIES7_ssh.nc   (time series)
Output: <MX_OUTDIR>/<SRC>/halfmonth_avg/<same name>_avg.nc
        every variable with a time axis (time or time_ssh) -> the same variable with a "win" axis instead,
        mean of the finite records; n_<axis>(win) = records in the window; win_label, win_t0, win_t1, win_kind
        (half / month / total). Variables without a time axis are copied.
        Windows with no record are left out.

Settings (environment): MX_SRC (comma separated, default: every folder in MX_OUTDIR), MX_OUTDIR,
                        MX_T0, MX_T1 (default 2022-08-01, 2022-12-01, [T0, T1))
Usage: MX_SRC=CMEMS julia --project=. model_validation/mooring/mooring_halfmonth_avg.jl
=#

using NCDatasets, Dates, Printf

const HOST = get(ENV, "MX_HOST", isdir("/expanse/lustre") ? "expanse" : "aurelia")
const OUT_DIR = get(ENV, "MX_OUTDIR", HOST == "expanse" ? "/expanse/lustre/projects/uso102/hchen54/mooring_extract" :
                                                         "/home/hsinyi/roms_data/mooring_extract")
const T0 = DateTime(get(ENV, "MX_T0", "2022-08-01"))
const T1 = DateTime(get(ENV, "MX_T1", "2022-12-01"))
const SRCS = haskey(ENV, "MX_SRC") ? split(ENV["MX_SRC"], ",") : filter(d -> isdir(joinpath(OUT_DIR, d)), readdir(OUT_DIR))

## windows: (label, kind, t0, t1)
function windows()
    W = Tuple{String, String, DateTime, DateTime}[]
    m = DateTime(year(T0), month(T0))
    while m < T1
        mid, nxt = m + Day(15), m + Month(1)
        push!(W, (Dates.format(m, "yyyy-mm") * "-01_15", "half", m, mid))
        push!(W, (Dates.format(m, "yyyy-mm") * "-16_" * Dates.format(nxt - Day(1), "dd"), "half", mid, nxt))
        push!(W, (Dates.format(m, "yyyy-mm"), "month", m, nxt))
        m = nxt
    end
    push!(W, ("total", "total", T0, T1))
    [(l, k, max(a, T0), min(b, T1)) for (l, k, a, b) in W if min(b, T1) > max(a, T0)]
end
const WIN = windows()

readall(v) = ndims(v) == 0 ? v[] : v[ntuple(_ -> Colon(), ndims(v))...]   # v[:] would flatten

function avg_file(fin, fout)
    NCDataset(fin) do ds
        tax = filter(d -> d in ("time", "time_ssh"), collect(keys(ds.dim)))
        T = Dict(d => ds[d][:] for d in tax)
        idx = Dict(d => [findall(t -> a <= t < b, T[d]) for (_, _, a, b) in WIN] for d in tax)
        # windows with records on the main axis (time; time_ssh when it is the only one)
        main = "time" in tax ? "time" : first(tax)
        keep = findall(!isempty, idx[main])
        isempty(keep) && (println("  skip $(basename(fin)): no record in [$T0, $T1)"); return)
        mkpath(dirname(fout))
        NCDataset(fout, "c") do o
            for (k, v) in ds.attrib; o.attrib[k] = v; end
            o.attrib["averaging"] = "mooring_halfmonth_avg.jl: mean of finite records per window [t0, t1); from $(basename(fin))"
            for (d, n) in ds.dim; d in tax || defDim(o, d, n); end
            defDim(o, "win", length(keep))
            defVar(o, "win_label", [WIN[i][1] for i in keep], ("win",))
            defVar(o, "win_kind", [WIN[i][2] for i in keep], ("win",), attrib = ["long_name" => "half / month / total"])
            defVar(o, "win_t0", [WIN[i][3] for i in keep], ("win",), attrib = ["units" => "hours since 2022-01-01 00:00:00"])
            defVar(o, "win_t1", [WIN[i][4] for i in keep], ("win",), attrib = ["units" => "hours since 2022-01-01 00:00:00",
                                                                                 "long_name" => "window end (exclusive)"])
            for d in tax
                defVar(o, "n_$d", Int32[length(idx[d][i]) for i in keep], ("win",),
                       attrib = ["long_name" => "records of $d in the window"])
            end
            for (name, v) in ds
                name in tax && continue
                dims = dimnames(v)
                at = [k => val for (k, val) in v.attrib if !(k in ("_FillValue", "missing_value"))]
                td = findfirst(in(tax), dims)
                if td === nothing
                    defVar(o, name, readall(v), dims; attrib = at)
                    continue
                end
                @assert td == length(dims) "$name: time axis not last"
                A = readall(v); A = Float32.(coalesce.(A, NaN32))
                d = dims[td]; sz = size(A)[1:end-1]
                M = fill(NaN32, sz..., length(keep))
                for (j, i) in enumerate(keep)
                    r = idx[d][i]; isempty(r) && continue
                    S = zeros(Float64, sz); N = zeros(Int, sz)
                    for t in r
                        a = selectdim(A, ndims(A), t)
                        S .+= ifelse.(isfinite.(a), a, 0); N .+= isfinite.(a)
                    end
                    selectdim(M, ndims(M), j) .= ifelse.(N .> 0, Float32.(S ./ max.(N, 1)), NaN32)
                end
                defVar(o, name, M, (dims[1:end-1]..., "win"); attrib = at, fillvalue = NaN32)
            end
        end
        @printf("  %-40s -> %s  (%d windows)\n", basename(fin), basename(fout), length(keep))
    end
end

for s in SRCS
    dir = joinpath(OUT_DIR, s)
    isdir(dir) || (println("skip $s: no $dir"); continue)
    println("===== $s")
    for f in sort(filter(f -> endswith(f, ".nc"), readdir(dir)))
        avg_file(joinpath(dir, f), joinpath(dir, "halfmonth_avg", replace(f, ".nc" => "_avg.nc")))
    end
end

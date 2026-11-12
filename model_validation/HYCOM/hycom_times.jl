# hycom_times.jl
#
# HYCOM snapshot times from the files' metadata, not their names.
# A GOFS file name gives the analysis (cycle) time and the forecast hour,
# e.g. gofs35_hycom_glbz04_2022091412_t0024_uv3z.nc; the time the data is
# valid for is its `time` variable (hours since 2000-01-01): 2022-09-15
# 12:00 for that file. Used by HYCOM_data.jl, time_capture_NCOM.jl and
# time_capture_CMEMS.jl.

using NCDatasets, Dates

"""
    hycom_file_time(file) -> (valid, tau, analysis)

`valid` = the `time` variable (hours since 2000-01-01), `tau` = forecast
hour, `analysis` = tau's time_origin (the cycle time).
"""
function hycom_file_time(file::AbstractString)
    NCDataset(file, "r") do ds
        # raw variables: indexing ds["tau"] makes NCDatasets try (and fail) to
        # parse its units "hours since analysis" as a CF time
        vtime, vtau = variable(ds, "time"), variable(ds, "tau")
        valid = DateTime(2000, 1, 1) + Millisecond(round(Int, vtime[1] * 3_600_000))
        tau = Float64(vtau[1])
        analysis = DateTime(vtau.attrib["time_origin"], dateformat"yyyy-mm-dd HH:MM:SS")
        (valid = valid, tau = tau, analysis = analysis)
    end
end

"""
    hycom_snapshots(dir) -> Vector of (valid, tau, analysis, files)

One entry per valid time found in `dir`, sorted by time. `files` maps
"ssh" / "ts3z" / "uv3z" to the file paths of that snapshot; the three are
checked to have the same valid time. When several cycles hold the same
valid time (t0024 of one cycle and t0000 of the next), the one with the
smallest tau — the most recent analysis — is kept.
"""
function hycom_snapshots(dir::AbstractString)
    groups = Dict{String,Dict{String,String}}()   # "<cycle>_t####" => kind => path
    for f in readdir(dir)
        m = match(r"^(gofs35_hycom_glbz04_\d{10}_t\d{4})_(ssh|ts3z|uv3z)\.nc$", f)
        m === nothing && continue
        get!(groups, m[1], Dict{String,String}())[m[2]] = joinpath(dir, f)
    end
    best = Dict{DateTime,Any}()
    for (key, files) in groups
        length(files) == 3 || (println("incomplete HYCOM snapshot $key (", join(keys(files), ", "), "), skipped"); continue)
        times = [hycom_file_time(files[k]) for k in ("ssh", "ts3z", "uv3z")]
        all(t -> t.valid == times[1].valid, times) || error("ssh/ts3z/uv3z valid times differ for $key")
        s = (valid = times[1].valid, tau = times[1].tau, analysis = times[1].analysis, files = files)
        if !haskey(best, s.valid) || s.tau < best[s.valid].tau
            best[s.valid] = s
        end
    end
    return sort(collect(values(best)); by = s -> s.valid)
end

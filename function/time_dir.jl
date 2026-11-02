# Time directory of a set of ROMS output files: which file and record holds each time step.
# Each time step is read once (a file's last record can repeat the next file's first).
# Needs `using NCDatasets, Dates, Printf` in the calling script.
#   td = time_dir(data_folder)                                            # all roms_his.*.nc, hourly
#   td = time_dir(data_folder; spinup = 3)                                # drop the first 3 days
#   td = time_dir(data_folder; tstart = DateTime(2022, 9, 1), tend = DateTime(2022, 9, 15))
#   td = time_dir(data_folder; prefix = "roms_avg.", dt = 86400)          # daily averages
#   td.files[td.TL[td.steps[1]][1]], td.TL[td.steps[1]][2]                # file name and record of the first step
#   for (fi, recs) in records_by_file(td)                                 # read file by file:
#       z = NCDataset(ds -> ds["zeta"].var[:, :, :], data_folder * td.files[fi])
#       for (k, n) in recs; ssh[:, :, n] .= z[:, :, k]; end               # record k of the file -> time step n
#   end
# Returns a NamedTuple:
#   files  file names (sorted), TL  Dict step -> (file index, record), steps  step numbers kept (sorted),
#   nt  number of steps, tdate  DateTime of each step, tsec  seconds since t_ref of each step

function time_dir(data_folder; prefix = "roms_his.", t_ref = DateTime(1994, 1, 1), dt = 3600,
                  spinup = 0.0, tstart = nothing, tend = nothing, verbose = true)
    files = sort(filter(f -> startswith(f, prefix) && endswith(f, ".nc"), readdir(data_folder)))
    isempty(files) && error("no $(prefix)*.nc files in $data_folder")
    TL = Dict{Int, Tuple{Int, Int}}()                                    # step since t_ref -> (file index, record)
    for (fi, f) in enumerate(files)
        t = NCDataset(ds -> ds["ocean_time"].var[:], joinpath(data_folder, f))   # seconds since t_ref (raw, no date decoding)
        for (k, tk) in enumerate(t); TL[round(Int, tk / dt)] = (fi, k); end
    end
    steps = sort(collect(keys(TL)))
    steps = filter(s -> s >= steps[1] + round(Int, spinup * 86400 / dt), steps)   # drop the spin-up [days]
    tstep(d) = round(Int, Dates.value(Millisecond(d - t_ref)) / 1000 / dt)
    tstart === nothing || (steps = filter(s -> s >= tstep(tstart), steps))
    tend   === nothing || (steps = filter(s -> s <= tstep(tend), steps))
    isempty(steps) && error("no time steps left in the requested window")
    tsec  = dt .* Float64.(steps)
    tdate = t_ref .+ Millisecond.(round.(Int, 1000 .* tsec))
    gaps  = diff(steps)
    verbose && @printf("%d files, %d records (dt %d s), %s .. %s, gaps: %d (largest %d dt)\n",
                       length(files), length(steps), dt, tdate[1], tdate[end], count(>(1), gaps), isempty(gaps) ? 0 : maximum(gaps))
    return (; files, TL, steps, nt = length(steps), tdate, tsec)
end

# Records grouped by file, in file order: [(file index, [(record in file, time step n), ...]), ...];
# lets a reader open each file once.
function records_by_file(td)
    byfile = Dict{Int, Vector{Tuple{Int, Int}}}()
    for (n, s) in enumerate(td.steps); (fi, k) = td.TL[s]; push!(get!(byfile, fi, Tuple{Int, Int}[]), (k, n)); end
    return sort([(fi, sort(v)) for (fi, v) in byfile])
end

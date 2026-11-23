#= meanflow_window.jl
Time windows for the background / wave split of wave_mean_flow.jl.
Needs `using NCDatasets, Dates, Printf` and function/time_dir.jl (loaded by load_all_hpc.jl).

  win = window_setup(data_folder, "u"; tstart = DateTime(2022, 9, 15), tend = DateTime(2022, 10, 15),
                     window_length = 72, window_overlap = 0.0)
  win.nwin                      number of windows in the span
  win.idx[w]                    records of window w, as indices into win.td.steps / win.td.tdate (n = 1..td.nt)
  win.t0[w], win.t1[w], win.tmid[w]   first / last record time and centre of window w
  win.td                        the time_dir of the span (files, TL, steps, nt, tdate, tsec); feed it to data_load

Span: [tstart, tend), so tend itself is not used: 09-15 00:00 -> 10-15 00:00 with hourly his records
is 720 records = 30 days, i.e. ten 3-day windows with no overlap.
Windows: length L = window_length [h] / dt records, step S = max(1, round(L * (1 - window_overlap))) records
  overlap 0 -> S = L (back to back), 0.5 -> S = L/2 (2-day window moves by 1 day), 1 -> S = 1 (running window).
  Window w covers records 1 + (w-1) S .. (w-1) S + L; nwin = floor((nt - L) / S) + 1. Records after the last full
  window (fewer than S) are left out and reported.
A window that spans missing his records (a gap in td.steps) is marked in win.gap.
=#

function window_setup(data_folder, var; tstart, tend, window_length = 72, window_overlap = 0.0,
                      prefix = "roms_his.", dt = 3600, t_ref = DateTime(1994, 1, 1), verbose = true)
    0 <= window_overlap <= 1 || error("window_overlap = $window_overlap, must be in [0, 1]")
    tend > tstart || error("tend $tend is not after tstart $tstart")

    # records in [tstart, tend): time_dir keeps both ends, so stop one record before tend
    td = time_dir(data_folder; prefix, t_ref, dt, tstart, tend = tend - Second(dt), verbose)
    NCDataset(joinpath(data_folder, td.files[td.TL[td.steps[1]][1]])) do ds
        haskey(ds, his_name(var)) || error("$var not in $(prefix)*.nc; variables: $(join(keys(ds), ", "))")
    end

    L = round(Int, window_length * 3600 / dt)              # window length [records]
    L >= 2 || error("window_length = $window_length h is shorter than two records (dt $dt s)")
    S = max(1, round(Int, L * (1 - window_overlap)))        # window step [records]
    nwin = td.nt >= L ? fld(td.nt - L, S) + 1 : 0
    nwin == 0 && error("span $(td.tdate[1]) .. $(td.tdate[end]) has $(td.nt) records, less than one window ($L)")

    idx  = [(1 + (w - 1) * S):((w - 1) * S + L) for w in 1:nwin]
    t0   = [td.tdate[first(r)] for r in idx]
    t1   = [td.tdate[last(r)] for r in idx]
    tmid = [a + (b - a) / 2 for (a, b) in zip(t0, t1)]
    gap  = [td.steps[last(r)] - td.steps[first(r)] != L - 1 for r in idx]   # missing records inside the window
    nleft = td.nt - last(idx[end])                          # records after the last full window

    if verbose
        @printf("%s: span %s .. %s (%d records), window %g h = %d records, overlap %g -> step %d records\n",
                var, tstart, tend, td.nt, window_length, L, window_overlap, S)
        @printf("%d windows, %d records left after the last one, %d windows with gaps\n", nwin, nleft, count(gap))
        for w in 1:nwin
            @printf("  %3d  %s .. %s  (records %d:%d)%s\n", w, t0[w], t1[w], first(idx[w]), last(idx[w]),
                    gap[w] ? "  GAP" : "")
        end
    end
    return (; var, td, L, S, nwin, idx, t0, t1, tmid, gap, nleft, window_length, window_overlap)
end

#= ---------------------------------------------------------------------------------------------------------------
Window means of one his variable, one NetCDF file per window.
Needs records_by_file (function/time_dir.jl) and uvp_masks (function/uvp_masks.jl), both loaded by load_all_hpc.jl.

  files = window_mean(win, data_folder, out_dir)                              # every window of win
  files = window_mean(win, data_folder, out_dir; windows = 1:3, grid_file)    # windows 1-3, land -> NaN
  files = window_mean(win, data_folder, out_dir; var = "ubar")                # another variable, same windows

Each his file is opened once and read in one block (all its records of var); every record is added to the sum of
every window that holds it, so with overlap a record is still read only once. A window is written as soon as its
last record is in, and its sum is dropped. Memory = one file block (4 records) + the sums of the windows open at the
same time (ceil(L / S), Float64): 3-D field ~0.4 GB + 0.2 GB per open window; overlap 0 or 0.5 -> 1-2 windows.
Output: out_dir/<name>_win<NNN>_<t0 yyyymmddTHH>.nc (name = mean_name(var): U, V, W, rho_window, zeta_window, ...), the mean in the variable's own grid (xi, eta[, s]) and dims,
time = window centre (ocean_time, seconds since 1994-01-01), plus t_start, t_end, nrec of the window.
Files are written under a temporary name and renamed when complete; existing files are skipped (overwrite = false),
so a stopped run can be started again. `windows` lets several jobs split the windows between them.
--------------------------------------------------------------------------------------------------------------- =#

# Names of the window means (background fields): his variable -> saved name (folder, file prefix, variable in the file)
#   u -> U, v -> V, w -> W, rho -> rho_window, zeta -> zeta_window; the others (temp, salt, ubar, vbar) keep the his name
#   rhoi (in-situ density of the split EOS, insitu_rho.jl; computed from rho, temp, salt, zeta) -> rhoi_window
const MEAN_NAME = Dict("u" => "U", "v" => "V", "w" => "W", "rho" => "rho_window", "zeta" => "zeta_window", "rhoi" => "rhoi_window")
mean_name(var) = get(MEAN_NAME, var, var)

window_file(out_dir, var, w, t0) = joinpath(out_dir, @sprintf("%s_win%03d_%s.nc", var, w, Dates.format(t0, "yyyymmddTHH")))

function window_mean(win, data_folder, out_dir; var = win.var, windows = 1:win.nwin, grid_file = nothing,
                     overwrite = false, t_ref = DateTime(1994, 1, 1), max_gb = 6.0, verbose = true)
    td = win.td
    mkpath(out_dir)
    name = mean_name(var)                                            # saved name of the mean (U, rho_window, ...)
    fnames = Dict(w => window_file(out_dir, name, w, win.t0[w]) for w in windows)
    for w in windows; rm(fnames[w] * ".tmp"; force = true); end            # left by a run stopped mid-write
    todo = sort([w for w in windows if overwrite || !window_done(fnames[w], name, win, w, t_ref)])
    verbose && length(todo) < length(windows) && println("$var: $(length(windows) - length(todo)) windows already done, skipped")
    isempty(todo) && return [fnames[w] for w in windows]

    # variable layout from the first file (Julia order: xi, eta[, s], time)
    dnames, sz, vt, vatt = NCDataset(joinpath(data_folder, td.files[td.TL[td.steps[1]][1]])) do ds
        v = ds[his_name(var)]
        dimnames(v), size(v), eltype(v.var), var == "rhoi" ?
            ["long_name" => "in-situ density anomaly of the split EOS (rho1 + qp1 dpth (1 - qp2 dpth))", "units" => "kilogram meter-3"] :
            [k => v.attrib[k] for k in ("long_name", "units") if haskey(v.attrib, k)]
    end
    sp = sz[1:end-1]; blk = map(n -> 1:n, sp); cs = ntuple(_ -> Colon(), length(sp))

    # memory check: one file block + the window sums open at once
    nopen = cld(win.L, win.S)
    gb = (prod(sp) * 4 * 4 + nopen * prod(sp) * 8) / 1e9
    verbose && @printf("%s %s: up to %d windows open, ~%.1f GB\n", var, sp, nopen, gb)
    gb > max_gb && error("needs ~$(round(gb, digits = 1)) GB (> max_gb = $max_gb): use a larger overlap step or tiles")

    if var == "rhoi"                                             # in-situ density: h, z0 of the rho grid, rho0
        grid_file === nothing && error("rhoi needs grid_file (h)")
        his1 = joinpath(data_folder, td.files[1])
        hr = NCDataset(g -> Float64.(g["h"].var[:, :]), grid_file)
        z0r = s_stretch(his1, hr)[1]; rho0 = NCDataset(d -> Float64(d.attrib["rho0"]), his1)
        rbuf = Array{vt}(undef, sp...)
    end

    land = nothing
    if grid_file !== nothing                                     # land points of the variable's grid
        mr = NCDataset(g -> g["mask_rho"].var[:, :], grid_file)
        mk = "xi_u" in dnames ? uvp_masks(mr)[1] : "eta_v" in dnames ? uvp_masks(mr)[2] : mr
        land = findall(mk .== 0)
    end

    need = falses(td.nt); for w in todo; need[win.idx[w]] .= true; end
    sums = Dict{Int, Array{Float64}}(); cnt = Dict{Int, Int}()
    bufs = Dict{Int, Array{vt}}()
    t0 = time(); nwritten = 0
    for (fi, recs) in records_by_file(td)
        recs = filter(r -> need[r[2]], recs)
        isempty(recs) && continue
        ks = first.(recs); kr = minimum(ks):maximum(ks)
        z = var == "rhoi" ? nothing : get!(() -> Array{vt}(undef, sp..., length(kr)), bufs, length(kr))
        if var == "rhoi"
            zb, ζb = NCDataset(d -> (read_his_block(d, var, :, :, kr), d["zeta"].var[:, :, kr]), joinpath(data_folder, td.files[fi]))
        else
            NCDataset(ds -> NCDatasets.load!(ds[var].var, z, blk..., kr), joinpath(data_folder, td.files[fi]))
        end
        for (k, n) in recs
            rec = var == "rhoi" ? record_of(zb, var, k - first(kr) + 1, rbuf, view(ζb, :, :, k - first(kr) + 1), hr, z0r, rho0) :
                                  view(z, cs..., k - first(kr) + 1)
            for w in todo
                n in win.idx[w] || continue
                get!(() -> zeros(Float64, sp...), sums, w) .+= rec
                cnt[w] = get(cnt, w, 0) + 1
                n == last(win.idx[w]) || continue
                m = Float32.(sums[w] ./ cnt[w])                   # window complete: mean, write, drop
                land === nothing || for p in land; m[p, ntuple(_ -> Colon(), length(sp) - 2)...] .= NaN32; end
                write_window_mean(fnames[w], name, m, dnames, vatt, win, w, cnt[w], t_ref)
                delete!(sums, w); nwritten += 1
                verbose && @printf("  %s window %d (%s .. %s, %d records) -> %s, %.1f min\n", var, w, win.t0[w],
                                   win.t1[w], cnt[w], basename(fnames[w]), (time() - t0) / 60)
            end
        end
    end
    isempty(sums) || error("windows $(sort(collect(keys(sums)))) never reached their last record")
    verbose && @printf("%s: %d windows written to %s, %.1f min\n", var, nwritten, out_dir, (time() - t0) / 60)
    return [fnames[w] for w in windows]
end

# A window file counts as done only if it opens, holds var, and its record count and first record match window w
# (a damaged file, or one from another span/setup, is redone).
function window_done(fname, var, win, w, t_ref; nrec = length(win.idx[w]), t0 = win.t0[w])
    isfile(fname) || return false
    ok = try
        NCDataset(fname) do ds
            haskey(ds, var) && ds["nrec"][1] == nrec &&
                ds["t_start"].var[1] ≈ Dates.value(Millisecond(t0 - t_ref)) / 1000
        end
    catch
        false
    end
    ok || println("  $(basename(fname)): exists but incomplete or from another setup, redone")
    return ok
end

function write_window_mean(fname, var, m, dnames, vatt, win, w, nrec, t_ref)
    sec(t) = Dates.value(Millisecond(t - t_ref)) / 1000
    tunits = "seconds since " * Dates.format(t_ref, "yyyy-mm-dd HH:MM:SS")
    tmp = fname * ".tmp"
    NCDataset(tmp, "c"; attrib = ["title" => "window mean of ROMS his $var",
                                  "window" => w, "window_length_h" => win.window_length,
                                  "window_overlap" => win.window_overlap,
                                  "span_start" => string(win.td.tdate[1]), "span_end" => string(win.td.tdate[end]),
                                  "created" => string(now())]) do ds
        for (d, n) in zip(dnames[1:end-1], size(m)); defDim(ds, d, n); end
        defDim(ds, "time", 1)
        v = defVar(ds, var, Float32, (dnames[1:end-1]..., "time"); attrib = [vatt...; "note" => "mean over the window; land = NaN if masked"])
        v[ntuple(_ -> Colon(), ndims(m))..., 1] = m
        defVar(ds, "ocean_time", [sec(win.tmid[w])], ("time",); attrib = ["units" => tunits, "long_name" => "window centre"])
        defVar(ds, "t_start", [sec(win.t0[w])], ("time",); attrib = ["units" => tunits, "long_name" => "first record of the window"])
        defVar(ds, "t_end", [sec(win.t1[w])], ("time",); attrib = ["units" => tunits, "long_name" => "last record of the window"])
        defVar(ds, "nrec", Int32[nrec], ("time",); attrib = ["long_name" => "his records averaged"])
    end
    mv(tmp, fname; force = true)
end

#= ---------------------------------------------------------------------------------------------------------------
Window means of several his variables, one folder per variable (mean_root/<var>/); the step wave_mean_flow.jl calls.
  mean_files = window_means(win, data_folder, mean_root, ["zeta", "u", "v"]; windows = 1:win.nwin, grid_file)
  mean_files["u"][w]     file of the mean of u in window w (saved as U: mean_root/U/U_win<NNN>_<t0>.nc)
Checks first that every variable is in the his files (before any long read); windows already written are skipped
(window_done), so this is also the quick way to get the file names when the means exist.
--------------------------------------------------------------------------------------------------------------- =#
function window_means(win, data_folder, mean_root, vars; windows = 1:win.nwin, grid_file = nothing, kw...)
    NCDataset(joinpath(data_folder, win.td.files[1])) do ds
        miss = filter(v -> !all(haskey(ds, x) for x in his_needs(v)), vars)
        isempty(miss) || error("not in the his files: $(join(miss, ", "))")
    end
    mean_files = Dict{String, Vector{String}}()
    for var in vars
        mean_files[var] = window_mean(win, data_folder, joinpath(mean_root, mean_name(var)); var, windows, grid_file, kw...)
        GC.gc()
    end
    return mean_files
end

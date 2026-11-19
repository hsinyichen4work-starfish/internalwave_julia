#= wave_perturb.jl
Wave part of a his variable at every hourly record (step 5 of wave_mean_flow.jl):
  rho_wave = ρ′ = ρ − ρ̄_window,  u_wave = u′ = u − ū_window,  v_wave = v′ = v − v̄_window, ...
on the s-levels of the record, on the variable's own grid (rho / u / v points), at the original time steps.
Needs `using NCDatasets, Dates, Printf`, records_by_file (time_dir.jl), uvp_masks, and meanflow_window.jl
(window_file, window_done) — the window means of section 2 must exist.

Which window mean a record uses: its own window (overlap 0). With overlap a record sits in several windows; it uses
the one whose centre is nearest, so every record is written once and the files tile the span.

  files = window_wave(win, data_folder, mean_root; var = "rho", windows = 1:win.nwin, grid_file)
Output: mean_root/<var>_wave/<var>_wave_win<NNN>_<t0>.nc, one file per window, holding all its records:
  <var>_wave (xi, eta[, s], time), ocean_time (every record), t_start, t_end, nrec  (~98 MB per 3-D record:
  a 3-day window of one 3-D variable is ~7 GB). Chunked one record per chunk, so a record reads fast.
Each his file is read once in one block; the window mean is read once per window. Written under .tmp and renamed
when complete; complete files are skipped on a rerun.
=#

# window whose mean each record uses: the window holding it with the nearest centre
function wave_owner(win)
    own = zeros(Int, win.td.nt)
    for n in 1:win.td.nt, w in 1:win.nwin
        n in win.idx[w] || continue
        cw(w) = abs(n - (first(win.idx[w]) + last(win.idx[w])) / 2)
        (own[n] == 0 || cw(w) < cw(own[n])) && (own[n] = w)
    end
    return own
end

function window_wave(win, data_folder, mean_root; var, windows = 1:win.nwin, grid_file = nothing, overwrite = false,
                     t_ref = DateTime(1994, 1, 1), verbose = true)
    td = win.td
    name = var * "_wave"
    out = joinpath(mean_root, name); mkpath(out)
    own = wave_owner(win)
    recs_of = Dict(w => findall(==(w), own) for w in windows)
    fn(w) = window_file(out, name, w, win.t0[w])
    fn_mean(w) = window_file(joinpath(mean_root, mean_name(var)), mean_name(var), w, win.t0[w])
    for w in windows; rm(fn(w) * ".tmp"; force = true); end
    todo = sort([w for w in windows if !isempty(recs_of[w]) &&
                 (overwrite || !window_done(fn(w), name, win, w, t_ref; nrec = length(recs_of[w]), t0 = td.tdate[recs_of[w][1]]))])
    verbose && @printf("%s: %d of %d windows to do\n", name, length(todo), length(windows))
    isempty(todo) && return [fn(w) for w in windows]
    for w in todo
        window_done(fn_mean(w), mean_name(var), win, w, t_ref) || error("window $w: no mean of $var in $(dirname(fn_mean(w))); run section 2 first")
    end

    dnames, sz, vt, vatt = NCDataset(joinpath(data_folder, td.files[td.TL[td.steps[1]][1]])) do ds
        v = ds[var]
        dimnames(v), size(v), eltype(v.var), [k => v.attrib[k] for k in ("long_name", "units") if haskey(v.attrib, k)]
    end
    sp = sz[1:end-1]; blk = map(n -> 1:n, sp); cs = ntuple(_ -> Colon(), length(sp))
    land = nothing
    if grid_file !== nothing
        mr = NCDataset(g -> g["mask_rho"].var[:, :], grid_file)
        mk = "xi_u" in dnames ? uvp_masks(mr)[1] : "eta_v" in dnames ? uvp_masks(mr)[2] : mr
        land = findall(mk .== 0)
    end
    tunits = "seconds since " * Dates.format(t_ref, "yyyy-mm-dd HH:MM:SS")
    sec(t) = Dates.value(Millisecond(t - t_ref)) / 1000

    # the open output: one window at a time (records come in time order, a window's records are contiguous)
    cur = 0; ds = nothing; m = nothing; slot = Dict{Int, Int}()
    function open_window(w)
        rs = recs_of[w]
        slot = Dict(n => i for (i, n) in enumerate(rs))
        m = NCDataset(d -> d[mean_name(var)].var[cs..., 1], fn_mean(w))           # window mean (xi, eta[, s])
        ds = NCDataset(fn(w) * ".tmp", "c"; attrib = ["title" => "wave part $var - window mean of ROMS his $var",
                                                     "window" => w, "window_length_h" => win.window_length,
                                                     "window_overlap" => win.window_overlap,
                                                     "mean_file" => fn_mean(w), "created" => string(now())])
        for (d, n) in zip(dnames[1:end-1], sp); defDim(ds, d, n); end
        defDim(ds, "time", length(rs)); defDim(ds, "one", 1)
        defVar(ds, name, Float32, (dnames[1:end-1]..., "time"); chunksizes = [sp..., 1],
               attrib = [vatt...; "long_name" => "$var - window mean of $var (wave part)"])
        defVar(ds, "ocean_time", sec.(td.tdate[rs]), ("time",); attrib = ["units" => tunits, "long_name" => "record time"])
        defVar(ds, "t_start", [sec(td.tdate[rs[1]])], ("one",); attrib = ["units" => tunits, "long_name" => "first record"])
        defVar(ds, "t_end", [sec(td.tdate[rs[end]])], ("one",); attrib = ["units" => tunits, "long_name" => "last record"])
        defVar(ds, "nrec", Int32[length(rs)], ("one",); attrib = ["long_name" => "records in this file"])
        cur = w
    end
    function close_window()
        close(ds); mv(fn(cur) * ".tmp", fn(cur); force = true)
        verbose && @printf("  %s window %d: %d records -> %s, %.1f min\n", name, cur, length(recs_of[cur]),
                           basename(fn(cur)), (time() - t0) / 60)
        cur = 0; ds = nothing; m = nothing; GC.gc()
    end

    t0 = time()
    need = falses(td.nt); for w in todo; need[recs_of[w]] .= true; end
    bufs = Dict{Int, Array{vt}}()
    wv = Array{Float32}(undef, sp...)
    for (fi, recs) in records_by_file(td)
        recs = filter(r -> need[r[2]], recs)
        isempty(recs) && continue
        ks = first.(recs); kr = minimum(ks):maximum(ks)
        z = get!(() -> Array{vt}(undef, sp..., length(kr)), bufs, length(kr))
        NCDataset(d -> NCDatasets.load!(d[var].var, z, blk..., kr), joinpath(data_folder, td.files[fi]))
        for (k, n) in recs
            w = own[n]
            w == cur || (cur == 0 || close_window(); open_window(w))
            wv .= view(z, cs..., k - first(kr) + 1) .- m
            land === nothing || for p in land; wv[p, ntuple(_ -> Colon(), length(sp) - 2)...] .= NaN32; end
            ds[name][cs..., slot[n]] = wv
            n == last(recs_of[w]) && close_window()
        end
    end
    cur == 0 || error("$name window $cur never reached its last record")
    verbose && @printf("%s: %d windows written to %s, %.1f min\n", name, length(todo), out, (time() - t0) / 60)
    return [fn(w) for w in windows]
end

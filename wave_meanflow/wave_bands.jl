#= wave_bands.jl
Step 5: wave parts at a fixed depth, every hourly record     rho_wave, u_wave, v_wave, zeta_wave
Step 5b: p_wave from rho_wave (hydrostatic, from the free surface down), every hourly record
Step 6: three frequency bands of each wave part               <field>_hp28, <field>_bp9_15, <field>_hp9
        (fields: rhoi_wave and p_wave from rhoi (in-situ density, insitu_rho.jl; p_from = "rho" for rho1 instead);
         rho_wave; u_wave; v_wave; zeta_wave)
Only the bands are saved; <var>_wave lives in memory between the two steps, one horizontal tile at a time.
No depth mean is removed here (barotropic / baroclinic split is a later step).
Needs `using NCDatasets, Dates, Printf, DSP`, function/butter_filters.jl (bandpass_butter, lowhighpass_butter),
load_all_hpc.jl (data_load, records_by_file, rho2u / rho2v), meanflow_window.jl (window_file, window_done, mean_name),
wave_perturb.jl (wave_owner) and mode_projection.jl (s_stretch).

Step 5, at every record t (hourly, original time steps), on the variable's own grid (rho, u or v points):
  var_wave(t, z) = var(t) − background(t) taken at the record's depths z = ζ(t) + (ζ(t) + h) z0   (fixed z)
  background = the window mean (U, V, rho_window on the levels of zeta_window, interpolated linearly in each column to
  the record's depths; end values held outside the mean profile), so the tidal heave of the s-levels through the
  mean profile is not a wave.
    background = "window"  the mean of the record's window (wave_owner: nearest centre with overlap). Piecewise
                           constant in time: steps every window, which a filter turns into short false signals.
    background = "interp"  linear in time between the means of the two windows whose centres bracket t (held before
                           the first / after the last centre): no steps.
  zeta_wave = zeta − zeta_window (2-D, no levels), same choice of background.
  At u / v points: h, zeta averaged from rho points (rho2u / rho2v), z0 from that h.
Step 5b (in the rho run, before filtering, since the layer thicknesses change with ζ(t)):
  p_wave_k = Σ_{levels above k} g rho_wave Hz + ½ g rho_wave_k Hz_k,   Hz = (ζ(t) + h) Δz0   [Pa], p_wave = 0 at z = ζ
  (as DFF/bcef_modes_puv.jl; only the internal part: the surface term g rho0 zeta_wave is depth-uniform and goes with
  the barotropic part later; no depth mean removed here)
Step 6: each band by a 4th-order Butterworth filter run forward and backward (filtfilt, butter_filters.jl), along
  time over the whole span at once (not window by window, so there are no window edges in the filter):
    hp28    28 h high-pass                  (diurnal, semidiurnal and faster)
    bp9_15  9-15 h band-pass                (semidiurnal)
    hp9     9 h high-pass                   (higher harmonics, short nonlinear waves)
  The first / last ~day of the span is affected by the filter's ends.

Output, per field and band: mean_root/<field>_<band>/<field>_<band>_win<NNN>_<t0>.nc, one file per window, all its records,
  (xi, eta, s, time) on the variable's grid, chunked like the his files (one tile x all levels x one record).
  3-D variable: ~7 GB per band per 3-day window. Files are filled tile by tile under .tmp (progress in
  <var>_wave_progress.txt, so a stopped run resumes at the next tile) and renamed when every tile is in.
=#

const BANDS = [(name = "hp28",   kind = :high, T = 28.0),
               (name = "bp9_15", kind = :band, T = (9.0, 15.0)),
               (name = "hp9",    kind = :high, T = 9.0)]

# ---------------------------------------------------------------------------------------------------------------
# step 5 helpers
# ---------------------------------------------------------------------------------------------------------------
# window-mean profile xbar (on its levels zbar) at the depths z = ζ + (ζ + h) z0 of a record, column by column
function bg_at_depth!(bg, ζ, h, z0, zbar, xbar)
    nx, ny, N = size(bg)
    @inbounds for j in 1:ny, i in 1:nx
        H = ζ[i, j] + h[i, j]; m = 1
        for k in 1:N
            zz = ζ[i, j] + H * z0[i, j, k]
            while m < N - 1 && zbar[i, j, m+1] < zz; m += 1; end
            w = clamp((zz - zbar[i, j, m]) / (zbar[i, j, m+1] - zbar[i, j, m]), 0f0, 1f0)
            bg[i, j, k] = xbar[i, j, m] + w * (xbar[i, j, m+1] - xbar[i, j, m])
        end
    end
    return bg
end

# for every record: (window 1, window 2, weight of window 2) of its background
function bg_weights(win, background)
    own = wave_owner(win)
    tm = Dates.value.(win.tmid .- win.tmid[1]) ./ 3.6e6                 # window centres [h]
    tr = Dates.value.(win.td.tdate .- win.tmid[1]) ./ 3.6e6             # records [h]
    map(1:win.td.nt) do n
        background == "window" && return (own[n], own[n], 0f0)
        background == "interp" || error("background = $background: use \"window\" or \"interp\"")
        tr[n] <= tm[1] && return (1, 1, 0f0)
        tr[n] >= tm[end] && return (win.nwin, win.nwin, 0f0)
        w1 = findlast(<=(tr[n]), tm)
        (w1, w1 + 1, Float32((tr[n] - tm[w1]) / (tm[w1+1] - tm[w1])))
    end
end

# grid of a variable: :rho, :u, :v from its dimension names
grid_of(dnames) = "xi_u" in dnames ? :u : "eta_v" in dnames ? :v : :rho
to_grid(f, g) = g == :u ? rho2u(f) : g == :v ? rho2v(f) : f

# horizontal tiles = the his chunks of the variable: [(xi range, eta range), ...]
function var_tiles(his_file, var)
    sz, ch = NCDataset(his_file) do ds; size(ds[var]), chunking(ds[var].var)[2]; end
    rx = [i:min(i + ch[1] - 1, sz[1]) for i in 1:ch[1]:sz[1]]
    ry = [j:min(j + ch[2] - 1, sz[2]) for j in 1:ch[2]:sz[2]]
    return vec([(ix, jy) for ix in rx, jy in ry]), ch
end

# step 5: var_wave on one tile (ix, jy) for every record of the span -> (nx_t, ny_t, N, nt) Float32
# G: grid of the variable (h, z0 full fields; ζ series full field, (xi, eta, nt)); xbar / zbar: window means on the tile
function wave_series_tile(var, ix, jy, win, data_folder, G, xbar, zbar, wts)
    td = win.td; N = size(G.z0, 3)
    x = Array{Float32}(undef, length(ix), length(jy), N, td.nt)
    h = G.h[ix, jy]; z0 = G.z0[ix, jy, :]
    bg1 = Array{Float32}(undef, length(ix), length(jy), N); bg2 = similar(bg1); rbuf = similar(bg1)
    for (fi, recs) in records_by_file(td)
        ks = first.(recs); kr = minimum(ks):maximum(ks)
        blk = NCDataset(d -> read_his_block(d, var, ix, jy, kr), joinpath(data_folder, td.files[fi]))   # one chunk per record
        for (k, n) in recs
            ζ = view(G.ζ, ix, jy, n)
            rec = record_of(blk, var, k - first(kr) + 1, rbuf, ζ, h, z0, G.rho0)   # rhoi: in-situ density of the record
            w1, w2, a = wts[n]
            bg_at_depth!(bg1, ζ, h, z0, zbar[w1], xbar[w1])
            if a > 0
                bg_at_depth!(bg2, ζ, h, z0, zbar[w2], xbar[w2]); bg1 .= (1 - a) .* bg1 .+ a .* bg2
            end
            @views x[:, :, :, n] .= rec .- bg1
        end
    end
    return x
end

# step 5b: p_wave (xi, eta, N, nt) from rho_wave x on a tile; ζ (xi, eta, nt), h, dz0 on the same tile
function p_from_rho(x, ζ, h, dz0; g = 9.81f0)
    nx, ny, N, nt = size(x); p = similar(x)
    @inbounds for n in 1:nt, j in 1:ny, i in 1:nx
        H = ζ[i, j, n] + h[i, j]; top = 0f0
        for k in N:-1:1
            dp = g * x[i, j, k, n] * H * dz0[i, j, k]
            p[i, j, k, n] = top + 0.5f0 * dp; top += dp
        end
    end
    return p
end

# ---------------------------------------------------------------------------------------------------------------
# step 6: one band of a series x (xi, eta[, s], time), filtfilt along time, one level at a time (Float64 inside)
# ---------------------------------------------------------------------------------------------------------------
function band_filter(x, band; dt_h = 1.0, order = 4)
    nt = size(x, ndims(x)); out = similar(x)
    levels = ndims(x) == 4 ? (1:size(x, 3)) : (1:1)
    for k in levels
        sl = ndims(x) == 4 ? view(x, :, :, k, :) : view(x, :, :, :)
        M = permutedims(reshape(Float64.(sl), :, nt))                    # (time, points)
        F = band.kind == :band ? bandpass_butter(M, band.T[1], band.T[2], dt_h, order) :
                                 lowhighpass_butter(M, band.T, dt_h, order, band.kind == :high ? "high" : "low")
        o = ndims(x) == 4 ? view(out, :, :, k, :) : out
        o .= reshape(permutedims(F), size(sl))
    end
    return out
end

# grid of a variable and the ζ series on it: h, z0 (3-D vars), ζ (xi, eta, nt) for the whole span, ζ of each window mean
function wave_grid(win, data_folder, mean_root, var, grid_file)
    his1 = joinpath(data_folder, win.td.files[1])
    dnames, sz, rho0 = NCDataset(ds -> (dimnames(ds[his_name(var)]), size(ds[his_name(var)]), Float64(ds.attrib["rho0"])), his1)
    is3d = length(sz) == 4; g = grid_of(dnames)
    hr = NCDataset(gf -> Float64.(gf["h"].var[:, :]), grid_file)
    hg = to_grid(hr, g)
    ζ = Float32.(to_grid(data_load(data_folder, "zeta", win.td; verbose = false), g))   # (xi, eta, nt) on the var's grid
    ζbar = [Float32.(to_grid(NCDataset(d -> d["zeta_window"].var[:, :, 1],
                                       window_file(joinpath(mean_root, "zeta_window"), "zeta_window", w, win.t0[w])), g))
            for w in 1:win.nwin]
    z0, dz0 = is3d ? s_stretch(his1, hg) : (nothing, nothing)
    return (h = Float32.(hg), z0, dz0, ζ, ζbar, is3d, g, dnames, sz, rho0)
end

# window means of var on a tile: xbar[w] (values) and zbar[w] (depths of its levels, from zeta_window)
function tile_means(win, mean_root, var, G, ix, jy; t_ref = DateTime(1994, 1, 1))
    mfile(w) = window_file(joinpath(mean_root, mean_name(var)), mean_name(var), w, win.t0[w])
    xbar = [NCDataset(d -> d[mean_name(var)].var[ix, jy, :, 1], mfile(w)) for w in 1:win.nwin]
    zbar = [G.ζbar[w][ix, jy] .+ (G.ζbar[w][ix, jy] .+ G.h[ix, jy]) .* G.z0[ix, jy, :] for w in 1:win.nwin]
    return xbar, zbar
end

# ---------------------------------------------------------------------------------------------------------------
# steps 5 (+5b) + 6 for one his variable (rho, u, v or zeta): bands of its wave fields, written per window
# ---------------------------------------------------------------------------------------------------------------
function window_wave_bands(win, data_folder, mean_root; var, grid_file, bands = BANDS, order = 4, background = "window",
                           p_from = "rhoi", t_ref = DateTime(1994, 1, 1), verbose = true)
    td = win.td
    his1 = joinpath(data_folder, td.files[1])
    own = wave_owner(win)
    recs_of = [findall(==(w), own) for w in 1:win.nwin]
    # p_wave (step 5b) from the density picked by p_from: "rhoi" (in-situ, split EOS) or "rho" (rho1, his rho)
    fields = var == p_from ? ["$(var)_wave", "p_wave"] : ["$(var)_wave"]
    name(f, b) = "$(f)_$(b.name)"
    fn(f, b, w) = window_file(joinpath(mean_root, name(f, b)), name(f, b), w, win.t0[w])
    done(f, b, w) = window_done(fn(f, b, w), name(f, b), win, w, t_ref; nrec = length(recs_of[w]), t0 = td.tdate[recs_of[w][1]])
    allfiles() = Dict(name(f, b) => [fn(f, b, w) for w in 1:win.nwin] for f in fields, b in bands)
    if all(done(f, b, w) for f in fields, b in bands, w in 1:win.nwin if !isempty(recs_of[w]))
        verbose && println("$(join(fields, ", ")): all bands and windows done, skipped")
        return allfiles()
    end
    ws = [w for w in 1:win.nwin if !isempty(recs_of[w])]

    G = wave_grid(win, data_folder, mean_root, var, grid_file)
    dnames, sz, is3d = G.dnames, G.sz, G.is3d
    vunits = NCDataset(ds -> get(ds[his_name(var)].attrib, "units", ""), his1)
    units(f) = f == "p_wave" ? "Pa" : vunits
    tiles, ch = is3d ? var_tiles(his1, his_name(var)) : ([(1:sz[1], 1:sz[2])], (sz[1], sz[2], 1))
    wts = bg_weights(win, background)

    # files: created once under .tmp; progress file lists the tiles already written to all of them
    prog = joinpath(mean_root, "$(var)_wave_progress.txt")
    tmp_ok = isfile(prog) && all(isfile(fn(f, b, w) * ".tmp") for f in fields, b in bands, w in ws)
    tiles_done = tmp_ok ? Set(parse.(Int, filter(!isempty, readlines(prog)))) : Set{Int}()
    if !tmp_ok
        rm(prog; force = true)
        tunits = "seconds since " * Dates.format(t_ref, "yyyy-mm-dd HH:MM:SS")
        sec(t) = Dates.value(Millisecond(t - t_ref)) / 1000
        for f in fields, b in bands, w in ws
            mkpath(dirname(fn(f, b, w))); rs = recs_of[w]
            what = f == "p_wave" ? "hydrostatic pressure of $(var)_wave ($(var == "rhoi" ? "in-situ density of the split EOS" : "his rho = rho1") minus its window background at fixed depth)" :
                                   "$var minus its window background at fixed depth"
            NCDataset(fn(f, b, w) * ".tmp", "c"; attrib = ["title" => "$(name(f, b)): $what, $(b.kind == :band ? "$(b.T[1])-$(b.T[2]) h band-pass" : "$(b.T) h $(b.kind)-pass")",
                                                       "filter" => "Butterworth order $order, filtfilt over the whole span $(td.tdate[1]) .. $(td.tdate[end])",
                                                       "background" => background, "window" => w, "density" => var,
                                                       "window_length_h" => win.window_length, "window_overlap" => win.window_overlap,
                                                       "created" => string(now())]) do ds
                for (d, n) in zip(dnames[1:end-1], sz[1:end-1]); defDim(ds, d, n); end
                defDim(ds, "time", length(rs)); defDim(ds, "one", 1)
                defVar(ds, name(f, b), Float32, (dnames[1:end-1]..., "time"); chunksizes = [ch[1:end-1]..., 1],
                       attrib = ["units" => units(f), "long_name" => "$(name(f, b)) ($f, band $(b.name))"])
                defVar(ds, "ocean_time", sec.(td.tdate[rs]), ("time",); attrib = ["units" => tunits, "long_name" => "record time"])
                defVar(ds, "t_start", [sec(td.tdate[rs[1]])], ("one",); attrib = ["units" => tunits])
                defVar(ds, "t_end", [sec(td.tdate[rs[end]])], ("one",); attrib = ["units" => tunits])
                defVar(ds, "nrec", Int32[length(rs)], ("one",); attrib = ["long_name" => "records in this file"])
            end
        end
    end
    verbose && @printf("%s: %d tiles (%d done), %d bands x %d windows, background %s\n", join(fields, ", "), length(tiles),
                       length(tiles_done), length(bands), length(ws), background)

    mfile(w) = window_file(joinpath(mean_root, mean_name(var)), mean_name(var), w, win.t0[w])
    for w in 1:win.nwin
        window_done(mfile(w), mean_name(var), win, w, t_ref) || error("window $w: no $(mean_name(var)); run section 2 first")
    end

    t0 = time()
    for (q, (ix, jy)) in enumerate(tiles)
        q in tiles_done && continue
        # step 5: var_wave on this tile, whole span
        if is3d
            xbar, zbar = tile_means(win, mean_root, var, G, ix, jy)
            x = wave_series_tile(var, ix, jy, win, data_folder, G, xbar, zbar, wts)
        else
            zb = G.ζbar
            x = similar(G.ζ)
            for n in 1:td.nt
                w1, w2, a = wts[n]
                @views x[:, :, n] .= G.ζ[:, :, n] .- ((1 - a) .* zb[w1] .+ a .* zb[w2])
            end
        end
        # step 5b: p_wave from rho_wave (before filtering)
        X = Dict(fields[1] => x)
        "p_wave" in fields && (X["p_wave"] = p_from_rho(x, view(G.ζ, ix, jy, :), G.h[ix, jy], G.dz0[ix, jy, :]))
        # step 6: each field and band, written into every window's file
        for f in fields, b in bands
            y = band_filter(X[f], b; order)
            for w in ws
                rs = recs_of[w]
                NCDataset(fn(f, b, w) * ".tmp", "a") do ds
                    is3d ? (ds[name(f, b)][ix, jy, :, :] = y[:, :, :, rs]) : (ds[name(f, b)][ix, jy, :] = y[:, :, rs])
                end
            end
            y = nothing
        end
        open(io -> println(io, q), prog, "a")
        verbose && @printf("  %s tile %d / %d (xi %d:%d, eta %d:%d): %d bands written, %.1f min\n", join(fields, ", "), q,
                           length(tiles), first(ix), last(ix), first(jy), last(jy), length(bands), (time() - t0) / 60)
        x = nothing; X = nothing; GC.gc()
    end
    for f in fields, b in bands, w in ws; mv(fn(f, b, w) * ".tmp", fn(f, b, w); force = true); end
    rm(prog; force = true)
    verbose && @printf("%s: %d bands x %d windows written, %.1f min\n", join(fields, ", "), length(bands), length(ws), (time() - t0) / 60)
    return allfiles()
end

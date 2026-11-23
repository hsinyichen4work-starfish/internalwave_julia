#= baroclinic_modes.jl
Step 8: baroclinic part of each band, every hourly record (computed on the fly, not saved: ~630 GB)
  u_bc = u_wave − u_bt,   v_bc = v_wave − v_bt                                  (same band on both sides)
  p_bc = (p_wave + ρ0 g zeta_wave) − p_bt = p_wave − (1/H) ∫ p_wave dz
         the total wave pressure is p_wave (internal, from rho_wave) + ρ0 g zeta_wave (free surface); p_bt holds the
         depth mean of that total, so the surface term cancels and (1/H) ∫ p_bc dz = 0, as for u_bc, v_bc.
  The _bt fields are 2-D, taken off every level of the column; (1/H) ∫ u_bc dz = 0 (checked: ±4e-9).
Step 9: projection of the baroclinic part on the vertical modes of the window (step 4, modes<NMODE>)
  f_n = (1/H) ∫ f_bc φn dz = Σ_k f_bc Hz φn / H     (Siyanbola et al. 2024 Eq. 15),   f = u, v, p,  n = 1..NMODE
  Hz = (ζ(t) + h) Δz0, H = h + ζ(t) of the record; φn of the record's window (wave_owner), on its levels, taken per
  s-level as in DFF/bcef_modes_puv.jl. u_bc, v_bc are moved to rho points first (u2rho!, v2rho!: average of the two
  neighbours, edges copied), so u_n, v_n, p_n all sit on the rho grid with the same φn and H.
  u_n, v_n along the grid xi / eta (not rotated). Depth-integrated mode-n energy flux = H p_n u_n (no cross-mode terms).
  Check per window: share of the baroclinic variance in the modes,
    <f>_nvar   (xi, eta, mode)  window mean of H f_n²          (= ∫ (f_n φn)² dz)
    <f>_bcvar  (xi, eta)        window mean of ∫ f_bc² dz
  Σ_n <f>_nvar / <f>_bcvar = share of the baroclinic variance held by modes 1..NMODE (≤ 1, orthogonality of φn).
Needs wave_bands.jl (wave_grid, BANDS), mode_projection.jl (u2rho!, v2rho!), meanflow_window.jl, wave_perturb.jl.

Output: mean_root/<f>_n_<band>/<f>_n_<band>_win<NNN>_<t0>.nc, per field, band and window:
  <f>_n (xi_rho, eta_rho, mode, time) [m s-1 or Pa], <f>_nvar, <f>_bcvar, ocean_time, t_start, t_end, nrec.
  ~0.55 GB per field, band and window for 10 modes (~50 GB for u, v, p x 3 bands x 10 windows).
=#

# step 8: f_bc = f_wave − f_bt on every level (x: (xi, eta, N) one record, bt: (xi, eta));
# for p, bt is passed as p_bt − ρ0 g zeta_wave (= depth mean of p_wave), see the header
bc_record!(xbc, x, bt) = (xbc .= x .- bt; xbc)

# step 9: projections of one record's f_bc (rho grid) on the window's φ: F (xi, eta, mode); also ∫ f_bc² dz -> fbc2
function project_bc!(F, fbc2, xbc, ζ, h, dz0, φ)
    nx, ny, N = size(xbc); nm = size(φ, 4)
    @inbounds for j in 1:ny, i in 1:nx
        H = ζ[i, j] + h[i, j]; s2 = 0f0
        for k in 1:N; s2 += xbc[i, j, k]^2 * H * dz0[i, j, k]; end
        fbc2[i, j] = s2
        for n in 1:nm
            a = 0f0
            for k in 1:N; a += xbc[i, j, k] * H * dz0[i, j, k] * φ[i, j, k, n]; end
            F[i, j, n] = a / H
        end
    end
    return F
end

function window_modal(win, data_folder, mean_root; field, band, grid_file, nmode = 10, windows = 1:win.nwin,
                      t_ref = DateTime(1994, 1, 1), verbose = true)
    td = win.td
    own = wave_owner(win)
    recs_of = [findall(==(w), own) for w in 1:win.nwin]
    ws = [w for w in windows if !isempty(recs_of[w])]
    nw, nb, nout = "$(field)_wave_$(band.name)", "$(field)_bt_$(band.name)", "$(field)_n_$(band.name)"
    nzw = "zeta_wave_$(band.name)"                                          # p: surface part of the total wave pressure
    mtag = "modes$nmode"
    fwin(nm, w) = window_file(joinpath(mean_root, nm), nm, w, win.t0[w])
    done(nm, w; full = true) = window_done(fwin(nm, w), nm, win, w, t_ref;
                                           nrec = full ? length(recs_of[w]) : length(win.idx[w]), t0 = full ? td.tdate[recs_of[w][1]] : win.t0[w])
    todo = [w for w in ws if !done(nout, w)]
    verbose && @printf("%s: %d of %d windows to do\n", nout, length(todo), length(ws))
    isempty(todo) && return [fwin(nout, w) for w in 1:win.nwin]
    for w in todo
        done(nw, w) || error("no $nw for window $w: run steps 5-6 first")
        done(nb, w) || error("no $nb for window $w: run step 7 first")
        field == "p" && !done(nzw, w) && error("no $nzw for window $w: run steps 5-6 for zeta first")
        window_done(fwin(mtag, w), "c", win, w, t_ref) || error("no $mtag for window $w: run step 4 with nmode = $nmode")
    end

    G = wave_grid(win, data_folder, mean_root, "rho", grid_file)            # rho grid: h, Δz0, ζ(t)
    nx, ny, N = size(G.dz0)
    sz_in = NCDataset(ds -> size(ds[nw])[1:3], fwin(nw, todo[1]))           # the field's own grid
    x = Array{Float32}(undef, sz_in...); xbc = similar(x); bt = Array{Float32}(undef, sz_in[1:2]...)
    rho0 = NCDataset(ds -> Float32(ds.attrib["rho0"]), joinpath(data_folder, td.files[1])); g = 9.81f0
    zw = field == "p" ? Array{Float32}(undef, sz_in[1:2]...) : nothing
    xr = field == "p" ? xbc : Array{Float32}(undef, nx, ny, N)              # f_bc on rho points
    F = Array{Float32}(undef, nx, ny, nmode); fbc2 = Array{Float32}(undef, nx, ny)
    units = field == "p" ? "Pa" : "meter second-1"
    tunits = "seconds since " * Dates.format(t_ref, "yyyy-mm-dd HH:MM:SS")
    sec(t) = Dates.value(Millisecond(t - t_ref)) / 1000

    t0 = time()
    for w in todo
        rs = recs_of[w]; fo = fwin(nout, w)
        mkpath(dirname(fo)); rm(fo * ".tmp"; force = true)
        φ = NCDataset(d -> d["phi"].var[:, :, :, :], fwin(mtag, w))
        size(φ, 4) == nmode || error("$(fwin(mtag, w)) has $(size(φ, 4)) modes, nmode = $nmode")
        nvar = zeros(Float64, nx, ny, nmode); bcvar = zeros(Float64, nx, ny)
        din = NCDataset(fwin(nw, w)); dbt = NCDataset(fwin(nb, w)); dzw = field == "p" ? NCDataset(fwin(nzw, w)) : nothing
        try
            NCDataset(fo * ".tmp", "c"; attrib = ["title" => "$nout: $(field)_bc = $nw $(field == "p" ? "+ rho0 g $nzw " : "")− $nb projected on the $nmode vertical modes of the window",
                                                 "modes_file" => fwin(mtag, w), "window" => w, "nmode" => nmode,
                                                 "created" => string(now())]) do ds
                defDim(ds, "xi_rho", nx); defDim(ds, "eta_rho", ny); defDim(ds, "mode", nmode)
                defDim(ds, "time", length(rs)); defDim(ds, "one", 1)
                v = defVar(ds, "$(field)_n", Float32, ("xi_rho", "eta_rho", "mode", "time"); chunksizes = [nx, ny, 1, 1],
                           attrib = ["units" => units, "long_name" => "mode-n amplitude of $(field)_bc ($(band.name)): (1/H) int $(field)_bc phi_n dz" *
                                     (field == "p" ? "" : ", along grid $(field == "u" ? "xi" : "eta"), at rho points")])
                defVar(ds, "ocean_time", sec.(td.tdate[rs]), ("time",); attrib = ["units" => tunits, "long_name" => "record time"])
                defVar(ds, "t_start", [sec(td.tdate[rs[1]])], ("one",); attrib = ["units" => tunits])
                defVar(ds, "t_end", [sec(td.tdate[rs[end]])], ("one",); attrib = ["units" => tunits])
                for (i, n) in enumerate(rs)
                    NCDatasets.load!(din[nw].var, x, :, :, :, i); NCDatasets.load!(dbt[nb].var, bt, :, :, i)
                    if field == "p"                                                            # p_bt − ρ0 g zeta_wave
                        NCDatasets.load!(dzw[nzw].var, zw, :, :, i); bt .-= rho0 * g .* zw
                    end
                    bc_record!(xbc, x, bt)                                                     # step 8
                    field == "u" ? u2rho!(xr, xbc) : field == "v" ? v2rho!(xr, xbc) : nothing
                    project_bc!(F, fbc2, xr, view(G.ζ, :, :, n), G.h, G.dz0, φ)                 # step 9
                    for m in 1:nmode
                        v[:, :, m, i] = F[:, :, m]
                        @views nvar[:, :, m] .+= (G.ζ[:, :, n] .+ G.h) .* F[:, :, m] .^ 2
                    end
                    bcvar .+= fbc2
                end
                nr = length(rs)
                defVar(ds, "$(field)_nvar", Float32.(nvar ./ nr), ("xi_rho", "eta_rho", "mode");
                       attrib = ["long_name" => "window mean of H $(field)_n^2 (mode-n share of the baroclinic variance, depth-integrated)",
                                 "units" => field == "p" ? "Pa2 m" : "m3 s-2"])
                defVar(ds, "$(field)_bcvar", Float32.(bcvar ./ nr), ("xi_rho", "eta_rho");
                       attrib = ["long_name" => "window mean of int $(field)_bc^2 dz (all baroclinic variance, depth-integrated)",
                                 "units" => field == "p" ? "Pa2 m" : "m3 s-2"])
                defVar(ds, "nrec", Int32[nr], ("one",); attrib = ["long_name" => "records in this file"])
            end
        finally
            close(din); close(dbt); dzw === nothing || close(dzw)
        end
        mv(fo * ".tmp", fo; force = true)
        verbose && @printf("  %s window %d: %d records, modes 1-%d hold %.1f%% of the baroclinic variance (mode 1 %.1f%%), %.1f min\n",
                           nout, w, length(rs), nmode, 100sum(nvar) / sum(bcvar), 100sum(nvar[:, :, 1]) / sum(bcvar), (time() - t0) / 60)
        φ = nothing; GC.gc()
    end
    return [fwin(nout, w) for w in 1:win.nwin]
end

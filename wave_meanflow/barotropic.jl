#= barotropic.jl
Step 7: barotropic part (depth mean, mode 0) of each band of the wave fields, every hourly record.
  u_bt = (1/H) ∫ u_wave dz,   v_bt = (1/H) ∫ v_wave dz,   on the u / v grid
  p_bt = ρ0 g zeta_wave + (1/H) ∫ p_wave dz                on the rho grid
  with Hz = (ζ(t) + h) Δz0 and H = h + ζ(t) of the record (Σ Hz = H), ζ(t) the full his zeta on the field's grid,
  per band: u_bt_hp28 from u_wave_hp28, p_bt_bp9_15 from p_wave_bp9_15 and zeta_wave_bp9_15, ...
  p_wave holds only the internal pressure of rho_wave (zero at the surface); ρ0 g zeta_wave adds the surface
  (free-surface) pressure, which is depth-uniform, so it is barotropic. ρ0 = rho0 of the his files (1027.5).
The baroclinic part is not saved: u_bc = u_wave − u_bt (same for v, p), computed when needed (one subtraction).
The bands were filtered before the depth mean; since Hz changes with ζ(t) the two do not commute exactly
(negligible in deep water, small on the shelf where ζ swings ~2 m).
Needs wave_bands.jl (wave_grid, BANDS), meanflow_window.jl, wave_perturb.jl (wave_owner); the band files of step 6.

Output: mean_root/<field>_bt_<band>/<field>_bt_<band>_win<NNN>_<t0>.nc, (xi, eta, time) per window, all its records.
Read one 3-D record at a time (~98 MB), so memory stays small; written under .tmp, renamed when complete, skipped on rerun.
=#

his_var(field) = field == "p" ? "rho" : field          # the his variable whose grid / ζ the field uses

function window_bt(win, data_folder, mean_root; field, grid_file, bands = BANDS, t_ref = DateTime(1994, 1, 1), verbose = true)
    td = win.td
    own = wave_owner(win)
    recs_of = [findall(==(w), own) for w in 1:win.nwin]
    ws = [w for w in 1:win.nwin if !isempty(recs_of[w])]
    nin(b) = "$(field)_wave_$(b.name)"; nout(b) = "$(field)_bt_$(b.name)"; nz(b) = "zeta_wave_$(b.name)"
    fwin(nm, w) = window_file(joinpath(mean_root, nm), nm, w, win.t0[w])
    done(nm, w) = window_done(fwin(nm, w), nm, win, w, t_ref; nrec = length(recs_of[w]), t0 = td.tdate[recs_of[w][1]])
    todo = [(b, w) for b in bands for w in ws if !done(nout(b), w)]
    verbose && @printf("%s_bt: %d of %d band-windows to do\n", field, length(todo), length(bands) * length(ws))
    isempty(todo) && return Dict(nout(b) => [fwin(nout(b), w) for w in 1:win.nwin] for b in bands)
    for (b, w) in todo
        done(nin(b), w) || error("no $(nin(b)) for window $w: run steps 5-6 first")
        field == "p" && !done(nz(b), w) && error("no $(nz(b)) for window $w: run steps 5-6 for zeta first")
    end

    G = wave_grid(win, data_folder, mean_root, his_var(field), grid_file)   # h, Δz0, ζ(t) on the field's grid
    rho0 = NCDataset(ds -> Float32(ds.attrib["rho0"]), joinpath(data_folder, td.files[1]))
    g = 9.81f0
    nx, ny, N = size(G.dz0)
    dn = G.dnames[1:2]
    tunits = "seconds since " * Dates.format(t_ref, "yyyy-mm-dd HH:MM:SS")
    sec(t) = Dates.value(Millisecond(t - t_ref)) / 1000
    units = field == "p" ? "Pa" : "meter second-1"
    x = Array{Float32}(undef, nx, ny, N); bt = Array{Float32}(undef, nx, ny)

    t0 = time()
    for (b, w) in todo
        rs = recs_of[w]; fo = fwin(nout(b), w)
        rm(fo * ".tmp"; force = true)
        din = NCDataset(fwin(nin(b), w)); dz = field == "p" ? NCDataset(fwin(nz(b), w)) : nothing
        try
            NCDataset(fo * ".tmp", "c"; attrib = ["title" => "$(nout(b)): barotropic (depth-mean) part of $(nin(b))" *
                                                            (field == "p" ? " plus rho0 g $(nz(b))" : ""),
                                                 "rho0" => rho0, "window" => w, "created" => string(now())]) do ds
                defDim(ds, dn[1], nx); defDim(ds, dn[2], ny); defDim(ds, "time", length(rs)); defDim(ds, "one", 1)
                v = defVar(ds, nout(b), Float32, (dn..., "time"); chunksizes = [nx, ny, 1],
                           attrib = ["units" => units, "long_name" => "depth mean of $(nin(b))" *
                                     (field == "p" ? " + rho0 g $(nz(b))" : "") * " (mode 0)"])
                defVar(ds, "ocean_time", sec.(td.tdate[rs]), ("time",); attrib = ["units" => tunits, "long_name" => "record time"])
                defVar(ds, "t_start", [sec(td.tdate[rs[1]])], ("one",); attrib = ["units" => tunits])
                defVar(ds, "t_end", [sec(td.tdate[rs[end]])], ("one",); attrib = ["units" => tunits])
                defVar(ds, "nrec", Int32[length(rs)], ("one",); attrib = ["long_name" => "records in this file"])
                for (i, n) in enumerate(rs)
                    NCDatasets.load!(din[nin(b)].var, x, :, :, :, i)
                    @inbounds for jj in 1:ny, ii in 1:nx
                        H = G.ζ[ii, jj, n] + G.h[ii, jj]; s = 0f0
                        for k in 1:N; s += x[ii, jj, k] * H * G.dz0[ii, jj, k]; end
                        bt[ii, jj] = s / H
                    end
                    field == "p" && (bt .+= rho0 * g .* dz[nz(b)].var[:, :, i])
                    v[:, :, i] = bt
                end
            end
        finally
            close(din); dz === nothing || close(dz)
        end
        mv(fo * ".tmp", fo; force = true)
        verbose && @printf("  %s window %d: %d records, %.1f min\n", nout(b), w, length(rs), (time() - t0) / 60)
    end
    return Dict(nout(b) => [fwin(nout(b), w) for w in 1:win.nwin] for b in bands)
end

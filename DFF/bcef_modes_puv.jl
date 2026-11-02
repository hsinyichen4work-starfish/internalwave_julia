using Distributed

# Baroclinic pressure and velocity projected on the first NMODE vertical modes (default 3), pn, un, vn [xi, eta, time],
# from the ROMS his files — the input for the directional Fourier filter of the energy flux in DFF_on_bcef.jl.
# Generalization of the mode-1 script (bcef_mode1_puv.jl); NMODE = 1 reproduces it.
#
# Why modes: the DFF needs fields that are linear in the waves (north + south waves add), like ssh. The energy flux
# p'u' is a product, so it cannot be filtered itself. Instead p', u', v' are filtered and the flux is formed from each
# direction's part. Projecting p', u', v' on the vertical modes gives 2-D fields per mode, filtered exactly like ssh:
#   p'(z, t) ≈ Σn pn(t) φn(z),   u'(z, t) ≈ Σn un(t) φn(z),   v'(z, t) ≈ Σn vn(t) φn(z)
#   with (1/H) ∫ φm φn dz = δmn (Siyanbola et al. 2024, Eqs. 10-15), so the depth-integrated flux is
#   F = ∫ p'u' dz = Σn H pn un   [W m-1]   — no mode-to-mode cross terms (orthogonality)
#
# p', u', v' are made as in energy_flux/model_check_energyflux.jl (energyflux_piece): rho' relative to the time-mean
# density at a fixed depth ("fixed_z"), hydrostatic p from rho' integrated down from the free surface, then the
# thickness-weighted depth mean removed from p, u, v. The mean density is over the whole record.
#
# Two passes over the his files (each ~1.2 GB of rho per file; the passes run on the workers, one file at a time):
#   pass 1  time-mean rho and zeta -> mean depths of the levels, N²(z), and the mode structures φn(z), cn at every
#           point (bcef_modes<NMODE>_structure.nc; kept, so a rerun skips this pass)
#   pass 2  every hour: p', u', v' -> projections pn, un, vn, written to bcef_modes<NMODE>_puv.nc batch by batch
#           (the main process never holds the whole record, so memory does not grow with NMODE or record length)
# Output variables p1, u1, v1, p2, u2, v2, ... [xi, eta, time] — same names and layout as the mode-1 file, so
# DFF_on_bcef.jl can read mode 1 unchanged and mode n by changing the variable name.
# Also saved: time sums of H pn² per mode and of ∫p'² dz at every point, for maps of each mode's share of p' variance.
# u, v, un, vn are along the grid's xi / eta directions (not rotated to east / north), like the saved fluxes.
#
# Settings (environment): NMODE (default 3), BCEF_OUT (output folder), BCEF_TEST=true (first 8 files only).

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "3")) - 1)   # -1 reserves a CPU for this main process
job_mem_mb = parse(Int, get(ENV, "SLURM_MEM_PER_NODE", "16000"))
# the main process now holds only one batch of results (pass 2 writes as it goes) — leave it 4 GB.
# each worker holds the structure (rho_mean, z_r_mean, φ for NMODE modes: ~(2 + NMODE) x 0.3 GB) plus one his record's 3-D fields
heap_mb = floor(Int, max(job_mem_mb - 4000, 4000) * 0.85 / n_workers)
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # several processes read the same his files; workers inherit this
nprocs() == 1 && addprocs(n_workers; exeflags = ["--project=$(Base.active_project())", "--heap-size-hint=$(heap_mb)M"])
println("running with $(nprocs() - 1) worker processes, heap size hint $(heap_mb) MB each")

@everywhere begin
    using NCDatasets, Dates, Statistics, LinearAlgebra
    include("/home/hchen54/internalwave_julia/function/uv2rho.jl")   # u2rho, v2rho
    include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")       # dff_run, run_dir, ddf_dir, fig_dir (set by DFF_RUN)

    const NMODE = parse(Int, get(ENV, "NMODE", "3"))
    grid_fname = "/expanse/lustre/projects/uso102/hchen54/input/grid/roms_grd_900m.nc"
    datadir = joinpath(run_dir, "after_joint")
    # output folder; BCEF_OUT overrides it (e.g. for a test run)
    out_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, "ddf_bcef"))
    # new file names (not the mode-1 ones), so the mode-1 structure file is never picked up by mistake
    struct_fname = joinpath(out_dir, "bcef_modes$(NMODE)_structure.nc")
    puv_fname = joinpath(out_dir, "bcef_modes$(NMODE)_puv.nc")

    his_files = joinpath.(datadir, sort(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(datadir))))
    # BCEF_TEST=true: only the first 8 files (32 hours) — for checking the script, not for the analysis
    get(ENV, "BCEF_TEST", "false") == "true" && (his_files = his_files[1:8])

    h = NCDataset(ds -> Float64.(ds["h"][:, :]), grid_fname)
    nx, ny = size(h)
    # vertical grid, as in energyflux_piece (set_depth.F, Vtransform 2): z = zeta + (zeta + h) * z0, with
    # z0 = (hc*sc + Cs*h) / (h + hc) fixed in time. k = 1 is the bottom level, nz the surface one.
    nz, hc, Cs_r, Cs_w, rho0 = NCDataset(his_files[1]) do ds
        ds.dim["s_rho"], ds.attrib["hc"], Float64.(ds.attrib["Cs_r"]), Float64.(ds.attrib["Cs_w"]), Float64(ds.attrib["rho0"])
    end
    NMODE <= nz - 1 || error("NMODE = $NMODE but only $(nz - 1) interior interfaces")
    sc_r = ((1:nz) .- nz .- 0.5) ./ nz
    sc_w = ((0:nz) .- nz) ./ nz
    z0_r = Float32.((hc .* reshape(sc_r, 1, 1, :) .+ reshape(Cs_r, 1, 1, :) .* h) ./ (h .+ hc))   # [xi, eta, s]
    z0_w = Float32.((hc .* reshape(sc_w, 1, 1, :) .+ reshape(Cs_w, 1, 1, :) .* h) ./ (h .+ hc))   # [xi, eta, s_w]
    h32 = Float32.(h)
    zlev(z0, zeta) = zeta .+ (zeta .+ h32) .* z0   # depths [xi, eta, s] for one snapshot zeta [xi, eta]
    g = 9.81f0

    # mean profile rho_mean (on the mean level depths z_mean, both [xi, eta, s]) looked up at the depths z[xi, eta, s]
    # by linear interpolation in each column; outside the mean profile the end value is held.
    # Same as rho_at_depth in model_check_energyflux.jl, for one snapshot.
    function rho_at_depth(z_mean, rho_mean, z)
        nx, ny, nz = size(z)
        out = similar(z)
        @inbounds for j in 1:ny, i in 1:nx
            m = 1
            for k in 1:nz
                zz = z[i, j, k]
                while m < nz - 1 && z_mean[i, j, m+1] < zz
                    m += 1
                end
                w = clamp((zz - z_mean[i, j, m]) / (z_mean[i, j, m+1] - z_mean[i, j, m]), 0f0, 1f0)
                out[i, j, k] = rho_mean[i, j, m] + w * (rho_mean[i, j, m+1] - rho_mean[i, j, m])
            end
        end
        out
    end

    # pass 1 on a worker: sums of rho [xi, eta, s] and zeta [xi, eta] over the records of some files (Float64)
    function mean_sums(files)
        rs, zs, n = zeros(nx, ny, nz), zeros(nx, ny), 0
        for f in files
            NCDataset(f) do ds
                for r in 1:ds.dim["time"]
                    rs .+= ds["rho"][:, :, :, r]
                    zs .+= ds["zeta"][:, :, r]
                    n += 1
                end
            end
        end
        rs, zs, n
    end

    # pass 2 on a worker: pn, un, vn [xi, eta, record, mode] for every record of one his file, plus time sums at every
    # point for the variance check: Σ H pn² [xi, eta, mode] and Σ ∫p'² dz [xi, eta]
    function project_file(f)   # uses this worker's S (read once in section 2)
        NCDataset(f) do ds
            nr = ds.dim["time"]
            pm, um, vm = (zeros(Float32, nx, ny, nr, NMODE) for _ in 1:3)
            emap = zeros(nx, ny, NMODE)   # Σ over records of H pn²
            tmap = zeros(nx, ny)          # Σ over records of ∫ p'² dz
            for r in 1:nr
                zeta = Float32.(ds["zeta"][:, :, r])
                z_r = zlev(S.z0_r, zeta)
                Hz = diff(zlev(S.z0_w, zeta), dims = 3)   # layer thickness [xi, eta, s]
                H = sum(Hz, dims = 3)                      # h + zeta  [xi, eta, 1]
                # rho' at a fixed depth, then hydrostatic p' integrated down from the surface (p = 0 at z = zeta)
                rho_p = Float32.(ds["rho"][:, :, :, r]) .- rho_at_depth(S.z_r_mean, S.rho_mean, z_r)
                p = similar(rho_p)
                p_top = zeros(Float32, nx, ny)
                for k in nz:-1:1
                    @views dp = g .* rho_p[:, :, k] .* Hz[:, :, k]
                    @views p[:, :, k] .= p_top .+ 0.5f0 .* dp
                    p_top .+= dp
                end
                rho_p = nothing
                u = u2rho(Float32.(ds["u"][:, :, :, r]))
                v = v2rho(Float32.(ds["v"][:, :, :, r]))
                # baroclinic parts (depth mean removed), then the projection on each φn: (1/H) ∫ f' φn dz  (Eq. 15)
                for (fld, out) in ((p, pm), (u, um), (v, vm))
                    fld .-= sum(fld .* Hz, dims = 3) ./ H
                    for n in 1:NMODE
                        @views out[:, :, r, n] .= dropdims(sum(fld .* S.φ[:, :, :, n] .* Hz, dims = 3) ./ H, dims = 3)
                    end
                end
                for n in 1:NMODE
                    @views emap[:, :, n] .+= dropdims(H, dims = 3) .* pm[:, :, r, n] .^ 2
                end
                tmap .+= dropdims(sum(p .^ 2 .* Hz, dims = 3), dims = 3)   # p is p' here (depth mean removed above)
            end
            pm, um, vm, emap, tmap
        end
    end
end   # @everywhere

mkpath(out_dir)
ocean_time = DateTime(1994, 1, 1) .+ Second.(round.(Int, reduce(vcat, [NCDataset(ds -> Float64.(ds["ocean_time"][:]), f) for f in his_files])))   # seconds since 1994-01-01 -> DateTime
nt = length(ocean_time)
println("$dff_run: $(length(his_files)) his files, $nt records, $(ocean_time[1]) -> $(ocean_time[end]); grid $nx x $ny x $nz; $NMODE modes")
all(==(Hour(1)), diff(ocean_time)) || @warn "records are not hourly and evenly spaced"
# what the run found: printed, and written to bcef_modes<NMODE>_analysis.txt at the end
report_fname = joinpath(out_dir, "bcef_modes$(NMODE)_analysis.txt")
report = ["$dff_run: $nt records, $(ocean_time[1]) -> $(ocean_time[end]); grid $nx x $ny x $nz; $NMODE modes"]
say(line) = (println(line); push!(report, line))
rng(lo_hi) = round.(Float64.(Tuple(lo_hi)), digits = 3)   # (min, max) for the report

# ============================================================================
# (1) pass 1: time-mean rho and zeta, then N² and the first NMODE mode structures at every point
# ============================================================================
# Modes of  W'' + (N² / c²) W = 0,  W = 0 at the bottom and the surface (rigid lid), on the mean level depths
# (Siyanbola et al. 2024 Eq. 10; Gerkema Ch. 5.1, hydrostatic 5.1.3):
# W on the interfaces, second differences over the layers, N² from the mean density between neighbouring levels.
# A W = λ B W with B = diag(N² Δz) is made symmetric by y = B^½ W, then the NMODE smallest eigenvalues λn = 1/cn².
# The pressure / horizontal velocity structure is φn = dWn/dz on the rho levels (Eq. 11), normalized so that
# (1/H) ∫ φn² dz = 1 (Eq. 12), with φn(surface) > 0. The discrete φn are exactly orthogonal: Σ φm φn Hz = Wmᵀ A Wn = 0.
# N² is held at ≥ 1e-7 s⁻² so a mixed (or slightly unstable) layer does not break the problem.
# Checked against constant N on a stretched grid: cn = N H / (nπ), φn = √2 cos(nπ z / H), Gram matrix = identity.
function modes(zr, zw, rho; nmode = NMODE, N2min = 1e-7)
    nz = length(zr)
    Hz = diff(zw)
    dzc = diff(zr)
    N2 = max.(-9.81 / rho0 .* diff(rho) ./ dzc, N2min)   # [s⁻²] on the interior interfaces 2..nz
    B = N2 .* dzc
    d = (1 ./ Hz[2:nz] .+ 1 ./ Hz[1:nz-1]) ./ B
    e = (-1 ./ Hz[2:nz-1]) ./ sqrt.(B[1:end-1] .* B[2:end])
    F = eigen(SymTridiagonal(d, e), 1:nmode)             # the nmode smallest eigenvalues, increasing
    c = 1 ./ sqrt.(F.values)                             # c1 > c2 > c3 ...
    φ = zeros(nz, nmode)
    for n in 1:nmode
        W = [0; F.vectors[:, n] ./ sqrt.(B); 0]          # Wn on the interfaces, 0 at bottom and surface
        φn = diff(W) ./ Hz                               # Eq. (11)
        φn ./= sqrt(sum(φn .^ 2 .* Hz) / sum(Hz))        # Eq. (12), normalized
        φn[end] < 0 && (φn .*= -1)                       # sign convention: positive at the surface
        φ[:, n] .= φn
    end
    c, φ, N2
end

if isfile(struct_fname)
    println("mode structure: using $struct_fname (delete it to recompute)")
else
    blocks = collect(Iterators.partition(his_files, cld(length(his_files), nworkers())))
    @time sums = pmap(mean_sums, blocks)
    n_rec = sum(s[3] for s in sums)
    n_rec == nt || error("pass 1 read $n_rec records, expected $nt")
    rho_mean = Float32.(sum(s[1] for s in sums) ./ n_rec)
    zeta_mean = Float32.(sum(s[2] for s in sums) ./ n_rec)
    sums = nothing
    # the depths are linear in zeta, so the mean depth of each level is its depth at the mean zeta
    z_r_mean, z_w_mean = zlev(z0_r, zeta_mean), zlev(z0_w, zeta_mean)
    # in a function (not a loop at the top level of the script), so Julia compiles it for the actual array types
    function all_modes(z_r_mean, z_w_mean, rho_mean)
        φ = zeros(Float32, nx, ny, nz, NMODE)
        c = zeros(Float32, nx, ny, NMODE)
        N2 = zeros(Float32, nx, ny, nz - 1)
        for j in 1:ny, i in 1:nx
            cc, ph, n2 = modes(Float64.(z_r_mean[i, j, :]), Float64.(z_w_mean[i, j, :]), Float64.(rho_mean[i, j, :]))
            c[i, j, :] .= cc
            φ[i, j, :, :] .= ph
            N2[i, j, :] .= n2
        end
        φ, c, N2
    end
    @time φ, c, N2 = all_modes(z_r_mean, z_w_mean, rho_mean)
    # orthogonality check on a sample of deep-enough points: (1/H) ∫ φm φn dz should be the identity matrix
    let worst = 0.0
        Hz_mean = diff(z_w_mean, dims = 3)
        for j in 1:25:ny, i in 1:25:nx
            h[i, j] > 50 || continue
            hz = Float64.(Hz_mean[i, j, :])
            P = Float64.(φ[i, j, :, :])
            G = (P .* hz)' * P ./ sum(hz)
            worst = max(worst, maximum(abs.(G - I)))
        end
        println("orthogonality: max |(1/H) ∫ φm φn dz - δmn| over sampled points = $(round(worst, sigdigits = 2))")
    end
    NCDataset(struct_fname * ".tmp", "c") do ds
        defDim(ds, "xi_rho", nx); defDim(ds, "eta_rho", ny); defDim(ds, "s_rho", nz); defDim(ds, "s_w_interior", nz - 1)
        defDim(ds, "mode", NMODE)
        defVar(ds, "h", h, ("xi_rho", "eta_rho"); attrib = ["units" => "meter"])
        defVar(ds, "zeta_mean", zeta_mean, ("xi_rho", "eta_rho"); attrib = ["units" => "meter"])
        defVar(ds, "z_r_mean", z_r_mean, ("xi_rho", "eta_rho", "s_rho"); attrib = ["long_name" => "mean depth of the rho levels", "units" => "meter"])
        defVar(ds, "rho_mean", rho_mean, ("xi_rho", "eta_rho", "s_rho"); attrib = ["long_name" => "time-mean density anomaly", "units" => "kg m-3"])
        defVar(ds, "N2", N2, ("xi_rho", "eta_rho", "s_w_interior"); attrib = ["long_name" => "buoyancy frequency squared from the mean density, interior interfaces, held >= 1e-7", "units" => "s-2"])
        defVar(ds, "phi", φ, ("xi_rho", "eta_rho", "s_rho", "mode"); attrib = ["long_name" => "mode structures of p and u on the rho levels, (1/H) int phi_m phi_n dz = delta_mn, positive at the surface"])
        defVar(ds, "c", c, ("xi_rho", "eta_rho", "mode"); attrib = ["long_name" => "mode eigenspeeds (no rotation)", "units" => "m s-1"])
        ds.attrib["source"] = datadir
        ds.attrib["records"] = "$nt records, $(ocean_time[1]) -> $(ocean_time[end])"
        ds.attrib["nmode"] = NMODE
    end
    mv(struct_fname * ".tmp", struct_fname; force = true)
    println("wrote $struct_fname")
end

# from the structure file, so a rerun that skips pass 1 reports them too
let c = NCDataset(ds -> ds["c"][:, :, :], struct_fname)
    for n in 1:NMODE
        cn = c[:, :, n]
        say("mode-$n speed c$n: $(round.(Float64.(extrema(cn)), digits = 2)) m/s, median $(round(median(cn), digits = 2)) m/s; " *
            "over h > 1000 m median $(round(median(cn[h .> 1000]), digits = 2)) m/s")
    end
end

# ============================================================================
# (2) pass 2: pn, un, vn every hour, written batch by batch
# ============================================================================
# The workers read the structure file once (mean density, mean level depths, φ for all modes), then project one his
# file at a time. The main process writes each batch's records into the output file at once and frees them.
@everywhere S = NCDataset($struct_fname) do ds
    ds.attrib["nmode"] == NMODE || error("$(basename(struct_fname)) has $(ds.attrib["nmode"]) modes, NMODE = $NMODE")
    (rho_mean = ds["rho_mean"][:, :, :], z_r_mean = ds["z_r_mean"][:, :, :], φ = ds["phi"][:, :, :, :], z0_r = z0_r, z0_w = z0_w)
end

rec_end = cumsum([NCDataset(ds -> ds.dim["time"], f) for f in his_files])
emap_sum = zeros(nx, ny, NMODE)   # Σ H pn² over all records, every point
tmap_sum = zeros(nx, ny)          # Σ ∫ p'² dz over all records, every point
var_range = Dict("$v$n" => [Inf32, -Inf32] for v in ("p", "u", "v") for n in 1:NMODE)   # [min, max] of each output variable
batch = 4 * nworkers()            # files per batch: one batch of results (~30 MB per file and mode) is held, then freed

isfile(puv_fname) && rm(puv_fname)
ds = NCDataset(puv_fname * ".tmp", "c")
try
    defDim(ds, "xi_rho", nx); defDim(ds, "eta_rho", ny); defDim(ds, "time", nt); defDim(ds, "mode", NMODE)
    defVar(ds, "ocean_time", ocean_time, ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
    for n in 1:NMODE
        for (name, ln, un) in (("p$n", "mode-$n amplitude of the baroclinic pressure p'", "Pa"),
                               ("u$n", "mode-$n amplitude of the baroclinic velocity u' along xi", "m s-1"),
                               ("v$n", "mode-$n amplitude of the baroclinic velocity v' along eta", "m s-1"))
            defVar(ds, name, Float32, ("xi_rho", "eta_rho", "time"); chunksizes = [nx, ny, 1],
                   attrib = ["long_name" => ln, "units" => un])
        end
    end

    @time for b in Iterators.partition(eachindex(his_files), batch)
        res = pmap(k -> project_file(his_files[k]), b)
        for (k, (pm, um, vm, emap, tmap)) in zip(b, res)
            idx = rec_end[k]-size(pm, 3)+1:rec_end[k]
            for n in 1:NMODE
                ds["p$n"][:, :, idx] = pm[:, :, :, n]
                ds["u$n"][:, :, idx] = um[:, :, :, n]
                ds["v$n"][:, :, idx] = vm[:, :, :, n]
                for (v, a) in (("p", pm), ("u", um), ("v", vm))
                    lo, hi = extrema(@view a[:, :, :, n])
                    var_range["$v$n"] .= (min(var_range["$v$n"][1], lo), max(var_range["$v$n"][2], hi))
                end
            end
            emap_sum .+= emap
            tmap_sum .+= tmap
        end
        res = nothing
        println("  pass 2: $(last(b)) / $(length(his_files)) files")
    end

    # share of the baroclinic pressure variance in each mode (all frequencies), whole domain and at every point
    ok = isfinite.(tmap_sum) .& (tmap_sum .> 0)
    for n in 1:NMODE
        en = emap_sum[:, :, n]
        say("mode $n holds $(round(100 * sum(en[ok]) / sum(tmap_sum[ok]), digits = 1))% of ∫p'² dz (all frequencies, whole domain and record)")
    end
    say("modes 1-$NMODE together hold $(round(100 * sum(sum(emap_sum[:, :, n][ok]) for n in 1:NMODE) / sum(tmap_sum[ok]), digits = 1))% of ∫p'² dz")
    for n in 1:NMODE
        say("p$n: $(rng(var_range["p$n"])) Pa, u$n: $(rng(var_range["u$n"])) m/s, v$n: $(rng(var_range["v$n"])) m/s")
    end
    defVar(ds, "pvar_mode", Float32.(emap_sum ./ nt), ("xi_rho", "eta_rho", "mode");
           attrib = ["long_name" => "time mean of H pn^2 (mode-n baroclinic pressure variance, depth-integrated)", "units" => "Pa2 m"])
    defVar(ds, "pvar_total", Float32.(tmap_sum ./ nt), ("xi_rho", "eta_rho");
           attrib = ["long_name" => "time mean of int p'^2 dz (all baroclinic pressure variance, depth-integrated)", "units" => "Pa2 m"])

    ds.attrib["note"] = "f'(z, t) ~ sum_n fn(t) phi_n(z) with phi in $(basename(struct_fname)); depth-integrated flux = sum_n H pn un " *
                        "(H = h + zeta), no cross-mode terms. p' from rho' at a fixed depth (record-mean density), depth mean removed, as in " *
                        "energy_flux/model_check_energyflux.jl. un, vn along grid xi / eta. pvar_mode / pvar_total = share of p' variance per mode."
    ds.attrib["source"] = datadir
    ds.attrib["nmode"] = NMODE
finally
    close(ds)
end
mv(puv_fname * ".tmp", puv_fname; force = true)
println("wrote $puv_fname")
open(io -> foreach(l -> println(io, l), report), report_fname, "w")
println("wrote $report_fname")

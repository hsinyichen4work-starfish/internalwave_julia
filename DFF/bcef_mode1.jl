using Distributed

# Mode-1 baroclinic pressure and velocity, p1, u1, v1 [xi, eta, time], from the ROMS his files — the input for the
# directional Fourier filter of the energy flux in DFF_on_bcef.jl.
#
# Why mode 1: the DFF needs fields that are linear in the waves (north + south waves add), like ssh. The energy flux
# p'u' is a product, so it cannot be filtered itself. Instead p', u', v' are filtered and the flux is formed from each
# direction's part. Filtering p', u', v' on all 128 levels would be ~400 DFF runs; projecting them on the first vertical
# mode gives three 2-D fields, filtered exactly like ssh:
#   p'(z, t) ≈ p1(t) φ1(z),   u'(z, t) ≈ u1(t) φ1(z),   v'(z, t) ≈ v1(t) φ1(z)
#   with (1/H) ∫ φ1² dz = 1, so the depth-integrated mode-1 flux is  F = ∫ p'u' dz = H p1 u1  [W m-1]
#
# p', u', v' are made as in energy_flux/model_check_energyflux.jl (energyflux_piece): rho' relative to the time-mean
# density at a fixed depth ("fixed_z"), hydrostatic p from rho' integrated down from the free surface, then the
# thickness-weighted depth mean removed from p, u, v. One difference: the mean density here is over the whole record,
# not over each 20-day window.
#
# Two passes over the his files (each ~1.2 GB of rho per file; the passes run on the workers, one file at a time):
#   pass 1  time-mean rho and zeta -> mean depths of the levels, N²(z), and the mode-1 structure φ1(z), c1 at every
#           point (bcef_mode1_structure.nc; kept, so a rerun skips this pass)
#   pass 2  every hour: p', u', v' -> projections p1, u1, v1 (bcef_mode1_puv.nc)
# u, v, u1, v1 are along the grid's xi / eta directions (not rotated to east / north), like the saved fluxes.

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "3")) - 1)   # -1 reserves a CPU for this main process
job_mem_mb = parse(Int, get(ENV, "SLURM_MEM_PER_NODE", "16000"))
# the main process holds p1, u1, v1 for the whole record (3 x 5.5 GB for 2353 hours) — leave it 20 GB
heap_mb = floor(Int, max(job_mem_mb - 20000, 4000) * 0.85 / n_workers)
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # several processes read the same his files; workers inherit this
nprocs() == 1 && addprocs(n_workers; exeflags = ["--project=$(Base.active_project())", "--heap-size-hint=$(heap_mb)M"])
println("running with $(nprocs() - 1) worker processes, heap size hint $(heap_mb) MB each")

@everywhere begin
    using NCDatasets, Dates, Statistics, LinearAlgebra
    include("/home/hchen54/internalwave_julia/function/uv2rho.jl")   # u2rho, v2rho
    include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")       # dff_run, run_dir, ddf_dir, fig_dir (set by DFF_RUN)

    grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"
    datadir = joinpath(run_dir, "after_joint")
    # output folder; BCEF_OUT overrides it (e.g. for a test run)
    out_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, "ddf_bcef"))
    struct_fname = joinpath(out_dir, "bcef_mode1_structure.nc")
    puv_fname = joinpath(out_dir, "bcef_mode1_puv.nc")

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

    # pass 2 on a worker: p1, u1, v1 [xi, eta, record] for every record of one his file, plus two sums for the check
    # of how much of p' mode 1 holds: Σ H p1² and Σ ∫p'² dz over the file's points and records
    function project_file(f)   # uses this worker's S (read once in section 2)
        NCDataset(f) do ds
            nr = ds.dim["time"]
            p1, u1, v1 = (zeros(Float32, nx, ny, nr) for _ in 1:3)
            e1, etot = 0.0, 0.0
            for r in 1:nr
                zeta = Float32.(ds["zeta"][:, :, r])
                z_r = zlev(S.z0_r, zeta)
                Hz = diff(zlev(S.z0_w, zeta), dims = 3)   # layer thickness [xi, eta, s]
                H = sum(Hz, dims = 3)                      # h + zeta
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
                # baroclinic parts (depth mean removed), then the projection on φ1: (1/H) ∫ f' φ1 dz
                for (fld, out) in ((p, p1), (u, u1), (v, v1))
                    fld .-= sum(fld .* Hz, dims = 3) ./ H
                    out[:, :, r] .= dropdims(sum(fld .* S.φ1 .* Hz, dims = 3) ./ H, dims = 3)
                end
                e1 += sum(H .* p1[:, :, r] .^ 2)
                etot += sum(p .^ 2 .* Hz)
            end
            p1, u1, v1, e1, etot
        end
    end
end   # @everywhere

mkpath(out_dir)
ocean_time = DateTime(1994, 1, 1) .+ Second.(round.(Int, reduce(vcat, [NCDataset(ds -> Float64.(ds["ocean_time"][:]), f) for f in his_files])))   # seconds since 1994-01-01 -> DateTime
nt = length(ocean_time)
println("$dff_run: $(length(his_files)) his files, $nt records, $(ocean_time[1]) -> $(ocean_time[end]); grid $nx x $ny x $nz")
all(==(Hour(1)), diff(ocean_time)) || @warn "records are not hourly and evenly spaced"

# ============================================================================
# (1) pass 1: time-mean rho and zeta, then N² and the mode-1 structure at every point
# ============================================================================
# Mode 1 of  W'' + (N² / c²) W = 0,  W = 0 at the bottom and the surface (rigid lid), on the mean level depths:
# W on the interfaces, second differences over the layers, N² from the mean density between neighbouring levels.
# This is a symmetric tridiagonal problem after scaling by √(N² Δz), solved for its smallest eigenvalue 1/c1².
# The pressure / horizontal velocity structure is φ1 = dW/dz on the rho levels, (1/H) ∫ φ1² dz = 1, φ1(surface) > 0.
# N² is held at ≥ 1e-7 s⁻² so a mixed (or slightly unstable) layer does not break the problem.
# Checked against constant N: c1 = N H / π and φ1 = √2 cos(π z / H) to 1e-4.
function mode1(zr, zw, rho; N2min = 1e-7)
    nz = length(zr)
    Hz = diff(zw)
    dzc = diff(zr)
    N2 = max.(-9.81 / rho0 .* diff(rho) ./ dzc, N2min)   # [s⁻²] on the interior interfaces 2..nz
    B = N2 .* dzc
    d = (1 ./ Hz[2:nz] .+ 1 ./ Hz[1:nz-1]) ./ B
    e = (-1 ./ Hz[2:nz-1]) ./ sqrt.(B[1:end-1] .* B[2:end])
    F = eigen(SymTridiagonal(d, e), 1:1)
    W = [0; F.vectors[:, 1] ./ sqrt.(B); 0]
    φ = diff(W) ./ Hz
    φ ./= sqrt(sum(φ .^ 2 .* Hz) / sum(Hz))
    φ[end] < 0 && (φ .*= -1)
    1 / sqrt(F.values[1]), φ, N2
end

if isfile(struct_fname)
    println("mode-1 structure: using $struct_fname (delete it to recompute)")
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
        φ1 = zeros(Float32, nx, ny, nz)
        c1 = zeros(Float32, nx, ny)
        N2 = zeros(Float32, nx, ny, nz - 1)
        for j in 1:ny, i in 1:nx
            c, φ, n2 = mode1(Float64.(z_r_mean[i, j, :]), Float64.(z_w_mean[i, j, :]), Float64.(rho_mean[i, j, :]))
            c1[i, j] = c
            φ1[i, j, :] .= φ
            N2[i, j, :] .= n2
        end
        φ1, c1, N2
    end
    @time φ1, c1, N2 = all_modes(z_r_mean, z_w_mean, rho_mean)
    println("mode-1 speed c1: $(round.(extrema(c1), digits = 2)) m/s, median $(round(median(c1), digits = 2)) m/s; ",
            "over h > 1000 m median $(round(median(c1[h .> 1000]), digits = 2)) m/s")
    NCDataset(struct_fname * ".tmp", "c") do ds
        defDim(ds, "xi_rho", nx); defDim(ds, "eta_rho", ny); defDim(ds, "s_rho", nz); defDim(ds, "s_w_interior", nz - 1)
        defVar(ds, "h", h, ("xi_rho", "eta_rho"); attrib = ["units" => "meter"])
        defVar(ds, "zeta_mean", zeta_mean, ("xi_rho", "eta_rho"); attrib = ["units" => "meter"])
        defVar(ds, "z_r_mean", z_r_mean, ("xi_rho", "eta_rho", "s_rho"); attrib = ["long_name" => "mean depth of the rho levels", "units" => "meter"])
        defVar(ds, "rho_mean", rho_mean, ("xi_rho", "eta_rho", "s_rho"); attrib = ["long_name" => "time-mean density anomaly", "units" => "kg m-3"])
        defVar(ds, "N2", N2, ("xi_rho", "eta_rho", "s_w_interior"); attrib = ["long_name" => "buoyancy frequency squared from the mean density, interior interfaces, held >= 1e-7", "units" => "s-2"])
        defVar(ds, "phi1", φ1, ("xi_rho", "eta_rho", "s_rho"); attrib = ["long_name" => "mode-1 structure of p and u on the rho levels, (1/H) int phi1^2 dz = 1, positive at the surface"])
        defVar(ds, "c1", c1, ("xi_rho", "eta_rho"); attrib = ["long_name" => "mode-1 eigenspeed (no rotation)", "units" => "m s-1"])
        ds.attrib["source"] = datadir
        ds.attrib["records"] = "$nt records, $(ocean_time[1]) -> $(ocean_time[end])"
    end
    mv(struct_fname * ".tmp", struct_fname; force = true)
    println("wrote $struct_fname")
end

# ============================================================================
# (2) pass 2: p1, u1, v1 every hour
# ============================================================================
# The workers read the structure file once (mean density, mean level depths, φ1: ~1 GB each), then project one his
# file at a time. The main process collects the results in batches and writes them at the end.
@everywhere S = NCDataset($struct_fname) do ds
    (rho_mean = ds["rho_mean"][:, :, :], z_r_mean = ds["z_r_mean"][:, :, :], φ1 = ds["phi1"][:, :, :], z0_r = z0_r, z0_w = z0_w)
end

p1, u1, v1 = (zeros(Float32, nx, ny, nt) for _ in 1:3)
rec_end = cumsum([NCDataset(ds -> ds.dim["time"], f) for f in his_files])
e1_sum, etot_sum = 0.0, 0.0
batch = 4 * nworkers()   # files per batch: the results of one batch (~30 MB per file) are copied in, then freed
@time for b in Iterators.partition(eachindex(his_files), batch)
    res = pmap(k -> project_file(his_files[k]), b)
    for (k, (a, bb, c, e1, etot)) in zip(b, res)
        idx = rec_end[k]-size(a, 3)+1:rec_end[k]
        p1[:, :, idx] .= a; u1[:, :, idx] .= bb; v1[:, :, idx] .= c
        global e1_sum += e1; global etot_sum += etot
    end
    println("  pass 2: $(last(b)) / $(length(his_files)) files")
end
println("mode 1 holds $(round(100 * e1_sum / etot_sum, digits = 1))% of ∫p'² dz (all frequencies, whole domain and record)")
println("p1: $(extrema(p1)) Pa, u1: $(extrema(u1)) m/s, v1: $(extrema(v1)) m/s")

isfile(puv_fname) && rm(puv_fname)
NCDataset(puv_fname * ".tmp", "c") do ds
    defDim(ds, "xi_rho", nx); defDim(ds, "eta_rho", ny); defDim(ds, "time", nt)
    defVar(ds, "ocean_time", ocean_time, ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
    for (name, a, ln, un) in (("p1", p1, "mode-1 amplitude of the baroclinic pressure p'", "Pa"),
                              ("u1", u1, "mode-1 amplitude of the baroclinic velocity u' along xi", "m s-1"),
                              ("v1", v1, "mode-1 amplitude of the baroclinic velocity v' along eta", "m s-1"))
        defVar(ds, name, a, ("xi_rho", "eta_rho", "time"); chunksizes = [nx, ny, 1],
               attrib = ["long_name" => ln, "units" => un])
    end
    ds.attrib["note"] = "f'(z, t) ~ f1(t) phi1(z) with phi1 in $(basename(struct_fname)); depth-integrated mode-1 flux = H p1 u1 " *
                        "(H = h + zeta). p' from rho' at a fixed depth (record-mean density), depth mean removed, as in " *
                        "energy_flux/model_check_energyflux.jl. u1, v1 along grid xi / eta."
    ds.attrib["source"] = datadir
    ds.attrib["mode1_share_of_p_variance"] = e1_sum / etot_sum
end
mv(puv_fname * ".tmp", puv_fname; force = true)
println("wrote $puv_fname")

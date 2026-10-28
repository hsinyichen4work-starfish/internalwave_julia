using NCDatasets, Dates, Statistics, FFTW

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # lustre: don't lock files that other jobs may be reading
include("/home/hchen54/internalwave_julia/DFF/dff_run.jl")   # dff_run, run_dir, ddf_dir, fig_dir (set by DFF_RUN)
include("/home/hchen54/internalwave_julia/DFF/dff_fun.jl")   # waxis, wdir, dff_sector, dff_spectrum

# Mode-1 baroclinic energy flux split into grid-north and grid-south going parts.
# The flux is a product (p'u'), so it is not filtered itself: the mode-1 pressure and velocity p1, u1, v1 (linear in
# the waves, made by bcef_mode1.jl) are each split by direction with the DFF, exactly as the ssh in DDF_on_ssh.jl, and
# the flux of each direction is formed from its own parts:
#   F_N = H p1_N (u1_N, v1_N),   F_S = H p1_S (u1_S, v1_S)        [W m-1], H = h + mean zeta
# The full mode-1 flux H p1 (u1, v1) = F_N + F_S + cross terms H (p1_N u1_S + p1_S u1_N). The cross terms come from
# the north and south waves crossing (a standing pattern): they average out over a record where the two are not
# phase-locked, but not where they are (e.g. near a reflecting slope); they are saved as a check.
FFTW.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")))
bcef_dir = get(ENV, "BCEF_OUT", joinpath(run_dir, "ddf_bcef"))
puv_fname = joinpath(bcef_dir, "bcef_mode1_puv.nc")
struct_fname = joinpath(bcef_dir, "bcef_mode1_structure.nc")
out_fname = joinpath(bcef_dir, "bcef_dff_flux.nc")

# ============================================================================
# (1) read p1, u1, v1 and the depth, trimmed to odd sizes like the ssh
# ============================================================================
ds_puv = NCDataset(puv_fname)
nx, ny, nt = ds_puv.dim["xi_rho"], ds_puv.dim["eta_rho"], ds_puv.dim["time"]
nx_f, ny_f = isodd(nx) ? nx : nx - 1, isodd(ny) ? ny : ny - 1
date = ds_puv["ocean_time"][:]   # DateTime
H, c1 = NCDataset(struct_fname) do ds
    Float32.(ds["h"][1:nx_f, 1:ny_f] .+ ds["zeta_mean"][1:nx_f, 1:ny_f]), ds["c1"][1:nx_f, 1:ny_f]
end
grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"
lon_rho, lat_rho, land_mask = NCDataset(grid_fname) do ds
    ds["lon_rho"][1:nx_f, 1:ny_f], ds["lat_rho"][1:nx_f, 1:ny_f], ds["mask_rho"][1:nx_f, 1:ny_f]
end
lon_rho[lon_rho .> 180] .-= 360
println("$dff_run: p1, u1, v1 $nx x $ny x $nt -> trimmed to $nx_f x $ny_f, $(date[1]) -> $(date[end])")

# ============================================================================
# (2) the DFF spectrum of each field (rfft in time, fft2 in space): 3 x ~5.5 GB for the 3-month run
# ============================================================================
# The time mean (frequency 0) is not in either band below, so no plane or mean is removed first: the baroclinic
# mode-1 fields have no barotropic tilt across the domain, which is what the plane took out of the ssh.
spec = Dict{String, Array{ComplexF32, 3}}()
for name in ("p1", "u1", "v1")
    a = nomissing(ds_puv[name][1:nx_f, 1:ny_f, :], NaN32)
    any(!isfinite, a) && error("$name has NaN/Inf")
    spec[name] = dff_spectrum(a)
    a = nothing
    GC.gc()
    println("$name: spectrum $(size(spec[name]))")
end
close(ds_puv)
nf = size(spec["p1"], 3)
freq_cph = (0:nf-1) ./ (nt * Dates.value(date[2] - date[1]) / 3.6e6)   # [cycles per hour]

# ============================================================================
# (3) directions and frequency bands
# ============================================================================
# sectors as for the ssh (θ from the grid xi axis, direction of travel; half-open); "all" keeps every direction, so
# its flux is the full mode-1 flux of the band, for the check F_all = F_N + F_S + cross
kx, ky = waxis(nx_f), waxis(ny_f)
θ = wdir(kx, ky)
sectors = [("all", 0, 360), ("north", 0, 180), ("south", 180, 360), ("north_smallsec", 60, 120), ("south_smallsec", 240, 300)]
sec_desc = Dict(s => (s == "all" ? "all directions (full mode-1 flux of the band)" : "$s sector ($a1-$a2 deg from grid xi)")
                for (s, a1, a2) in sectors)
sec_desc["cross"] = "cross terms H (p1_N u1_S + p1_S u1_N) of the half-plane sectors = all - north - south"
# bands as the saved flux pieces (energyflux_dum): M2 = periods 9–15 h; hp24 = periods shorter than 24 h; hp9 = periods
# shorter than 9 h (M4 and higher harmonics, nonlinear internal waves). To add a band, add a line. Sharp in
# frequency here (FFT bins), Butterworth in the flux pieces.
bands = [("M2", (1 / 15 .<= freq_cph .<= 1 / 9), "M2 band, periods 9-15 h"),
         ("hp24", (freq_cph .> 1 / 24), "periods shorter than 24 h"),
         ("hp9", (freq_cph .> 1 / 9), "periods shorter than 9 h")]
for (b, fk, d) in bands
    println("band $b ($d): $(count(fk)) frequencies")
end

# time ranges for the means: the whole record and each month (Aug–Sep together, as in the ssh analysis)
chunk(d1, d2) = (a = findfirst(>=(DateTime(d1)), date); b = findlast(<(DateTime(d2)), date);
                 a === nothing || b === nothing || b <= a ? nothing : a:b)
periods = filter(x -> x[2] !== nothing,
                 [("full", 1:nt), ("augsep", chunk(Date(2022, 8, 1), Date(2022, 10, 1))),
                  ("oct", chunk(Date(2022, 10, 1), Date(2022, 11, 1))), ("nov", chunk(Date(2022, 11, 1), Date(2022, 12, 1)))])
println("periods: ", join(["$p $(date[t[1]]) -> $(date[t[end]])" for (p, t) in periods], ", "))

# the three Hovmöller lines of the ssh analysis (grid columns cutting the domain into 4 along xi): the flux along them
# is kept every hour
tr_lines = [("west", round(Int, 0.5 + nx_f / 4)), ("mooring", round(Int, 0.5 + nx_f / 2)), ("east", round(Int, 0.5 + 3nx_f / 4))]

# ============================================================================
# (4) per band and sector: p1, u1, v1 of that direction -> flux -> time means and the lines
# ============================================================================
# One sector at a time (its p1, u1, v1: 3 x 5.5 GB for the 3-month run). The half-plane sectors split every wavenumber
# exactly once, so p1_all = p1_N + p1_S (same for u1, v1) and the cross terms are F_all − F_N − F_S; that split is
# checked at one point.
Fmean = Dict{Tuple{String, String, String}, Array{Float32, 3}}()   # (band, sector, "Fx"/"Fy") => [xi, eta, period]
Fline = Dict{Tuple{String, String, String, String}, Matrix{Float32}}()   # (band, sector, comp, line) => [eta, time]
ixc, iyc = cld(nx_f, 2), cld(ny_f, 2)   # check point (domain centre)

function flux_means(p, u, v)
    Fx = zeros(Float32, nx_f, ny_f, length(periods))
    Fy = zeros(Float32, nx_f, ny_f, length(periods))
    for (ip, (_, tidx)) in enumerate(periods)
        for t in tidx
            @views Fx[:, :, ip] .+= H .* p[:, :, t] .* u[:, :, t]
            @views Fy[:, :, ip] .+= H .* p[:, :, t] .* v[:, :, t]
        end
        Fx[:, :, ip] ./= length(tidx)
        Fy[:, :, ip] ./= length(tidx)
    end
    Fx, Fy
end

for (b, fkeep, bdesc) in bands
    pcheck = Dict{String, Vector{Float32}}()   # p1 at the check point, per sector
    for (sec, a1, a2) in sectors
        mask = a1 .<= θ .< a2
        @time p, u, v = (dff_sector(spec[n], mask, nt, land_mask; fkeep = fkeep) for n in ("p1", "u1", "v1"))
        Fx, Fy = flux_means(p, u, v)
        Fmean[(b, sec, "Fx")], Fmean[(b, sec, "Fy")] = Fx, Fy
        for (l, ix) in tr_lines
            Fline[(b, sec, "Fx", l)] = H[ix, :] .* p[ix, :, :] .* u[ix, :, :]
            Fline[(b, sec, "Fy", l)] = H[ix, :] .* p[ix, :, :] .* v[ix, :, :]
        end
        println("$b $sec: domain-mean flux (full record) Fx $(round(mean(Fx[:, :, 1]), sigdigits = 3)), ",
                "Fy $(round(mean(Fy[:, :, 1]), sigdigits = 3)) W/m; p1 std $(round(std(p), sigdigits = 3)) Pa")
        pcheck[sec] = p[ixc, iyc, :]
        p = u = v = nothing
        GC.gc()
    end
    # check: north + south = all at the check point; then the cross terms
    err = maximum(abs, pcheck["north"] .+ pcheck["south"] .- pcheck["all"])
    println("$b check at (xi, eta) = ($ixc, $iyc): max |p1_N + p1_S − p1_all| = $(round(err, sigdigits = 2)) Pa ",
            "(p1_all std $(round(std(pcheck["all"]), sigdigits = 3)) Pa)")
    for c in ("Fx", "Fy")
        Fmean[(b, "cross", c)] = Fmean[(b, "all", c)] .- Fmean[(b, "north", c)] .- Fmean[(b, "south", c)]
        println("$b $c: domain-mean |cross| / |all| = ",
                "$(round(mean(abs, Fmean[(b, "cross", c)][:, :, 1]) / mean(abs, Fmean[(b, "all", c)][:, :, 1]), digits = 2)) (full record)")
    end
end

# ============================================================================
# (5) save
# ============================================================================
isfile(out_fname) && rm(out_fname)
NCDataset(out_fname * ".tmp", "c") do ds
    defDim(ds, "xi_rho", nx_f); defDim(ds, "eta_rho", ny_f); defDim(ds, "period", length(periods)); defDim(ds, "time", nt)
    defVar(ds, "lon_rho", lon_rho, ("xi_rho", "eta_rho"); attrib = ["units" => "degree_east"])
    defVar(ds, "lat_rho", lat_rho, ("xi_rho", "eta_rho"); attrib = ["units" => "degree_north"])
    defVar(ds, "H", H, ("xi_rho", "eta_rho"); attrib = ["long_name" => "h + mean zeta", "units" => "meter"])
    defVar(ds, "c1", c1, ("xi_rho", "eta_rho"); attrib = ["long_name" => "mode-1 eigenspeed", "units" => "m s-1"])
    defVar(ds, "ocean_time", date, ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
    ds.attrib["periods"] = join(["$p = $(date[t[1]]) - $(date[t[end]])" for (p, t) in periods], "; ")
    for (b, _, bdesc) in bands, sec in [first.(sectors); "cross"], c in ("Fx", "Fy")
        defVar(ds, "$(c)_$(b)_$(sec)", Fmean[(b, sec, c)], ("xi_rho", "eta_rho", "period");
               attrib = ["long_name" => "time-mean mode-1 energy flux along grid $(c == "Fx" ? "xi" : "eta"), $bdesc, $(sec_desc[sec])",
                         "units" => "W m-1"])
    end
    ds.attrib["note"] = "F = H p1 (u1, v1) from bcef_mode1.jl; each direction's p1, u1, v1 from the DFF (DDF_on_ssh.jl method). " *
                        "Fx along grid xi, Fy along grid eta (not east / north). all = north + south + cross."
end
mv(out_fname * ".tmp", out_fname; force = true)
println("wrote $out_fname")

# flux along the three lines every hour, one file per line (as the ssh transects)
for (l, ix) in tr_lines
    fname = joinpath(bcef_dir, "bcef_dff_flux_transect_$(l)_xi$(ix).nc")
    isfile(fname) && rm(fname)
    NCDataset(fname, "c") do ds
        defDim(ds, "eta_rho", ny_f); defDim(ds, "time", nt)
        defVar(ds, "ocean_time", date, ("time",); attrib = ["units" => "seconds since 1994-01-01 00:00:00"])
        defVar(ds, "H", H[ix, :], ("eta_rho",); attrib = ["units" => "meter"])
        defVar(ds, "lat_rho", lat_rho[ix, :], ("eta_rho",); attrib = ["units" => "degree_north"])
        for (b, _, bdesc) in bands, sec in first.(sectors), c in ("Fx", "Fy")
            defVar(ds, "$(c)_$(b)_$(sec)", Fline[(b, sec, c, l)], ("eta_rho", "time");
                   attrib = ["long_name" => "mode-1 energy flux along grid $(c == "Fx" ? "xi" : "eta"), $bdesc, $sec", "units" => "W m-1"])
        end
        ds.attrib["line"] = "$l line, grid column xi = $ix (Julia 1-based, trimmed DFF grid)"
    end
    println("wrote $fname")
end

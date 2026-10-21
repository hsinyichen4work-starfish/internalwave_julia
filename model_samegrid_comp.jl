using Distributed

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "32")) - 1)   # -1 reserves a CPU for this main process
# heap-size-hint makes each worker GC before it grows past ~3 GB (one time step's
# 3D fields + z levels for both runs peak around 2–3 GB)
addprocs(n_workers; exeflags = ["--project=/home/hchen54/internalwave_julia", "--heap-size-hint=3G"])
println("running with $(nprocs() - 1) worker processes")

@everywhere begin
    using NCDatasets, Dates, Statistics, CairoMakie
    CairoMakie.activate!()
    include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")
    include("/home/hchen54/internalwave_julia/function/zlevs3.jl")
    include("/home/hchen54/internalwave_julia/function/uv2rho.jl")

    grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # shared by both runs
    datadir1   = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_pure_or/after_joint"      # run 1: pure orlanski
    datadir2   = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon_1008_2/after_joint"  # run 2: 3-month run
    label1, label2 = "pure_or", "3mon_1008_2"
    title1, title2 = "pure Orlanski bry", "dynamic bry"   # names used in figure titles

    figure_path = "/expanse/lustre/projects/uso101/hchen54/figure/orlanski_comp"

    t_ref = DateTime(1994, 1, 1, 0, 0, 0)   # ocean_time is "seconds since 1994/01/01"

    ## GRID — same grid for both runs, so read it once
    mask_rho, lon_rho, lat_rho, h, pm, pn, grid_angle = NCDataset(grid_fname) do ds
        ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
        ds["h"][:, :], ds["pm"][:, :], ds["pn"][:, :], ds["angle"][:, :]
    end
    lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180

    lon_lim = extrema(lon_rho)
    lat_lim = extrema(lat_rho)
    bathy_levels = [500, 1000, 2000]

    ## WHAT TO PLOT
    # depth slices: (key used in filenames, label for titles, z in m; nothing = top s-level)
    depth_slices = [("surf", "surface", nothing),
                    ("100m", "100 m",   -100.0),
                    ("200m", "200 m",   -200.0)]

    # (key, title label, units, snapshot subfolder, colormap for the two run panels)
    # u/v are rotated to true east/north before plotting.
    plot_vars = [("ssh",  "SSH",       "m",   "ssh_snapshot",  :viridis),
                 ("temp", "Temp",      "°C",  "temp_snapshot", :thermal),
                 ("u",    "u (east)",  "m/s", "u_snapshot",    :balance),
                 ("v",    "v (north)", "m/s", "v_snapshot",    :balance)]
    var_info = Dict(v[1] => v for v in plot_vars)

    # fixed snapshot color limits per (var, depth) = ((min, max), symmetric diff limit),
    # so frames are comparable over time. Set from sampled ranges over the overlap period.
    snap_lims = Dict(
        ("ssh",  "")     => ((-2.0, 2.0),   0.2),
        ("temp", "surf") => ((26.0, 30.0),  0.5),
        ("temp", "100m") => ((14.0, 28.0),  2.0),
        ("temp", "200m") => ((10.0, 18.0),  1.0),
        ("u",    "surf") => ((-1.5, 1.5),   0.5),
        ("u",    "100m") => ((-1.2, 1.2),   0.3),
        ("u",    "200m") => ((-0.6, 0.6),   0.2),
        ("v",    "surf") => ((-1.5, 1.5),   0.5),
        ("v",    "100m") => ((-1.2, 1.2),   0.3),
        ("v",    "200m") => ((-0.6, 0.6),   0.2),
    )

    # Running sums for the time mean, keyed by (var, depth key) -> (sum1, sum2, count1, count2).
    # Every process gets its OWN copy of this Dict (@everywhere does not share memory),
    # so each worker adds only the times it was handed; the main process collects and
    # adds all workers' copies at the end. Counts are per pixel because a 200 m slice
    # near the shelf break is below the seafloor (NaN) at some tides and not at others.
    field_sums = Dict{Tuple{String, String}, NTuple{4, Matrix{Float64}}}()
end

mkpath(figure_path)
for v in plot_vars
    mkpath(joinpath(figure_path, v[4]))
end

## HIS FILE TIME INDEX
# Each roms_his.*.nc holds several records (4 hourly records per file), so build a
# lookup from DateTime -> (file, record index) by reading ocean_time from every file.
# The dir also has dia/ext/mooring files mixed in, so filter on the "roms_his." prefix.
function his_time_index(datadir)
    files = sort(filter(f -> startswith(f, "roms_his.") && endswith(f, ".nc"), readdir(datadir)))
    index = Dict{DateTime, Tuple{String, Int}}()
    for f in files
        fname = joinpath(datadir, f)
        ocean_time = NCDataset(fname) do ds
            ds["ocean_time"][:]
        end
        for (it, ot) in enumerate(ocean_time)
            t = t_ref + Millisecond(round(Int, ot * 1000))
            index[t] = (fname, it)   # if a time repeats across files (restart overlap), the later file wins
        end
    end
    return index
end

# built once on the main process, then copied to every worker
his1 = his_time_index(datadir1)
his2 = his_time_index(datadir2)
println(label1, ": ", length(his1), " records, ", minimum(keys(his1)), " -> ", maximum(keys(his1)))
println(label2, ": ", length(his2), " records, ", minimum(keys(his2)), " -> ", maximum(keys(his2)))
@everywhere his1 = $his1
@everywhere his2 = $his2

# times that exist in BOTH runs, in chronological order
common_times = sort(collect(intersect(keys(his1), keys(his2))))
isempty(common_times) && error("no overlapping his times between $label1 and $label2")
println("overlap: ", length(common_times), " records, ", first(common_times), " -> ", last(common_times))

@everywhere begin
    ## READ ONE RECORD AT ONE TIME
    # Julia/NCDatasets dims are reversed from ncdump: 2D vars are (xi, eta, time),
    # 3D vars are (xi, eta, s_rho, time). Returns the field with the time dim dropped.
    function read_his_var(index, t, varname)
        fname, it = index[t]
        NCDataset(fname) do ds
            v = ds[varname]
            ndims(v) == 3 ? v[:, :, it] : v[:, :, :, it]
        end
    end

    ## VERTICAL SLICING
    # For one target depth, find per column the s-level k below it and the linear
    # weight w toward level k+1. Done once per run per time from that run's own z,
    # then reused for temp/u/v. k = 0 marks "target is below the seafloor" (NaN).
    function depth_weights(z, target)   # z: (xi, eta, s_rho), ascending in s
        M, L, N = size(z)
        k = zeros(Int32, M, L)
        w = zeros(Float32, M, L)
        for j in 1:L, i in 1:M
            zcol = view(z, i, j, :)
            if target < zcol[1]
                continue                          # below seafloor
            elseif target >= zcol[N]
                k[i, j], w[i, j] = N - 1, 1f0     # above top rho level: use top level
            else
                kk = searchsortedlast(zcol, target)
                k[i, j] = kk
                w[i, j] = (target - zcol[kk]) / (zcol[kk + 1] - zcol[kk])
            end
        end
        return k, w
    end

    function apply_weights(F, (k, w))
        out = fill(NaN32, size(k))
        for j in axes(k, 2), i in axes(k, 1)
            kk = k[i, j]
            kk == 0 && continue
            out[i, j] = (1 - w[i, j]) * F[i, j, kk] + w[i, j] * F[i, j, kk + 1]
        end
        return out
    end

    # Dict depth key -> weights (nothing for the surface, which is just the top s-level)
    function run_weights(index, t, zeta)
        fname, _ = index[t]
        theta_s, theta_b, hc, N = NCDataset(fname) do ds
            ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"], ds.dim["s_rho"]
        end
        z_dum, _ = zlevs3(h, zeta, theta_s, theta_b, hc, N, "r", "new2008")
        z = permutedims(z_dum, (2, 3, 1))   # (s, xi, eta) -> (xi, eta, s)
        return Dict(key => (zt === nothing ? nothing : depth_weights(z, zt)) for (key, _, zt) in depth_slices)
    end

    # read a 3D rho-point field (already moved to rho points by `prep`) and slice every depth
    function slice_all(F, W)
        Dict(key => (W[key] === nothing ? Float32.(F[:, :, end]) : apply_weights(F, W[key]))
             for (key, _, _) in depth_slices)
    end

    # returns Dict var key -> Dict depth key -> 2D slice, for one run at one time
    function run_slices(index, t, W)
        out = Dict{String, Dict{String, Matrix{Float32}}}()
        out["temp"] = slice_all(read_his_var(index, t, "temp"), W)
        u_s = slice_all(u2rho(read_his_var(index, t, "u")), W)   # grid-relative (xi direction)
        v_s = slice_all(v2rho(read_his_var(index, t, "v")), W)   # grid-relative (eta direction)
        # rotate grid xi/eta into true east/north (slicing is linear in the vertical,
        # so slicing first and rotating after gives the same answer and is cheaper)
        out["u"] = Dict(k => u_s[k] .* cos.(grid_angle) .- v_s[k] .* sin.(grid_angle) for k in keys(u_s))
        out["v"] = Dict(k => u_s[k] .* sin.(grid_angle) .+ v_s[k] .* cos.(grid_angle) for k in keys(u_s))
        return out
    end

    ## 3-PANEL COMPARISON FIGURE: (1) dynamic bry, (2) pure Orlanski bry, (3) dynamic - pure Orlanski
    landmask(field) = ifelse.(mask_rho .== 0, NaN32, Float32.(field))

    function plot_comp3(outname, field2, field1, timestr; varlabel, units, clim, dclim, cmap = :viridis)
        diff = field2 .- field1
        fig = Figure(size = (2000, 650))
        panels = ((field2, "$title2", cmap, (clim[1], clim[2])),
                  (field1, "$title1", cmap, (clim[1], clim[2])),
                  (diff,   "$title2 − $title1", Reverse(:RdBu), (-dclim, dclim)))
        for (i, (field, name, cm, cr)) in enumerate(panels)
            ax = topdown_axis3(fig[1, 2i - 1]; title = "$varlabel, $name\n$timestr")
            xlims!(ax, lon_lim...)
            ylims!(ax, lat_lim...)
            sp = plot_curvilinear!(ax, lon_rho, lat_rho, landmask(field); colormap = cm, colorrange = cr,
                                   nan_color = :lightgray)   # gray = land or below the seafloor
            contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels, color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            Colorbar(fig[1, 2i], sp, label = units)
        end
        save(outname, fig)
        println("saved plot to ", outname)
    end

    # "ssh" has no depth; everything else is "<var>_<depth>"
    file_tag(var, dkey) = dkey == "" ? var : "$(var)_$(dkey)"
    title_label(var, dkey, dlabel) = dkey == "" ? var_info[var][2] : "$(var_info[var][2]) at $dlabel"

    function accumulate!(key, f1, f2)
        s1, s2, c1, c2 = get!(field_sums, key) do
            (zeros(size(h)), zeros(size(h)), zeros(size(h)), zeros(size(h)))
        end
        for i in eachindex(f1)
            isnan(f1[i]) || (s1[i] += f1[i]; c1[i] += 1)
            isnan(f2[i]) || (s2[i] += f2[i]; c2[i] += 1)
        end
    end

    # add this time to the running sums, then plot the snapshot unless it already exists
    function handle_field(t, var, dkey, dlabel, f1, f2)
        accumulate!((var, dkey), f1, f2)
        info = var_info[var]
        outname = joinpath(figure_path, info[4], "$(file_tag(var, dkey))_comp_$(Dates.format(t, "yyyymmdd_HHMM")).png")
        isfile(outname) && return   # re-runs only redo the time means
        clim, dclim = snap_lims[(var, dkey)]
        plot_comp3(outname, f2, f1, Dates.format(t, "yyyy-mm-dd HH:MM");
                   varlabel = title_label(var, dkey, dlabel), units = info[3],
                   clim = clim, dclim = dclim, cmap = info[5])
    end

    ## ONE TIME STEP: ssh + temp/u/v at every depth, both runs
    function process_time(t)
        zeta1 = read_his_var(his1, t, "zeta")
        zeta2 = read_his_var(his2, t, "zeta")

        # skip incomplete/fill-valued records (e.g. a file still being written)
        if maximum(abs, zeta1) > 1e30 || maximum(abs, zeta2) > 1e30
            println("skipping $t — fill-valued record")
            return
        end
        handle_field(t, "ssh", "", "", zeta1, zeta2)

        S1 = run_slices(his1, t, run_weights(his1, t, zeta1))
        S2 = run_slices(his2, t, run_weights(his2, t, zeta2))
        for var in ("temp", "u", "v"), (dkey, dlabel, _) in depth_slices
            handle_field(t, var, dkey, dlabel, S1[var][dkey], S2[var][dkey])
        end
        GC.gc()   # the full 3D fields are large; free them before this worker takes the next time
        return
    end

    # Named function (not an anonymous closure) so it reads the CALLED worker's
    # own sums — a closure written in Main would ship the main process's (empty) copy.
    get_field_sums() = field_sums
end

## SNAPSHOTS (parallel over time)
# pmap hands one time step at a time to whichever worker is free. No retry on
# error: a retried time step would be added to the sums twice.
pmap(process_time, common_times; on_error = ex -> println("a time step failed: ", ex))

## TIME MEANS
# gather each worker's partial sums onto the main process and add them up
total = Dict{Tuple{String, String}, NTuple{4, Matrix{Float64}}}()
for w in workers()
    for (key, parts) in remotecall_fetch(get_field_sums, w)
        total[key] = haskey(total, key) ? total[key] .+ parts : parts
    end
end

ocean = mask_rho .== 1
mean_period = "$(Dates.format(first(common_times), "yyyy-mm-dd")) to $(Dates.format(last(common_times), "yyyy-mm-dd"))"
for ((var, dkey), (s1, s2, c1, c2)) in sort(collect(total); by = first)
    mean1 = Float32.(ifelse.(c1 .> 0, s1 ./ c1, NaN))   # Float32 for plotting, like the snapshots
    mean2 = Float32.(ifelse.(c2 .> 0, s2 ./ c2, NaN))
    nrec = Int(maximum(c1))
    println("time mean $(file_tag(var, dkey)): up to $nrec of $(length(common_times)) records per pixel")

    # mean fields are much smoother than snapshots, so set their limits from the data
    vals = filter(isfinite, vcat(mean1[ocean], mean2[ocean]))
    dvals = filter(isfinite, abs.((mean2 .- mean1)[ocean]))
    info = var_info[var]
    clim = info[5] == :balance ? (q = quantile(abs.(vals), 0.99); (-q, q)) : (quantile(vals, 0.005), quantile(vals, 0.995))
    dclim = quantile(dvals, 0.99)
    dlabel = dkey == "" ? "" : only(d[2] for d in depth_slices if d[1] == dkey)
    plot_comp3(joinpath(figure_path, "$(file_tag(var, dkey))_comp_timemean.png"), mean2, mean1,
               "mean $mean_period ($nrec records)";
               varlabel = title_label(var, dkey, dlabel), units = info[3],
               clim = clim, dclim = dclim, cmap = info[5])
    GC.gc()   # free the previous figure before drawing the next one
end

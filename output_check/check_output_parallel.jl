using Distributed

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "7")) - 1)   # -1 reserves a CPU for this main process
addprocs(n_workers)
println("running with $(nprocs() - 1) worker processes")

@everywhere begin
    using NCDatasets, CairoMakie, Dates, Statistics
    CairoMakie.activate!()
    include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
    include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

    # which run to plot: CHECK_RUN=cmems_2700m (default), cmems_2700m_3mon or 900m_pure_or
    # (the workers inherit ENV from the main process)
    run_name = get(ENV, "CHECK_RUN", "cmems_2700m")
    proj = "/expanse/lustre/projects/uso101/hchen54/"
    run_cfg = Dict(
        # 2700 m CMEMS ini/bry + ERA5 forcing + TPXO10 tides, 378 x 509 rho points
        "cmems_2700m"  => (grid = proj * "input/grid/roms_cmems_grd_2700m.nc",
                           moor = proj * "input/grid/roms_cmems_grd_2700m_mor_edata.nc",
                           fold = "amazon_2700m_cmems_test", skip = 22),
        # same setup, 2022-08-24 -> 11-30
        "cmems_2700m_3mon" => (grid = proj * "input/grid/roms_cmems_grd_2700m.nc",
                               moor = proj * "input/grid/roms_cmems_grd_2700m_mor_edata.nc",
                               fold = "amazon_2700m_cmems_3mon", skip = 22),
        # 900 m NCOM run, 686 x 856 rho points
        "900m_pure_or" => (grid = proj * "input/grid/roms_grd_900m.nc",
                           moor = proj * "input/roms_grd_900m_mor_edata.nc",
                           fold = "amazon_900m_pure_or", skip = 40))[run_name]

    grid_fname = run_cfg.grid   # the grid_file listed in the .nc's global attributes
    mooring_file = run_cfg.moor   # french, M1..M4, CPIES1..9
    datadir = proj * run_cfg.fold * "/after_joint"   # HPC output dir — contains avg/dia/his/rst files mixed together
    figure_path = proj * "figure/" * run_cfg.fold
    mkpath(figure_path)

    ##
    mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
        ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
        ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
    end
    lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
    umask, vmask, mask_p = uvp_masks(mask_rho)
    pm_u, pn_v = rho2u(pm), rho2v(pn)   # 1/dx on U-points, 1/dy on V-points (1/m)
    lon_psi, lat_psi = rho2p(lon_rho), rho2p(lat_rho)
    # no native lon_psi/lat_psi in this grid file — approximate via corner averaging

    lon_lim = extrema(lon_rho)   # shared x/y axis limits for every plot below
    lat_lim = extrema(lat_rho)

    # h here is the ROMS grid's own bathymetry (positive-down, no missing values),
    # already on the same lon_rho/lat_rho grid as everything plotted below — unlike
    # check_NCOM_comp.jl, no sign flip / coalesce-to-NaN is needed before contouring.
    bathy_levels = [500, 1000, 2000]

    # map axis shared by every plot below: topdown_axis3 plus the common
    # lon/lat limits. Axis3 puts the y ticks/label on whichever side faces
    # the camera; at exactly -pi/2 that lands on the right, and this tiny
    # nudge (far too small to see) tips it over to the left instead
    function map_axis(pos; title = "")
        ax = topdown_axis3(pos; title = title)
        ax.azimuth = -pi / 2 - 1e-4
        xlims!(ax, lon_lim...)
        ylims!(ax, lat_lim...)
        return ax
    end

    # every marker/arrow/label size below is given for the 18 cm canvas; the SSH
    # gradient plot passes this as `scale` (now also 18 cm wide, so 1)
    small_scale = 1.0

    # mooring locations (lon in degrees_east, -180..180 like the grid; lat in degrees_north),
    # taken from the global attributes of mooring_file, e.g.
    #   :M1_mor_info = "indices for M1_mor in roms_grd_900m.nc , location:(-45.4802;2.5415)"
    # (the *_mor variables themselves hold fractional grid indices, not lon/lat)
    moor_name, moor_lon, moor_lat = NCDataset(mooring_file) do ds
        names = [replace(v, "_mor" => "") for v in keys(ds) if endswith(v, "_mor")]
        locs  = [match(r"location:\(([-\d.]+);([-\d.]+)\)", ds.attrib["$(n)_mor_info"]) for n in names]
        names, [parse(Float64, m[1]) for m in locs], [parse(Float64, m[2]) for m in locs]
    end
    moor_group = [replace(n, r"\d+$" => "") for n in moor_name]   # "M3" -> "M", "CPIES7" -> "CPIES"

    # one marker style per mooring type, so the three arrays can be told apart
    moor_style = Dict("french" => (marker = :star5,   markersize = 18, color = :magenta),
                      "M"      => (marker = :diamond, markersize = 11, color = :cyan),
                      "CPIES"  => (marker = :circle,  markersize = 10, color = :yellow))

    # marks every mooring on the map; z = 0 puts them on the same flat plane as
    # plot_curvilinear!'s surface, and overdraw keeps them from being hidden
    # behind that surface. Call it last so the markers sit on top of any arrows.
    function add_mooring!(ax; scale = 1)
        for g in unique(moor_group)
            k = findall(==(g), moor_group)
            st = get(moor_style, g, (marker = :rect, markersize = 10, color = :white))
            scatter!(ax, Point3f.(moor_lon[k], moor_lat[k], 0);
                     marker = st.marker, markersize = st.markersize * scale, color = st.color,
                     strokecolor = :black, strokewidth = 1.2 * scale, overdraw = true)
        end
    end

    skip = run_cfg.skip   # downsample for legible quiver arrows (~17 x 21 arrows on either grid)
    # lengthscale = degrees of lon/lat drawn per 1 m/s
    quiver_kw(scale) = (lengthscale = 0.25, color = :black, shaftwidth = 1.5 * scale,
                        tipwidth = 5 * scale, tiplength = 6 * scale)

    # reference arrow for the quiver scale, in the empty lower-right corner
    # of the axis (the rotated grid leaves all four corners blank)
    ref_speed = 1.0   # m/s
    ref_lon = lon_lim[2] - 0.22 * (lon_lim[2] - lon_lim[1])
    ref_lat = lat_lim[1] + 0.07 * (lat_lim[2] - lat_lim[1])

    # surface-current arrows + their reference arrow, in one call
    function add_quiver!(ax, u_east, v_north; scale = 1)
        quiver_curvilinear!(ax, lon_rho, lat_rho, u_east, v_north; skip = skip, quiver_kw(scale)...)
        arrows2d!(ax, [ref_lon], [ref_lat], [0.0], [ref_speed], [0.0], [0.0]; quiver_kw(scale)...)
        text!(ax, Point3f(ref_lon, ref_lat, 0); text = "$(ref_speed) m/s", fontsize = 10 * scale,
              align = (:left, :bottom), offset = (0, 6 * scale), overdraw = true)
    end

    # SSH gradient magnitude |∇zeta| = sqrt((dzeta/dx)^2 + (dzeta/dy)^2) on the
    # RHO grid, dimensionless (m/m). The differences between neighboring RHO
    # cells naturally live on U-points (xi direction) and V-points (eta
    # direction), so compute them there, then average back onto RHO. The
    # magnitude is rotation-invariant, so no grid_angle rotation is needed.
    #
    # multiplying by umask/vmask zeroes any difference taken across a
    # land/sea boundary, and dividing by the averaged mask afterwards turns
    # the U/V -> RHO average into "mean of the wet neighbors only" — so a
    # land cell's zeta never leaks into the gradient at the coast.
    function zeta_gradient(zeta)
        dzdx_u = diff(zeta, dims = 1) .* pm_u .* umask
        dzdy_v = diff(zeta, dims = 2) .* pn_v .* vmask
        dzdx = u2rho(dzdx_u) ./ u2rho(umask)
        dzdy = v2rho(dzdy_v) ./ v2rho(vmask)
        return sqrt.(dzdx .^ 2 .+ dzdy .^ 2)
    end

    grad_scale = 1e6        # plot in units of 10^-6, i.e. mm of SSH change per km
    grad_range = (0, 10)    # ~90th percentile offshore; the shelf saturates

end

@everywhere function process_file(fname)
    ocean_time = NCDataset(fname) do ds
        ds["ocean_time"][:]
    end
    println(fname, " => size(ocean_time) = ", size(ocean_time))
    ntime = length(ocean_time)   # length(), not size() — size() returns a Tuple like (4,), not a plain number
    t_ref = DateTime(1994, 1, 1, 0, 0, 0)
    realtime = t_ref .+ Millisecond.(round.(Int, ocean_time .* 1000))   # Vector{DateTime}
    str1 = Dates.format.(realtime, "yyyy-mm-dd HH:MM:SS")                # for plot titles
    str2 = Dates.format.(realtime, "yyyymmdd_HHMM")                      # for filenames, e.g. 20220824_0730
 
    ## Whole-file skip — if this file's last time step already produced all
    # five plot types, assume the whole file was fully processed last run and
    # skip reading zeta/temp/u/v entirely (not just skip the plotting).
    outname_zeta_last = joinpath(figure_path, "zeta_plot_$(str2[ntime]).png")
    outname_temp_last = joinpath(figure_path, "temp_depth_plot_$(str2[ntime]).png")
    outname_speed_last = joinpath(figure_path, "speed_plot_$(str2[ntime]).png")
    outname_vor_last = joinpath(figure_path, "vorticity_plot_$(str2[ntime]).png")
    outname_grad_last = joinpath(figure_path, "zeta_grad_plot_$(str2[ntime]).png")
 
    if isfile(outname_zeta_last) && isfile(outname_temp_last) &&
       isfile(outname_speed_last) && isfile(outname_vor_last) && isfile(outname_grad_last)
        println("all plots already exist for ", fname, ", skipping file entirely")
        return
    end
 
    ## SSH DATA
    # NCDataset(fname) opens the file; the `do...end` block auto-closes it when done
    # (like fopen/fclose in MATLAB, but you don't have to remember to close it)
    # the last line of the block is what gets returned into `zeta` below
    zeta = NCDataset(fname) do ds
        ds["zeta"][:, :, :]   # sea surface height, dims xi_rho x eta_rho x time
    end
    println(fname, " => size(zeta) = ", size(zeta))
 
    zeta_masked = ifelse.(mask_rho .== 0, NaN32, zeta)
    zeta_grad_masked = ifelse.(mask_rho .== 0, NaN, zeta_gradient(zeta))
 
    # theta_s/theta_b/hc/N are all cheap metadata — attributes, plus a
    # variable's shape without reading its actual data — so we can always
    # fetch them without waiting on the (possibly skippable) full temp read
    # below. This is what z/z_p (needed by every plot type) depend on.
    theta_s, theta_b, hc, N = NCDataset(fname) do ds
        ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"], size(ds["temp"], 3)
    end
 
    z = zeros(size(zeta, 1), size(zeta, 2), N, ntime)
    Cs = nothing
    for t in 1:ntime
        z_dum, Cs = zlevs3(h, zeta[:, :, t], theta_s, theta_b, hc, N, "r", "new2008")
        z[:, :, :, t] = permutedims(z_dum, (2, 3, 1))
    end
    z_p = rho2p(z)
 
    ## TEMP DATA — only actually read the (large) temp array if the temp
    # plot for this file's last time step isn't already sitting on disk
    if isfile(outname_temp_last)
        println("temp plots already exist for ", fname, ", skipping temp read")
    else
        temp = NCDataset(fname) do ds
            ds["temp"][:, :, :,:]
        end
        println(fname, " => size(temp) = ", size(temp))
        temp_masked = ifelse.(mask_rho .== 0, NaN32, temp)
    end
 
    ## vel data for speed + vorticity + SSH gradient (its arrows) — same idea,
    # skip the read if every plot that needs it already exists for this
    # file's last time step
    if isfile(outname_speed_last) && isfile(outname_vor_last) && isfile(outname_grad_last)
        println("speed/vorticity/SSH gradient plots already exist for ", fname, ", skipping u/v read")
    else
        u,v = NCDataset(fname) do ds
            ds["u"][:, :, :,:]  , ds["v"][:, :, :,:]
        end
        println(fname, " => size(u) = ", size(u))
        println(fname, " => size(v) = ", size(v))
 
        ## speed
        # average raw (unmasked) u, v onto the rho grid first, then mask the
        # result — masking beforehand would spread NaN from land onto valid
        # ocean cells one row/column too far, same reasoning as vorticity below
        u_rho = u2rho(u)   # still grid-relative (xi/eta), not east/north
        v_rho = v2rho(v)
 
        # speed is rotation-invariant, so no need to rotate before this
        speed = sqrt.(u_rho .^ 2 .+ v_rho .^ 2)
        speed_masked = ifelse.(mask_rho .== 0, NaN32, speed)
 
        # but quiver arrows plotted against true lon/lat DO need rotating out of
        # the grid's local xi/eta directions into true east/north, using the
        # grid's rotation angle (radians) from the grid file
        u_east = u_rho .* cos.(grid_angle) .- v_rho .* sin.(grid_angle)
        v_north = u_rho .* sin.(grid_angle) .+ v_rho .* cos.(grid_angle)
        u_east_masked = ifelse.(mask_rho .== 0, NaN32, u_east)
        v_north_masked = ifelse.(mask_rho .== 0, NaN32, v_north)
 
        ## VORTICITY
        # use the raw (unmasked) u, v here — masking with NaN before differencing
        # would spread NaN into neighboring cells; mask the result afterward instead
        vor_psi = vorticity_cal(u, v, pm, pn)   # relative vorticity on the PSI grid
        vor_psi_masked = ifelse.(mask_p .== 0, NaN32, vor_psi)
    end
 
 
    ## PLOTS — one set of files per ocean_time step
    for t in 1:ntime
 
        # skip incomplete/fill-valued records — e.g. the last time step of a
        # file still being actively written, where every variable is left at
        # the raw NetCDF fill value (~9.97e36) instead of real data
        if maximum(abs, zeta[:, :, t]) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end
 
        # -- SSH --
        outname_zeta = joinpath(figure_path, "zeta_plot_$(str2[t]).png")
        if isfile(outname_zeta)
            println("already exists, skipping: ", outname_zeta)
        else
            fig = Figure(; figkw(18, 13.5)...)
            ax = map_axis(fig[1, 1]; title = "Sea surface height (zeta), $(str1[t])")
            sp = plot_curvilinear!(ax, lon_rho, lat_rho, zeta_masked[:, :, t];
                                    colormap = Reverse(:RdBu), colorrange = (-1, 1))
            Colorbar(fig[1, 2], sp, label = "meters")
            contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            add_mooring!(ax)
            figsave(outname_zeta, fig)
            println("saved plot to ", outname_zeta)
        end
 
        # -- TEMP --
        # slice_at_depth does per-column vertical interpolation only (same
        # horizontal grid in/out) — z and temp_masked here are one time step's
        # (M, L, N) fields; target depth is in meters, negative below the surface.
        outname_temp = joinpath(figure_path, "temp_depth_plot_$(str2[t]).png")
        if isfile(outname_temp)
            println("already exists, skipping: ", outname_temp)
        else
            temp_1m = slice_at_depth(z[:, :, :, t], temp_masked[:, :, :, t], -1.0)
            temp_100m = slice_at_depth(z[:, :, :, t], temp_masked[:, :, :, t], -100.0)
 
            fig = Figure(; figkw(18, 7.7)...)
            for (col, (field, depth_label, crange)) in enumerate(((temp_1m, "1 m", (24, 30)), (temp_100m, "100 m", (15, 30))))
                ax = map_axis(fig[1, 2col - 1]; title = "Temperature at $depth_label, $(str1[t])")
                sp = plot_curvilinear!(ax, lon_rho, lat_rho, field; colormap = :thermal, colorrange = crange)
                Colorbar(fig[1, 2col], sp, label = "°C")
                contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                         color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
                add_mooring!(ax)
            end
            figsave(outname_temp, fig)
            println("saved plot to ", outname_temp)
        end
 
        # -- SPEED --
        outname_speed = joinpath(figure_path, "speed_plot_$(str2[t]).png")
        if isfile(outname_speed)
            println("already exists, skipping: ", outname_speed)
        else
            speed_1m = slice_at_depth(z[:, :, :, t], speed_masked[:, :, :, t], -1.0)
            u_east_1m = slice_at_depth(z[:, :, :, t], u_east_masked[:, :, :, t], -1.0)
            v_north_1m = slice_at_depth(z[:, :, :, t], v_north_masked[:, :, :, t], -1.0)
 
            fig = Figure(; figkw(18, 13.5)...)
            ax = map_axis(fig[1, 1]; title = "Speed at 1 m, $(str1[t])")
            sp = plot_curvilinear!(ax, lon_rho, lat_rho, speed_1m; colormap = :speed, colorrange = (0, 2))
            Colorbar(fig[1, 2], sp, label = "m/s")
            contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            add_quiver!(ax, u_east_1m, v_north_1m)
            add_mooring!(ax)
            figsave(outname_speed, fig)
            println("saved plot to ", outname_speed)
        end
 
        # -- VORTICITY --
        outname_vor = joinpath(figure_path, "vorticity_plot_$(str2[t]).png")
        if isfile(outname_vor)
            println("already exists, skipping: ", outname_vor)
        else
            # speed's u_east_1m/v_north_1m may have been skipped above (if the
            # speed plot already existed), so recompute them here too — cheap
            # relative to the rest of this block, and keeps this block able to
            # run independently of the speed one.
            u_east_1m = slice_at_depth(z[:, :, :, t], u_east_masked[:, :, :, t], -1.0)
            v_north_1m = slice_at_depth(z[:, :, :, t], v_north_masked[:, :, :, t], -1.0)
            vor_1m = slice_at_depth(z_p[:, :, :, t], vor_psi_masked[:, :, :, t], -1.0)
 
            # clim = 5 * mean(abs, filter(!isnan, vor_1m))   # symmetric color limits centered on 0
            clim = 5e-5;
            fig = Figure(; figkw(18, 13.5)...)
            ax = map_axis(fig[1, 1]; title = "Relative vorticity, surface, $(str1[t])")
            sp = plot_curvilinear!(ax, lon_psi, lat_psi, vor_1m;
                                    colormap = Reverse(:RdBu), colorrange = (-clim, clim))
            Colorbar(fig[1, 2], sp, label = "s⁻¹")
            contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            add_quiver!(ax, u_east_1m, v_north_1m)
            add_mooring!(ax)
            figsave(outname_vor, fig)
            println("saved plot to ", outname_vor)
        end
 
        # -- SSH GRADIENT --
        outname_grad = joinpath(figure_path, "zeta_grad_plot_$(str2[t]).png")
        if isfile(outname_grad)
            println("already exists, skipping: ", outname_grad)
        else
            # recomputed here for the same reason as in the vorticity block
            u_east_1m = slice_at_depth(z[:, :, :, t], u_east_masked[:, :, :, t], -1.0)
            v_north_1m = slice_at_depth(z[:, :, :, t], v_north_masked[:, :, :, t], -1.0)
 
            # small canvas = relatively larger text; the extra bottom padding
            # (left, right, bottom, top) keeps the "Longitude" label from being cut off
            fig = Figure(; figkw(18, 15.3)..., figure_padding = (11, 11, 29, 11))
            ax = map_axis(fig[1, 1]; title = "SSH gradient magnitude |∇ζ|, $(str1[t])")
            sp = plot_curvilinear!(ax, lon_rho, lat_rho, zeta_grad_masked[:, :, t] .* grad_scale;
                                    colormap = :amp, colorrange = grad_range)
            Colorbar(fig[1, 2], sp, label = "10⁻⁶ (mm/km)")
            contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            add_quiver!(ax, u_east_1m, v_north_1m; scale = small_scale)
            add_mooring!(ax; scale = small_scale)
            figsave(outname_grad, fig)
            println("saved plot to ", outname_grad)
        end
    end
end

# the HPC output dir mixes roms_avg/dia/his/rst files together, so filter
# to just the "his" files (they carry zeta/temp/u/v, everything the plots
# below need) — sorted so files are processed in chronological order,
# which works here because the filenames embed a sortable timestamp
# files = sort(filter(f -> occursin("roms_avg", basename(f)) && endswith(f, ".nc"), readdir(datadir, join=true)))
# files = [joinpath(datadir, "roms_avg.20220908210000.nc")]
files = sort(filter(f -> occursin("roms_his", basename(f)) &&
                         endswith(f, ".nc") &&
                         basename(f) <= "roms_his.20221130240000.nc",
                    readdir(datadir, join=true)))
println("found $(length(files)) his files in $datadir")
# pmap hands files out to whichever worker is free, one at a time, and
# blocks here until every file is done — no manual scheduling needed.
# pmap(process_file, files)
pmap(process_file, files; on_error = ex -> println("a file failed: ", ex))


## MOVIES
# stitches each set of PNGs (across all his files/timesteps) into an mp4,
# in chronological order (filenames sort correctly since str2 timestamps
# are lexicographically ordered). Requires ffmpeg on PATH — on the HPC
# you probably need `module load ffmpeg` (or similar) before launching julia.

using FFMPEG_jll
function make_movie(prefix::String; fps = 4)
    pngs = sort(filter(f -> startswith(basename(f), prefix) && endswith(f, ".png"),
                        readdir(figure_path, join = true)))
    if isempty(pngs)
        println("no PNGs found for prefix \"$prefix\", skipping movie")
        return
    end

    listfile = joinpath(figure_path, "$(prefix)_filelist.txt")
    open(listfile, "w") do io
        for p in pngs
            println(io, "file '$(p)'")
        end
    end

    outname = joinpath(figure_path, "$(prefix)_movie.mp4")
    FFMPEG_jll.ffmpeg() do ffmpeg_path
        run(`$ffmpeg_path -y -r $fps -f concat -safe 0 -i $listfile -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p $outname`)
    end
    rm(listfile)
    println("saved movie to ", outname)
end

make_movie("zeta_plot")
make_movie("temp_depth_plot")
make_movie("speed_plot")
make_movie("vorticity_plot")
make_movie("zeta_grad_plot")
using Distributed

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "7")) - 1)   # -1 reserves a CPU for this main process
addprocs(n_workers)
println("running with $(nprocs() - 1) worker processes")

@everywhere begin
    using NCDatasets, CairoMakie, Dates, Statistics
    CairoMakie.activate!()
    include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
    include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

    grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
    datadir = "/expanse/lustre/projects/uso101/hchen54/test_63_2"   # HPC output dir — contains avg/dia/his/rst files mixed together
    figure_path = "/home/hchen54/figure/wo_dbry900m"

    ##
    mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
        ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
        ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
    end
    lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
    _, _, mask_p = uvp_masks(mask_rho)
    lon_psi, lat_psi = rho2p(lon_rho), rho2p(lat_rho)
    # no native lon_psi/lat_psi in this grid file — approximate via corner averaging

    lon_lim = extrema(lon_rho)   # shared x/y axis limits for every plot below
    lat_lim = extrema(lat_rho)

    # h here is the ROMS grid's own bathymetry (positive-down, no missing values),
    # already on the same lon_rho/lat_rho grid as everything plotted below — unlike
    # check_NCOM_comp.jl, no sign flip / coalesce-to-NaN is needed before contouring.
    bathy_levels = [500, 1000, 2000]

    skip = 30   # downsample for legible quiver arrows — plotting all 686x856 would be unreadable

    # depths (meters, negative below surface) plotted as the 6 subplots below
    depths = [-1.0, -50.0, -100.0, -150.0, -200.0, -250.0]
    depth_labels = ["1 m", "50 m", "100 m", "150 m", "200 m", "250 m"]
    depth_positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]   # row, col within the 2x3 subplot grid

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

    ## Whole-file skip — if this file's last time step already produced the
    # vorticity plot, assume the whole file was fully processed last run and
    # skip reading zeta/u/v entirely (not just skip the plotting).
    outname_vor_last = joinpath(figure_path, "vorticity_depthplot_$(str2[ntime]).png")

    if isfile(outname_vor_last)
        println("vorticity plot already exists for ", fname, ", skipping file entirely")
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

    # theta_s/theta_b/hc/N are all cheap metadata — attributes, plus a
    # variable's shape without reading its actual data — so we can always
    # fetch them without waiting on the u/v read below. This is what z/z_p
    # (needed by the depth slicing) depend on.
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

    ## vel data for vorticity
    u,v = NCDataset(fname) do ds
        ds["u"][:, :, :,:]  , ds["v"][:, :, :,:]
    end
    println(fname, " => size(u) = ", size(u))
    println(fname, " => size(v) = ", size(v))

    # average raw (unmasked) u, v onto the rho grid first — masking beforehand
    # would spread NaN from land onto valid ocean cells one row/column too far,
    # same reasoning as vorticity below
    u_rho = u2rho(u)   # still grid-relative (xi/eta), not east/north
    v_rho = v2rho(v)

    # quiver arrows plotted against true lon/lat need rotating out of the
    # grid's local xi/eta directions into true east/north, using the grid's
    # rotation angle (radians) from the grid file
    u_east = u_rho .* cos.(grid_angle) .- v_rho .* sin.(grid_angle)
    v_north = u_rho .* sin.(grid_angle) .+ v_rho .* cos.(grid_angle)
    u_east_masked = ifelse.(mask_rho .== 0, NaN32, u_east)
    v_north_masked = ifelse.(mask_rho .== 0, NaN32, v_north)

    ## VORTICITY
    # use the raw (unmasked) u, v here — masking with NaN before differencing
    # would spread NaN into neighboring cells; mask the result afterward instead
    vor_psi = vorticity_cal(u, v, pm, pn)   # relative vorticity on the PSI grid
    vor_psi_masked = ifelse.(mask_p .== 0, NaN32, vor_psi)

    ## PLOTS — one multi-depth figure per ocean_time step
    for t in 1:ntime

        # skip incomplete/fill-valued records — e.g. the last time step of a
        # file still being actively written, where every variable is left at
        # the raw NetCDF fill value (~9.97e36) instead of real data
        if maximum(abs, zeta[:, :, t]) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end

        outname_vor = joinpath(figure_path, "vorticity_depthplot_$(str2[t]).png")
        if isfile(outname_vor)
            println("already exists, skipping: ", outname_vor)
            continue
        end

        # clim = 5 * mean(abs, filter(!isnan, vor_1m))   # symmetric color limits centered on 0
        clim = 5e-5
        fig = Figure(size = (1800, 1000))
        Label(fig[0, 1:3], "Relative vorticity, $(str1[t])", fontsize = 20)

        sp = nothing   # last heatmap handle, reused for the shared Colorbar below
        for (depth, label, (row, col)) in zip(depths, depth_labels, depth_positions)
            ax = topdown_axis3(fig[row, col]; title = label)
            xlims!(ax, lon_lim...)
            ylims!(ax, lat_lim...)

            vor_d = slice_at_depth(z_p[:, :, :, t], vor_psi_masked[:, :, :, t], depth)
            sp = plot_curvilinear!(ax, lon_psi, lat_psi, vor_d;
                                    colormap = Reverse(:RdBu), colorrange = (-clim, clim))
            contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)

            u_east_d = slice_at_depth(z[:, :, :, t], u_east_masked[:, :, :, t], depth)
            v_north_d = slice_at_depth(z[:, :, :, t], v_north_masked[:, :, :, t], depth)
            quiver_curvilinear!(ax, lon_rho, lat_rho, u_east_d, v_north_d;
                                 skip = skip, lengthscale = 0.5, color = :black,
                                 shaftwidth = 1.5, tipwidth = 5, tiplength = 6)
        end
        Colorbar(fig[1:2, 4], sp, label = "s⁻¹")

        save(outname_vor, fig)
        println("saved plot to ", outname_vor)
    end
end

# the HPC output dir mixes roms_avg/dia/his/rst files together, so filter
# to just the "his" files (they carry zeta/u/v, everything the plots below
# need) — sorted so files are processed in chronological order, which works
# here because the filenames embed a sortable timestamp
files = sort(filter(f -> occursin("roms_his", basename(f)) &&
                         endswith(f, ".nc") &&
                         basename(f) <= "roms_his.20220923210000.nc",
                    readdir(datadir, join=true)))
println("found $(length(files)) his files in $datadir")
# pmap hands files out to whichever worker is free, one at a time, and
# blocks here until every file is done — no manual scheduling needed.
pmap(process_file, files; on_error = ex -> println("a file failed: ", ex))


## MOVIE
# stitches the vorticity PNGs (across all his files/timesteps) into an mp4,
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

make_movie("vorticity_depthplot")

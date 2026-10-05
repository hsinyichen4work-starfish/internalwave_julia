using NCDatasets, PyPlot, Dates, ColorSchemes, Statistics

# cmocean's "thermal" colormap, via ColorSchemes.jl, converted to a
# matplotlib colormap object for use with PyPlot's pcolormesh/cmap=...
cmap_thermal = PyPlot.matplotlib.colors.ListedColormap([[c.r, c.g, c.b] for c in colorschemes[:thermal]])
cmap_speed = PyPlot.matplotlib.colors.ListedColormap([[c.r, c.g, c.b] for c in colorschemes[:speed]])

include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
# path relative to where you launch julia (your Documents/Julia folder)

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_2"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hchen54/figure/dbry900m"

##
mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
    ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
end
lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
_, _, mask_p = uvp_masks(mask_rho)
lon_psi, lat_psi = rho2p(lon_rho), rho2p(lat_rho)   # no native lon_psi/lat_psi in this grid file — approximate via corner averaging


# the HPC output dir mixes roms_avg/dia/his/rst files together, so filter
# to just the "his" files (they carry zeta/temp/u/v, everything the plots
# below need) — sorted so files are processed in chronological order,
# which works here because the filenames embed a sortable timestamp
# files = sort(filter(f -> occursin("roms_avg", basename(f)) && endswith(f, ".nc"), readdir(datadir, join=true)))
# files = [joinpath(datadir, "roms_avg.20220908210000.nc")]
files = sort(filter(f -> occursin("roms_avg", basename(f)) &&
                         endswith(f, ".nc") &&
                         basename(f) <= "roms_avg.20220923210000.nc",
                    readdir(datadir, join=true)))
println("found $(length(files)) avg files in $datadir")

skip = 30   # downsample for legible quiver arrows — plotting all 686x856 would be unreadable

for fname in files

    ocean_time = NCDataset(fname) do ds
        ds["ocean_time"][:]
    end
    println(fname, " => size(ocean_time) = ", size(ocean_time))
    ntime = length(ocean_time)   # length(), not size() — size() returns a Tuple like (4,), not a plain number
    t_ref = DateTime(1994, 1, 1, 0, 0, 0)
    realtime = t_ref .+ Millisecond.(round.(Int, ocean_time .* 1000))   # Vector{DateTime}
    str1 = Dates.format.(realtime, "yyyy-mm-dd HH:MM:SS")                # for plot titles
    str2 = Dates.format.(realtime, "yyyymmdd_HHMM")                      # for filenames, e.g. 20220824_0730

    ## SSH DATA
    # NCDataset(fname) opens the file; the `do...end` block auto-closes it when done
    # (like fopen/fclose in MATLAB, but you don't have to remember to close it)
    # the last line of the block is what gets returned into `zeta` below
    zeta = NCDataset(fname) do ds
        ds["zeta"][:, :, :]   # sea surface height, dims xi_rho x eta_rho x time
    end
    println(fname, " => size(zeta) = ", size(zeta))

    zeta_masked = ifelse.(mask_rho .== 0, NaN32, zeta)

    ## TEMP DATA
    temp = NCDataset(fname) do ds
        ds["temp"][:, :, :,:]
    end
    println(fname, " => size(temp) = ", size(temp))

    theta_s, theta_b, hc = NCDataset(fname) do ds
        ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"]
    end
    N = size(temp,3)

    z = zeros(size(temp))
    Cs = nothing
    for t in 1:ntime
        z_dum, Cs = zlevs3(h, zeta[:, :, t], theta_s, theta_b, hc, N, "r", "new2008")
        z[:, :, :, t] = permutedims(z_dum, (2, 3, 1))
    end
    z_p = rho2p(z)
    temp_masked = ifelse.(mask_rho .== 0, NaN32, temp)

    ## vel data for vorticity
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
        # the grid is a rotated rectangle (xi/eta axes aren't aligned with lon/lat),
        # so lon_rho/lat_rho are full 2D fields, not simple 1D axes.
        # pcolormesh (from matplotlib, via PyPlot) accepts those 2D coordinate arrays
        # directly and draws each grid cell as its true rotated quadrilateral —
        # this is the Julia equivalent of MATLAB's pcolor(lon, lat, zeta).
        # Cells outside the rotated domain (the corners of its lon/lat bounding box)
        # are simply not drawn, and NaN-masked land points show as gaps too.
        fig, ax = subplots(figsize = (8, 6))
        pc = ax.pcolormesh(lon_rho, lat_rho, zeta_masked[:, :, t], shading = "auto", cmap = "RdBu_r", vmin = -1, vmax = 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Sea surface height (zeta), $(str1[t])")
        colorbar(pc, ax = ax, label = "meters")

        outname = joinpath(figure_path, "zeta_plot_$(str2[t]).png")
        savefig(outname)
        println("saved plot to ", outname)
        close(fig)

        # -- TEMP --
        # slice_at_depth does per-column vertical interpolation only (same
        # horizontal grid in/out) — z and temp_masked here are one time step's
        # (M, L, N) fields; target depth is in meters, negative below the surface.
        temp_1m = slice_at_depth(z[:, :, :, t], temp_masked[:, :, :, t], -1.0)
        temp_10m = slice_at_depth(z[:, :, :, t], temp_masked[:, :, :, t], -10.0)

        fig, axs = subplots(1, 2, figsize = (14, 6))
        for (ax, field, depth_label) in ((axs[1], temp_1m, "1 m"), (axs[2], temp_10m, "10 m"))
            pc = ax.pcolormesh(lon_rho, lat_rho, field, shading = "auto", cmap = cmap_thermal, vmin = 24, vmax = 30)
            ax.set_aspect("equal")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Temperature at $depth_label, $(str1[t])")
            colorbar(pc, ax = ax, label = "°C")
        end

        outname = joinpath(figure_path, "temp_depth_plot_$(str2[t]).png")
        savefig(outname)
        println("saved plot to ", outname)
        close(fig)

        # -- SPEED --
        speed_1m = slice_at_depth(z[:, :, :, t], speed_masked[:, :, :, t], -1.0)
        u_east_1m = slice_at_depth(z[:, :, :, t], u_east_masked[:, :, :, t], -1.0)
        v_north_1m = slice_at_depth(z[:, :, :, t], v_north_masked[:, :, :, t], -1.0)

        fig, ax = subplots(figsize = (8, 6))
        pc = ax.pcolormesh(lon_rho, lat_rho, speed_1m, shading = "auto", cmap = cmap_speed)
        colorbar(pc, ax = ax, label = "m/s")

        ax.quiver(lon_rho[1:skip:end, 1:skip:end], lat_rho[1:skip:end, 1:skip:end],
                  u_east_1m[1:skip:end, 1:skip:end], v_north_1m[1:skip:end, 1:skip:end],
                  color = "k", scale = 40)

        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Speed at 1 m, $(str1[t])")

        outname = joinpath(figure_path, "speed_plot_$(str2[t]).png")
        savefig(outname)
        println("saved plot to ", outname)
        close(fig)

        # -- VORTICITY --
        vor_1m = slice_at_depth(z_p[:, :, :, t], vor_psi_masked[:, :, :, t], -1.0)

        clim = 5 * mean(abs, filter(!isnan, vor_1m))   # symmetric color limits centered on 0
        fig, ax = subplots(figsize = (8, 6))
        pc = ax.pcolormesh(lon_psi, lat_psi, vor_1m, shading = "auto", cmap = "RdBu_r", vmin = -clim, vmax = clim)

        ax.quiver(lon_rho[1:skip:end, 1:skip:end], lat_rho[1:skip:end, 1:skip:end],
                  u_east_1m[1:skip:end, 1:skip:end], v_north_1m[1:skip:end, 1:skip:end],
                  color = "k", scale = 40)

        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Relative vorticity, surface, $(str1[t])")
        colorbar(pc, ax = ax, label = "s⁻¹")

        outname = joinpath(figure_path, "vorticity_plot_$(str2[t]).png")
        savefig(outname)
        #outname = joinpath(figure_path, "vorticity_plot_$(str2[t]).html")
        println("saved plot to ", outname)
        close(fig)
    end
end

## MOVIES
# stitches each set of PNGs (across all his files/timesteps) into an mp4,
# in chronological order (filenames sort correctly since str2 timestamps
# are lexicographically ordered). Requires ffmpeg on PATH — on the HPC
# you probably need `module load ffmpeg` (or similar) before launching julia.
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
    run(`ffmpeg -y -r $fps -f concat -safe 0 -i $listfile -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p $outname`)
    rm(listfile)
    println("saved movie to ", outname)
end

make_movie("zeta_plot")
make_movie("temp_depth_plot")
make_movie("speed_plot")
make_movie("vorticity_plot")

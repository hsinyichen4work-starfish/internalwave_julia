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
    mooring_file = "/expanse/lustre/projects/uso101/hchen54/input/roms_grd_900m_mor_edata.nc"   # french, M1..M4, CPIES1..9
    datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_0927"   # HPC output dir — contains avg/dia/his/rst files mixed together
    figure_path = "/home/hchen54/figure/dbry_0927/zeta_gradient"
    mkpath(figure_path)

    ##
    mask_rho, lon_rho, lat_rho, h, pm, pn = NCDataset(grid_fname) do ds
        ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
        ds["h"][:, :], ds["pm"][:, :], ds["pn"][:, :]
    end
    lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
    umask, vmask, _ = uvp_masks(mask_rho)
    pm_u, pn_v = rho2u(pm), rho2v(pn)   # 1/dx on U-points, 1/dy on V-points (1/m)

    lon_lim = extrema(lon_rho)   # shared x/y axis limits for every plot below
    lat_lim = extrema(lat_rho)

    # h here is the ROMS grid's own bathymetry (positive-down, no missing values),
    # already on the same lon_rho/lat_rho grid as everything plotted below — unlike
    # check_NCOM_comp.jl, no sign flip / coalesce-to-NaN is needed before contouring.
    bathy_levels = [500, 1000, 2000]

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
    moor_style = Dict("french" => (marker = :star5,   markersize = 11, color = :magenta),
                      "M"      => (marker = :diamond, markersize = 7,  color = :cyan),
                      "CPIES"  => (marker = :circle,  markersize = 6,  color = :yellow))

    # name labels sit to the right of each marker by default; these few go on
    # the left instead so neighboring labels don't run into each other
    moor_label_left = ("french", "M1", "CPIES1", "CPIES5", "CPIES9")

    # marks every mooring on the map; z = 0 puts them on the same flat plane as
    # plot_curvilinear!'s surface, and overdraw keeps them from being hidden
    # behind that surface. text! writes each mooring's own name next to its marker.
    function add_mooring!(ax)
        for g in unique(moor_group)
            k = findall(==(g), moor_group)
            st = get(moor_style, g, (marker = :rect, markersize = 6, color = :white))
            scatter!(ax, Point3f.(moor_lon[k], moor_lat[k], 0);
                     marker = st.marker, markersize = st.markersize, color = st.color,
                     strokecolor = :black, strokewidth = 0.75, overdraw = true)
        end
        for (n, lon, lat) in zip(moor_name, moor_lon, moor_lat)
            left = n in moor_label_left
            kw = (text = n, fontsize = 7, font = :bold, overdraw = true,
                  align = (left ? :right : :left, :center), offset = (left ? -5 : 5, 0))
            # white outline drawn first, underneath, so the black name stays
            # readable where it crosses a dark wave front
            text!(ax, Point3f(lon, lat, 0); kw..., color = :white, strokecolor = :white, strokewidth = 2)
            text!(ax, Point3f(lon, lat, 0); kw..., color = :black)
        end
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

    ## Whole-file skip — if this file's last time step already produced its
    # zeta gradient plot, assume the whole file was fully processed last run
    # and skip reading zeta entirely (not just skip the plotting).
    outname_grad_last = joinpath(figure_path, "zeta_grad_plot_$(str2[ntime]).png")
    if isfile(outname_grad_last)
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

    zeta_grad_masked = ifelse.(mask_rho .== 0, NaN, zeta_gradient(zeta))

    ## PLOTS — one file per ocean_time step
    for t in 1:ntime

        # skip incomplete/fill-valued records — e.g. the last time step of a
        # file still being actively written, where every variable is left at
        # the raw NetCDF fill value (~9.97e36) instead of real data
        if maximum(abs, zeta[:, :, t]) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end

        # -- SSH GRADIENT --
        outname_grad = joinpath(figure_path, "zeta_grad_plot_$(str2[t]).png")
        if isfile(outname_grad)
            println("already exists, skipping: ", outname_grad)
        else
            # small canvas = relatively larger text; the extra bottom padding
            # (left, right, bottom, top) keeps the "Longitude" label from being cut off
            fig = Figure(size = (500, 425), figure_padding = (16, 16, 40, 16))
            ax = topdown_axis3(fig[1, 1]; title = "SSH gradient magnitude |∇ζ|, $(str1[t])")
            xlims!(ax, lon_lim...)
            ylims!(ax, lat_lim...)
            sp = plot_curvilinear!(ax, lon_rho, lat_rho, zeta_grad_masked[:, :, t] .* grad_scale;
                                    colormap = :amp, colorrange = grad_range)
            Colorbar(fig[1, 2], sp, label = "10⁻⁶ (mm/km)")
            contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            add_mooring!(ax)
            save(outname_grad, fig; px_per_unit = 3.2)   # 1600x1360 pixels, so the movie stays as sharp as before
            println("saved plot to ", outname_grad)
        end
    end
end

# the HPC output dir mixes roms_avg/dia/his/rst files together, so filter
# to just the "his" files (they carry zeta, everything the plot below
# needs) — sorted so files are processed in chronological order,
# which works here because the filenames embed a sortable timestamp
files = sort(filter(f -> occursin("roms_his", basename(f)) &&
                         endswith(f, ".nc") &&
                         basename(f) <= "roms_his.20221130240000.nc",
                    readdir(datadir, join=true)))
println("found $(length(files)) his files in $datadir")
# pmap hands files out to whichever worker is free, one at a time, and
# blocks here until every file is done — no manual scheduling needed.
mkpath(figure_path)
pmap(process_file, files; on_error = ex -> println("a file failed: ", ex))


## MOVIE
# stitches the zeta gradient PNGs (across all his files/timesteps) into an mp4,
# in chronological order (filenames sort correctly since str2 timestamps
# are lexicographically ordered). ffmpeg comes from FFMPEG_jll, so no
# `module load ffmpeg` is needed.

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

make_movie("zeta_grad_plot")

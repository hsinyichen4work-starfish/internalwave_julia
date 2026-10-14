using Distributed

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")) - 1)   # -1 reserves a CPU for this main process
addprocs(n_workers)
println("running with $(nprocs() - 1) worker processes")

@everywhere begin
    using NCDatasets, CairoMakie, Dates, Statistics
    CairoMakie.activate!()
    include("/home/hsinyi/Documents/Julia/function/load_all.jl")
    include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")

    ## path setting
    child_grid = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"
    parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"
    mooring_file = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"   # french, M1..M4, CPIES1..9
    datadir = "/home/hsinyi/roms_data/NCOM_DATA_NC/"   # NCOM daily files: <yyyymmddhh>_ssh.nc, _uv.nc, ...
    figure_path = "/home/hsinyi/figure/20261009_region/NCOM/zeta_gradient"
    mkpath(figure_path)

    ## child (ROMS 900m) grid — only used for its outline and to zoom the
    # NCOM plots to the same extent as the ROMS zeta gradient plots
    lon_chd, lat_chd = NCDataset(child_grid) do ds
        ds["lon_rho"][:, :], ds["lat_rho"][:, :]
    end
    lon_chd[lon_chd .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
    lon_chd_b, lat_chd_b = grid_boundary(lon_chd, lat_chd)
    lon_lim = extrema(lon_chd)   # shared x/y axis limits for every plot below
    lat_lim = extrema(lat_chd)

    ## parent (NCOM) grid
    lon_par, lat_par, h_par, mask, dx_par, dy_par = NCDataset(parent_grid) do ds
        ds["lon"][:, :], ds["lat"][:, :], ds["h"][:, :],
        ds["mask"][:, :], ds["dx"][:, :], ds["dy"][:, :]
    end
    # h is negative-down with `missing` over land — coalesce to NaN and flip
    # sign so the bathy_levels below can be given as positive depths
    depth_par = abs.(coalesce.(h_par, NaN32))
    lon_par_f = Float64.(lon_par)
    lat_par_f = Float64.(lat_par)
    bathy_levels = [500, 1000, 2000]

    # NCOM's ssh/u/v all sit on one unstaggered grid, but the gradient is
    # still computed ROMS-style below: differences between neighboring cells
    # live "between" them (U/V points), then get averaged back onto the grid
    umask, vmask, _ = uvp_masks(mask)
    pm_u = rho2u(1 ./ Float64.(dx_par))   # 1/dx between xi-neighbors (1/m)
    pn_v = rho2v(1 ./ Float64.(dy_par))   # 1/dy between eta-neighbors (1/m)

    # mooring locations (lon in degrees_east, -180..180; lat in degrees_north),
    # taken from the global attributes of mooring_file, e.g.
    #   :M1_mor_info = "indices for M1_mor in roms_grd_900m.nc , location:(-45.4802;2.5415)"
    moor_name, moor_lon, moor_lat = NCDataset(mooring_file) do ds
        names = [replace(v, "_mor" => "") for v in keys(ds) if endswith(v, "_mor")]
        locs  = [match(r"location:\(([-\d.]+);([-\d.]+)\)", ds.attrib["$(n)_mor_info"]) for n in names]
        names, [parse(Float64, m[1]) for m in locs], [parse(Float64, m[2]) for m in locs]
    end
    moor_group = [replace(n, r"\d+$" => "") for n in moor_name]   # "M3" -> "M", "CPIES7" -> "CPIES"

    moor_style = Dict("french" => (marker = :star5,   markersize = 11, color = :magenta),
                      "M"      => (marker = :diamond, markersize = 7,  color = :cyan),
                      "CPIES"  => (marker = :circle,  markersize = 6,  color = :yellow))

    function add_mooring!(ax)
        for g in unique(moor_group)
            k = findall(==(g), moor_group)
            st = get(moor_style, g, (marker = :rect, markersize = 6, color = :white))
            scatter!(ax, Point3f.(moor_lon[k], moor_lat[k], 0);
                     marker = st.marker, markersize = st.markersize, color = st.color,
                     strokecolor = :black, strokewidth = 0.75, overdraw = true)
        end
    end

    # SSH gradient magnitude |∇ssh| (m/m), same method as the ROMS version:
    # differences on U/V points, masked across land/sea boundaries, then
    # averaged back over wet neighbors only. NCOM's ssh is NaN over land,
    # and NaN * 0 is still NaN, so land is zero-filled before differencing.
    function zeta_gradient(zeta)
        zeta0 = ifelse.(isnan.(zeta), 0.0, zeta)
        dzdx_u = diff(zeta0, dims = 1) .* pm_u .* umask
        dzdy_v = diff(zeta0, dims = 2) .* pn_v .* vmask
        dzdx = u2rho(dzdx_u) ./ u2rho(umask)
        dzdy = v2rho(dzdy_v) ./ v2rho(vmask)
        return sqrt.(dzdx .^ 2 .+ dzdy .^ 2)
    end

    grad_scale = 1e6        # plot in units of 10^-6, i.e. mm of SSH change per km
    grad_range = (0, 10)    # same color range as the ROMS plots, for direct comparison

    skip = 50   # downsample for legible quiver arrows on this ~1244x1334 grid
    quiver_kw = (lengthscale = 0.25, color = :black, shaftwidth = 1, tipwidth = 3.5, tiplength = 4)

    # reference arrow for the quiver scale, in the blank (no NCOM data) lower-left corner
    ref_speed = 1.0   # m/s
    ref_lon = lon_lim[1] + 0.05 * (lon_lim[2] - lon_lim[1])
    ref_lat = lat_lim[1] + 0.025 * (lat_lim[2] - lat_lim[1])
    function add_quiver_key!(ax)
        arrows2d!(ax, [ref_lon], [ref_lat], [0.0], [ref_speed], [0.0], [0.0]; quiver_kw...)
        text!(ax, Point3f(ref_lon, ref_lat, 0); text = "$(ref_speed) m/s", fontsize = 9,
              align = (:left, :bottom), offset = (0, 4), overdraw = true)
    end
end

@everywhere function ncom_time(fname)
    ocean_time = NCDataset(fname) do ds
        ds["MT"][:]
    end
    ntime = length(ocean_time)
    t_ref = DateTime(1900, 12, 31, 0, 0, 0)
    realtime = t_ref .+ Millisecond.(round.(Int, ocean_time .* 86400 .* 1000))   # days -> ms
    str1 = Dates.format.(realtime, "yyyy-mm-dd HH:MM:SS")                # for plot titles
    str2 = Dates.format.(realtime, "yyyymmdd_HHMM")                      # for filenames, e.g. 20220824_0730
    return ntime, str1, str2
end

@everywhere function process_file(fname)
    ntime, str1, str2 = ncom_time(fname)
    println(fname, " => ntime = ", ntime)

    # check every timestep, not just the last — NCOM's daily files overlap
    # at midnight (hour 24 of one file == hour 0 of the next), and pmap does
    # not dispatch files in chronological order
    outname(t) = joinpath(figure_path, "zeta_grad_ncom_plot_$(str2[t]).png")
    if all(t -> isfile(outname(t)), 1:ntime)
        println("all plots already exist for ", fname, ", skipping file entirely")
        return
    end

    ## SSH DATA
    zeta = NCDataset(fname) do ds
        ds["ssh"][:, :, :]   # dims xi x eta x time, NaN over land
    end
    zeta_grad_masked = ifelse.(mask .== 0, NaN, zeta_gradient(zeta))

    ## SURFACE VELOCITY — from the matching _uv.nc file (same MT times as
    # _ssh.nc). Level 1 is the top layer (zm3 ≈ -0.5 m); only that level is
    # read, since the full 100-level u/v is ~1.3 GB per time step.
    # NCOM's u/v are already true east/north on the unstaggered grid, so
    # no u2rho/v2rho averaging or rotation is needed.
    uv_fname = replace(fname, r"_ssh\.nc$" => "_uv.nc")
    u_east, v_north = NCDataset(uv_fname) do ds
        ds["u_velocity"][:, :, 1, :], ds["v_velocity"][:, :, 1, :]   # dims xi x eta x time
    end
    u_east_masked = ifelse.(mask .== 0, NaN, coalesce.(u_east, NaN))
    v_north_masked = ifelse.(mask .== 0, NaN, coalesce.(v_north, NaN))

    ## PLOTS — one file per time step
    for t in 1:ntime
        # skip incomplete/fill-valued records (raw NetCDF fill ~9.97e36)
        finite_z = filter(isfinite, zeta[:, :, t])
        if !isempty(finite_z) && maximum(abs, finite_z) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end

        outname_grad = outname(t)
        if isfile(outname_grad)
            println("already exists, skipping: ", outname_grad)
            continue
        end

        fig = Figure(size = (500, 425), figure_padding = (16, 16, 40, 16))
        ax = topdown_axis3(fig[1, 1]; title = "NCOM SSH gradient |∇ζ|, $(str1[t])")
        ax.azimuth = -pi / 2 - 1e-4   # keeps the y ticks/label on the left side
        sp = plot_curvilinear!(ax, lon_par, lat_par, zeta_grad_masked[:, :, t] .* grad_scale;
                                colormap = :amp, colorrange = grad_range)
        Colorbar(fig[1, 2], sp, label = "10⁻⁶ (mm/km)")
        contour!(ax, lon_par_f, lat_par_f, depth_par; levels = bathy_levels,
                 color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
        lines!(ax, lon_chd_b, lat_chd_b, zeros(length(lon_chd_b));
               color = :black, linewidth = 2)   # ROMS child-grid outline
        # xlim/ylim keep arrows from poking in from outside the zoomed view
        quiver_curvilinear!(ax, lon_par, lat_par, u_east_masked[:, :, t], v_north_masked[:, :, t];
                             skip = skip, xlim = lon_lim, ylim = lat_lim, quiver_kw...)
        add_quiver_key!(ax)
        add_mooring!(ax)   # last, so the markers sit on top of the arrows
        xlims!(ax, lon_lim...)
        ylims!(ax, lat_lim...)
        save(outname_grad, fig; px_per_unit = 3.2)
        println("saved plot to ", outname_grad)
    end
end

files = sort(filter(f -> endswith(f, "_ssh.nc"),   # every ssh file in datadir (2022-08-22 .. 2022-11-30)
#                         "2022082400_ssh.nc" <= basename(f) <= "2022092300_ssh.nc",
#                         "2022090100_ssh.nc" <= basename(f) <= "2022090100_ssh.nc",
                    readdir(datadir, join=true)))
println("found $(length(files)) ssh files in $datadir")
pmap(process_file, files; on_error = ex -> println("a file failed: ", ex))


## MOVIE
# stitches the PNGs into an mp4 in chronological order (str2 timestamps
# sort lexicographically). ffmpeg comes from FFMPEG_jll.

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

make_movie("zeta_grad_ncom_plot")

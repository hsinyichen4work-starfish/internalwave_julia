using NCDatasets, CairoMakie, Dates, Statistics, DSP
CairoMakie.activate!()
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_0927"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hchen54/figure/dbry_0927/energy_flux"   # separate from his_rejoint so the two scripts' skip checks don't interfere
mkpath(figure_path)

##
mask_rho, lon_rho, lat_rho = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :]
end
land = mask_rho .== 0
rho0 = 1027.5   # ds.attrib["rho0"] — up, vp in the file are per unit rho0 (m4 s-3); * rho0 -> W/m

# every record of every dia file, in time order: frames[k] = (file, record within the file, date)
dia_files = sort(filter(f -> occursin(r"^roms_dia\.\d{14}\.nc$", f), readdir(datadir)))
frames_all = [(file = joinpath(datadir, f), rec = n, date = DateTime(1994, 1, 1) + Second(round(Int, t)))
          for f in dia_files for (n, t) in enumerate(NCDataset(ds -> ds["ocean_time"][:], joinpath(datadir, f)))]
frames = frames_all
#frames = filter(fr -> DateTime(2022, 9, 1) <= fr.date < DateTime(2022, 9, 3), frames_all)
println("dia: $(length(frames)) records in $(length(dia_files)) files, ", frames[1].date, " -> ", frames[end].date)

# one record -> up, vp, F_mag on rho points, land blanked, in kW/m   [xi_rho, eta_rho]
#   up is on u points (685 x 856), vp on v points (686 x 855) — moved onto rho points (686 x 856) so they can be combined
function load_flux(fr)
    up, vp = NCDataset(fr.file) do ds
        u2rho(coalesce.(ds["up"][:, :, fr.rec], NaN)), v2rho(coalesce.(ds["vp"][:, :, fr.rec], NaN))
    end
    up .*= rho0 / 1e3
    vp .*= rho0 / 1e3
    up[land] .= NaN
    vp[land] .= NaN
    up, vp, sqrt.(up.^2 .+ vp.^2)
end

# called after each record. Every file opened leaves read buffers behind inside the HDF5 library (the one NetCDF reads
# with): ~10 MB of address space per record that Julia's own cleanup cannot reach. It is not memory in use (that stays
# flat at ~1 GB), but the login node caps address space at 8 GB (ulimit -v), so without this the loop dies with
# OutOfMemoryError after ~200 records. H5garbage_collect makes HDF5 let go of those buffers; GC.gc + malloc_trim
# (same pair as in model_check_energyflux.jl) drop Julia's temporary arrays and hand the memory back to the system.
const libhdf5 = NCDatasets.NetCDF_jll.HDF5_jll.libhdf5
function free_memory()
    GC.gc()
    ccall((:H5garbage_collect, libhdf5), Cint, ())
    Sys.islinux() && ccall(:malloc_trim, Cint, (Cint,), 0)
    nothing
end

##
# one png per record in frame_path, then all pngs joined into one movie.
# The figure is built once; each frame only swaps the data and the title.
# A frame whose png already exists is skipped, so an interrupted run can simply be started again.
frame_path = joinpath(figure_path, "dia_frames")
mkpath(frame_path)
flux_lim = 50   # kW/m
stride = 2      # plot every 2nd grid point: a panel is ~450 px across for 686 points, so nothing visible is lost,
                # and drawing is ~4x faster (CairoMakie draws every grid cell as its own little polygon)
sub(a) = a[1:stride:end, 1:stride:end]
frame_fname(fr) = joinpath(frame_path, "dia_flux_$(Dates.format(fr.date, "yyyymmddHH")).png")

up_o, vp_o, F_o = Observable.(sub.(load_flux(frames[1])))
title_o = Observable("")
fig = Figure(size = (1500, 600))
fields = [("up (along xi)", up_o, :balance, (-flux_lim, flux_lim)),
          ("vp (along eta)", vp_o, :balance, (-flux_lim, flux_lim)),
          ("F_mag", F_o, :viridis, (0, flux_lim))]
for (n, (title, data, colormap, colorrange)) in enumerate(fields)
    ax = topdown_axis3(fig[1, 2n-1]; title = title, ylabel = n == 1 ? "Latitude" : "")
    # same flat-surface trick as plot_curvilinear!, written out here because the colour is an Observable
    hm = surface!(ax, sub(lon_rho), sub(lat_rho), zeros(size(sub(lon_rho))); color = data, shading = NoShading,
                  colormap = colormap, colorrange = colorrange)
    Colorbar(fig[1, 2n], hm, label = "kW m⁻¹")
end
Label(fig[0, :], title_o, fontsize = 18)

@time for (k, fr) in enumerate(frames)
    isfile(frame_fname(fr)) && continue
    up_o[], vp_o[], F_o[] = sub.(load_flux(fr))
    title_o[] = "ROMS dia baroclinic energy flux (1-hr mean)   $(Dates.format(fr.date, "yyyy-mm-dd HH:MM"))"
    save(frame_fname(fr), fig)
    free_memory()
    k % 50 == 0 && println("frame $k / $(length(frames))")
end

##
# join the pngs into a movie (file names sort in time order). Uses the ffmpeg that comes with Makie.
movie_fname = joinpath(figure_path, "dia_flux_movie.mp4")
framerate = 12   # frames (= model hours) per second of movie
run(`$(Makie.FFMPEG_jll.ffmpeg()) -y -loglevel error -framerate $framerate -pattern_type glob -i $(joinpath(frame_path, "dia_flux_*.png"))
     -vf scale=1500:-2 -c:v libx264 -pix_fmt yuv420p $movie_fname`)
println("saved ", movie_fname)


##
# time mean over the run, leaving out the first skip_days (the model's running mean of p starts from zero, so the
# first days are not real flux). Records are added to a running sum one at a time, so memory stays small however
# long the period is. Uses frames_all, not frames, so narrowing the movie to a few days does not change the mean.
skip_days = 5
t0 = frames_all[1].date + Day(skip_days)
frames_avg = filter(fr -> fr.date >= t0, frames_all)
println("mean: $(length(frames_avg)) records, ", frames_avg[1].date, " -> ", frames_avg[end].date)

up_sum = zeros(size(lon_rho))
vp_sum = zeros(size(lon_rho))
@time for (k, fr) in enumerate(frames_avg)
    up, vp, _ = load_flux(fr)   # kW/m, on rho points
    up_sum .+= up
    vp_sum .+= vp
    free_memory()
    k % 100 == 0 && println("record $k / $(length(frames_avg))")
end
up_mean = up_sum ./ length(frames_avg)
vp_mean = vp_sum ./ length(frames_avg)
F_mag_mean = sqrt.(up_mean.^2 .+ vp_mean.^2)   # magnitude of the mean flux (not the mean of the hourly magnitudes)
println("mean flux [kW/m]:  up ", max_min(filter(isfinite, up_mean)), "  vp ", max_min(filter(isfinite, vp_mean)))

##
mean_lim = 30   # kW/m
fig_mean = Figure(size = (1500, 600))
fields_mean = [("up (along xi)", up_mean, :balance, (-mean_lim, mean_lim)),
               ("vp (along eta)", vp_mean, :balance, (-mean_lim, mean_lim)),
               ("F_mag", F_mag_mean, :viridis, (0, mean_lim))]
for (n, (title, data, colormap, colorrange)) in enumerate(fields_mean)
    ax = topdown_axis3(fig_mean[1, 2n-1]; title = title, ylabel = n == 1 ? "Latitude" : "")
    hm = plot_curvilinear!(ax, lon_rho, lat_rho, data; colormap = colormap, colorrange = colorrange)
    Colorbar(fig_mean[1, 2n], hm, label = "kW m⁻¹")
end
Label(fig_mean[0, :], "ROMS dia baroclinic energy flux, mean of $(Dates.format(frames_avg[1].date, "yyyy-mm-dd HH:MM")) -> " *
      "$(Dates.format(frames_avg[end].date, "yyyy-mm-dd HH:MM"))  ($(length(frames_avg)) hourly records)", fontsize = 18)
mean_fname = joinpath(figure_path, "dia_flux_mean_$(Dates.format(frames_avg[1].date, "yyyymmdd"))_$(Dates.format(frames_avg[end].date, "yyyymmdd")).png")
save(mean_fname, fig_mean)
println("saved ", mean_fname)
fig_mean

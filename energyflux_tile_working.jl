using Distributed

# worker processes for the parallel plotting loop. Added only once, so re-running this block in the same session is safe.
# Each worker is told how much memory it may use before it must clean up (its share of the job's memory).
n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "7")) - 1)   # -1 reserves a CPU for this main process
job_mem_mb = parse(Int, get(ENV, "SLURM_MEM_PER_NODE", "64000"))
heap_mb = floor(Int, (job_mem_mb - 4000) * 0.85 / n_workers)
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # several processes read the same files; workers inherit this
nprocs() == 1 && addprocs(n_workers; exeflags = "--heap-size-hint=$(heap_mb)M")
println("running with $(nprocs() - 1) worker processes")

# everything below up to "end   # @everywhere" is set up on every process: packages, paths, grid, the list of saved
# pieces, and the functions that read and plot them
@everywhere begin
    using NCDatasets, CairoMakie, Dates, Statistics
    CairoMakie.activate!()
    include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
    include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

    grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
    datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_0927"   # HPC output dir — contains avg/dia/his/rst files mixed together
    figure_path = "/home/hchen54/figure/dbry_0927/energy_flux"   # separate from his_rejoint so the two scripts' skip checks don't interfere
    nc_dum_save = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_0927/energyflux_dum"
    mkpath(figure_path)
    mkpath(nc_dum_save)

    mask_rho, lon_rho, lat_rho, h, grid_angle = NCDataset(grid_fname) do ds
        ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :], ds["h"][:, :], ds["angle"][:, :]
    end

    # (1) which pieces are there. File names are energyflux_w<window>_tile_<ix>_<iy>.nc, one per tile and time window
    piece_files = filter(f -> occursin(r"^energyflux_w\d+_tile_\d+_\d+\.nc$", f), readdir(nc_dum_save))
    piece_id(f) = Tuple(parse.(Int, match(r"^energyflux_w(\d+)_tile_(\d+)_(\d+)\.nc$", f).captures))   # (iw, ix, iy)
    piece_ids = piece_id.(piece_files)
    nwindow = maximum(id -> id[1], piece_ids)
    ndomx, ndomy = maximum(id -> id[2], piece_ids), maximum(id -> id[3], piece_ids)

    # (2) read one time window: put every tile's core back at its place in the full grid.
    #   Fx["M2"], Fy["hp24"], ...   depth-integrated energy flux [W m-1], [xi, eta, time], along the grid's xi / eta directions
    #   record   index of each time in the whole run's his records;   date   the matching DateTime
    # Points no tile covered stay NaN, and a missing tile is reported.
    function read_window(iw)
        files = [f for (f, id) in zip(piece_files, piece_ids) if id[1] == iw]
        record, ocean_time, bands = NCDataset(joinpath(nc_dum_save, files[1])) do ds
            ds["record"][:], ds["ocean_time"][:], [k[4:end] for k in keys(ds) if startswith(k, "Fx_")]
        end
        Fx = Dict(b => fill(NaN32, size(mask_rho)..., length(record)) for b in bands)
        Fy = Dict(b => fill(NaN32, size(mask_rho)..., length(record)) for b in bands)
        for f in files
            NCDataset(joinpath(nc_dum_save, f)) do ds
                ds["record"][:] == record || error("$f covers different records than $(files[1])")
                i, j = ds["xi_rho"][:], ds["eta_rho"][:]   # where this tile sits in the full grid
                for b in bands
                    Fx[b][i, j, :] = ds["Fx_$b"][:, :, :]
                    Fy[b][i, j, :] = ds["Fy_$b"][:, :, :]
                end
            end
        end
        nhole = count(isnan, @view Fx[bands[1]][:, :, 1])
        nhole == 0 || @warn "window $iw: $nhole grid points were not covered by any tile"
        date = DateTime(1994, 1, 1) .+ Second.(round.(Int, ocean_time))
        println("window $iw: read $(length(files)) tiles, bands $bands, $(length(record)) records  ", date[1], " -> ", date[end])
        (; record, date, Fx, Fy)
    end

    # (2b) read one time only, over the whole domain — light, does not need the full read in (3).
    #   returns Fx["M2"], Fy["hp24"], ... each [xi, eta] in W m-1 (along the grid's xi / eta directions), plus the date
    #   and the record. The window that kept this time is found from the files; a time no window kept is an error
    #   (the first and last 3 days of the run are filter edges and were never saved).
    function read_time(target::DateTime)
        for iw in 1:nwindow
            files = [f for (f, id) in zip(piece_files, piece_ids) if id[1] == iw]
            record, ocean_time, bands = NCDataset(joinpath(nc_dum_save, files[1])) do ds
                ds["record"][:], ds["ocean_time"][:], [k[4:end] for k in keys(ds) if startswith(k, "Fx_")]
            end
            date = DateTime(1994, 1, 1) .+ Second.(round.(Int, ocean_time))
            n = findfirst(==(target), date)   # position of this time in the window's files
            n === nothing && continue
            Fx = Dict(b => fill(NaN32, size(mask_rho)) for b in bands)
            Fy = Dict(b => fill(NaN32, size(mask_rho)) for b in bands)
            for f in files
                NCDataset(joinpath(nc_dum_save, f)) do ds
                    ds["record"][n] == record[n] || error("$f covers different records than $(files[1])")
                    i, j = ds["xi_rho"][:], ds["eta_rho"][:]   # where this tile sits in the full grid
                    for b in bands
                        Fx[b][i, j] = ds["Fx_$b"][:, :, n]
                        Fy[b][i, j] = ds["Fy_$b"][:, :, n]
                    end
                end
            end
            nhole = count(isnan, Fx[bands[1]])
            nhole == 0 || @warn "$target: $nhole grid points were not covered by any tile"
            println("read $target from window $iw (record $(record[n]) of the run), bands $bands")
            return (; date = target, record = record[n], Fx, Fy)
        end
        error("$target is not in any saved window")
    end

    # called after each time step. Every file opened leaves read buffers behind inside the HDF5 library (the one NetCDF
    # reads with) that Julia's own cleanup cannot reach; H5garbage_collect makes HDF5 let go of them, GC.gc + malloc_trim
    # drop Julia's temporary arrays and hand the memory back to the system (same as in model_dia_output.jl).
    const libhdf5 = NCDatasets.NetCDF_jll.HDF5_jll.libhdf5
    function free_memory()
        GC.gc()
        ccall((:H5garbage_collect, libhdf5), Cint, ())
        Sys.islinux() && ccall(:malloc_trim, Cint, (Cint,), 0)
        nothing
    end

    # plot one time: one figure per band, panels Fx (along xi), Fy (along eta) and |F| — same layout as model_dia_output.jl
    # band_info["M2"] = "band-pass 9-15 hr, Butterworth order 4", read from the saved files so it stays right if filters change
    band_info = NCDataset(joinpath(nc_dum_save, piece_files[1])) do ds
        Dict(k[4:end] => split(ds[k].attrib["long_name"], "along xi, ")[end] for k in keys(ds) if startswith(k, "Fx_"))
    end
    # one folder per band, one png per time step:  figure_path/m2flux, figure_path/hp24flux, ...
    frame_dir(b) = joinpath(figure_path, "$(lowercase(b))flux")
    foreach(b -> mkpath(frame_dir(b)), keys(band_info))
    flux_lim = 50   # kW/m
    stride = 2      # plot every 2nd grid point: a panel is ~450 px across for 686 points, so nothing visible is lost,
                    # and drawing is ~4x faster (CairoMakie draws every grid cell as its own little polygon)
    sub(a) = a[1:stride:end, 1:stride:end]
    frame_fname(b, target) = joinpath(frame_dir(b), "energyflux_$(b)_$(Dates.format(target, "yyyymmddHH")).png")

    # read one time and save its figures (one per band). Times whose pngs all exist already are skipped unless
    # overwrite = true, so an interrupted run can simply be started again. Returns the figures (nothing when skipped).
    function plot_time(target::DateTime; overwrite = false)
        bands = sort(collect(keys(band_info)))
        if !overwrite && all(b -> isfile(frame_fname(b, target)), bands)
            return nothing
        end
        snap = read_time(target)
        figs = Dict{String, Figure}()
        for b in bands
            Fmag = sqrt.(snap.Fx[b] .^ 2 .+ snap.Fy[b] .^ 2)   # flux magnitude |F|, W m-1
            fig = Figure(size = (1500, 600))
            fields = [("Fx (along xi)", snap.Fx[b] ./ 1e3, :balance, (-flux_lim, flux_lim)),   # W/m -> kW/m
                      ("Fy (along eta)", snap.Fy[b] ./ 1e3, :balance, (-flux_lim, flux_lim)),
                      ("|F|", Fmag ./ 1e3, :viridis, (0, flux_lim))]
            for (n, (title, data, colormap, colorrange)) in enumerate(fields)
                ax = topdown_axis3(fig[1, 2n-1]; title = title, ylabel = n == 1 ? "Latitude" : "")
                hm = plot_curvilinear!(ax, sub(lon_rho), sub(lat_rho), sub(data); colormap = colormap, colorrange = colorrange)
                Colorbar(fig[1, 2n], hm, label = "kW m⁻¹")
            end
            Label(fig[0, :], "baroclinic energy flux, $b ($(band_info[b]))   $(Dates.format(target, "yyyy-mm-dd HH:MM"))", fontsize = 18)
            save(frame_fname(b, target), fig)
            figs[b] = fig
        end
        free_memory()
        figs
    end

end   # @everywhere

##
println("$(length(piece_files)) pieces: $nwindow time windows × $ndomx × $ndomy tiles")
for iw in 1:nwindow
    absent = [(ix, iy) for ix in 1:ndomx, iy in 1:ndomy if (iw, ix, iy) ∉ piece_ids]
    info = NCDataset(joinpath(nc_dum_save, piece_files[findfirst(id -> id[1] == iw, piece_ids)])) do ds; ds.attrib["window"] end
    println("  ", info, isempty(absent) ? "   — all $(ndomx * ndomy) tiles there" : "   — MISSING tiles (ix, iy): $absent")
end

##
# one time step, interactively
snap = read_time(DateTime(2022, 9, 2, 1))   # snap.Fx["M2"], snap.Fy["M2"], ...  [xi, eta]

# flux magnitude |F| = sqrt(Fx^2 + Fy^2) in each band [W m-1]:  Fmag["M2"], Fmag["hp24"]   [xi, eta]
# (the magnitude does not depend on the direction the axes point, so no rotation by grid_angle is needed for it)
Fmag = Dict(b => sqrt.(snap.Fx[b] .^ 2 .+ snap.Fy[b] .^ 2) for b in keys(snap.Fx))
for b in keys(Fmag)
    println("|F| $b: median ", median(Fmag[b]), "   max ", maximum(Fmag[b]), " W m-1")
end

size(snap.Fx["M2"])     # (686, 856)  ->  [xi, eta]
size(snap.Fx["hp24"])   # (686, 856)
max_min(snap.Fx["M2"])
max_min(snap.Fx["hp24"])

# its figures, saved to frame_dir(band) and shown here
figs = plot_time(DateTime(2022, 9, 2, 1); overwrite = true)
isinteractive() && display(figs["M2"])   # shown only in an interactive session, not in a batch job

##
# every saved time step, in parallel: pmap hands the times out to whichever worker is free and blocks until all are
# done. The saved times are listed from one piece of each window (all tiles of a window hold the same times).
all_dates = reduce(vcat, [NCDataset(joinpath(nc_dum_save, piece_files[findfirst(id -> id[1] == iw, piece_ids)])) do ds
                              DateTime(1994, 1, 1) .+ Second.(round.(Int, ds["ocean_time"][:]))
                          end for iw in 1:nwindow])
println("plotting $(length(all_dates)) time steps x $(length(band_info)) bands into ", join(frame_dir.(sort(collect(keys(band_info)))), ", "))
@time pmap(all_dates; on_error = ex -> println("a time step failed: ", ex)) do target
    plot_time(target)
    nothing   # don't send the figures back to the main process
end

##
# time mean over all saved times (or a shorter period: set mean_start / mean_end), one band at a time.
# Read tile by tile and added to a running sum, so the whole time series is never in memory at once.
#   Fx_mean["M2"], Fy_mean["M2"], ... [xi, eta] in kW m-1 (along the grid's xi / eta directions)
#   F_mag_mean[b] = magnitude of the mean flux (not the mean of the hourly magnitudes), as in model_dia_output.jl
mean_start, mean_end = first(all_dates), last(all_dates)
nmean = count(d -> mean_start <= d <= mean_end, all_dates)
println("mean: $nmean hourly records, ", mean_start, " -> ", mean_end)
Fx_sum = Dict(b => zeros(size(mask_rho)) for b in keys(band_info))
Fy_sum = Dict(b => zeros(size(mask_rho)) for b in keys(band_info))
@time for f in piece_files
    NCDataset(joinpath(nc_dum_save, f)) do ds
        date = DateTime(1994, 1, 1) .+ Second.(round.(Int, ds["ocean_time"][:]))
        n = findall(d -> mean_start <= d <= mean_end, date)   # this piece's times inside the mean period
        isempty(n) && return
        i, j = ds["xi_rho"][:], ds["eta_rho"][:]   # where this tile sits in the full grid
        for b in keys(band_info)
            Fx_sum[b][i, j] .+= dropdims(sum(Float64, ds["Fx_$b"][:, :, n], dims = 3), dims = 3)
            Fy_sum[b][i, j] .+= dropdims(sum(Float64, ds["Fy_$b"][:, :, n], dims = 3), dims = 3)
        end
    end
    free_memory()   # each file read leaves HDF5 buffers behind; without this the loop runs out of address space
end
Fx_mean = Dict(b => Fx_sum[b] ./ nmean ./ 1e3 for b in keys(band_info))   # W/m -> kW/m
Fy_mean = Dict(b => Fy_sum[b] ./ nmean ./ 1e3 for b in keys(band_info))
F_mag_mean = Dict(b => sqrt.(Fx_mean[b] .^ 2 .+ Fy_mean[b] .^ 2) for b in keys(band_info))
Fx_sum = Fy_sum = nothing
for b in sort(collect(keys(band_info)))
    println("mean flux $b [kW/m]:  Fx ", max_min(Fx_mean[b]), "  Fy ", max_min(Fy_mean[b]), "  |F| max ", maximum(F_mag_mean[b]))
end

##
# plot the mean, one figure per band, saved in figure_path
mean_lim = 30   # kW/m
for b in sort(collect(keys(band_info)))
    fig_mean = Figure(size = (1500, 600))
    fields_mean = [("Fx (along xi)", Fx_mean[b], :balance, (-mean_lim, mean_lim)),
                   ("Fy (along eta)", Fy_mean[b], :balance, (-mean_lim, mean_lim)),
                   ("|mean F|", F_mag_mean[b], :viridis, (0, mean_lim))]
    for (n, (title, data, colormap, colorrange)) in enumerate(fields_mean)
        ax = topdown_axis3(fig_mean[1, 2n-1]; title = title, ylabel = n == 1 ? "Latitude" : "")
        hm = plot_curvilinear!(ax, lon_rho, lat_rho, data; colormap = colormap, colorrange = colorrange)
        Colorbar(fig_mean[1, 2n], hm, label = "kW m⁻¹")
    end
    Label(fig_mean[0, :], "baroclinic energy flux, $b ($(band_info[b])), mean of $(Dates.format(mean_start, "yyyy-mm-dd HH:MM")) -> " *
          "$(Dates.format(mean_end, "yyyy-mm-dd HH:MM"))  ($nmean hourly records)", fontsize = 18)
    mean_fname = joinpath(figure_path, "energyflux_mean_$(b)_$(Dates.format(mean_start, "yyyymmdd"))_$(Dates.format(mean_end, "yyyymmdd")).png")
    save(mean_fname, fig_mean)
    println("saved ", mean_fname)
    isinteractive() && display(fig_mean)
end

# ##
# # (3) all windows, joined in time. The saved pieces already hold only each window's kept records: where two windows
# #     overlapped, the overlap was split at its midpoint when the pieces were made, so every record was taken from the
# #     window where it is farthest from an edge. Nothing has to be chosen here — the windows just go one after another.
# ws = [read_window(iw) for iw in 1:nwindow]   # ws[1] is the first window on its own
# record = reduce(vcat, [w.record for w in ws])
# record == first(record):last(record) || error("the windows do not join up: a record is missing or appears twice")
# date = reduce(vcat, [w.date for w in ws])
# Fx = Dict(b => cat((w.Fx[b] for w in ws)...; dims = 3) for b in keys(ws[1].Fx))   # [xi, eta, time] over the whole run
# Fy = Dict(b => cat((w.Fy[b] for w in ws)...; dims = 3) for b in keys(ws[1].Fy))
# ws = nothing   # free memory: everything is in Fx, Fy now
# println("joined $nwindow windows: records $(first(record))-$(last(record)), ", date[1], " -> ", date[end], ",  ", size(first(values(Fx))))

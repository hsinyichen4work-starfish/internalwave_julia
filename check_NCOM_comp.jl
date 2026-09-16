using NCDatasets, CairoMakie, Dates, Statistics

CairoMakie.activate!()   # explicit, in case another Makie backend gets loaded too

include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")

## path setting
child_grid = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"
parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"
datadir = "/home/hsinyi/roms_data/NCOM_DATA_NC/"
figure_path = "/home/hsinyi/figure/20260914_julia_outputtest/NCOM_FIG"

##
lon_chd, lat_chd, h_chd, angle_chd = NCDataset(child_grid) do ds
    ds["lon_rho"][:, :], ds["lat_rho"][:, :], ds["h"][:,:], ds["angle"][:,:]
end
lon_chd[lon_chd .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
lon_chd_b, lat_chd_b = grid_boundary(lon_chd, lat_chd)   # child-grid outline, for overlaying on parent-grid plots
lon_chd_lims = extrema(lon_chd)   # child-grid extent, to zoom parent-grid plots to it (xlim/ylim equivalent)
lat_chd_lims = extrema(lat_chd)

lon_par, lat_par, h_par, angle_par, mask, zm3, kb = NCDataset(parent_grid) do ds
    ds["lon"][:, :], ds["lat"][:, :], ds["h"][:,:], ds["ang"][:,:],
    ds["mask"][:, :], ds["zm3"][:,:,:], ds["kb"][:,:]
end
# h is negative-down (~-5 to -5078 m) with `missing` over land — contour!
# can't dim-convert a Union{Missing,_} matrix, and levels are meant as
# positive depths, so coalesce to NaN and flip sign before contouring.
depth_par = abs.(coalesce.(h_par, NaN32))
lon_par_f = Float64.(lon_par)
lat_par_f = Float64.(lat_par)

##
files_ssh = sort(filter(f -> endswith(f, "_ssh.nc") &&
#                         "2022082400_ssh.nc" <= basename(f) <= "2022092300_ssh.nc",
                         "2022082400_ssh.nc" <= basename(f) <= "2022082400_ssh.nc",
                    readdir(datadir, join=true)))
println("found $(length(files_ssh)) ssh files in $datadir")

for fname in files_ssh
    ocean_time = NCDataset(fname) do ds
        ds["MT"][:]
    end
    println(fname, " => size(ocean_time) = ", size(ocean_time))
    ntime = length(ocean_time)   # length(), not size() — size() returns a Tuple like (4,), not a plain number
    t_ref = DateTime(1900, 12, 31, 0, 0, 0)
    realtime = t_ref .+ Millisecond.(round.(Int, ocean_time .* 86400 .* 1000))   # days -> ms
    str1 = Dates.format.(realtime, "yyyy-mm-dd HH:MM:SS")                # for plot titles
    str2 = Dates.format.(realtime, "yyyymmdd_HHMM")                      # for filenames, e.g. 20220824_0730

    println(str1)

    zeta = NCDataset(fname) do ds
        ds["ssh"][:, :, :]   # sea surface height, dims xi_rho x eta_rho x time
    end
    println(fname, " => size(zeta) = ", size(zeta))
 
    zeta_masked = ifelse.(mask .== 0, NaN32, zeta)

    for t in 1:ntime
 
        # skip incomplete/fill-valued records — e.g. the last time step of a
        # file still being actively written, where every variable is left at
        # the raw NetCDF fill value (~9.97e36) instead of real data
        if maximum(abs, zeta[:, :, t]) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end
 
        # -- SSH --
        outname_zeta = joinpath(figure_path, "zeta_ncom_plot_$(str2[t]).png")
        if isfile(outname_zeta)
            println("already exists, skipping: ", outname_zeta)
        else
            fig = Figure(size = (800, 600))
            ax = topdown_axis3(fig[1, 1]; title = "Sea surface height (zeta), $(str1[t])")
            sp = plot_curvilinear!(ax, lon_par, lat_par, zeta_masked[:, :, t];
                                    colormap = Reverse(:RdBu), colorrange = (-1, 1))
            Colorbar(fig[1, 2], sp, label = "meters")
            contour!(ax, lon_par_f, lat_par_f, depth_par; levels = [500, 1000, 2000],
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            lines!(ax, lon_chd_b, lat_chd_b, zeros(length(lon_chd_b));
                   color = :black, linewidth = 2)
            xlims!(ax, lon_chd_lims...)
            ylims!(ax, lat_chd_lims...)
            save(outname_zeta, fig)
            println("saved plot to ", outname_zeta)
        end
    end
end



##
files_ts = sort(filter(f -> endswith(f, "_ts.nc") &&
#                         "2022082400_ts.nc" <= basename(f) <= "2022092300_ts.nc",
                         "2022082400_ts.nc" <= basename(f) <= "2022082400_ts.nc",
                    readdir(datadir, join=true)))
println("found $(length(files_ts)) temp files in $datadir")

for fname in files_ts
    ocean_time = NCDataset(fname) do ds
        ds["MT"][:]
    end
    println(fname, " => size(ocean_time) = ", size(ocean_time))
    ntime = length(ocean_time)   # length(), not size() — size() returns a Tuple like (4,), not a plain number
    t_ref = DateTime(1900, 12, 31, 0, 0, 0)
    realtime = t_ref .+ Millisecond.(round.(Int, ocean_time .* 86400 .* 1000))   # days -> ms
    str1 = Dates.format.(realtime, "yyyy-mm-dd HH:MM:SS")                # for plot titles
    str2 = Dates.format.(realtime, "yyyymmdd_HHMM")                      # for filenames, e.g. 20220824_0730

    println(str1)

    # layer_temperature is (xi, eta, z, time) at ~1.3 GB per time step
    # (Float64) on this grid — reading all 25 steps at once like the ssh
    # loop does would need ~33 GB, so ds_ts is kept open and indexed one
    # time step at a time inside the loop below instead.
    ds_ts = NCDataset(fname)

    for t in 1:ntime
        temp_t = ds_ts["layer_temperature"][:, :, :, t]

        # two-step mask: land (horizontal, mask==0) and below-seafloor
        # (vertical, level index > kb[i,j] at that column) — both are
        # already NaN in the raw file (checked directly), but masking
        # explicitly means temp_masked doesn't depend on that assumption
        # holding for every file.
        kb0 = coalesce.(kb, 0)                          # missing (land) -> 0, so every level there gets masked
        zidx = reshape(1:size(temp_t, 3), 1, 1, :)
        temp_masked = ifelse.((mask .== 0) .| (zidx .> kb0), NaN32, temp_t)

        # skip incomplete/fill-valued records, same idea as the ssh loop —
        # but layer_temperature is expected to already contain NaN below
        # kb[i,j] (below the seafloor at that column), so those NaNs must
        # be excluded first or they'd swallow a real 1e30 fill value too
        # (maximum() propagates NaN rather than ignoring it).
        finite_temp = filter(isfinite, temp_t)
        if !isempty(finite_temp) && maximum(abs, finite_temp) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end

        # -- TEMP --
        outname_temp = joinpath(figure_path, "temp_ncom_plot_$(str2[t]).png")
        if isfile(outname_temp)
            println("already exists, skipping: ", outname_temp)
        else
            temp_1m = slice_at_depth_ncom(zm3, kb, temp_masked, -1.0)
            temp_100m = slice_at_depth_ncom(zm3, kb, temp_masked, -100.0)

            fig = Figure(size = (1400, 600))
            for (col, (field, depth_label, crange)) in enumerate(((temp_1m, "1 m", (24, 30)), (temp_100m, "100 m", (15, 30))))
                ax = topdown_axis3(fig[1, 2col - 1]; title = "Temperature at $depth_label, $(str1[t])")
                sp = plot_curvilinear!(ax, lon_par, lat_par, field; colormap = :thermal, colorrange = crange)
                Colorbar(fig[1, 2col], sp, label = "°C")
            end
            save(outname_temp, fig)
            println("saved plot to ", outname_temp)
        end
    end

    close(ds_ts)
end
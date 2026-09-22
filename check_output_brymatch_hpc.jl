using Distributed

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "7")) - 1)   # -1 reserves a CPU for this main process
addprocs(n_workers)
println("running with $(nprocs() - 1) worker processes")

@everywhere begin

using NCDatasets, CairoMakie, Dates, Statistics

CairoMakie.activate!()   # explicit, in case another Makie backend gets loaded too

include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")   # topdown_axis3, plot_curvilinear! (surface!-as-pcolormesh trick, reused below for time/depth)
# path relative to where you launch julia (your Documents/Julia folder)

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc" ;  # the grid_file listed in the .nc's global attributes
bry_dir = "/expanse/lustre/projects/uso101/hchen54/input/bry_63"
# each day's bry/dbry file is named after that day at hour 00, e.g. 2022090100
bry_fname_for(date)  = joinpath(bry_dir, "roms_bry_900m_$(Dates.format(date, "yyyymmdd"))00.nc")
dbry_fname_for(date) = joinpath(bry_dir, "roms_dbry_flux_900m_$(Dates.format(date, "yyyymmdd"))00.nc")

datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_2"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hchen54/figure/dbry900m/bry_match"   # figpath(subfolder, filename) comes from load_all_hpc.jl -> load_usefultool_fun.jl

##
mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
    ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
end
lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
t_ref = DateTime(1994, 1, 1, 0, 0, 0)

# ---- boundary-generic indexing -------------------------------------------
# Every ROMS field comes back from NCDatasets with xi as dimension 1 and eta
# as dimension 2 (west=1/east=end along xi; south=1/north=end along eta) —
# true whether the field lives on the rho, u, or v sub-grid, so one rule
# covers zeta/ubar/vbar/temp/salt/u/v and the grid statics alike.
bnd_dim(boundary) = boundary in ("east", "west") ? 1 : 2
bnd_edge_index(A, boundary) = boundary in ("east", "north") ? size(A, bnd_dim(boundary)) : 1

# 1D boundary line pulled out of an already-loaded grid array (mask/lon/lat/h)
bnd_line(A, boundary) = selectdim(A, bnd_dim(boundary), bnd_edge_index(A, boundary))

# Reads *only* the boundary slice directly off disk — the his files are ~6 GB
# each, so this avoids ever loading a full field just to keep one edge of it.
function read_bnd(ds, varname, boundary)
    var = ds[varname]
    dim = bnd_dim(boundary)
    i = bnd_edge_index(var, boundary)
    nd = ndims(var)
    idx = dim == 1 ? (i, ntuple(_ -> Colon(), nd - 1)...) : (Colon(), i, ntuple(_ -> Colon(), nd - 2)...)
    return var[idx...]
end

# 3 sample points per boundary, as fractions of that boundary's length —
# these fractions reproduce the original [200, 400, 600] picks exactly for
# east/west (856 eta_rho points) and scale proportionally for north/south
# (686 xi_rho points).
nchoose_fracs = [200, 400, 600] ./ 856

target_depths = Dict("1m" => -1.0, "10m" => -10.0, "100m" => -100.0)
depth_order = ["1m", "10m", "100m"]   # explicit order since target_depths is a Dict (unordered)

# up_east/up_west/vp_north/vp_south are the HF baroclinic energy flux
# (rho0-normalized) at each open boundary, read straight from dbry_fname —
# these already live on the boundary line (eta_rho for east/west, xi_rho for
# north/south), no u/v extraction needed. A sign flip here means the flux —
# and with it the dynamic (radiation-type) boundary condition switch — has
# changed between incoming and outgoing.
flux_specs = [
    ("up_east",  "east"),
    ("up_west",  "west"),
    ("vp_north", "north"),
    ("vp_south", "south"),
]

# zlevs3 requires h/zeta as 2D (M,L) arrays (it does `M, L = size(h)`), but a
# boundary line is 1D — so h/zeta here are reshaped to (Npts,1) rather than
# left as plain Vectors, matching the (M,L) convention with a singleton L.
#
# extract_depth_timeseries: slice_at_depth expects z/F shaped (M,L,N) with
# depth last, called once per time step, whereas z/z_bry here are
# (Npts,N,ntime): depth in the *middle*, time not yet looped. So each time
# slice gets reshaped to (Npts,1,N) — a dummy singleton L=1.
function extract_depth_timeseries(z, F, target_depth)
    Npts, N, ntime = size(z)
    out = zeros(Float32, Npts, ntime)
    for t in 1:ntime
        zt = reshape(view(z, :, :, t), Npts, 1, N)
        Ft = reshape(view(F, :, :, t), Npts, 1, N)
        out[:, t] = slice_at_depth(zt, Ft, target_depth)[:, 1]
    end
    return out
end

# same 3-row (points) x 1-col timeseries layout for zeta/ubar/vbar
function plot_bry_timeseries(varname, unit_label, boundary, datestr, datestr_save, lon_line, lat_line, nchoose,
                              bry_realtime, model_realtime, bry_series, model_series)
    subfolder = "$(varname)_$(boundary)_bry"
    outname = figpath(subfolder, "$(subfolder)_$(datestr_save).png")
    fig = Figure(size = (800, 650))
    for (i, idx) in enumerate(nchoose)
        ax = Axis(fig[i, 1], xlabel = "time", ylabel = varname,
        title = "$varname at lon = $(round(lon_line[idx], digits=2)), lat = $(round(lat_line[idx], digits=2))")
        scatterlines!(ax, bry_realtime, bry_series[idx,:], color = :blue,
            linewidth = 3, marker = :circle, markersize = 18)
        scatterlines!(ax, model_realtime, model_series[idx,:], color = :orange,
            linewidth = 2, marker = :circle, markersize = 10, linestyle = :dash)
    end
    Label(fig[0, :], "$varname ($unit_label) at $datestr — $boundary boundary", fontsize = 24, font = :bold, tellwidth = false)
    rowsize!(fig.layout, 0, Fixed(40))   # keep the label row thin so it doesn't steal space from the axes
    save(outname, fig)
    println("saved plot to ", outname)
end

end   # @everywhere begin — shared setup/data/helpers, available on every worker

# Runs the full extract+plot pipeline for one boundary only. Called once per
# boundary from process_day, so only one boundary's worth of his-file arrays
# is ever live in memory at a time — the previous boundary's arrays are local
# to the previous call and get GC'd afterwards.
@everywhere function process_boundary(boundary::String, bry_fname::String, his_files::Vector{String}, theta_s, theta_b, hc)
    println("\n=== processing boundary: $boundary ===")

    temp_b, salt_b, u_b, v_b, ubar_b, vbar_b, zeta_b, bry_time = NCDataset(bry_fname) do ds
        ds["temp_$boundary"][:,:,:], ds["salt_$boundary"][:,:,:], ds["u_$boundary"][:,:,:], ds["v_$boundary"][:,:,:],
        ds["ubar_$boundary"][:,:], ds["vbar_$boundary"][:,:], ds["zeta_$boundary"][:,:], ds["bry_time"][:]
    end
    bry_realtime = t_ref .+ Millisecond.(round.(Int, bry_time .* 86400 .* 1000))   # bry_time is in DAYS, not seconds
    datestr = Dates.format.(t_ref .+ Millisecond.(round.(Int, mean(bry_time) .* 86400 .* 1000)), "yyyy-mm-dd")
    datestr_save = Dates.format.(t_ref .+ Millisecond.(round.(Int, mean(bry_time) .* 86400 .* 1000)), "yyyymmdd")

    lon_line = bnd_line(lon_rho, boundary)
    lat_line = bnd_line(lat_rho, boundary)
    h_line = bnd_line(h, boundary)
    Lline = length(lon_line)
    nchoose = clamp.(round.(Int, nchoose_fracs .* Lline), 1, Lline)

    # the day is chunked into 6 his files (000000, 040000, ..., 200000) — loop
    # through all of them and concatenate, instead of reading just the first.
    # only this boundary's slice (via read_bnd) is pulled per file, not the
    # full zeta field — these files are ~6 GB each, so loading the whole
    # thing per chunk just to throw away everything but one edge would be
    # wasteful.
    zeta_ex_list = Vector{Matrix{Float32}}()
    ubar_ex_list = Vector{Matrix{Float32}}()
    vbar_ex_list = Vector{Matrix{Float32}}()
    realtime_list = Vector{Vector{DateTime}}()
    for fname in his_files
        zeta_f, ubar_f, vbar_f, time_f = NCDataset(fname) do ds
            read_bnd(ds, "zeta", boundary), read_bnd(ds, "ubar", boundary), read_bnd(ds, "vbar", boundary), ds["ocean_time"][:]
        end
        println(fname, " => size(ocean_time) = ", size(time_f))
        push!(zeta_ex_list, zeta_f)
        push!(ubar_ex_list, ubar_f)
        push!(vbar_ex_list, vbar_f)
        push!(realtime_list, t_ref .+ Millisecond.(round.(Int, time_f .* 1000)))
    end
    zeta_ex = hcat(zeta_ex_list...)   # (line_pts, total_ntime), chronological since his_files is sorted
    ubar_ex = hcat(ubar_ex_list...)
    vbar_ex = hcat(vbar_ex_list...)
    realtime = vcat(realtime_list...)
    zeta_ex_list = ubar_ex_list = vbar_ex_list = realtime_list = nothing   # drop ASAP, next block rebuilds its own lists

    plot_bry_timeseries("zeta", "m", boundary, datestr, datestr_save, lon_line, lat_line, nchoose,
        bry_realtime, realtime, zeta_b, zeta_ex)

    # ubar/vbar are already in the grid's local xi/eta directions in both the
    # bry file and the his output (no east/north rotation needed, unlike a
    # true speed/quiver plot), so they can be compared directly.
    plot_bry_timeseries("ubar", "m/s", boundary, datestr, datestr_save, lon_line, lat_line, nchoose,
        bry_realtime, realtime, ubar_b, ubar_ex)

    # on east/west boundaries vbar/v live on eta_v (one point shorter than
    # eta_rho); on north/south boundaries the same offset hits ubar/u on
    # xi_u instead. Either way, reusing the rho-grid-based nchoose indices
    # here is a fine approximation for this diagnostic, not an exact
    # eta_rho/eta_v (or xi_rho/xi_u) alignment.
    plot_bry_timeseries("vbar", "m/s", boundary, datestr, datestr_save, lon_line, lat_line, nchoose,
        bry_realtime, realtime, vbar_b, vbar_ex)

    ##
    temp_ex_list = Vector{Array{Float32,3}}()
    salt_ex_list = Vector{Array{Float32,3}}()
    u_ex_list = Vector{Array{Float32,3}}()
    v_ex_list = Vector{Array{Float32,3}}()
    realtime_list = Vector{Vector{DateTime}}()
    for fname in his_files
        temp_f, salt_f, u_f, v_f, time_f = NCDataset(fname) do ds
            read_bnd(ds, "temp", boundary), read_bnd(ds, "salt", boundary),
            read_bnd(ds, "u", boundary), read_bnd(ds, "v", boundary), ds["ocean_time"][:]
        end
        println(fname, " => size(ocean_time) = ", size(time_f))
        push!(temp_ex_list, temp_f)
        push!(salt_ex_list, salt_f)
        push!(u_ex_list, u_f)
        push!(v_ex_list, v_f)
        push!(realtime_list, t_ref .+ Millisecond.(round.(Int, time_f .* 1000)))
    end
    temp_ex = cat(temp_ex_list...; dims=3)   # concat along time (dim 3), not hcat (dim 2)
    salt_ex = cat(salt_ex_list...; dims=3)
    u_ex = cat(u_ex_list...; dims=3)
    v_ex = cat(v_ex_list...; dims=3)
    realtime = vcat(realtime_list...)
    temp_ex_list = salt_ex_list = u_ex_list = v_ex_list = realtime_list = nothing

    ntime = length(realtime)
    N = size(temp_ex, 2)   # s_rho
    Npts = length(nchoose)

    h_pts = reshape(h_line[nchoose], Npts, 1)
    z = zeros(Npts, N, ntime); Cs = nothing
    for t in 1:ntime
        zeta_t = reshape(zeta_ex[nchoose, t], Npts, 1)
        z_dum, Cs = zlevs3(h_pts, zeta_t, theta_s, theta_b, hc, N, "r", "new2008")   # (N, Npts, 1)
        z[:, :, t] = dropdims(permutedims(z_dum, (2, 3, 1)); dims=2)   # (N,Npts,1) -> (Npts,1,N) -> (Npts,N)
    end

    # same z-level computation as above, but sourced from the bry file's own
    # zeta/bry_time instead of the model output's zeta_ex/realtime
    ntime_bry = length(bry_time)
    N_bry = size(temp_b, 2)   # s_rho, matches N (both 128) but kept independent for clarity

    z_bry = zeros(Npts, N_bry, ntime_bry); Cs_bry = nothing
    for t in 1:ntime_bry
        zeta_t_bry = reshape(zeta_b[nchoose, t], Npts, 1)
        z_dum_bry, Cs_bry = zlevs3(h_pts, zeta_t_bry, theta_s, theta_b, hc, N_bry, "r", "new2008")   # (N_bry, Npts, 1)
        z_bry[:, :, t] = dropdims(permutedims(z_dum_bry, (2, 3, 1)); dims=2)   # (N_bry,Npts,1) -> (Npts,1,N_bry) -> (Npts,N_bry)
    end
    # z[i,:,:] is (N,ntime) — depth shifts slightly with the tide (via zeta in
    # zlevs3), so it's a genuinely 2D coordinate, which plain heatmap!/contour!
    # can't take (they require a 1D y). Reusing the surface!-as-pcolormesh
    # trick from plotting_fun.jl instead — x=time, y=depth, both full 2D,
    # color=data, viewed top-down via topdown_axis3. surface! also needs a
    # numeric x, so time is converted to hours-since-start first.
    time_num = Dates.value.(realtime .- realtime[1]) ./ (1000 * 3600)   # hours since realtime[1]
    time_numbry = Dates.value.(bry_realtime .- bry_realtime[1]) ./ (1000 * 3600)   # hours since bry_realtime[1]

    # Axis3 has no linkxaxes! (Makie only defines it for Axis), and this is a
    # static CairoMakie export anyway (no interactive zoom to keep in sync), so
    # the practical equivalent is just forcing every panel to the same xlims —
    # model and bry time axes don't naturally match otherwise
    xlims_shared = (min(time_num[1], time_numbry[1]), max(time_num[end], time_numbry[end]))

    # full water-column range at the chosen points (model and bry combined),
    # for the "whole depth" version of the u/v plots below — the regular
    # version clips to -200..0 since that's where the boundary comparison
    # usually matters most.
    full_depth_range = (floor(min(minimum(z), minimum(z_bry))), 0)

    # same model-vs-bry, 2-col x 3-row comparison, factored into a function
    # since temp/salt/u/v repeat the exact same layout. depth_range/suffix
    # let the same function produce both the near-surface (-200..0, default)
    # and whole-depth versions without duplicating the plotting code.
    function plot_bry_compare_2d(varname, C_model_full, C_bry_full; colorrange, colormap, contour_levels, cbar_label,
                                  depth_range = (-200, 0), suffix = "")
        subfolder = "$(varname)$(suffix)_$(boundary)_bry"
        outname = figpath(subfolder, "$(subfolder)_$(datestr_save).png")
        fig = Figure(size = (1100, 700), figure_padding = (10, 80, 60, 10))   # extra right padding for the offset ylabel, extra bottom for the xlabel — both were being clipped at the default padding
        hm = nothing   # last-drawn heatmap handle, reused below for the one shared colorbar
        for (i, idx) in enumerate(nchoose)
            loc = "lon = $(round(lon_line[idx], digits=2)), lat = $(round(lat_line[idx], digits=2))"

            ax_model = topdown_axis3(fig[i, 1];
                title = "model $varname, $loc",
                xlabel = i == length(nchoose) ? "hours since $(Dates.format(realtime[1], "yyyy-mm-dd HH:MM"))" : "",
                ylabel = "depth (m)",
                aspect = (3, 1, 1), ylabeloffset = 80)
            X = repeat(reshape(time_num, 1, :), N, 1)
            Y = z[i, :, :]
            C = C_model_full[idx, :, :]
            hm = plot_curvilinear!(ax_model, X, Y, C; colormap = colormap, colorrange = colorrange,
                contour_levels = contour_levels, contour_color = RGBf(0.6, 0.6, 0.6))
            ylims!(ax_model, depth_range...)
            xlims!(ax_model, xlims_shared...)

            ax_bry = topdown_axis3(fig[i, 2];
                title = "bry $varname, $loc",
                xlabel = i == length(nchoose) ? "hours since $(Dates.format(bry_realtime[1], "yyyy-mm-dd HH:MM"))" : "",
                ylabel = "",
                aspect = (3, 1, 1))
            X_bry = repeat(reshape(time_numbry, 1, :), N_bry, 1)
            Y_bry = z_bry[i, :, :]
            C_bry = C_bry_full[idx, :, :]
            hm = plot_curvilinear!(ax_bry, X_bry, Y_bry, C_bry; colormap = colormap, colorrange = colorrange,
                contour_levels = contour_levels, contour_color = RGBf(0.6, 0.6, 0.6))
            ylims!(ax_bry, depth_range...)
            xlims!(ax_bry, xlims_shared...)
        end
        Colorbar(fig[1:length(nchoose), 3], hm, label = cbar_label)
        depth_note = suffix == "" ? "" : " (whole depth)"
        Label(fig[0, :], "$varname at $datestr — model (left) vs bry (right), $boundary boundary$depth_note", fontsize = 24, font = :bold, tellwidth = false)
        rowgap!(fig.layout, 8)
        colgap!(fig.layout, 8)
        rowsize!(fig.layout, 0, Fixed(40))
        save(outname, fig)
        println("saved plot to ", outname)
    end

    # colorranges below were tuned by checking this dataset's east-boundary
    # extrema directly (temp/salt) or typical current speeds (u/v) — if a
    # west/north/south plot looks saturated or washed out, these may need
    # per-boundary adjustment.
    plot_bry_compare_2d("temp", temp_ex, temp_b; colorrange = (15, 32), colormap = :thermal,
        contour_levels = 15:1:32, cbar_label = "°C")

    # salt: PSU, 34-36.5 range covers this Amazon-plume boundary (checked
    # directly against the east-boundary data — not the open-ocean ~34-37
    # you'd assume by default)
    plot_bry_compare_2d("salt", salt_ex, salt_b; colorrange = (34, 36.5), colormap = :viridis,
        contour_levels = 34:0.25:36.5, cbar_label = "PSU")

    # u/v: velocity, so symmetric colorrange around 0 and a diverging colormap
    # (checked actual east-boundary extrema first: u in [-1.42, 0.72], v in [-0.83, 0.88] m/s)
    plot_bry_compare_2d("u", u_ex, u_b; colorrange = (-1.5, 1.5), colormap = :balance,
        contour_levels = -1.5:0.25:1.5, cbar_label = "m/s")
    plot_bry_compare_2d("u", u_ex, u_b; colorrange = (-1.5, 1.5), colormap = :balance,
        contour_levels = -1.5:0.25:1.5, cbar_label = "m/s", depth_range = full_depth_range, suffix = "_fulldepth")

    plot_bry_compare_2d("v", v_ex, v_b; colorrange = (-1.0, 1.0), colormap = :balance,
        contour_levels = -1.0:0.2:1.0, cbar_label = "m/s")
    plot_bry_compare_2d("v", v_ex, v_b; colorrange = (-1.0, 1.0), colormap = :balance,
        contour_levels = -1.0:0.2:1.0, cbar_label = "m/s", depth_range = full_depth_range, suffix = "_fulldepth")

    # extract temp/salt/u/v at 1 m/10 m/100 m for the nchoose points
    vars_bnd = Dict(
        "temp" => (temp_ex[nchoose, :, :], temp_b[nchoose, :, :]),
        "salt" => (salt_ex[nchoose, :, :], salt_b[nchoose, :, :]),
        "u"    => (u_ex[nchoose, :, :],    u_b[nchoose, :, :]),
        "v"    => (v_ex[nchoose, :, :],    v_b[nchoose, :, :]),
    )

    # series_model["temp_1m"] :: (Npts, ntime) matrix, series_bry["temp_1m"] :: (Npts, ntime_bry)
    series_model = Dict{String,Matrix{Float32}}()
    series_bry = Dict{String,Matrix{Float32}}()
    for (varname, (F_model, F_bry)) in vars_bnd, (depthname, target_depth) in target_depths
        key = "$(varname)_$(depthname)"
        series_model[key] = extract_depth_timeseries(z, F_model, target_depth)
        series_bry[key] = extract_depth_timeseries(z_bry, F_bry, target_depth)
    end

    # same 3-row (points) x 3-col (depths) layout, factored into a function
    # since temp/salt/u/v repeat it exactly — reuses series_model/series_bry
    function plot_depth_compare(varname, unit_label)
        subfolder = "$(varname)_depth_$(boundary)_bry"
        outname = figpath(subfolder, "$(subfolder)_$(datestr_save).png")
        fig = Figure(size = (1200, 650))
        for (i, idx) in enumerate(nchoose)
            for (col, depthname) in enumerate(depth_order)
                key = "$(varname)_$(depthname)"
                ax = Axis(fig[i, col], xlabel = "time", ylabel = varname,
                    title = "$varname ($depthname) at lon = $(round(lon_line[idx], digits=2)), lat = $(round(lat_line[idx], digits=2))")
                scatterlines!(ax, bry_realtime, series_bry[key][i, :], color = :blue,
                    linewidth = 3, marker = :circle, markersize = 18)
                scatterlines!(ax, realtime, series_model[key][i, :], color = :orange,
                    linewidth = 2, marker = :circle, markersize = 10, linestyle = :dash)
            end
        end
        Label(fig[0, :], "$varname ($unit_label) at $datestr — $boundary boundary", fontsize = 24, font = :bold, tellwidth = false)
        rowsize!(fig.layout, 0, Fixed(40))
        save(outname, fig)
        println("saved plot to ", outname)
    end

    plot_depth_compare("temp", "°C")
    plot_depth_compare("salt", "PSU")
    plot_depth_compare("u", "m/s")
    plot_depth_compare("v", "m/s")

    return nothing
end

## Runs everything (all 4 boundaries + the flux check) for one calendar day:
# builds that day's bry/dbry/his_files paths, skips cleanly if a day's input
# is missing (e.g. a gap in the his output), then delegates to
# process_boundary per boundary and does the flux plots inline. This is the
# unit of work handed out to worker processes by pmap below — one call per
# day, each call producing that day's own set of figures (datestr/datestr_save
# below always come from that day's own bry_time/flux_time), so the figures
# show one day at a time even though many days now run at once.
@everywhere function process_day(date::Date)
    datestr_files = Dates.format(date, "yyyymmdd")
    bry_fname = bry_fname_for(date)
    dbry_fname = dbry_fname_for(date)

    if !isfile(bry_fname) || !isfile(dbry_fname)
        println("!! skipping $datestr_files — missing bry/dbry file ($bry_fname / $dbry_fname)")
        return nothing
    end

    theta_s, theta_b, hc = NCDataset(bry_fname) do ds
        # theta_s/theta_b/hc live on a size-1 "one" dimension in this file, so
        # [:] would give a 1-element Vector; zlevs3 needs plain scalars, hence [1]
        ds["theta_s"][1], ds["theta_b"][1], ds["hc"][1]
    end

    his_files = sort(filter(f -> occursin("roms_his.$datestr_files", basename(f)) && endswith(f, ".nc"),
                             readdir(datadir, join=true)))
    if isempty(his_files)
        println("!! skipping $datestr_files — no his files found in $datadir")
        return nothing
    end
    println("found $(length(his_files)) his files in $datadir for $datestr_files")

    ## run one boundary fully (extract + plot) before moving to the next, so only
    # one boundary's his-file arrays are ever resident in memory at a time.
    for boundary in ["east", "west", "north", "south"]
        process_boundary(boundary, bry_fname, his_files, theta_s, theta_b, hc)
        GC.gc()   # this boundary's arrays are all local to process_boundary and out of scope now — reclaim before the next one
    end

    ## flux check (see flux_specs comment above for what these variables mean)
    for (varname, boundary) in flux_specs
        # dbry_fname declares bry_time:units = "days since 1900-12-31 00:00:00",
        # which NCDatasets would auto-decode to DateTime — but that reference
        # date is wrong: e.g. for the "2022090100" file the raw value is
        # 10470.0, and t_ref (1994-01-01) + 10470 days = 2022-09-01 (matches
        # the filename) whereas 1900-12-31 + 10470 days = 1929-08-31
        # (nonsense). So bry_time here is actually a plain day-count since
        # t_ref, same as every other bry_time in this script — read via
        # `.var[:]` to bypass NCDatasets' (mistaken) CF auto-decoding.
        Fp, flux_time = NCDataset(dbry_fname) do ds
            ds[varname][:, :], ds["bry_time"].var[:]
        end
        flux_realtime = t_ref .+ Millisecond.(round.(Int, flux_time .* 86400 .* 1000))
        datestr = Dates.format(t_ref + Millisecond(round(Int, mean(flux_time) * 86400 * 1000)), "yyyy-mm-dd")
        datestr_save = Dates.format(t_ref + Millisecond(round(Int, mean(flux_time) * 86400 * 1000)), "yyyymmdd")

        lon_line = bnd_line(lon_rho, boundary)
        lat_line = bnd_line(lat_rho, boundary)
        Lline = length(lon_line)
        nchoose = clamp.(round.(Int, nchoose_fracs .* Lline), 1, Lline)

        outname = figpath("flux_timeplot", "$(varname)_$(datestr_save).png")
        fig = Figure(size = (800, 650))
        for (i, idx) in enumerate(nchoose)
            ax = Axis(fig[i, 1], xlabel = "time", ylabel = varname,
                title = "$varname at lon = $(round(lon_line[idx], digits=2)), lat = $(round(lat_line[idx], digits=2))")
            scatterlines!(ax, flux_realtime, Fp[idx, :], color = :blue,
                linewidth = 3, marker = :circle, markersize = 18)
            hlines!(ax, [0.0]; color = :gray, linestyle = :dot)   # sign flip = flux direction switches
        end
        Label(fig[0, :], "$varname (m⁴/s³) at $datestr", fontsize = 24, font = :bold, tellwidth = false)
        rowsize!(fig.layout, 0, Fixed(40))
        save(outname, fig)
        println("saved plot to ", outname)
    end

    return nothing
end

## drive the whole diagnostic across the run: one process_day call per
# calendar day from 2022-08-24 through 2022-09-23 (inclusive), handed out
# across worker processes by pmap — like check_output_parallel.jl's per-file
# pmap, but the unit of work here is a day instead of a single his file.
# pmap blocks until every day is done; on_error logs a failed day instead of
# aborting the whole run. Each day's figures are saved under the same
# figpath(subfolder, ...) layout as before, just with that day's own
# datestr_save in the filename, so nothing overwrites another day.
start_date = Date(2022, 8, 24)
end_date   = Date(2022, 9, 23)
dates = collect(start_date:Day(1):end_date)
println("processing $(length(dates)) days across $(nprocs() - 1) worker processes")
pmap(process_day, dates; on_error = ex -> println("a day failed: ", ex))

using Distributed

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "7")) - 1)   # -1 reserves a CPU for this main process
addprocs(n_workers)
println("running with $(nprocs() - 1) worker processes")

@everywhere begin

using NCDatasets, CairoMakie, Dates, Statistics

CairoMakie.activate!()   # explicit, in case another Makie backend gets loaded too

include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc" ;
# bry_dir = "/expanse/lustre/projects/uso101/hchen54/input/bry_63"
datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_2"
figure_path = "/home/hchen54/figure/dbry900m/sideview"

##
mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
    ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
end
lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
lon_chd_b, lat_chd_b = grid_boundary(lon_rho, lat_rho)
lon_u, lat_u = rho2u(lon_rho), rho2u(lat_rho)
lon_v, lat_v = rho2v(lon_rho), rho2v(lat_rho)

nchoose = [200, 400, 600]
colors = [:red, :blue, :green]

end   # closes @everywhere begin — only shared *data* lives above this line;
      # the one-off check plot below runs on the master only, not per-worker.

##
outname = joinpath(figure_path, "crossline.png")
fig = Figure()
ax = Axis(fig[1, 1],aspect = DataAspect())
lines!(ax, lon_chd_b, lat_chd_b, zeros(length(lon_chd_b)), color = :black, linewidth = 2)
for (i, n) in enumerate(nchoose)
    lines!(ax, lon_rho[:, n], lat_rho[:, n], color = colors[i], linewidth = 1)
end
fig
save(outname, fig)
println("saved plot to ", outname)

##
@everywhere function daily_process(date::Date)
    t_ref = DateTime(1994, 1, 1, 0, 0, 0)
    datestr_files = Dates.format(date, "yyyymmdd")

    his_files = sort(filter(f -> occursin("roms_his.$datestr_files", basename(f)) && endswith(f, ".nc"),
                             readdir(datadir, join=true)))
    if isempty(his_files)
        println("!! skipping $datestr_files — no his files found in $datadir")
        return nothing
    end
    fname = his_files[1]

    theta_s, theta_b, hc, ocean_time= NCDataset(fname) do ds
        ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"], ds["ocean_time"][:]
    end
    zeta, ubar, vbar = NCDataset(fname) do ds
        ds["zeta"][:, nchoose, :], ds["ubar"][:, nchoose, :], ds["vbar"][:, nchoose, :]
    end
    u, v, temp, salt = NCDataset(fname) do ds
        ds["u"][:, nchoose, :, :], ds["v"][:, nchoose, :, :],
        ds["temp"][:, nchoose, :, :], ds["salt"][:, nchoose, :, :]
    end
    realtime = t_ref .+ Millisecond.(round.(Int, ocean_time .* 1000))

    ##
    # z grid for the 3 crossections: h[:, nchoose]/zeta[:, :, t] are already
    # (xi_rho, 3) — zlevs3's (M, L) convention — so no reshaping is needed here,
    # unlike a 1D boundary line (see check_output_brymatch.jl).
    N = size(temp, 3)   # s_rho
    ntime = size(zeta, 3)
    h_cross = h[:, nchoose]   # (xi_rho, 3)
    z = zeros(Float32, size(h_cross)..., N, ntime)   # (xi_rho, 3, N, ntime)
    for t in 1:ntime
        zeta_t = zeta[:, :, t]   # (xi_rho, 3)
        z_dum, _ = zlevs3(h_cross, zeta_t, theta_s, theta_b, hc, N, "r", "new2008")   # (N, xi_rho, 3)
        z[:, :, :, t] = permutedims(z_dum, (2, 3, 1))   # (N,xi_rho,3) -> (xi_rho,3,N)
    end

    # u/v cross-section z-grids: average h and zeta to u-/v-points on the FULL
    # grid, THEN subset to nchoose — same reasoning as lon_u/lon_v/lat_u/lat_v
    # above. nchoose picks 3 non-adjacent eta rows, so averaging after
    # subsetting (e.g. rho2v(z)) would blend unrelated transects together
    # instead of shifting the eta index by one the way rho2v is meant to.
    zeta_full = NCDataset(fname) do ds
        ds["zeta"][:, :, :]
    end

    h_u_cross    = rho2u(h)[:, nchoose]              # (xi_u, 3)
    h_v_cross    = rho2v(h)[:, nchoose]              # (xi_rho, 3)
    zeta_u_cross = rho2u(zeta_full)[:, nchoose, :]   # (xi_u, 3, ntime)
    zeta_v_cross = rho2v(zeta_full)[:, nchoose, :]   # (xi_rho, 3, ntime)

    z_u = zeros(Float32, size(h_u_cross)..., N, ntime)   # (xi_u, 3, N, ntime)
    z_v = zeros(Float32, size(h_v_cross)..., N, ntime)   # (xi_rho, 3, N, ntime)
    for t in 1:ntime
        zu_dum, _ = zlevs3(h_u_cross, zeta_u_cross[:, :, t], theta_s, theta_b, hc, N, "r", "new2008")
        zv_dum, _ = zlevs3(h_v_cross, zeta_v_cross[:, :, t], theta_s, theta_b, hc, N, "r", "new2008")
        z_u[:, :, :, t] = permutedims(zu_dum, (2, 3, 1))
        z_v[:, :, :, t] = permutedims(zv_dum, (2, 3, 1))
    end

    function plot_sideview(n, t)
        fig = Figure(size = (800, 850), figure_padding = (10, 80, 60, 10))

        ax = Axis(fig[1, 1],aspect = DataAspect())
        lines!(ax, lon_chd_b, lat_chd_b, zeros(length(lon_chd_b)), color = :black, linewidth = 2)
        lines!(ax, lon_rho[:, nchoose[n]], lat_rho[:, nchoose[n]], color = colors[n], linewidth = 1)

        ax1 = Axis(fig[2, 1], title = "ssh (m)")
        lines!(ax1, lon_rho[:, nchoose[n]], zeta[:,n,t])
        ax2 = Axis(fig[3, 1], title = "ubar (m/s)")
        lines!(ax2, lon_u[:, nchoose[n]], ubar[:,n,t])
        ax3 = Axis(fig[4, 1], title = "vbar (m/s)")
        lines!(ax3, lon_v[:, nchoose[n]], vbar[:,n,t])
        ax4 = Axis(fig[1, 2], title = "temp (°C)")
        hm = plot_curvilinear!(ax4,repeat(lon_rho[:, nchoose[n]], 1, N),z[:,n,:,t], temp[:,n,:,t];
            colorrange = (15,32),colormap = :thermal,contour_levels = Float64.(15:1:32),
            contour_color = RGBf(0.6, 0.6, 0.6),contour_linewidth = 1)
        ylims!(ax4, -200,0)
        Colorbar(fig[1, 3],hm)
        ax5 = Axis(fig[2, 2], title = "salt (PSU)")
        hm = plot_curvilinear!(ax5,repeat(lon_rho[:, nchoose[n]], 1, N),z[:,n,:,t], salt[:,n,:,t];
            colorrange = (33.5,36.5),colormap = :viridis,contour_levels = Float64.(33.5:0.1:36.5),
            contour_color = RGBf(0.6, 0.6, 0.6),contour_linewidth = 1)
        ylims!(ax5, -200,0)
        Colorbar(fig[2, 3],hm)
        ax6 = Axis(fig[3, 2], title = "u (m/s)")
        hm = plot_curvilinear!(ax6,repeat(lon_u[:, nchoose[n]], 1, N),z_u[:,n,:,t], u[:,n,:,t];
            colorrange = (-1.5,1.5),colormap = :balance,contour_levels = Float64.(-1.5:0.25:1.5),
            contour_color = RGBf(0.6, 0.6, 0.6),contour_linewidth = 1)
        ylims!(ax6, -200,0)
        ax7 = Axis(fig[4, 2], title = "v (m/s)")
        hm = plot_curvilinear!(ax7,repeat(lon_v[:, nchoose[n]], 1, N),z_v[:,n,:,t], v[:,n,:,t];
            colorrange = (-1.5,1.5),colormap = :balance,contour_levels = Float64.(-1.5:0.25:1.5),
            contour_color = RGBf(0.6, 0.6, 0.6),contour_linewidth = 1)
        ylims!(ax7, -200,0)
        Colorbar(fig[3:4, 3], hm)
        Label(fig[0, :], Dates.format.(realtime[t], "yyyy/mm/dd HH:MM"), fontsize = 18)
        return fig
    end

    ##
    for n in eachindex(nchoose), t in 1:ntime
        fig = plot_sideview(n, t)
        outname = figpath("n_$(nchoose[n])", "sideview_n$(nchoose[n])_$(Dates.format(realtime[t], "yyyymmddHHMM")).png")
        save(outname, fig)
        println("saved plot to ", outname)
    end
end

start_date = Date(2022, 8, 24)
end_date   = Date(2022, 9, 23)
dates = collect(start_date:Day(1):end_date)
println("processing $(length(dates)) days across $(nprocs() - 1) worker processes")
pmap(daily_process, dates; on_error = ex -> println("a day failed: ", ex))

# dbry_check_plotting.jl
#
# High-frequency baroclinic energy flux at the four open boundaries, from the
# daily MAT files written by dbry_make.jl (raw W/m, NOT rho0-normalized):
#   (a) dbry_hovmoller_<dir>.png   x: time, y: along-boundary lon (N/S) or lat (E/W),
#                                  color: Fx, Fy, |F| (one panel each)
#   (b) dbry_timemean.png          time mean Fx, Fy, |F| along each boundary
# and prints the net inward flux through each boundary (MW).
#
# Fx/Fy are grid-relative (xi/eta), as ROMS uses them: the boundary-normal
# component is Fx on west/east and Fy on south/north. Inward is
# +Fx (west), -Fx (east), +Fy (south), -Fy (north).
#
#     julia --project=. input_file_make/dbry_check_plotting.jl

using MAT, NCDatasets, CairoMakie, Dates, Statistics, Printf
include("/home/hchen54/internalwave_julia/function/fig_setup.jl")

##
dx = 2700
mat_path = "/expanse/lustre/projects/uso102/hchen54/input_cmems/dbry_mat/"
grid_file = "/expanse/lustre/projects/uso102/hchen54/input_cmems/grid/roms_cmems_grd_$(dx)m.nc"
figure_path = "/expanse/lustre/projects/uso102/hchen54/figure/dbry_cmems_$(dx)m/"
mkpath(figure_path)
t_epoch = DateTime(1994, 1, 1)   # bry_time origin (days)

dirs = ("north", "south", "east", "west")
normal = Dict("west" => ("Fx", 1), "east" => ("Fx", -1), "south" => ("Fy", 1), "north" => ("Fy", -1))
along_name = Dict(d => d in ("north", "south") ? "lon" : "lat" for d in dirs)

## read the daily MAT files; consecutive days share their first/last sample
files = sort(filter(f -> startswith(f, "bryfile_dynamic$(dx)_") && endswith(f, ".mat"), readdir(mat_path)))
isempty(files) && error("no bryfile_dynamic$(dx)_*.mat in $mat_path")
println("reading $(length(files)) daily MAT files")

time = Float64[]
F = Dict(d => Dict("Fx" => Matrix{Float64}[], "Fy" => Matrix{Float64}[]) for d in dirs)
along = Dict{String,Vector{Float64}}()
for (n, f) in enumerate(files)
    matopen(joinpath(mat_path, f)) do io
        t = vec(read(io, "time"))
        keep = n == 1 ? (1:length(t)) : (2:length(t))
        append!(time, t[keep])
        for d in dirs
            n == 1 && (along[d] = vec(read(io, "$(d)_$(along_name[d])")))
            for k in ("Fx", "Fy")
                push!(F[d][k], read(io, "$(d)_$(k)")[:, keep])
            end
        end
    end
end
F = Dict(d => Dict(k => reduce(hcat, v) for (k, v) in F[d]) for d in dirs)
for d in dirs
    F[d]["|F|"] = hypot.(F[d]["Fx"], F[d]["Fy"])
end
dates = t_epoch .+ Millisecond.(round.(Int, time .* 86_400_000))
tday = time .- time[1]   # days since the first sample, the plotting x axis
println("time: $(dates[1]) .. $(dates[end]), $(length(time)) samples")

## along-boundary spacing (m) for the net flux: 1/pm along xi (N/S), 1/pn along eta (E/W)
ds_along = NCDataset(grid_file) do ds
    pm, pn = ds["pm"][:, :], ds["pn"][:, :]
    Dict("south" => 1 ./ pm[:, 1], "north" => 1 ./ pm[:, end],
         "west" => 1 ./ pn[1, :], "east" => 1 ./ pn[end, :])
end

## summary: how big is it?
println("\nboundary   max|F|     mean|F|    mean inward normal   net inward (time mean)")
println("           (kW/m)     (kW/m)     (kW/m)               (MW)")
for d in dirs
    k, s = normal[d]
    Fn = s .* F[d][k]
    net = sum(vec(mean(Fn; dims = 2)) .* ds_along[d]) / 1e6
    @printf("%-8s %9.3f  %9.3f  %12.4f         %10.2f\n", d,
        maximum(F[d]["|F|"]) / 1e3, mean(F[d]["|F|"]) / 1e3, mean(Fn) / 1e3, net)
end

## date ticks every ~2 weeks
tick_dates = collect(Date(dates[1]):Day(14):Date(dates[end]))
tick_pos = [Dates.value(DateTime(d) - dates[1]) / 86_400_000 for d in tick_dates]
tick_lab = Dates.format.(tick_dates, "mm/dd")

# color limits from the 99th percentile, so single spikes don't wash it out
q99(a) = quantile(vec(abs.(a)), 0.99)

## (a) time x along-boundary, one figure per boundary
for d in dirs
    y = along[d]
    order = sortperm(y)   # heatmap wants an increasing y
    clim = max(q99(F[d]["Fx"]), q99(F[d]["Fy"]), eps()) / 1e3
    fig = Figure(; figkw(18, 18)...)
    inward = normal[d][2] > 0 ? "+" : "−"
    for (row, k) in enumerate(("Fx", "Fy", "|F|"))
        lab = k == normal[d][1] ? "$k (normal, inward $(inward))" : k
        ax = Axis(fig[row, 1]; title = "$d boundary: $lab", ylabel = along_name[d] == "lon" ? "Longitude" : "Latitude",
                  xticks = (tick_pos, tick_lab), xlabel = row == 3 ? "2022" : "")
        a = F[d][k][order, :]' ./ 1e3   # (time, along)
        hm = k == "|F|" ?
             heatmap!(ax, tday, y[order], a; colormap = :amp, colorrange = (0, max(q99(F[d]["|F|"]) / 1e3, eps()))) :
             heatmap!(ax, tday, y[order], a; colormap = Reverse(:RdBu), colorrange = (-clim, clim))
        Colorbar(fig[row, 2], hm; label = "kW/m")
    end
    out = joinpath(figure_path, "dbry_hovmoller_$(d).png")
    figsave(out, fig)
    println("saved ", out)
end

## (b) time mean along each boundary
fig = Figure(; figkw(18, 14)...)
for (n, d) in enumerate(dirs)
    ax = Axis(fig[(n - 1) ÷ 2 + 1, (n - 1) % 2 + 1]; title = "$d boundary, time mean",
              xlabel = along_name[d] == "lon" ? "Longitude" : "Latitude", ylabel = "kW/m")
    y = along[d]
    hlines!(ax, [0]; color = :gray, linewidth = 0.8)
    for (k, c) in (("Fx", :royalblue), ("Fy", :firebrick), ("|F|", :black))
        lines!(ax, y, vec(mean(F[d][k]; dims = 2)) ./ 1e3; color = c)
    end
end
# one shared legend below the panels (normal component: Fx on W/E, Fy on S/N)
Legend(fig[3, 1:2], [LineElement(color = c) for c in (:royalblue, :firebrick, :black)],
       ["Fx (normal on W/E)", "Fy (normal on S/N)", "|F|"]; orientation = :horizontal, framevisible = false)
Label(fig[0, :], "HF (< 28 h) baroclinic energy flux, $(Date(dates[1])) – $(Date(dates[end])), $(dx) m CMEMS bry")
out = joinpath(figure_path, "dbry_timemean.png")
figsave(out, fig)
println("saved ", out)

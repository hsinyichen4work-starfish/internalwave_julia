using NCDatasets, CairoMakie, Dates, Statistics
CairoMakie.activate!()
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

# ---- EDIT THESE TWO PATHS ----
grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"
fname      = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_2/roms_his.20220901160000.nc"  # <-- pick one file
# -------------------------------

figure_path = "/home/hchen54/figure/dbry900m/hisfile"  # test PNGs land here as temp_TEST_*.png

## Grid
mask_rho, lon_rho, lat_rho, h = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :], ds["h"][:, :]
end
lon_rho[lon_rho .> 180] .-= 360
bathy_levels = [500, 1000, 2000]

## Time — just use the first non-fill-valued time step in the file
ocean_time = NCDataset(fname) do ds
    ds["ocean_time"][:]
end
t_ref = DateTime(1994, 1, 1, 0, 0, 0)
realtime = t_ref .+ Millisecond.(round.(Int, ocean_time .* 1000))
str1 = Dates.format.(realtime, "yyyy-mm-dd HH:MM:SS")
println("time steps in file: ", str1)

## Zeta + vertical metadata
zeta = NCDataset(fname) do ds
    ds["zeta"][:, :, :]
end
theta_s, theta_b, hc, N = NCDataset(fname) do ds
    ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"], size(ds["temp"], 3)
end

# print out whatever transform-related attributes exist, so we can check
# they actually match "new2006" being passed into zlevs3 below
NCDataset(fname) do ds
    for key in ("Vtransform", "Vstretching", "theta_s", "theta_b", "hc", "N")
        if haskey(ds.attrib, key)
            println(key, " = ", ds.attrib[key])
        else
            println(key, " = <not found in file attributes>")
        end
    end
end

t = 1   # just test the first time step
println("\nUsing time step t = $t  ($(str1[t]))")

z_dum, Cs = zlevs3(h, zeta[:, :, t], theta_s, theta_b, hc, N, "r", "new2008")
z = permutedims(z_dum, (2, 3, 1))   # (xi, eta, N) for this one time step

println("size(z) = ", size(z))
println("global z range (all columns): ", extrema(z))

## Pick a deep-water point to sanity-check the z profile
# (adjust i,j if this lands on land / shallow water for your grid — check mask_rho[i,j] and h[i,j])
i, j = size(z, 1) ÷ 2, size(z, 2) ÷ 2
println("mask_rho[i,j] = ", mask_rho[i, j], "   h[i,j] = ", h[i, j], " m")
println("z profile at (i=$i, j=$j):")
println(z[i, j, :])

## Temp data for this one time step only
temp = NCDataset(fname) do ds
    ds["temp"][:, :, :, t]   # just this time step: (xi, eta, N)
end
temp_masked = ifelse.(mask_rho .== 0, NaN32, temp)

println("\ntemp profile at (i=$i, j=$j):")
println(temp_masked[i, j, :])

## Now the actual slice_at_depth calls being tested
temp_1m   = slice_at_depth(z, temp_masked, -1.0)
temp_100m  = slice_at_depth(z, temp_masked, -100.0)
temp_200m  = slice_at_depth(z, temp_masked, -200.0)
temp_3000m = slice_at_depth(z, temp_masked, -3000.0)

println("\ntemp_1m   valid-range: ", extrema(filter(!isnan, temp_1m)))
println("temp_100m  valid-range: ", extrema(filter(!isnan, temp_100m)))
println("temp_200m  valid-range: ", extrema(filter(!isnan, temp_200m)))
println("temp_3000m valid-range: ", extrema(filter(!isnan, temp_3000m)))
println("\nAt the single point (i,j): 1m=$(temp_1m[i,j])  100m=$(temp_10m[i,j])  200m=$(temp_50m[i,j])  3000m=$(temp_100m[i,j])")

## Plot all four in a 2x2 grid so you can see whether they actually diverge
fig = Figure(size = (1200, 1000))
panels = (
    (temp_1m,   "1 m"),
    (temp_100m,  "100 m"),
    (temp_200m,  "200 m"),
    (temp_3000m, "3000 m"),
)
for (idx, (field, label)) in enumerate(panels)
    row = (idx - 1) ÷ 2 + 1
    col = (idx - 1) % 2 + 1
    # each panel gets its own 2-wide sub-layout: [plot | colorbar], so
    # colorbars sit right next to their own panel instead of at the row end
    gl = fig[row, col] = GridLayout()
    ax = topdown_axis3(gl[1, 1]; title = "Temp @ $label, $(str1[t])")
    sp = plot_curvilinear!(ax, lon_rho, lat_rho, field; colormap = :thermal, colorrange = (15, 30))
    Colorbar(gl[1, 2], sp, label = "°C")
    contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels, color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
end

outname = joinpath(figure_path, "temp_TEST_$(Dates.format(realtime[t], "yyyymmdd_HHMM")).png")
save(outname, fig)
println("\nsaved test comparison figure to ", outname)
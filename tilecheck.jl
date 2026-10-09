using NCDatasets, CairoMakie, Statistics, Printf

base = "/expanse/lustre/projects/uso101/hchen54"
rec  = get(ARGS, 1, "roms_his.20220824040000.nc")
runs = ["old" => joinpath(base, "amazon_900m_3mon_old", rec),
        "new" => joinpath(base, "amazon_900m_3mon", "after_joint", rec)]
grid = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"
outdir = ARGS[2]
pm, pn, mask = NCDataset(grid) do ds; ds["pm"][:, :], ds["pn"][:, :], ds["mask_rho"][:, :]; end

# relative vorticity at psi points / f-free, in 1/s
function vort(u, v, pm, pn)
    dx = 1 ./ pm; dy = 1 ./ pn
    dvdx = (v[2:end, :] .- v[1:end-1, :]) ./ (0.5 .* (dx[2:end, 1:end-1] .+ dx[1:end-1, 2:end]))
    dudy = (u[:, 2:end] .- u[:, 1:end-1]) ./ (0.5 .* (dy[1:end-1, 2:end] .+ dy[2:end, 1:end-1]))
    dvdx .- dudy
end

# seam metric: mean |2nd difference| across x (per column) and across y (per row)
function seam(a)
    a = replace(a, NaN => 0.0)
    sx = vec(mean(abs.(a[3:end, :] .- 2a[2:end-1, :] .+ a[1:end-2, :]), dims=2))
    sy = vec(mean(abs.(a[:, 3:end] .- 2a[:, 2:end-1] .+ a[:, 1:end-2]), dims=1))
    sx, sy
end
function spikes(s)
    m = median(s)
    findall(s .> 3m) .+ 1, maximum(s) / m
end

res = Dict()
for (name, f) in runs
    ds = NCDataset(f)
    nt = size(ds["zeta"], 3); ns = size(ds["u"], 3)
    u = Float64.(coalesce.(ds["u"][:, :, ns, nt], NaN))
    v = Float64.(coalesce.(ds["v"][:, :, ns, nt], NaN))
    h = Float64.(coalesce.(ds["hbls"][:, :, nt], NaN))
    t = ds["ocean_time"][nt]; close(ds)
    h[mask .== 0] .= NaN
    z = vort(u, v, pm, pn)
    res[name] = (; z, h, t)
    for (lab, a) in (("vort", z), ("hbls", h))
        sx, sy = seam(a)
        ix, rx = spikes(sx); iy, ry = spikes(sy)
        @printf("%-4s %-5s  x: max/median=%.1f spikes at i=%s\n", name, lab, rx, string(ix))
        @printf("%-4s %-5s  y: max/median=%.1f spikes at j=%s\n", name, lab, ry, string(iy))
    end
    println("$name time = $t")
end

fig = Figure(size=(1400, 1300))
for (c, name) in enumerate(["old", "new"])
    r = res[name]
    ax = Axis(fig[1, c], title="$name surface vorticity  $rec", aspect=DataAspect())
    hm = heatmap!(ax, r.z, colormap=:balance, colorrange=(-2e-4, 2e-4))
    c == 2 && Colorbar(fig[1, 3], hm)
    ax2 = Axis(fig[2, c], title="$name hbls (m)", aspect=DataAspect())
    hm2 = heatmap!(ax2, r.h, colormap=:viridis, colorrange=(0, 60))
    c == 2 && Colorbar(fig[2, 3], hm2)
end
save(joinpath(outdir, "tilecheck_" * replace(rec, ".nc" => "") * ".png"), fig)

# difference of hbls seam profile, for plotting
fig2 = Figure(size=(1200, 600))
for (k, lab) in enumerate(("x (per i)", "y (per j)"))
    ax = Axis(fig2[1, k], title="hbls mean |2nd diff| along $lab", yscale=log10)
    for name in ["old", "new"]
        s = seam(res[name].h)[k]
        lines!(ax, s .+ 1e-6, label=name)
    end
    axislegend(ax)
end
save(joinpath(outdir, "seam_" * replace(rec, ".nc" => "") * ".png"), fig2)

using NCDatasets, CairoMakie, Dates, Statistics

CairoMakie.activate!()

include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")   # zlevs3, slice_at_depth, figpath

# Follow-up to check_output_brymatch_hpc.jl: the boundary mismatch shows odd
# jumps on 08/24-08/27, so here temp and zeta are read along the *whole*
# east/west boundary line (not just 3 sample points) and the 4 days are
# stitched into one continuous record, so a jump at a day/file transition
# shows up directly in boundary-position vs time (Hovmöller) plots.

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"
bry_dir = "/expanse/lustre/projects/uso101/hchen54/input/bry"
bry_fname_for(date) = joinpath(bry_dir, "roms_bry_900m_$(Dates.format(date, "yyyymmdd"))00.nc")

datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_0927/"
figure_path = "/home/hchen54/figure/dbry_0927/bry_match_tempjump"   # used by figpath()
mkpath(figure_path)

# whole run split into consecutive window_len-day windows (0824-0827,
# 0828-0831, 0901-0904, ...), one set of figures per window; the last
# window is clipped at run_end
run_start  = Date(2022, 8, 24)
run_end    = Date(2022, 9, 23)
window_len = 4
windows = [(d, min(d + Day(window_len - 1), run_end)) for d in run_start:Day(window_len):run_end]
boundaries = ["east", "west"]

target_depths = [("1m", -1.0), ("10m", -10.0), ("100m", -100.0), ("300m", -300.0)]

##
mask_rho, lon_rho, lat_rho, h = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :], ds["h"][:, :]
end
lon_rho[lon_rho .> 180] .-= 360
t_ref = DateTime(1994, 1, 1, 0, 0, 0)

# same boundary-generic indexing as check_output_brymatch_hpc.jl
bnd_dim(boundary) = boundary in ("east", "west") ? 1 : 2
bnd_edge_index(A, boundary) = boundary in ("east", "north") ? size(A, bnd_dim(boundary)) : 1
bnd_line(A, boundary) = selectdim(A, bnd_dim(boundary), bnd_edge_index(A, boundary))

function read_bnd(ds, varname, boundary)
    var = ds[varname]
    dim = bnd_dim(boundary)
    i = bnd_edge_index(var, boundary)
    nd = ndims(var)
    idx = dim == 1 ? (i, ntuple(_ -> Colon(), nd - 1)...) : (Colon(), i, ntuple(_ -> Colon(), nd - 2)...)
    return var[idx...]
end

# x axis of every plot = days since the window start, so day transitions sit on integers
days_since_t0(t, t0) = Dates.value(t - t0) / (1000 * 86400)

# keep the first occurrence of each time stamp (consecutive files can share
# their edge record, e.g. bry day d ends at 24h = day d+1 00h) — heatmap!
# needs strictly increasing x
function dedupe_time(times, arrs...)
    keep = [findfirst(==(t), times) == i for (i, t) in enumerate(times)]
    perm = sortperm(times[keep])
    return times[keep][perm], map(A -> selectdim(A, ndims(A), findall(keep)[perm]) |> copy, arrs)...
end

# z (Npts,N,ntime) at rho points along the whole boundary line, from that
# line's h and the time-varying zeta
function boundary_z(h_line, zeta_line, theta_s, theta_b, hc, N)
    Npts, ntime = size(zeta_line)
    hh = reshape(h_line, Npts, 1)
    z = zeros(Npts, N, ntime)
    for t in 1:ntime
        z_dum, _ = zlevs3(hh, reshape(zeta_line[:, t], Npts, 1), theta_s, theta_b, hc, N, "r", "new2008")   # (N,Npts,1)
        z[:, :, t] = permutedims(z_dum[:, :, 1], (2, 1))
    end
    return z
end

# same as in check_output_brymatch_hpc.jl
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

# model record nearest to each bry time (within tol), so model - bry can be
# formed on the bry time axis; NaN where no model record is close enough
function match_to_times(model_t, model_F, target_t; tol = Minute(30))
    out = fill(NaN32, size(model_F, 1), length(target_t))
    for (j, t) in enumerate(target_t)
        dt, k = findmin(abs.(model_t .- t))
        dt <= tol && (out[:, j] = model_F[:, k])
    end
    return out
end

nanq(A, q) = (v = filter(isfinite, vec(A)); isempty(v) ? 1.0 : quantile(v, q))
sym_range(A) = (r = max(nanq(abs.(A), 0.99), 1e-6); (-r, r))
rms_along(D) = [sqrt(mean(filter(isfinite, view(D, :, j)) .^ 2)) for j in 1:size(D, 2)]
maxabs_along(D) = [maximum(abs, filter(isfinite, view(D, :, j)); init = 0.0) for j in 1:size(D, 2)]

# CairoMakie can't write .jpg itself ("Unsupported mime: image/jpeg"), so
# render to a pixel buffer and hand that to FileIO; JPEG has no alpha
# channel, hence the ARGB -> RGB conversion first
save_jpg(fname, fig) = CairoMakie.FileIO.save(fname, Makie.RGB{Makie.Colors.N0f8}.(colorbuffer(fig)))

# fixed colorbar for the temp model − bry panels (the _fixcb version), so
# windows can be compared against each other directly
dT_fixed_range = (-0.5, 0.5)

function mark_days!(ax, ndays)
    vlines!(ax, 0:ndays; color = (:black, 0.5), linestyle = :dash, linewidth = 1)
end

##
function process_window(start_date, end_date)
dates = collect(start_date:Day(1):end_date)
ndays = length(dates)
t0 = DateTime(start_date)
for boundary in boundaries
    println("\n=== $boundary boundary, $(start_date) to $(end_date) ===")

    mask_line = bnd_line(mask_rho, boundary)
    lat_line = collect(bnd_line(lat_rho, boundary))
    h_line = collect(bnd_line(h, boundary))
    Npts = length(lat_line)
    # y axis: latitude if it is monotonic along the boundary (heatmap! needs
    # that), otherwise fall back to the eta index
    if issorted(lat_line) || issorted(lat_line, rev = true)
        yvals, ylab = lat_line, "latitude"
    else
        yvals, ylab = collect(1:Npts), "eta index"
    end
    land = mask_line .== 0

    zeta_b_l = Matrix{Float32}[]; temp_b_l = Array{Float32,3}[]; tb_l = Vector{DateTime}[]
    zeta_m_l = Matrix{Float32}[]; temp_m_l = Array{Float32,3}[]; tm_l = Vector{DateTime}[]
    theta_s = theta_b = hc = nothing

    for date in dates
        ds_str = Dates.format(date, "yyyymmdd")
        bry_fname = bry_fname_for(date)
        if !isfile(bry_fname)
            println("!! missing $bry_fname — skipping $ds_str")
            continue
        end
        zb, Tb, tb, theta_s, theta_b, hc = NCDataset(bry_fname) do ds
            ds["zeta_$boundary"][:, :], ds["temp_$boundary"][:, :, :], ds["bry_time"][:],
            ds["theta_s"][1], ds["theta_b"][1], ds["hc"][1]
        end
        push!(zeta_b_l, zb); push!(temp_b_l, Tb)
        push!(tb_l, t_ref .+ Millisecond.(round.(Int, tb .* 86400 .* 1000)))   # bry_time is in days
        println("bry $ds_str: ", length(tb), " records, ", tb_l[end][1], " → ", tb_l[end][end])

        his_files = sort(filter(f -> occursin("roms_his.$ds_str", basename(f)) && endswith(f, ".nc"),
                                readdir(datadir, join = true)))
        isempty(his_files) && println("!! no his files for $ds_str")
        for fname in his_files
            zm, Tm, tm = NCDataset(fname) do ds
                read_bnd(ds, "zeta", boundary), read_bnd(ds, "temp", boundary), ds["ocean_time"][:]
            end
            push!(zeta_m_l, zm); push!(temp_m_l, Tm)
            push!(tm_l, t_ref .+ Millisecond.(round.(Int, tm .* 1000)))   # ocean_time is in seconds
            println("  ", basename(fname), ": ", length(tm), " records")
        end
    end

    if isempty(tb_l) || isempty(tm_l)
        println("!! no bry or his data for $boundary in $(start_date) – $(end_date), skipping")
        continue
    end

    # ---- bry file-to-file continuity: when day d's last record and day d+1's
    # first record share a time stamp they should be identical — a non-zero
    # difference here is a jump baked into the bry files themselves
    println("\n-- bry continuity at file transitions ($boundary) --")
    for k in 1:length(tb_l)-1
        t_end, t_next = tb_l[k][end], tb_l[k+1][1]
        if t_end == t_next
            dz = maximum(abs, filter(isfinite, zeta_b_l[k][:, end] .- zeta_b_l[k+1][:, 1]); init = 0.0)
            dT = maximum(abs, filter(isfinite, temp_b_l[k][:, :, end] .- temp_b_l[k+1][:, :, 1]); init = 0.0)
            println("  $t_end: max|Δzeta| = $(round(dz, digits=4)) m, max|Δtemp| = $(round(dT, digits=4)) °C")
        else
            println("  no shared record: file $k ends $t_end, file $(k+1) starts $t_next (gap = $(t_next - t_end))")
        end
    end

    tb, zeta_b, temp_b = dedupe_time(vcat(tb_l...), hcat(zeta_b_l...), cat(temp_b_l...; dims = 3))
    tm, zeta_m, temp_m = dedupe_time(vcat(tm_l...), hcat(zeta_m_l...), cat(temp_m_l...; dims = 3))
    zeta_b_l = temp_b_l = tb_l = zeta_m_l = temp_m_l = tm_l = nothing
    GC.gc()
    println("stitched: bry $(length(tb)) records, model $(length(tm)) records")

    xb = days_since_t0.(tb, t0)
    xm = days_since_t0.(tm, t0)
    xlab = "days since $(start_date)"
    tag = "$(Dates.format(start_date, "yyyymmdd"))_$(Dates.format(end_date, "yyyymmdd"))"

    # ================= zeta =================
    zeta_m_on_b = match_to_times(tm, zeta_m, tb)
    dzeta = zeta_m_on_b .- zeta_b
    zplot_m = copy(zeta_m); zplot_m[land, :] .= NaN
    zplot_b = copy(zeta_b); zplot_b[land, :] .= NaN
    dzeta[land, :] .= NaN

    zr = (min(nanq(zplot_b, 0.01), nanq(zplot_m, 0.01)), max(nanq(zplot_b, 0.99), nanq(zplot_m, 0.99)))
    fig = Figure(size = (1300, 1000))
    for (row, (ttl, x, D, cr, cm)) in enumerate([
            ("model zeta", xm, zplot_m, zr, :viridis),
            ("bry zeta", xb, zplot_b, zr, :viridis),
            ("model − bry zeta (model at nearest bry time)", xb, dzeta, sym_range(dzeta), :balance)])
        ax = Axis(fig[row, 1], title = ttl, xlabel = row == 3 ? xlab : "", ylabel = ylab)
        hm = heatmap!(ax, x, yvals, permutedims(D); colormap = cm, colorrange = cr, nan_color = :gray80)
        mark_days!(ax, ndays)
        xlims!(ax, 0, ndays)
        Colorbar(fig[row, 2], hm, label = "m")
    end
    Label(fig[0, :], "zeta along the whole $boundary boundary, $(start_date) – $(end_date)", fontsize = 22, font = :bold, tellwidth = false)
    rowsize!(fig.layout, 0, Fixed(35))
    outname = figpath("zeta_hovmoller_$(boundary)", "zeta_hovmoller_$(boundary)_$(tag).png")
    save(outname, fig); println("saved plot to ", outname)

    # boundary-wide mismatch summary: where in time does it jump?
    fig = Figure(size = (1200, 600))
    ax1 = Axis(fig[1, 1], ylabel = "m", title = "zeta mismatch along $boundary boundary: RMS (blue), max|·| (orange)")
    scatterlines!(ax1, xb, rms_along(dzeta), color = :blue, markersize = 6)
    scatterlines!(ax1, xb, maxabs_along(dzeta), color = :orange, markersize = 6)
    mark_days!(ax1, ndays)
    ax2 = Axis(fig[2, 1], xlabel = xlab, ylabel = "m", title = "boundary-mean zeta: bry (blue) vs model (orange)")
    wet = .!land
    scatterlines!(ax2, xb, vec(mean(zeta_b[wet, :], dims = 1)), color = :blue, markersize = 6)
    lines!(ax2, xm, vec(mean(zeta_m[wet, :], dims = 1)), color = :orange, linestyle = :dash)
    mark_days!(ax2, ndays)
    linkxaxes!(ax1, ax2); xlims!(ax2, 0, ndays)
    outname = figpath("zeta_mismatch_$(boundary)", "zeta_mismatch_$(boundary)_$(tag).png")
    save(outname, fig); println("saved plot to ", outname)

    # ================= temp at fixed depths =================
    N = size(temp_m, 2)
    z_m = boundary_z(h_line, zeta_m, theta_s, theta_b, hc, N)
    z_b = boundary_z(h_line, zeta_b, theta_s, theta_b, hc, size(temp_b, 2))

    Tm_d = Dict{String,Matrix{Float32}}(); Tb_d = Dict{String,Matrix{Float32}}(); dT_d = Dict{String,Matrix{Float32}}()
    for (dname, zt) in target_depths
        Tm_d[dname] = extract_depth_timeseries(z_m, temp_m, zt)
        Tb_d[dname] = extract_depth_timeseries(z_b, temp_b, zt)
        dT_d[dname] = match_to_times(tm, Tm_d[dname], tb) .- Tb_d[dname]
        for D in (Tm_d[dname], Tb_d[dname], dT_d[dname])
            D[land, :] .= NaN
        end
    end

    # drawn twice: auto colorbar for the mismatch row (.png), then the same
    # figure with the mismatch row fixed at dT_fixed_range (_fixcb.jpg)
    for fixcb in (false, true)
    fig = Figure(size = (1900, 1100))
    for (col, (dname, _)) in enumerate(target_depths)
        Tr = (min(nanq(Tb_d[dname], 0.01), nanq(Tm_d[dname], 0.01)), max(nanq(Tb_d[dname], 0.99), nanq(Tm_d[dname], 0.99)))
        dTr = fixcb ? dT_fixed_range : sym_range(dT_d[dname])
        for (row, (ttl, x, D, cr, cm, lab)) in enumerate([
                ("model temp $dname", xm, Tm_d[dname], Tr, :thermal, "°C"),
                ("bry temp $dname", xb, Tb_d[dname], Tr, :thermal, "°C"),
                ("model − bry temp $dname", xb, dT_d[dname], dTr, :balance, "°C")])
            ax = Axis(fig[row, 2col-1], title = ttl, xlabel = row == 3 ? xlab : "", ylabel = col == 1 ? ylab : "")
            hm = heatmap!(ax, x, yvals, permutedims(D); colormap = cm, colorrange = cr, nan_color = :gray80)
            mark_days!(ax, ndays)
            xlims!(ax, 0, ndays)
            Colorbar(fig[row, 2col], hm, label = lab)
        end
    end
    Label(fig[0, :], "temp along the whole $boundary boundary, $(start_date) – $(end_date)", fontsize = 22, font = :bold, tellwidth = false)
    rowsize!(fig.layout, 0, Fixed(35))
    if fixcb
        outname = figpath("temp_hovmoller_$(boundary)", "temp_hovmoller_$(boundary)_$(tag)_fixcb.jpg")
        save_jpg(outname, fig)
    else
        outname = figpath("temp_hovmoller_$(boundary)", "temp_hovmoller_$(boundary)_$(tag).png")
        save(outname, fig)
    end
    println("saved plot to ", outname)
    end   # fixcb

    fig = Figure(size = (1200, 800))
    for (row, (dname, _)) in enumerate(target_depths)
        ax = Axis(fig[row, 1], xlabel = row == length(target_depths) ? xlab : "", ylabel = "°C",
            title = "temp $dname mismatch along $boundary boundary: RMS (blue), max|·| (orange)")
        scatterlines!(ax, xb, rms_along(dT_d[dname]), color = :blue, markersize = 6)
        scatterlines!(ax, xb, maxabs_along(dT_d[dname]), color = :orange, markersize = 6)
        mark_days!(ax, ndays)
        xlims!(ax, 0, ndays)
    end
    outname = figpath("temp_mismatch_$(boundary)", "temp_mismatch_$(boundary)_$(tag).png")
    save(outname, fig); println("saved plot to ", outname)

    GC.gc()
end
end   # process_window

for (wstart, wend) in windows
    println("\n######## window $(wstart) – $(wend) ########")
    process_window(wstart, wend)
    GC.gc()   # free this window's arrays/figures before the next one
end

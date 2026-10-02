# Independent check of the NCOM data used in mooring_comp.jl.
#
# mooring_comp.jl takes NCOM from NCOM_french.nc, which was made in two steps:
#   binary flat files --(MATLAB NCOM2nc)--> NCOM_DATA_NC/*.nc --(mooring_NCOM_extraction.jl)--> NCOM_french.nc
# Here the NCOM column at the French mooring is read STRAIGHT from the binary flat files in
# /home/mbui/ModelOutput/NCOM (grid and data), with none of that code and none of those nc files, and then
# compared with NCOM_french.nc and plotted against the mooring and ROMS.
#
# What the binary files are (from /home/mbui/ModelOutput/NCOM/readme.txt and MATLAB_codes/):
#   - big-endian Float32, no record markers, stored (i, j, k) with i fastest; nest 2 is 1244 x 1334 x 100
#   - seatmp is in Kelvin
#   - uucurr / vvcurr are on the C-grid faces and along the GRID axes (the grid is turned about -27°):
#       u(i, j) sits on the WEST face of cell (i, j), v(i, j) on the SOUTH face
#     so they are averaged to the cell centre and then rotated by grdang to true east / north
#   - one folder per day, with hours 00..24; hour 24 is the same time as hour 00 of the next day
using Printf

include("/home/hsinyi/Documents/Julia/mooring_comp.jl")   # mooring / roms / ncom (from the nc file), helper and plot functions
##
ncom_root = "/home/mbui/ModelOutput/NCOM"
bin_fname = "/home/hsinyi/roms_data/NCOM_mooring/NCOM_french_from_binary.nc"   # cache of the extracted column
redo_extract = false          # true = read the binaries again even if bin_fname exists
moor_lon, moor_lat = -45.13, 3.95
nx, ny, nz = 1244, 1334, 100
bin_days = Date(model_time[1]):Day(1):Date(model_time[end])

##
# ---- low-level readers ----
# Read only a small (i, j) block of every level of a flat file, by jumping to the right bytes: a 3-D file is
# 664 MB and only a few values per level are needed. Returns Float64 (length(irange), length(jrange), nlev).
function read_block(fname, irange, jrange, nlev)
    out = Array{Float32}(undef, length(irange), length(jrange), nlev)
    buf = Vector{Float32}(undef, length(irange))
    open(fname) do io
        for k in 1:nlev, (n, j) in enumerate(jrange)
            seek(io, 4 * (((k - 1) * ny + (j - 1)) * nx + (first(irange) - 1)))
            read!(io, buf)
            out[:, n, k] = buf
        end
    end
    return Float64.(ntoh.(out))          # the files are big-endian
end
function flat_name(day, field, hour; nlev = 1, kind = "fcstfld")
    lev = nlev == 1 ? "sfc_000000_000000" : @sprintf("mod_000001_%06d", nlev)
    tag = Dates.format(day, "yyyymmdd") * "00"
    joinpath(ncom_root, "data", tag, @sprintf("%s_%s_2o%04dx%04d_%s_%04d0000_%s", field, lev, nx, ny, tag, hour, kind))
end

##
# ---- horizontal grid, from ohgrd_2.A: lon, lat, dx, dy, h, ang (6 of its 7 records are used here) ----
glon, glat, gh, gang = open(joinpath(ncom_root, "grid", "ohgrd_2.A")) do io
    rec() = Float64.(ntoh.(read!(io, Array{Float32}(undef, nx, ny))))
    lon = rec(); lat = rec(); rec(); rec(); h = rec(); ang = rec()
    lon, lat, h, ang                          # h: water depth, negative; ang: degrees counter-clockwise from east
end
glon[glon .> 180] .-= 360

# the grid cell that contains the mooring and the bilinear weights of its 4 corners
xy(i, j) = ((glon[i, j] - moor_lon) * cosd(moor_lat), glat[i, j] - moor_lat)    # local flat coordinates (degrees)
inear, jnear = Tuple(argmin([hypot(xy(i, j)...) for i in 1:nx, j in 1:ny]))
ei = xy(inear + 1, jnear) .- xy(inear, jnear)                                   # one grid step along i and along j
ej = xy(inear, jnear + 1) .- xy(inear, jnear)
a, b = [ei[1] ej[1]; ei[2] ej[2]] \ collect(.-xy(inear, jnear))                 # mooring = nearest point + a*ei + b*ej
i0, j0 = inear + floor(Int, a), jnear + floor(Int, b)
cx, cy = a - floor(a), b - floor(b)
wgt = [(1 - cx) * (1 - cy)  (1 - cx) * cy; cx * (1 - cy)  cx * cy]              # wgt[di+1, dj+1] for corner (i0+di, j0+dj)
bil(c) = sum(wgt .* c)                                                          # c is a 2x2 block of corner values
println("mooring cell: i = $i0:$(i0+1), j = $j0:$(j0+1), weights = ", round.(wgt, digits = 3))
println("  check: interpolated lon/lat = ", round(bil(glon[i0:i0+1, j0:j0+1]), digits = 4), ", ",
    round(bil(glat[i0:i0+1, j0:j0+1]), digits = 4), "; depth = ", round(bil(gh[i0:i0+1, j0:j0+1]), digits = 1),
    " m; grid angle = ", round(bil(gang[i0:i0+1, j0:j0+1]), digits = 2), "°")

##
# ---- vertical grid, from ovgrd_2.A: depth of the 101 layer interfaces in every column (static, negative down) ----
zw = let vg = read_block(joinpath(ncom_root, "grid", "ovgrd_2.A"), i0:i0+1, j0:j0+1, nz + 1)
    [bil(vg[:, :, k]) for k in 1:nz+1]
end
zc = (zw[1:end-1] .+ zw[2:end]) ./ 2          # layer centres, k = 1 at the surface
println("layer centres: ", round.(zc[1:3], digits = 2), " … ", round.(zc[end-1:end], digits = 1), " m")

##
# ---- extract the column at the mooring for every hour of the model period ----
function read_column(day, hour)
    # tracers at the 4 corner cells, size (2, 2, nz)
    T = read_block(flat_name(day, "seatmp", hour; nlev = nz), i0:i0+1, j0:j0+1, nz) .- 273.15       # Kelvin -> °C
    S = read_block(flat_name(day, "salint", hour; nlev = nz), i0:i0+1, j0:j0+1, nz)
    # faces -> centres: centre(i) = (u(i) + u(i+1)) / 2, centre(j) = (v(j) + v(j+1)) / 2
    uf = read_block(flat_name(day, "uucurr", hour; nlev = nz), i0:i0+2, j0:j0+1, nz)
    vf = read_block(flat_name(day, "vvcurr", hour; nlev = nz), i0:i0+1, j0:j0+2, nz)
    ug = (uf[1:2, :, :] .+ uf[2:3, :, :]) ./ 2
    vg = (vf[:, 1:2, :] .+ vf[:, 2:3, :]) ./ 2
    # grid axes -> true east / north, with the grid angle of each corner cell
    ang = gang[i0:i0+1, j0:j0+1]
    ue = ug .* cosd.(ang) .- vg .* sind.(ang)
    vn = ug .* sind.(ang) .+ vg .* cosd.(ang)
    ssh = bil(read_block(flat_name(day, "seahgt", hour), i0:i0+1, j0:j0+1, 1)[:, :, 1])
    col(f) = [bil(f[:, :, k]) for k in 1:nz]
    return col(T), col(S), col(ue), col(vn), ssh
end

if redo_extract || !isfile(bin_fname)
    # The time goes into waiting for the disk, so the hours are read in parallel: start Julia with threads
    # (julia -t 32, or "julia.NumThreads" in VS Code). About 5 min with 32 threads, over an hour with 1.
    jobs = [(day, hour) for day in bin_days for hour in 0:23]      # hour 24 is hour 00 of the next folder
    cols = Vector{Any}(undef, length(jobs))
    println("reading ", length(jobs), " hours from the binary files with ", Threads.nthreads(), " thread(s) …")
    Threads.@threads for n in eachindex(jobs)
        cols[n] = read_column(jobs[n]...)
    end
    bt = [DateTime(day) + Hour(hour) for (day, hour) in jobs]
    bT, bS, bU, bV = (getindex.(cols, m) for m in 1:4)
    bssh = Float64.(getindex.(cols, 5))
    NCDataset(bin_fname, "c") do ds
        ds.attrib["title"] = "NCOM at the French mooring, read directly from the binary flat files"
        ds.attrib["source"] = "mooring_comp_from_binary.jl, $ncom_root"
        defDim(ds, "k", nz); defDim(ds, "time", length(bt))
        defVar(ds, "time", bt, ("time",), attrib = ["units" => "hours since 2022-01-01 00:00:00"])
        defVar(ds, "z", zc, ("k",), attrib = ["long_name" => "static depth of the layer centres", "units" => "m", "positive" => "up"])
        for (name, x, long, units) in (("temp", bT, "temperature", "degC"), ("salt", bS, "salinity", "psu"),
                                       ("u", bU, "eastward velocity", "m/s"), ("v", bV, "northward velocity", "m/s"))
            defVar(ds, name, stack(x), ("k", "time"), attrib = ["long_name" => long, "units" => units])
        end
        defVar(ds, "ssh", bssh, ("time",), attrib = ["long_name" => "sea surface height", "units" => "m"])
    end
    println("saved ", bin_fname)
end

##
# ---- the binary-derived NCOM in the same frame as the other structures (see mooring_comp.jl) ----
# Unlike `ncom` (already interpolated to fixed 5 m levels), this one keeps NCOM's own 100 layers, k = 1 at the surface.
ncom_bin = NCDataset(bin_fname) do ds
    it = findall(model_time[1] .<= ds["time"][:] .<= model_time[end])
    (
        time = ds["time"][it],
        z    = repeat(abs.(ds["z"][:]), 1, length(it)),     # static layer depths taken as depth below the surface
        u    = ds["u"][:, it],
        v    = ds["v"][:, it],
        temp = ds["temp"][:, it],
        salt = ds["salt"][:, it],
        zeta = ds["ssh"][it],
    )
end
println("ncom_bin: ", ncom_bin.time[1], " to ", ncom_bin.time[end], ", size(temp) = ", size(ncom_bin.temp))

##
# ---- binary-derived NCOM against NCOM_french.nc, on the nc file's own depth levels (0–1000 m every 5 m) ----
ncom.time == ncom_bin.time || error("the two NCOM records are not on the same times")
zcheck = ncom.z[:, 1]
println("\nNCOM read from binary  minus  NCOM_french.nc   (all hours, 0–1000 m)")
println("variable   mean diff     RMS diff      max |diff|    typical size of the signal (std)")
for name in (:temp, :salt, :u, :v)
    d = to_zgrid(ncom_bin.z, getfield(ncom_bin, name), zcheck) .- getfield(ncom, name)
    d = filter(!isnan, d)                                    # the top 0.5 m is above NCOM's first layer centre
    @printf("%-8s %12.2e %12.2e %12.2e %12.3f\n", name, mean(d), sqrt(mean(d .^ 2)), maximum(abs.(d)), std(getfield(ncom, name)))
end
d = ncom_bin.zeta .- ncom.zeta
@printf("%-8s %12.2e %12.2e %12.2e %12.3f\n", "ssh", mean(d), sqrt(mean(d .^ 2)), maximum(abs.(d)), std(ncom.zeta))

##
# ---- figures: the same line plots as mooring_comp.jl, with the binary-derived NCOM added as a 4th line ----
# NCOM from the nc file is drawn first and NCOM from binary on top of it, so where only the binary line is
# visible the two agree.
figure_path = joinpath(figure_path, "from_binary")
mkpath(figure_path)
bin_color = "#1baf7a"

for (name, long, ylabel, fmoor) in ((:temp, "Temperature", "Temperature (°C)", mooring.temp),
                                    (:salt, "Salinity", "Salinity (PSU)", mooring.salt))
    plot_lines(ctd_titles, (
            ("Mooring CTD", mooring.time_ts, fmoor, src_colors.mooring),
            ("ROMS 900 m", roms.time, at_mooring_depth(roms, getfield(roms, name)), src_colors.roms),
            ("NCOM (NCOM_french.nc)", ncom.time, at_mooring_depth(ncom, getfield(ncom, name)), src_colors.ncom),
            ("NCOM (read from binary)", ncom_bin.time, at_mooring_depth(ncom_bin, getfield(ncom_bin, name)), bin_color));
        ylabel = ylabel, title = "$long at the CTDs, $site", fname = "$(name)_lines_at_ctd.png")
end

for (name, long, fmoor) in ((:u, "Eastward velocity u", mooring.u), (:v, "Northward velocity v", mooring.v))
    plot_lines(uv_titles, (
            ("Mooring ADCP", mooring.time_uv, to_zgrid(mooring.z_uv, fmoor, uv_depths), src_colors.mooring),
            ("ROMS 900 m", roms.time, to_zgrid(roms.z, getfield(roms, name), uv_depths), src_colors.roms),
            ("NCOM (NCOM_french.nc)", ncom.time, to_zgrid(ncom.z, getfield(ncom, name), uv_depths), src_colors.ncom),
            ("NCOM (read from binary)", ncom_bin.time, to_zgrid(ncom_bin.z, getfield(ncom_bin, name), uv_depths), bin_color));
        ylabel = "$name (m/s)", zeroline = true,
        title = "$long at fixed depths, $site", fname = "$(name)_lines_at_depth.png")
end

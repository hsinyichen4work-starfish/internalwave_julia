# river_input.jl
#
# ROMS (UCLA) river forcing from the Dai & Trenberth coastal-station discharge
# (coastal-stns-Vol-monthly.updated-May2019.nc), following the approach of
# ucla-tools/river_pipe_tools/fill_frc_rivers.m but with automatic placement:
#
#   1. pick rivers whose mouth is within max_dist_km of the grid coastline
#   2. discharge at the mouth = FLOW * ratio_m2s, monthly climatology over clim_years
#      (data end 2018-12, so runs after that use the climatology)
#   3. put each river on the nearest coastal LAND cells (mask_rho = 0 with a wet
#      neighbour), spread over ceil(Q / q_cell_max) cells; river k's cells hold
#      k + 1/n_k in `river_flux`, like fill_frc_rivers.m
#   4. write the grid (mask_rho with the closed water + `river_flux`) and the forcing file
#      (river_volume, river_tracer, river_time with cycle_length = 365.25)
#
# The input grid is always the untouched backup (grid_src); grid_out and frc_out
# are rebuilt from it on every run, so re-running gives the same result.
#
# Run from the Documents/Julia project, e.g.
#     julia --project=/home/hsinyi/Documents/Julia input_file_make/river_input.jl

using NCDatasets
using Dates
using Statistics

## settings
dx = 2700
river_file = "/home/hsinyi/roms_data/riverfile/coastal-stns-Vol-monthly.updated-May2019.nc"
grid_src   = "/home/hsinyi/roms_data/grid/new_design/roms_grd_$(dx)m_1104_before_riverfix.nc"  # untouched input
grid_out   = "/home/hsinyi/roms_data/grid/new_design/roms_grd_$(dx)m_1104.nc"   # rebuilt: new mask_rho + river_flux
grid_file  = grid_src
isfile(grid_src) || error("missing $grid_src: copy the original roms_grd_$(dx)m_1104.nc there first")
frc_out    = "/home/hsinyi/roms_data/riverfile/roms_riv_$(dx)m_1104.nc"

max_dist_km = 60.0        # keep a river if its mouth is within this of a coastal land cell
min_vol     = 0.0         # km3/yr, drop rivers smaller than this (0 = keep all)
keep_rivers = ["Amazon", "Tocantins"]   # riv_name to keep among the selected ones ([] = keep all)
clim_years  = 1980:2018   # years used for the monthly climatology
q_cell_max  = 5000.0      # m3/s per cell, the river is spread over ceil(Qmean/q_cell_max) cells
riv_temp    = 28.0        # degC  (ucla-tools uses 17, that is for California)
riv_salt    = 0.0         # psu   (ucla-tools uses 1)
bdry_cells  = 20          # no source cell within this many cells of the open boundary
# split a river's discharge over named branches (fractions sum to 1); each branch
# is its own river in the forcing file
amazon_split = Dict("Amazon_N" => 0.65, "Amazon_S" => 0.35)
# the dataset mouth points sit inside the estuary, in channels that run into the
# southern open boundary of this grid: put the sources at the outlets instead (deg E, deg N)
mouth_override = Dict("Amazon_N"  => (-50.45 + 360, 0.30),    # head of the North Channel
                      "Amazon_S"  => (-50.15 + 360, -0.12),   # head of the South Channel
                      "Tocantins" => (-48.7 + 360, -1.4))     # Para estuary
# fill (close) the wet cells inside these polygons, to simplify the Amazon mouth
# (deg E negative = west; only wet cells are changed, so the polygon may cover land)
fill_polys = Dict(
    "Amazon middle channel (green)" => [(-50.37, 0.225), (-50.32, 0.215), (-50.08, 0.33), (-50.08, 0.52), (-50.36, 0.25)],
    "Amazon strait (cyan)"          => [(-49.90, -0.06), (-49.835, -0.085), (-49.60, 0.08), (-49.57, 0.17),
                                        (-49.65, 0.24), (-49.76, 0.12)])
# close the estuary upstream of a cut line across it, and put the river on the
# new land along the cut (spread evenly across the channel width):
#   river => (line end 1, line end 2, a point on the upstream side)   (deg E, deg N)
mouth_cut = Dict("Tocantins" => ((-48.92 + 360, -1.25), (-48.56 + 360, -1.55), (-49.0 + 360, -1.8)))
cut_radius_km = 80.0      # only water within this of the cut line middle is closed
# river source line: a wall of land across the channel (p1 -> p2, deg E, deg N);
# the water it cuts off from the sea is closed, and the river goes on the wall face,
# plus the bank cells nearest the line if q_cell_max asks for more cells
mouth_wall = Dict("Amazon_N" => ((-50.645 + 360, 0.325), (-50.46 + 360, 0.243)),
                  "Amazon_S" => ((-50.31 + 360, 0.03),   (-50.30 + 360, -0.12)))

## helpers
const R_earth = 6371.0
function haversine(lon1, lat1, lon2, lat2)
    φ1, φ2 = deg2rad(lat1), deg2rad(lat2)
    dφ, dλ = φ2 - φ1, deg2rad(lon2 - lon1)
    a = sin(dφ / 2)^2 + cos(φ1) * cos(φ2) * sin(dλ / 2)^2
    return 2R_earth * asin(sqrt(min(a, 1.0)))
end

# char(chars, station) -> Vector{String}
charstr(c) = [String(strip(String(collect(skipmissing(c[:, i]))))) for i in axes(c, 2)]

## read the river data
function read_rivers(file)
    NCDataset(file) do ds
        t = ds["time"][:]                                   # YYYYMM
        (; time    = Date.(t .÷ 100, t .% 100, 1),
           id      = ds["id"][:],
           lon_mou = mod.(Float64.(ds["lon_mou"][:]), 360),  # 0-360 like the grid
           lat_mou = Float64.(ds["lat_mou"][:]),
           ratio   = Float64.(coalesce.(ds["ratio_m2s"][:], 1.0)),
           vol_stn = Float64.(coalesce.(ds["vol_stn"][:], NaN)),
           yrb     = ds["yrb"][:], yre = ds["yre"][:],
           riv_name = charstr(ds["riv_name"][:, :]),
           flow    = Float64.(coalesce.(ds["FLOW"][:, :], NaN)))  # (station, time), m3/s
    end
end

## coastal land cells: interior mask_rho = 0 points with a wet 4-neighbour
function coastal_cells(file; bdry_cells = 1)
    lon, lat, mask, h = NCDataset(file) do ds
        mod.(ds["lon_rho"][:, :], 360), ds["lat_rho"][:, :], ds["mask_rho"][:, :], ds["h"][:, :]
    end
    nx, ny = size(mask)
    coast = CartesianIndex{2}[]
    nb = max(1, bdry_cells)
    for j in 1+nb:ny-nb, i in 1+nb:nx-nb
        mask[i, j] == 0 || continue
        (mask[i-1, j] + mask[i+1, j] + mask[i, j-1] + mask[i, j+1]) > 0 && push!(coast, CartesianIndex(i, j))
    end
    isempty(coast) && error("$file has no land cells next to water, rivers cannot be placed")
    return (; lon, lat, mask, h, coast)
end

dist_to(grd, x, y) = [haversine(x, y, grd.lon[c], grd.lat[c]) for c in grd.coast]

## pick rivers whose mouth is near the grid coastline
function select_rivers(riv, grd; max_dist_km, min_vol)
    sel = Int[]; dist = Float64[]
    lo0, lo1 = extrema(grd.lon); la0, la1 = extrema(grd.lat)
    pad = max_dist_km / 111 + 0.5
    for s in eachindex(riv.lon_mou)
        x, y = riv.lon_mou[s], riv.lat_mou[s]
        (lo0 - pad <= x <= lo1 + pad && la0 - pad <= y <= la1 + pad) || continue
        isnan(riv.vol_stn[s]) || riv.vol_stn[s] >= min_vol || continue
        d = minimum(dist_to(grd, x, y))
        d <= max_dist_km || continue
        push!(sel, s); push!(dist, d)
    end
    return sel, dist
end

## mouth discharge and monthly climatology (river, 12); empty months -> annual mean
function monthly_clim(q, time, years)
    clim = fill(NaN, size(q, 1), 12)
    for m in 1:12
        it = findall(t -> month(t) == m && year(t) in years, time)
        for r in axes(q, 1)
            v = filter(!isnan, q[r, it])
            isempty(v) && (v = filter(!isnan, q[r, month.(time) .== m]))  # outside clim_years
            isempty(v) || (clim[r, m] = mean(v))
        end
    end
    for r in axes(clim, 1)
        bad = isnan.(clim[r, :])
        any(bad) && (clim[r, bad] .= mean(clim[r, .!bad]))
    end
    return clim
end

## coastal land cells of a mask: land with a wet 4-neighbour, away from the open boundary
function coast_of(mask; bdry_cells = 1)
    nx, ny = size(mask); nb = max(1, bdry_cells)
    [CartesianIndex(i, j) for j in 1+nb:ny-nb for i in 1+nb:nx-nb
     if mask[i, j] < 0.5 && (mask[i-1, j] + mask[i+1, j] + mask[i, j-1] + mask[i, j+1]) > 0]
end

## put river k on its n_k nearest free coastal land cells (river k gets k + 1/n_k
# in river_flux); biggest rivers first so they get the cells closest to their mouth
function place_rivers!(cells, lon, lat, coast, mouths, qmean, todo; q_cell_max)
    taken = Set(c for (k, ck) in enumerate(cells) if !(k in todo) for c in ck)
    for k in sort(todo, by = k -> -qmean[k])
        n = max(1, ceil(Int, qmean[k] / q_cell_max))
        x, y = mouths[k]
        d = [c in taken ? Inf : haversine(x, y, lon[c], lat[c]) for c in coast]
        cells[k] = coast[partialsortperm(d, 1:n)]
        union!(taken, cells[k])
    end
    return cells
end

function build_rflx(sz, cells)
    rflx = zeros(sz)
    for (k, ck) in enumerate(cells); rflx[ck] .= k + 1 / length(ck); end
    return rflx
end

# point in polygon (ray casting), poly = [(x, y), ...]
function inpoly(x, y, poly)
    inside = false; n = length(poly)
    for a in 1:n
        (x1, y1), (x2, y2) = poly[a], poly[mod1(a + 1, n)]
        ((y1 > y) != (y2 > y)) && (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1) && (inside = !inside)
    end
    inside
end

## write river_flux into a copy of the grid
function write_grid_rflx(grid_src, grid_out, mask, rflx)
    tmp = grid_out * ".tmp"
    cp(grid_src, tmp; force = true)
    NCDataset(tmp, "a") do ds
        nchg = count((ds["mask_rho"][:, :] .> 0.5) .!= (mask .> 0.5))
        ds["mask_rho"][:, :] = mask
        v = defVar(ds, "river_flux", Float64, ("xi_rho", "eta_rho"),
                   attrib = ["long_name" => "river volume flux partition", "units" => "none"])
        v[:, :] = rflx
        ds.attrib["river_history"] = "$(today()) river_input.jl: mask_rho closed upstream/inland of the river " *
            "sources ($nchg wet cells closed) and river_flux added; " *
            "original grid: $(basename(grid_src))"
    end
    mv(tmp, grid_out; force = true)
end

## write the forcing file: 12 mid-month climatology records, cycle_length 365.25
function write_river_frc(frc_out, q_clim, names; riv_temp, riv_salt)
    nriv = size(q_clim, 1)
    rtime = [Float64(dayofyear(Date(2001, m, 15))) - 1 for m in 1:12]   # yearday of mid-month
    trc = zeros(nriv, 2, 12); trc[:, 1, :] .= riv_temp; trc[:, 2, :] .= riv_salt
    NCDataset(frc_out, "c") do ds      # "c" overwrites
        ds.attrib["title"] = "ROMS river forcing, monthly climatology"
        ds.attrib["source"] = basename(river_file) * " (Dai & Trenberth), FLOW*ratio_m2s, clim $(clim_years)"
        ds.attrib["rivers"] = join(["$k:$(names[k])" for k in 1:nriv], ", ")
        ds.attrib["history"] = "created $(today()) by river_input.jl"
        defDim(ds, "nriver", nriv); defDim(ds, "ntracers", 2); defDim(ds, "river_time", 12)
        defVar(ds, "river_time", rtime, ("river_time",),
               attrib = ["long_name" => "river data time", "units" => "yearday", "cycle_length" => 365.25])
        defVar(ds, "river_volume", q_clim, ("nriver", "river_time"),
               attrib = ["long_name" => "River volume flux", "units" => "m^3/s"])
        defVar(ds, "river_tracer", trc, ("nriver", "ntracers", "river_time"),
               attrib = ["long_name" => "River tracer data (temperature, salinity)", "units" => "variable"])
    end
end

## run
riv = read_rivers(river_file)
grd = coastal_cells(grid_file; bdry_cells)
sel, dist = select_rivers(riv, grd; max_dist_km, min_vol)
if !isempty(keep_rivers)
    k = findall(s -> riv.riv_name[s] in keep_rivers, sel)
    missing_riv = setdiff(keep_rivers, riv.riv_name[sel[k]])
    isempty(missing_riv) || @warn "not found near the grid coast: $missing_riv"
    sel, dist = sel[k], dist[k]
end
# output rivers: (name, dataset station, fraction of its discharge)
names = String[]; src = Int[]; frac = Float64[]
for s in sel
    if riv.riv_name[s] == "Amazon" && !isempty(amazon_split)
        sum(values(amazon_split)) ≈ 1 || @warn "amazon_split fractions sum to $(sum(values(amazon_split)))"
        for nm in sort(collect(keys(amazon_split)))
            push!(names, nm); push!(src, s); push!(frac, amazon_split[nm])
        end
    else
        push!(names, riv.riv_name[s]); push!(src, s); push!(frac, 1.0)
    end
end
q_mou  = riv.flow[src, :] .* riv.ratio[src] .* frac        # (river, time), m3/s at the mouth
q_clim = monthly_clim(q_mou, riv.time, clim_years)
qmean  = vec(mean(q_clim, dims = 2))
mouths = [get(mouth_override, names[k], (riv.lon_mou[src[k]], riv.lat_mou[src[k]])) for k in eachindex(names)]
cells  = [CartesianIndex{2}[] for _ in names]

## close inland water that is not downstream of the sources (test only: the
# new mask goes to the figure, the grid file is NOT changed)
const NB4 = CartesianIndex.([(1, 0), (-1, 0), (0, 1), (0, -1)])

# breadth-first step distance through `pass` from `seeds`; -1 = not reached
function bfs(pass, seeds)
    D = fill(-1, size(pass)); q = CartesianIndex{2}[]
    for s in seeds; pass[s] && D[s] < 0 && (D[s] = 0; push!(q, s)); end
    k = 1
    while k <= length(q)
        c = q[k]; k += 1
        for d in NB4
            n = c + d
            (checkbounds(Bool, pass, n) && pass[n] && D[n] < 0) || continue
            D[n] = D[c] + 1; push!(q, n)
        end
    end
    D
end

boxany(A, r) = [any(@view A[max(i-r, 1):min(i+r, end), max(j-r, 1):min(j+r, end)])
                for i in axes(A, 1), j in axes(A, 2)]
boxall(A, r) = .!boxany(.!A, r)

"""
close_inland(mask, h, cells; ...) -> new mask_rho

 h_deep   : wet cells deeper than this are 'the sea' (seeds)
 r_wide   : water narrower than 2r_wide+1 cells counts as a channel
 max_chan : close cells more than this many steps up a narrow channel
 dam_r    : half-width (cells) of the temporary dam around the source cells
 tol      : close cells whose path to the sea grows by more than tol steps when dammed
 path_tol : keep the downstream corridor, the wet cells on paths from a river's
            dam to the sea at most path_tol steps longer than the shortest one
            (so a narrow outlet channel is not closed)
"""
function close_inland(mask, h, cells; h_deep = 50.0, r_wide = 2, max_chan = 4,
                      dam_r = 2, tol = 10, path_tol = 10)
    wet  = mask .> 0.5
    deep = findall(wet .& (h .>= h_deep))

    # (2) narrow inland channels
    wide = wet .& boxany(boxall(wet, r_wide), r_wide)        # morphological opening
    open = bfs(wide, deep) .>= 0                             # wide AND connected to the sea
    Dc   = bfs(wet, findall(open))                           # steps from open water
    narrow = Dc .> max_chan

    # (1) behind the sources: dam the outlet water, compare paths to the sea
    D0 = bfs(wet, deep)
    dam = falses(size(wet)); corridor = falses(size(wet))
    for ck in cells
        damk = falses(size(wet))
        for s in ck, di in -dam_r:dam_r, dj in -dam_r:dam_r
            n = s + CartesianIndex(di, dj)
            checkbounds(Bool, wet, n) && wet[n] && (damk[n] = true)
        end
        dam .|= damk
        # (3) downstream corridor: steps from the dam + steps to the sea ~ shortest path
        reach = filter(i -> D0[i] >= 0, findall(damk))
        isempty(reach) && continue
        dmin = minimum(D0[reach])
        Ds = bfs(wet, findall(damk))
        corridor .|= (Ds .>= 0) .& (D0 .>= 0) .& (Ds .+ D0 .<= dmin + path_tol)
    end
    D1 = bfs(wet .& .!dam, filter(i -> !dam[i], deep))
    behind = .!dam .& ((D1 .< 0) .| (D1 .- D0 .> tol))

    close = wet .& (narrow .| behind .| (D0 .< 0)) .& .!dam .& .!corridor   # keep outlet water + path to sea
    newwet = wet .& .!close
    newwet .&= bfs(newwet, deep) .>= 0                       # final: connected to the sea
    m = Float64.(newwet)

    # every source cell must still touch water
    for (k, ck) in enumerate(cells)
        nwet = count(c -> any(d -> checkbounds(Bool, m, c+d) && m[c+d] > 0, NB4), ck)
        nwet < length(ck) && @warn "river $k: $(length(ck)-nwet) of $(length(ck)) source cells lost their wet neighbour"
    end
    println("closed $(count(close)) wet cells")
    return m
end

# close wet cells upstream of a cross-channel cut line (p1->p2), within radius_km
function cut_upstream!(mask, lon, lat, p1, p2, upstream_pt; radius_km = 80)
    side(x, y) = sign((p2[1]-p1[1])*(y-p1[2]) - (p2[2]-p1[2])*(x-p1[1]))
    s_up = side(upstream_pt...)
    mid = ((p1[1]+p2[1])/2, (p1[2]+p2[2])/2)
    for I in CartesianIndices(mask)
        mask[I] > 0.5 || continue
        haversine(lon[I], lat[I], mid...) < radius_km || continue
        side(lon[I], lat[I]) == s_up && (mask[I] = 0)
    end
    mask
end

# (a) fill the polygons
mask_cut = copy(grd.mask)
for (nm, poly) in fill_polys
    pp = [(x + 360, y) for (x, y) in poly]
    in_p = [mask_cut[I] > 0.5 && inpoly(grd.lon[I], grd.lat[I], pp) for I in CartesianIndices(mask_cut)]
    mask_cut[in_p] .= 0
    println("fill $nm: closed $(count(in_p)) wet cells")
end

# (a2) source walls: close the wall cells (segment extended 1 cell at both ends)
function seg_dist_km(x, y, p1, p2)       # distance to the segment, local flat-earth
    c = cosd((p1[2] + p2[2]) / 2) * 111.2
    ax, ay = (p2[1] - p1[1]) * c, (p2[2] - p1[2]) * 111.2
    px, py = (x - p1[1]) * c, (y - p1[2]) * 111.2
    t = clamp((px * ax + py * ay) / (ax^2 + ay^2), 0, 1)
    hypot(px - t * ax, py - t * ay)
end
function extend(p1, p2, km)
    c = cosd((p1[2] + p2[2]) / 2) * 111.2
    L = hypot((p2[1] - p1[1]) * c, (p2[2] - p1[2]) * 111.2); f = km / L
    d = (p2[1] - p1[1], p2[2] - p1[2])
    (p1[1] - f * d[1], p1[2] - f * d[2]), (p2[1] + f * d[1], p2[2] + f * d[2])
end
walls = Dict{String, Vector{CartesianIndex{2}}}()
for (nm, (p1, p2)) in mouth_wall
    nm in names || continue
    q1, q2 = extend(p1, p2, dx / 1000)
    w = [I for I in CartesianIndices(mask_cut)
         if mask_cut[I] > 0.5 && seg_dist_km(grd.lon[I], grd.lat[I], q1, q2) <= 0.8 * dx / 1000]
    mask_cut[w] .= 0
    walls[nm] = w
end
if !isempty(walls)
    n0 = count(mask_cut .> 0.5)
    deep0 = findall((mask_cut .> 0.5) .& (grd.h .>= 50.0))
    mask_cut[(mask_cut .> 0.5) .& (bfs(mask_cut .> 0.5, deep0) .< 0)] .= 0   # cut off from the sea
    println("walls: $(sum(length, values(walls))) wall cells, $(n0 - count(mask_cut .> 0.5)) cut-off wet cells closed")
end
is_cut = falses(length(names))
coast_w = coast_of(mask_cut; bdry_cells)
# steps through the water from each wall's face water: a bank cell belongs to the
# wall whose water it touches first (so a branch never takes the other branch's banks)
wetw = mask_cut .> 0.5
Dw = Dict(nm => bfs(wetw, [I + d for I in w for d in NB4 if wetw[I + d]]) for (nm, w) in walls)
stepw(D, c) = minimum((D[c + d] for d in NB4 if wetw[c + d] && D[c + d] >= 0); init = typemax(Int))
for (k, nm) in enumerate(names)
    haskey(walls, nm) || continue
    p1, p2 = mouth_wall[nm]
    face = [I for I in walls[nm] if any(d -> mask_cut[I + d] > 0.5, NB4)]
    isempty(face) && (@warn "$nm: source wall touches no open water, placed normally"; continue)
    n = max(length(face), ceil(Int, qmean[k] / q_cell_max))
    taken = Set(c for ck in cells for c in ck)
    own(c) = all(o -> o == nm || stepw(Dw[nm], c) <= stepw(Dw[o], c), keys(Dw)) && stepw(Dw[nm], c) < typemax(Int)
    bank = sort([c for c in coast_w if !(c in face) && !(c in taken) && own(c)],
                by = c -> seg_dist_km(grd.lon[c], grd.lat[c], p1, p2))
    cells[k] = vcat(face, bank[1:n-length(face)]); is_cut[k] = true
    println("$nm: wall face $(length(face)) cells + $(n - length(face)) bank cells, ",
            round(qmean[k] / n, digits = 1), " m3/s each")
end

# (b) apply the cuts, put those rivers on the cut face (closed cells next to open water)
for (k, nm) in enumerate(names)
    haskey(mouth_cut, nm) || continue
    p1, p2, up = mouth_cut[nm]
    m0 = copy(mask_cut)
    cut_upstream!(mask_cut, grd.lon, grd.lat, p1, p2, up; radius_km = cut_radius_km)
    # face = newly closed cells next to open water AND on the cut segment itself
    # (not where the cut radius ends in some other channel)
    mid = ((p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)
    half = haversine(p1..., p2...) / 2 + dx / 1000
    face = [I for I in findall((m0 .> 0.5) .& (mask_cut .< 0.5))
            if haversine(grd.lon[I], grd.lat[I], mid...) <= half &&
               any(d -> checkbounds(Bool, mask_cut, I + d) && mask_cut[I + d] > 0.5, NB4)]
    isempty(face) && (@warn "$nm: cut line closes nothing next to open water, placed normally"; continue)
    cells[k] = face; is_cut[k] = true
    println("$nm: cut closed $(count((m0 .> 0.5) .& (mask_cut .< 0.5))) cells, river on $(length(face)) face cells, ",
            round(qmean[k] / length(face), digits = 1), " m3/s each")
end

# (c) the other rivers: nearest free coastal land cells of the edited mask, then
# close_inland; if it dried a source cell's water, place again on the new coast (pass 2)
todo = findall(.!is_cut)
lostwet(m, ck) = any(c -> !any(d -> m[c + d] > 0.5, NB4), ck)
place_rivers!(cells, grd.lon, grd.lat, coast_of(mask_cut; bdry_cells), mouths, qmean, todo; q_cell_max)
mask_new = redirect_stderr(devnull) do; close_inland(mask_cut, grd.h, cells; path_tol = 20); end
if any(k -> lostwet(mask_new, cells[k]), todo)
    println("pass 2: re-placing on the coast left by close_inland")
    place_rivers!(cells, grd.lon, grd.lat, coast_of(mask_new; bdry_cells), mouths, qmean, todo; q_cell_max)
    mask_new = close_inland(mask_cut, grd.h, cells; path_tol = 20)   # 20 keeps both Amazon outlet channels
end
rflx = build_rflx(size(grd.mask), cells)
println("wet cells removed in total: ", count((grd.mask .> 0.5) .& (mask_new .< 0.5)))

println("grid: ", grid_file)
println(rpad("k", 4), rpad("stn", 5), rpad("river", 18), rpad("record", 11), rpad("ratio", 7), rpad("frac", 6),
        rpad("Qmean m3/s", 12), rpad("ncell", 7), "cells (i,j)")
for (k, s) in enumerate(src)
    println(rpad(k, 4), rpad(s, 5), rpad(names[k], 18), rpad("$(riv.yrb[s])-$(riv.yre[s])", 11),
            rpad(round(riv.ratio[s], digits = 2), 7), rpad(frac[k], 6), rpad(round(qmean[k], digits = 1), 12),
            rpad(length(cells[k]), 7), join(Tuple.(cells[k][1:min(3, end)]), " "), length(cells[k]) > 3 ? " ..." : "")
end

# write the outputs: grid_out is rebuilt from grid_src, frc_out is overwritten
write_grid_rflx(grid_src, grid_out, mask_new, rflx); println("wrote ", grid_out)
write_river_frc(frc_out, q_clim, names; riv_temp, riv_salt); println("wrote ", frc_out)

## figure: grid land mask + coastline, river source cells coloured by flux per cell
fig_out = "/home/hsinyi/roms_data/riverfile/river_sources_$(dx)m_1104.png"
child_grid = "/home/hsinyi/roms_data/grid/new_design/roms_grd_900m_1104.nc"   # outline drawn on the figure (nothing = skip)

using CairoMakie
# closed outline of a curvilinear grid (outer rho points)
outline(lon, lat) = vcat([Point2f(lon[i, 1], lat[i, 1]) for i in axes(lon, 1)],
                         [Point2f(lon[end, j], lat[end, j]) for j in axes(lon, 2)],
                         [Point2f(lon[i, end], lat[i, end]) for i in reverse(axes(lon, 1))],
                         [Point2f(lon[1, j], lat[1, j]) for j in reverse(axes(lon, 2))])

function plot_sources(fig_out, grd, cells, qmean, names; mask_new = grd.mask, child_grid = nothing)
    child_edge = isnothing(child_grid) ? nothing : NCDataset(child_grid) do ds
        outline(mod.(ds["lon_rho"][:, :], 360) .- 360, ds["lat_rho"][:, :])
    end
    child_col = "#e87b1c"
    lon = grd.lon .- 360                      # degrees east, negative = west
    lat = grd.lat
    # per-cell source amount: river mean flux shared equally over its cells
    pts  = [Point2f(lon[c], lat[c]) for k in eachindex(cells) for c in cells[k]]
    qpts = [qmean[k] / length(cells[k]) for k in eachindex(cells) for c in cells[k]]
    blues = cgrad(["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])   # one-hue sequential ramp
    land = cgrad(["#ffffff", "#d9d8d3", "#d62728"], 3, categorical = true)  # wet, land, closed by close_inland
    cls  = ifelse.(grd.mask .< 0.5, 1.0, ifelse.(mask_new .< 0.5, 2.0, 0.0))

    # zoom box around all source cells
    xs, ys = first.(pts), last.(pts)
    zx = (minimum(xs) - 0.8, maximum(xs) + 1.6); zy = (minimum(ys) - 0.6, maximum(ys) + 0.6)

    fig = Figure(size = (1300, 700), fontsize = 14)
    for (col, ttl, lims) in ((1, "ROMS grid $(dx) m (1104)", nothing),
                             (2, "River source cells", (zx, zy)))
        ax = Axis(fig[1, col], title = ttl, xlabel = "Longitude (°E)", ylabel = "Latitude (°N)",
                  aspect = DataAspect(), xgridvisible = false, ygridvisible = false)
        surface!(ax, lon, lat, zeros(size(lon)); color = cls, colormap = land,
                 colorrange = (0, 2), shading = NoShading)
        contour!(ax, lon, lat, grd.mask; levels = [0.5], color = :gray35, linewidth = 1)
        # grid outline
        lines!(ax, outline(lon, lat), color = :gray20, linewidth = 1.5)
        isnothing(child_edge) || lines!(ax, child_edge, color = child_col, linewidth = 2)
        sc = scatter!(ax, pts; color = qpts, colormap = blues, colorrange = extrema(qpts) .+ (-1, 1),
                      markersize = col == 1 ? 9 : 14, strokecolor = :white, strokewidth = 1)
        if col == 1
            lines!(ax, Point2f[(zx[1], zy[1]), (zx[2], zy[1]), (zx[2], zy[2]), (zx[1], zy[2]), (zx[1], zy[1])],
                   color = :black, linestyle = :dash, linewidth = 1)
        else
            limits!(ax, zx..., zy...)
            for k in eachindex(cells)      # direct label: river name + total flux
                c = cells[k][argmax(lon[cells[k]])]          # east-most cell, label to its right
                text!(ax, lon[c], lat[c], text = "$(names[k])\n$(round(Int, qmean[k])) m³/s, $(length(cells[k])) cells",
                      align = (:left, :center), offset = (22, 0), fontsize = 13, color = :gray10)
            end
            Colorbar(fig[1, 3], sc, label = "Mean discharge per source cell (m³/s)")
            els  = Any[PolyElement(color = "#d62728")]
            labs = ["closed: fills, cuts, close_inland ($(count((grd.mask .> 0.5) .& (mask_new .< 0.5))) cells)"]
            isnothing(child_edge) || (push!(els, LineElement(color = child_col, linewidth = 2));
                                      push!(labs, "child grid: $(basename(child_grid))"))
            axislegend(ax, els, labs, position = :lb, framevisible = true, backgroundcolor = (:white, 0.85))
        end
    end
    save(fig_out, fig, px_per_unit = 2)
end
plot_sources(fig_out, grd, cells, qmean, names; mask_new, child_grid)
println("wrote ", fig_out)

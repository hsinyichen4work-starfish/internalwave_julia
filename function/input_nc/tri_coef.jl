# tri_coef.jl
#
# Horizontal (triangle-based linear) interpolation from a parent grid to
# child points. Ported from get_tri_coef.m, gnomonic.m, tsearch.m,
# fix_outside_child.m and fillmask.m (h2r / Jeroen Molemaker, UCLA).
#
# The MATLAB versions call delaunay + tsearch(n) on the parent points.
# Every parent grid in this pipeline is a *structured* (Mp x Lp) grid, so
# here the Delaunay triangle around a point is found directly instead:
# locate the grid cell that holds the point, then split that cell along
# the diagonal the Delaunay (empty circumcircle) criterion picks. For a
# near-orthogonal grid this is the same triangle delaunay() builds, but
# it avoids triangulating the whole grid (minutes for the full NCOM grid
# with DelaunayTriangulation.jl) and gives the same answer every run.
# The one difference: a point outside the grid but inside its convex hull
# is reported as "outside" here, where MATLAB would have used one of the
# sliver triangles along the hull.

"""
    gnomonic(lon, lat, lon0, lat0) -> (xg, yg)

Gnomonic projection (distance preserving for interpolation purposes).
lon/lat in degrees; lon0/lat0 the center of the domain.
"""
function gnomonic(lon::AbstractArray, lat::AbstractArray, lon0::Real, lat0::Real)
    if maximum(lon) - minimum(lon) > 100 || maximum(lat) - minimum(lat) > 100
        println("This area is too large for gnomonic projections!!")
        return float.(lon), float.(lat)
    end
    d2r = pi / 180
    latr, lonr = lat .* d2r, lon .* d2r
    lat0r, lon0r = lat0 * d2r, lon0 * d2r

    cosc = sin(lat0r) .* sin.(latr) .+ cos(lat0r) .* cos.(latr) .* cos.(lonr .- lon0r)
    xg = cos.(latr) .* sin.(lonr .- lon0r) ./ cosc
    yg = (cos(lat0r) .* sin.(latr) .- sin(lat0r) .* cos.(latr) .* cos.(lonr .- lon0r)) ./ cosc
    return xg, yg
end

# barycentric weights of (x,y) in triangle 1-2-3
@inline function _bary(x1, y1, x2, y2, x3, y3, x, y)
    d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / d
    w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / d
    return w1, w2, 1 - w1 - w2
end

# true if point d lies strictly inside the circumcircle of a-b-c
@inline function _incircle(ax, ay, bx, by, cx, cy, dx, dy)
    adx, ady = ax - dx, ay - dy
    bdx, bdy = bx - dx, by - dy
    cdx, cdy = cx - dx, cy - dy
    det = (adx^2 + ady^2) * (bdx * cdy - cdx * bdy) -
          (bdx^2 + bdy^2) * (adx * cdy - cdx * ady) +
          (cdx^2 + cdy^2) * (adx * bdy - bdx * ady)
    orient = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return det * orient > 0
end

"""
    tri_locate(xp, yp, xc, yc) -> (elem, coef)

For every child point (xc,yc) find the Delaunay triangle of the
structured parent grid (xp,yp), size (Mp,Lp), that contains it.

- `elem :: (size(xc)..., 3)` linear indices into the (Mp,Lp) parent
  arrays, all 0 for a point outside the parent grid (MATLAB's tsearch
  returns NaN there)
- `coef :: (size(xc)..., 3)` barycentric weights

Stands in for `tri = delaunay(xp,yp)` + `tsearch`/`tsearchn`.
"""
function tri_locate(xp::AbstractMatrix, yp::AbstractMatrix, xc::AbstractArray, yc::AbstractArray)
    Mp, Lp = size(xp)
    (Mp >= 2 && Lp >= 2) || error("tri_locate: parent grid must be at least 2 x 2")
    Nc = length(xc)
    tol = 1e-10

    tree = KDTree(permutedims(hcat(vec(xp), vec(yp))))
    near, _ = nn(tree, permutedims(hcat(vec(xc), vec(yc))))

    elem = zeros(Int, Nc, 3)
    coef = zeros(Float64, Nc, 3)
    lin = LinearIndices((Mp, Lp))

    Threads.@threads for ic in 1:Nc
        x, y = xc[ic], yc[ic]
        j0, i0 = Tuple(CartesianIndices((Mp, Lp))[near[ic]])

        best = -Inf
        # cells touching the nearest node first; widen once for distorted cells
        for ring in 1:2
            for j in max(1, j0 - ring):min(Mp - 1, j0 + ring - 1), i in max(1, i0 - ring):min(Lp - 1, i0 + ring - 1)
                p1, p2, p3, p4 = lin[j, i], lin[j+1, i], lin[j+1, i+1], lin[j, i+1]
                # Delaunay split of the cell: keep diagonal 1-3 unless 4 is inside circle(1,2,3)
                tris = _incircle(xp[p1], yp[p1], xp[p2], yp[p2], xp[p3], yp[p3], xp[p4], yp[p4]) ?
                       ((p1, p2, p4), (p2, p3, p4)) : ((p1, p2, p3), (p1, p3, p4))
                for (a, b, c) in tris
                    w1, w2, w3 = _bary(xp[a], yp[a], xp[b], yp[b], xp[c], yp[c], x, y)
                    wmin = min(w1, w2, w3)
                    if wmin > best
                        best = wmin
                        elem[ic, 1], elem[ic, 2], elem[ic, 3] = a, b, c
                        coef[ic, 1], coef[ic, 2], coef[ic, 3] = w1, w2, w3
                    end
                end
            end
            best >= -tol && break
        end
        if !(best >= -tol)      # outside the parent grid
            elem[ic, :] .= 0
            coef[ic, :] .= 0
        end
    end
    return reshape(elem, size(xc)..., 3), reshape(coef, size(xc)..., 3)
end

"""
    fix_outside_child(lonc, latc, outside) -> (lonc, latc)

Move every child point flagged in `outside` onto its nearest inside
child point (those points should be masked!). Port of fix_outside_child.m.
"""
function fix_outside_child(lonc::AbstractArray, latc::AbstractArray, outside::AbstractArray{Bool})
    println("Fixing outside child points, make sure these are masked")
    inside = findall(.!vec(outside))
    isempty(inside) && error("fix_outside_child: no child point lies inside the parent grid")
    tree = KDTree(permutedims(hcat(vec(lonc)[inside], vec(latc)[inside])))
    lonc, latc = copy(lonc), copy(latc)
    for ic in findall(vec(outside))
        idx, _ = nn(tree, [lonc[ic], latc[ic]])
        lonc[ic] = lonc[inside[idx]]
        latc[ic] = latc[inside[idx]]
    end
    return lonc, latc
end

"""
    tsearch_fix(xp, yp, xc, yc; label) -> (elem, coef)

`tri_locate`, followed by the outside-point fix every h2r routine applies
(warn, move outside child points onto their nearest inside neighbor,
search again).
"""
function tsearch_fix(xp, yp, xc, yc; label = "tsearch_fix")
    elem, coef = tri_locate(xp, yp, xc, yc)
    outside = selectdim(elem, ndims(elem), 1) .== 0
    if any(outside)
        println("Warning in $label: outside point(s) detected.")
        xc, yc = fix_outside_child(xc, yc, outside)
        elem, coef = tri_locate(xp, yp, xc, yc)
    end
    return elem, coef
end

"""
    get_tri_coef(lonp, latp, lonc, latc, maskp) -> (elem, coef, nnel)

Port of get_tri_coef.m.

- `lonp, latp` :: (Mp,Lp) parent grid, `lonc, latc` :: (Mc,Lc) child grid
- `elem` :: (Mc,Lc,3) pointers into the (Mp,Lp) parent data (3 per child point)
- `coef` :: (Mc,Lc,3) linear interpolation coefficients
- `nnel` :: (Mp,Lp) pointer to the nearest non-masked parent neighbor

Interpolate with `Fc = apply_tri_coef(elem, coef, Fp)`
(MATLAB: `sum(coef.*Fp(elem),3)`).
"""
function get_tri_coef(lonp::AbstractMatrix, latp::AbstractMatrix,
                      lonc::AbstractMatrix, latc::AbstractMatrix, maskp::AbstractMatrix)
    # project lon, lat with a gnomonic projection for accurate distances
    width = max(maximum(lonp) - minimum(lonp), maximum(latp) - minimum(latp),
                maximum(lonc) - minimum(lonc), maximum(latc) - minimum(latc))
    if width < 100
        lon0 = sum(lonc) / length(lonc)
        lat0 = sum(latc) / length(latc)
        xp, yp = gnomonic(lonp, latp, lon0, lat0)
        xc, yc = gnomonic(lonc, latc, lon0, lat0)
    else
        println("too big for gnomonic projection")
        xp, yp = float.(lonp), float.(latp)
        xc, yc = float.(lonc), float.(latc)
    end

    # child points outside the parent grid are moved onto the nearest inside one
    elem, coef = tsearch_fix(xp, yp, xc, yc; label = "get_tri_coef")

    # nearest non-masked parent neighbor of every parent point
    nnel = collect(LinearIndices(size(lonp)))
    valid = findall(!=(0), vec(maskp))
    if !isempty(valid) && length(valid) < length(maskp)
        vtree = KDTree(permutedims(hcat(vec(xp)[valid], vec(yp)[valid])))
        masked = findall(==(0), vec(maskp))
        idx, _ = nn(vtree, permutedims(hcat(vec(xp)[masked], vec(yp)[masked])))
        nnel[masked] = valid[idx]
    end

    return elem, coef, nnel
end

"""
    apply_tri_coef(elem, coef, Fp) -> Fc

`sum(coef.*Fp(elem),3)`: interpolate the 2D parent field `Fp` onto the
child grid with coefficients from `get_tri_coef`.
"""
function apply_tri_coef(elem::AbstractArray{<:Integer,3}, coef::AbstractArray{<:Real,3}, Fp::AbstractMatrix)
    Mc, Lc, _ = size(elem)
    Fc = zeros(Float64, Mc, Lc)
    @inbounds for c in 1:3, j in 1:Lc, i in 1:Mc
        Fc[i, j] += coef[i, j, c] * Fp[elem[i, j, c]]
    end
    return Fc
end

"""
    fillmask(f, type, maskp, nnel) -> fm

Put values in the masked points: nearest non-masked neighbor
(`type == 1`) or zero (`type == 0`). `f` is a 2D (Mp,Lp) field or a 3D
(N,Mp,Lp) one. Port of fillmask.m.
"""
function fillmask(f::AbstractArray, type::Integer, maskp::AbstractMatrix, nnel::AbstractMatrix)
    fm = float.(f)
    masked = findall(==(0), vec(maskp))
    if ndims(f) == 3
        n = size(f, 1)
        for k in 1:n
            fk = view(fm, k, :, :)
            if type == 1
                fk[masked] = fk[nnel[masked]]
            else
                fk[masked] .= 0.0 .* fk[masked]   # 0*NaN stays NaN, as in MATLAB
            end
        end
    else
        if type == 1
            fm[masked] = fm[nnel[masked]]
        else
            fm[masked] .= 0.0 .* fm[masked]
        end
    end
    return fm
end

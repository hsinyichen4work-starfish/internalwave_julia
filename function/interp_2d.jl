"""
    interp_2d.jl

Case (1): from (x,y) to (x',y') — horizontal-only interpolation on
scattered or curvilinear grids, direct translation of `get_tri_coef.m`
(+ `gnomonic.m` for the projection, and the nearest-neighbor logic from
`fix_outside_child.m` / `fillmask.m` for masked/out-of-hull points).

Requires:
    ] add DelaunayTriangulation NearestNeighbors

NOTE ON PORTABILITY: `DelaunayTriangulation.jl`'s point-location API
(`find_triangle`, ghost-triangle detection) has changed names across
versions. The call below matches recent releases; if your installed
version errors on `find_triangle` or `is_ghost_triangle`, check
`? DelaunayTriangulation` for the current point-location function name
(older versions used `jump_and_march`) and swap it in — the rest of the
algorithm (barycentric weights, fallback logic) is unaffected.
"""
module Interp2D

using DelaunayTriangulation
using NearestNeighbors

export gnomonic, get_2d_coef, apply_2d_coef, interp_2d

"""
    gnomonic(lon, lat, lon0, lat0) -> (xg, yg)

Gnomonic (great-circle-distance-preserving) projection so that
Euclidean triangulation/distances in (x,y) approximate true spherical
distance. Direct translation of `gnomonic.m`. lon/lat in degrees.
"""
function gnomonic(lon::AbstractArray, lat::AbstractArray, lon0::Real, lat0::Real)
    if maximum(lon) - minimum(lon) > 100 || maximum(lat) - minimum(lat) > 100
        @warn "Domain too large for a gnomonic projection; returning lon/lat unprojected."
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

"""
    barycentric(p1, p2, p3, q) -> (w1, w2, w3)

Barycentric weights of point `q` w.r.t. triangle (p1,p2,p3), each an
(x,y) tuple.
"""
function barycentric(p1, p2, p3, q)
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x, y = q
    d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / d
    w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / d
    w3 = 1 - w1 - w2
    return w1, w2, w3
end

"""
    get_2d_coef(lonp, latp, lonc, latc; maskp=nothing, project=true) -> (elem, coef, nnel)

Horizontal (2D) interpolation coefficients, analog of `get_tri_coef.m`.
`lonp,latp` (parent) and `lonc,latc` (child) are any-shaped arrays of
matching-shaped coordinates (typically 2D curvilinear grids).

Returns:
- `elem :: Matrix{Int}`     size (Nc,3): flat parent-point indices of the
                             enclosing triangle for each child point
- `coef :: Matrix{Float64}` size (Nc,3): barycentric weights
- `nnel :: Vector{Int}`     size (Np,): nearest *unmasked* parent index for
                             every parent point (fillmask.m equivalent)

Apply with: `Fc = sum(coef .* Fp[elem]; dims=2)` or `apply_2d_coef`.

Child points outside the parent's convex hull fall back to their
single nearest parent point (playing the role of `fix_outside_child.m`,
which instead re-triangulates after nudging outside points away —
functionally equivalent nearest-valid-data behavior).
"""
function get_2d_coef(lonp::AbstractArray, latp::AbstractArray,
                      lonc::AbstractArray, latc::AbstractArray;
                      maskp::Union{Nothing,AbstractArray}=nothing,
                      project::Bool=true)

    Np, Nc = length(lonp), length(lonc)

    xp, yp = vec(float.(lonp)), vec(float.(latp))
    xc, yc = vec(float.(lonc)), vec(float.(latc))

    if project
        lon0, lat0 = sum(xc) / Nc, sum(yc) / Nc
        xp2, yp2 = gnomonic(reshape(xp, size(lonp)), reshape(yp, size(lonp)), lon0, lat0)
        xc2, yc2 = gnomonic(reshape(xc, size(lonc)), reshape(yc, size(lonc)), lon0, lat0)
        xp, yp, xc, yc = vec(xp2), vec(yp2), vec(xc2), vec(yc2)
    end

    points = [(xp[i], yp[i]) for i in 1:Np]
    tri = triangulate(points)

    elem = ones(Int, Nc, 3)
    coef = zeros(Float64, Nc, 3)

    tree = KDTree(permutedims(hcat(xp, yp)))  # fallback for outside-hull points

    for ic in 1:Nc
        q = (xc[ic], yc[ic])
        v = try
            find_triangle(tri, q)
        catch
            (0, 0, 0)
        end
        i1, i2, i3 = v
        outside = i1 == 0 || DelaunayTriangulation.is_ghost_triangle(v)
        if outside
            idx, _ = nn(tree, [q[1], q[2]])
            elem[ic, :] .= idx
            coef[ic, 1] = 1.0
        else
            w1, w2, w3 = barycentric((xp[i1], yp[i1]), (xp[i2], yp[i2]), (xp[i3], yp[i3]), q)
            elem[ic, :] = [i1, i2, i3]
            coef[ic, :] = [w1, w2, w3]
        end
    end

    nnel = collect(1:Np)
    if maskp !== nothing
        mp = vec(maskp)
        valid = findall(!=(0), mp)
        if !isempty(valid) && length(valid) < Np
            vtree = KDTree(permutedims(hcat(xp[valid], yp[valid])))
            for i in 1:Np
                if mp[i] == 0
                    j, _ = nn(vtree, [xp[i], yp[i]])
                    nnel[i] = valid[j]
                end
            end
        end
    end

    return elem, coef, nnel
end

"""
    apply_2d_coef(coef, elem, Fp) -> Fc

Apply precomputed 2D coefficients to a (flattened) parent field `Fp`.
"""
function apply_2d_coef(coef::AbstractMatrix, elem::AbstractMatrix, Fp::AbstractVector)
    vec(sum(coef .* Fp[elem]; dims=2))
end

"""
    interp_2d(lonp, latp, Fp, lonc, latc; maskp=nothing) -> Fc

One-shot 2D horizontal interpolation, reshaped to `size(lonc)`. Masked
parent points (where `maskp .== 0`) are filled from their nearest
unmasked neighbor before interpolating (fillmask.m equivalent).
"""
function interp_2d(lonp, latp, Fp, lonc, latc; maskp=nothing)
    elem, coef, nnel = get_2d_coef(lonp, latp, lonc, latc; maskp=maskp)
    Fpf = vec(float.(Fp))
    if maskp !== nothing
        mp = vec(maskp)
        Fpf = copy(Fpf)
        bad = mp .== 0
        Fpf[bad] .= Fpf[nnel[bad]]
    end
    Fc = apply_2d_coef(coef, elem, Fpf)
    return reshape(Fc, size(lonc))
end

end # module
# Bilinear interpolation from any (regular or curvilinear) source grid to target points.
# Used by model_validation/tidal/ssh_tidalcomp_fes_models.jl and DFF/m2_modes_models_plot.jl.
#   L = locator(slon, slat, tlon, tlat)     # source lon / lat (2-D), target lon / lat (any shape), same lon convention
#   F_t = bilin(L, F_s)                     # source field (2-D, real or complex) -> target points; NaN outside or
#                                           # when any of the 4 source corners is NaN
# The cell of each target point is found by walking from the previous point's cell (target points visited in grid
# order), with a global nearest-node search when the walk fails; weights from the inverse bilinear map.

function invbilin(X, Y, x, y)                            # corners 00, 10, 01, 11 -> (a, b), may be outside [0, 1]
    a, b = 0.5, 0.5
    for _ in 1:30
        fx = (1 - a) * (1 - b) * X[1] + a * (1 - b) * X[2] + (1 - a) * b * X[3] + a * b * X[4] - x
        fy = (1 - a) * (1 - b) * Y[1] + a * (1 - b) * Y[2] + (1 - a) * b * Y[3] + a * b * Y[4] - y
        xa = (1 - b) * (X[2] - X[1]) + b * (X[4] - X[3]); xb = (1 - a) * (X[3] - X[1]) + a * (X[4] - X[2])
        ya = (1 - b) * (Y[2] - Y[1]) + b * (Y[4] - Y[3]); yb = (1 - a) * (Y[3] - Y[1]) + a * (Y[4] - Y[2])
        det = xa * yb - xb * ya; abs(det) < 1e-20 && return NaN, NaN
        da = (yb * fx - xb * fy) / det; db = (-ya * fx + xa * fy) / det
        a -= da; b -= db
        abs(da) + abs(db) < 1e-12 && break
    end
    return a, b
end

function locator(slon, slat, tlon, tlat)
    nx, ny = size(slon)
    I0 = zeros(Int, size(tlon)); J0 = zeros(Int, size(tlon)); A = zeros(size(tlon)); Bw = zeros(size(tlon))
    ci, cj = 1, 1
    function trycell(i, j, x, y, c)
        X = c .* (slon[i, j], slon[i+1, j], slon[i, j+1], slon[i+1, j+1]); Y = (slat[i, j], slat[i+1, j], slat[i, j+1], slat[i+1, j+1])
        return invbilin(X, Y, c * x, y)
    end
    for p in CartesianIndices(tlon)
        x, y = tlon[p], tlat[p]; c = cosd(y); found = false
        for _ in 1:200                                    # walk
            a, b = trycell(ci, cj, x, y, c)
            isfinite(a) || break
            if -1e-9 <= a <= 1 + 1e-9 && -1e-9 <= b <= 1 + 1e-9
                I0[p], J0[p], A[p], Bw[p] = ci, cj, clamp(a, 0, 1), clamp(b, 0, 1); found = true; break
            end
            ni = clamp(ci + floor(Int, a), 1, nx - 1); nj = clamp(cj + floor(Int, b), 1, ny - 1)
            (ni, nj) == (ci, cj) && break                 # stuck at the edge: outside
            ci, cj = ni, nj
        end
        if !found                                         # global nearest node, then its 4 cells
            d = ((slon .- x) .* c) .^ 2 .+ (slat .- y) .^ 2; q = argmin(d)
            for i in (q[1] - 1, q[1]), j in (q[2] - 1, q[2])
                (1 <= i < nx && 1 <= j < ny) || continue
                a, b = trycell(i, j, x, y, c)
                if -1e-9 <= a <= 1 + 1e-9 && -1e-9 <= b <= 1 + 1e-9
                    I0[p], J0[p], A[p], Bw[p] = i, j, clamp(a, 0, 1), clamp(b, 0, 1); ci, cj = i, j; break
                end
            end
        end
    end
    return (; I0, J0, A, B = Bw)
end

function bilin(L, F)
    out = fill(convert(eltype(F), NaN), size(L.I0))
    for p in CartesianIndices(L.I0)
        i, j = L.I0[p], L.J0[p]; i == 0 && continue
        a, b = L.A[p], L.B[p]
        out[p] = (1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i+1, j] + (1 - a) * b * F[i, j+1] + a * b * F[i+1, j+1]
    end
    return out
end

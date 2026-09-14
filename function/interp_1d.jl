"""
    interp_1d.jl

Case (4): from (x) to (x')  — e.g. one water column's sigma/z levels
interpolated onto a new set of target depths.

Direct translation of `get_1d_coef.m` (+ the implicit apply step used
throughout the MATLAB pipeline). No external package dependencies.
"""
module Interp1D

export get_1d_coef, apply_1d_coef, interp_1d

"""
    get_1d_coef(xp, xc) -> (coef, elem)

Linear interpolation weights mapping a parent 1D grid `xp` (ascending)
onto arbitrary target locations `xc`.

Returns:
- `coef :: Matrix{Float64}` size (length(xc), 2)
- `elem :: Matrix{Int}`     size (length(xc), 2), 1-based indices into `xp`

Points at/below `xp[1]` or at/above `xp[end]` are nearest-neighbor
filled (no extrapolation), matching the original MATLAB behavior.

Apply with: `Fc = coef[:,1].*Fp[elem[:,1]] .+ coef[:,2].*Fp[elem[:,2]]`
(or call `apply_1d_coef`).
"""
function get_1d_coef(xp::AbstractVector{<:Real}, xc::AbstractVector{<:Real})
    Np = length(xp)
    Nc = length(xc)
    coef = zeros(Float64, Nc, 2)
    elem = ones(Int, Nc, 2)

    ip = 1
    for ic in 1:Nc
        while ip < Np && xp[ip] < xc[ic]
            ip += 1
        end
        if ip == 1 || xp[ip] < xc[ic]
            # xc[ic] at/below xp[1], or at/above xp[end]: nearest-neighbor
            coef[ic, 1] = 1.0
            elem[ic, 1] = ip
            continue
        end
        alp = (xc[ic] - xp[ip-1]) / (xp[ip] - xp[ip-1])
        coef[ic, 1] = alp
        elem[ic, 1] = ip
        coef[ic, 2] = 1 - alp
        elem[ic, 2] = ip - 1
    end
    return coef, elem
end

"""
    apply_1d_coef(coef, elem, Fp) -> Fc

Apply precomputed 1D coefficients (from `get_1d_coef`) to parent data `Fp`.
"""
function apply_1d_coef(coef::AbstractMatrix, elem::AbstractMatrix, Fp::AbstractVector)
    @views coef[:, 1] .* Fp[elem[:, 1]] .+ coef[:, 2] .* Fp[elem[:, 2]]
end

"""
    interp_1d(xp, Fp, xc) -> Fc

One-shot convenience wrapper. For repeated interpolation of many
fields on the same (xp -> xc) pair, call `get_1d_coef` once and reuse
the coefficients with `apply_1d_coef` instead.
"""
function interp_1d(xp::AbstractVector, Fp::AbstractVector, xc::AbstractVector)
    coef, elem = get_1d_coef(xp, xc)
    return apply_1d_coef(coef, elem, Fp)
end

end # module
# uv2rho.jl
#
# Ported from u2rho.m / v2rho.m (useful_tools) — interpolates U-/V-point
# data onto the RHO grid. Interior points are averaged from their two
# neighbors; the two edge rows/columns are copied from the nearest
# U/V point (no second neighbor to average with there).
#
# Generalized here (like rho2uvp.jl) to work on any array with extra
# trailing dimensions (depth, time), not just 2D/3D.

function u2rho(var_u::AbstractArray)
    nd = ndims(var_u)
    trailing = ntuple(_ -> Colon(), nd - 1)
    interior = 0.5 .* (view(var_u, 1:size(var_u, 1)-1, trailing...) .+ view(var_u, 2:size(var_u, 1), trailing...))
    return cat(var_u[1:1, trailing...], interior, var_u[end:end, trailing...]; dims = 1)
end

function v2rho(var_v::AbstractArray)
    nd = ndims(var_v)
    trailing = ntuple(_ -> Colon(), nd - 2)
    interior = 0.5 .* (view(var_v, :, 1:size(var_v, 2)-1, trailing...) .+ view(var_v, :, 2:size(var_v, 2), trailing...))
    return cat(var_v[:, 1:1, trailing...], interior, var_v[:, end:end, trailing...]; dims = 2)
end

#= gradients.jl
Step 3 of wave_mean_flow.jl: horizontal and vertical derivatives for the energy terms.
Arrays in Julia order (xi, eta, k[, ...]), k = 1 the bottom, as in ROMS. NaN (land) is carried through.

  dx, dy = rho_spacing(pm, pn)            distance between neighbouring rho points: dx (nx-1, ny), dy (nx, ny-1)
  fz      = ddz_rho(f, zr)                ∂f/∂z on the rho levels (centred over two layers, ends extrapolated)
  fx, fy, fz = hor_deriv(f, zr, dx, dy)   ∂f/∂x, ∂f/∂y at constant z (sigma correction) + ∂f/∂z, for 3-D background
                                          fields on rho points: U, V (after u2rho!/v2rho!), B, φn
  qz      = ddz_w2r(q, zw)                ∂q/∂z of a w-level field q (N+1) on the rho levels (N): e.g. q = N²Φn (An4)
  fz      = ddz_r2r(f, zr, zw)            ∂f/∂z of a rho-level field, differenced to the interior w levels and averaged
                                          back to rho levels (staggered; alternative to ddz_rho for ∂U/∂z, ∂V/∂z)
  ax, ay  = grad2d(a, dx, dy)             ∂a/∂x, ∂a/∂y of 2-D fields (xi, eta[, time / mode ...]): un, vn, pn, H un, cn
  φ0, Φ0, c0 = mode0(zr, zw, h, zeta_window)   barotropic mode: φ0 = 1, Φ0 = z + H, c0 = √(gH)

hor_deriv is a port of Oladeji Siyanbola's hor_deriv.m (USM, 2023):
  ∂f/∂x|_z = ∂f/∂x|_σ − (∂f/∂z)(∂z/∂x|_σ),  centred differences over two cells, ∂f/∂z centred over two layers with
  linear extrapolation of ∂f/∂z to the bottom and surface levels.
  One change: the edges that hor_deriv.m sets to 0 (xi edges of fx, eta edges of fy, and the eta edges of fx /
  xi edges of fy) are NaN here, so they cannot be mistaken for data.
u, v, U, V, un, vn are along the grid ξ / η: keep every vector and gradient in grid coordinates and the dot products in
the energy terms are right without rotating. On a strongly curvilinear grid the divergence would need metric terms
(pm pn [∂ξ(Hu/pn) + ∂η(Hv/pm)]); grad2d ignores them (fine for a near-uniform grid).
=#

const G_ACC = 9.81

# distance between neighbouring rho points along xi (nx-1, ny) and eta (nx, ny-1), from the grid metrics pm = 1/dx, pn = 1/dy
rho_spacing(pm, pn) = (2 ./ (pm[1:end-1, :] .+ pm[2:end, :]), 2 ./ (pn[:, 1:end-1] .+ pn[:, 2:end]))

# ∂f/∂z on the rho levels, as hor_deriv.m lines 18-22
function ddz_rho(f, zr)
    N = size(f, 3); fz = f .* 0
    @views fz[:, :, 2:N-1] .= (f[:, :, 3:N] .- f[:, :, 1:N-2]) ./ (zr[:, :, 3:N] .- zr[:, :, 1:N-2])
    @views fz[:, :, 1] .= fz[:, :, 2] .- (zr[:, :, 2] .- zr[:, :, 1]) .* (fz[:, :, 3] .- fz[:, :, 2]) ./ (zr[:, :, 3] .- zr[:, :, 2])
    @views fz[:, :, N] .= fz[:, :, N-1] .- (zr[:, :, N-1] .- zr[:, :, N]) .* (fz[:, :, N-2] .- fz[:, :, N-1]) ./
                          (zr[:, :, N-2] .- zr[:, :, N-1])
    return fz
end

# ∂f/∂x, ∂f/∂y at constant z and ∂f/∂z, for a 3-D field f on rho points (xi, eta, k); zr the depths of its levels
# ∂f/∂x|z = ∂f/∂x|sig - ∂f/∂z * ∂z/∂x|sig = (along sig lay) - (verti grad * grid slope)
function hor_deriv(f, zr, dx, dy)
    nx, ny, N = size(f)
    fz = ddz_rho(f, zr)
    d2x = dx[1:end-1, :] .+ dx[2:end, :]            # distance i-1 -> i+1  (nx-2, ny)
    d2y = dy[:, 1:end-1] .+ dy[:, 2:end]            # distance j-1 -> j+1  (nx, ny-2)
    fx = f .* 0; fy = f .* 0
    @views fx[2:nx-1, :, :] .= (f[3:nx, :, :] .- f[1:nx-2, :, :]) ./ d2x .-
                               fz[2:nx-1, :, :] .* (zr[3:nx, :, :] .- zr[1:nx-2, :, :]) ./ d2x
    @views fy[:, 2:ny-1, :] .= (f[:, 3:ny, :] .- f[:, 1:ny-2, :]) ./ d2y .-
                               fz[:, 2:ny-1, :] .* (zr[:, 3:ny, :] .- zr[:, 1:ny-2, :]) ./ d2y
    T = eltype(fx)
    fx[[1, nx], :, :] .= T(NaN); fx[:, [1, ny], :] .= T(NaN)
    fy[:, [1, ny], :] .= T(NaN); fy[[1, nx], :, :] .= T(NaN)
    return fx, fy, fz
end

# ∂q/∂z of a w-level field q (xi, eta, N+1) on the rho levels (xi, eta, N): exact staggering, e.g. q = N² Φn for An4
ddz_w2r(q, zw) = @views (q[:, :, 2:end] .- q[:, :, 1:end-1]) ./ (zw[:, :, 2:end] .- zw[:, :, 1:end-1])

# ∂f/∂z of a rho-level field: one-layer differences on the interior w levels 2..N, then averaged back to the rho
# levels (bottom / surface rho level take the nearest interior w value). Sharper than ddz_rho in a thin shear layer.
function ddz_r2r(f, zr, zw)
    N = size(f, 3)
    fw = @views (f[:, :, 2:N] .- f[:, :, 1:N-1]) ./ (zr[:, :, 2:N] .- zr[:, :, 1:N-1])   # w levels 2..N  (N-1)
    fz = f .* 0
    @views fz[:, :, 2:N-1] .= 0.5 .* (fw[:, :, 1:N-2] .+ fw[:, :, 2:N-1])
    @views fz[:, :, 1] .= fw[:, :, 1]; @views fz[:, :, N] .= fw[:, :, N-1]
    return fz
end

# ∂a/∂x, ∂a/∂y of a 2-D field on rho points, any trailing dims (time, mode, ...): plain centred differences, edges NaN
# ∇ = ∂/∂x i + ∂/∂y j
function grad2d(a, dx, dy) 
    sz = size(a); nx, ny = sz[1], sz[2]
    A = reshape(a, nx, ny, :)
    ax = A .* 0; ay = A .* 0
    @views ax[2:nx-1, :, :] .= (A[3:nx, :, :] .- A[1:nx-2, :, :]) ./ (dx[1:end-1, :] .+ dx[2:end, :])
    @views ay[:, 2:ny-1, :] .= (A[:, 3:ny, :] .- A[:, 1:ny-2, :]) ./ (dy[:, 1:end-1] .+ dy[:, 2:end])
    T = eltype(ax)
    ax[[1, nx], :, :] .= T(NaN); ax[:, [1, ny], :] .= T(NaN)
    ay[:, [1, ny], :] .= T(NaN); ay[[1, nx], :, :] .= T(NaN)
    return reshape(ax, sz), reshape(ay, sz)
end

# barotropic mode (free-surface): φ0 = 1 on the rho levels, Φ0 = z + H on the w levels (0 at the bottom, H at the
# surface), c0 = √(g H), H = h + zeta_window. Satisfies dΦ0/dz = φ0, Φ0 = (c0²/g) dΦ0/dz at the surface, and Eq. 7 to
# O(N² H / g). Orthogonal to the rigid-lid φn (∫ φn dz = 0); its overlap with the rigid-lid Φn is ~ cn² φn(0), small.
function mode0(zr, zw, h, zeta; g = G_ACC)
    H = h .+ zeta
    φ0 = zr .* 0 .+ 1
    Φ0 = zw .+ h                                     # z + h, with zw = -h at the bottom: 0 at the bottom, H at the surface
    c0 = sqrt.(g .* H)
    return φ0, Φ0, c0
end
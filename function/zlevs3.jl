# Ported from zlevs3.m (ROMSTOOLS, Penven et al.) — computes depths (m) of
# RHO- or W-points for ROMS given the S-coordinate stretching parameters.
#
# type   : "r" (rho point) or "w" (w point)
# scoord : "old1994" (Song, 1994), "new2008"/"new2006" (Shchepetkin, 2008), "kau2015"

function zlevs3(h, zeta, theta_s, theta_b, hc, N, type, scoord)
    ϵ = 1e-20

    M, L = size(h)

    # Set S-curves in domain [-1, 0] at vertical W- and RHO-points
    if type == "w"
        sc = collect(((0:N) .- N) ./ N)
        N = N + 1
    else
        sc = collect(((1:N) .- N .- 0.5) ./ N)
    end

    if scoord == "new2008" || scoord == "kau2015"
        Cs = CSF(sc, theta_s, theta_b)
    else
        # for 'old1994' and 'new2006'
        cff1 = 1 / sinh(theta_s)
        cff2 = 0.5 / tanh(0.5 * theta_s)
        Cs = (1 - theta_b) .* cff1 .* sinh.(theta_s .* sc) .+
             theta_b .* (cff2 .* tanh.(theta_s .* (sc .+ 0.5)) .- 0.5)
    end

    z = zeros(N, M, L)

    if scoord == "old1994"
        hinv = 1 ./ h
        cff = hc .* (sc .- Cs)
        cff1 = Cs
        for k in 1:N
            z0 = cff[k] .+ cff1[k] .* h
            z[k, :, :] = z0 .+ zeta .* (1 .+ z0 .* hinv)
        end
    elseif scoord == "kau2015"
        Gcoord = 0.16 .+ abs.(sc) .^ 0.8 .* (1 .+ sc) .^ 0.2 .+
                 (0.3 .* exp.(-1 ./ max.(abs.(sc), ϵ))) ./ (max.(abs.(sc), ϵ) .^ 1.7)
        cff = hc .* sc .* Gcoord
        cff1 = Cs
        for k in 1:N
            hinv = 1 ./ (h .+ hc * Gcoord[k])
            z[k, :, :] = zeta .+ (zeta .+ h) .* (cff[k] .+ cff1[k] .* h) .* hinv
        end
    else
        # covers 'new2008' and 'new2006'
        hinv = 1 ./ (h .+ hc)
        cff = hc .* sc
        cff1 = Cs
        for k in 1:N
            z[k, :, :] = zeta .+ (zeta .+ h) .* (cff[k] .+ cff1[k] .* h) .* hinv
        end
    end

    return z, Cs
end

function CSF(sc, theta_s, theta_b)
    if theta_s > 0.0
        csrf = (1.0 .- cosh.(theta_s .* sc)) ./ (cosh(theta_s) - 1.0)
    else
        csrf = -sc .^ 2
    end
    sc1 = csrf .+ 1.0

    if theta_b > 0.0
        Cs = (exp.(theta_b .* sc1) .- 1.0) ./ (exp(theta_b) - 1.0) .- 1.0
    else
        Cs = csrf
    end
    return Cs
end

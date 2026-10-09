using NCDatasets, CairoMakie, Dates, Statistics
CairoMakie.activate!()
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_0927"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hchen54/figure/dbry_0927/energy_flux"   # separate from his_rejoint so the two scripts' skip checks don't interfere
mkpath(figure_path)

##
# fname = joinpath(datadir, "roms_dia.20220829010000.nc")
# ocean_time, up,vp = NCDataset(fname) do ds
#     show(ds)
#     ds["ocean_time"][:], ds["up"][:,:,:], ds["vp"][:,:,:]
# end

mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
    ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
end

# Split the rho grid into ndomx × ndomy tiles, each padded by `overlap` points (clipped at the domain edge).
# Per tile:
#   i, j           rho-point ranges to read from file (core + halo)
#   iu, jv         matching xi_u / eta_v ranges, so the chunk is itself a staggered mini-grid
#                  (u -> [iu, j], v -> [i, jv])
#   icore, jcore   global rho ranges this tile owns (no halo; tiles' cores cover the grid exactly once)
#   iloc, jloc     where the core sits inside the chunk that was read: chunk[iloc, jloc] -> global[icore, jcore]
function tile_ranges(nx, ny, ndomx, ndomy; overlap = 3)
    function split1d(n, ndom)
        edges = round.(Int, range(0, n, length = ndom + 1))
        map(1:ndom) do k
            core = edges[k]+1:edges[k+1]
            halo = max(1, first(core) - overlap):min(n, last(core) + overlap)
            loc = (first(core)-first(halo)+1):(last(core)-first(halo)+1)
            (; core, halo, loc)
        end
    end
    xs, ys = split1d(nx, ndomx), split1d(ny, ndomy)
    [(i = x.halo, j = y.halo,
      iu = first(x.halo):last(x.halo)-1, jv = first(y.halo):last(y.halo)-1,
      icore = x.core, jcore = y.core, iloc = x.loc, jloc = y.loc)
     for x in xs, y in ys]
end

ndomx, ndomy = 4, 4
tiles = tile_ranges(size(mask_rho)..., ndomx, ndomy; overlap = 2)   # ndomx × ndomy matrix of NamedTuples

his_files = sort(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", f), readdir(datadir)))   # joined his files only, in time order
his_files = joinpath.(datadir, his_files)

# read rho for one tile over every his file -> rho_t[xi, eta, s, time]
t = tiles[1, 1]
nt_each = [NCDataset(f) do ds; ds.dim["time"] end for f in his_files]   # records per file
nz = NCDataset(his_files[1]) do ds; ds.dim["s_rho"] end
t_end = cumsum(nt_each)
println("rho_t: ", (length(t.i), length(t.j), nz, t_end[end]), "  ≈ ",
        round(length(t.i) * length(t.j) * nz * t_end[end] * 4 / 2^30, digits = 1), " GB")

rho_t = Array{Float32}(undef, length(t.i), length(t.j), nz, t_end[end])
rho_time = Vector{Float64}(undef, t_end[end])
for (k, f) in enumerate(his_files)
    it = t_end[k]-nt_each[k]+1:t_end[k]
    NCDataset(f) do ds
        rho_time[it] = ds["ocean_time"][:]
        rho_t[:, :, :, it] = ds["rho"][t.i, t.j, :, :]
    end
    k % 20 == 0 && println("read $k / $(length(his_files))")
end

fname = his_files[1]
his_time,zeta,u,v,rho = NCDataset(fname) do ds
    show(ds)
    ds["ocean_time"][:], ds["zeta"][:,:,:], ds["u"][:,:,:,:], ds["v"][:,:,:,:], ds["rho"][:,:,:,:]
end
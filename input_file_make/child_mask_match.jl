# child_mask_match.jl
#
# After river_input.jl closed water in the 2700 m (parent) mask, make the 900 m
# (child) mask follow it: child wet cells whose nearest parent rho cell was wet
# before the river fix and is land after it are closed, then child water that is
# cut off from the sea by that is closed too. Mismatches that existed before the
# fix (resolution differences along the coast) are left alone.
#
# The input is always the untouched child backup (child_src), so re-running gives
# the same result. Run from the Documents/Julia project, e.g.
#     julia --project=/home/hsinyi/Documents/Julia input_file_make/child_mask_match.jl

using NCDatasets
using NearestNeighbors
using Dates

## settings
d = "/home/hsinyi/roms_data/grid/new_design/"
parent_old = d * "roms_grd_2700m_1104_before_riverfix.nc"   # parent before the river fix
parent_new = d * "roms_grd_2700m_1104.nc"                   # parent after the river fix
child_src  = d * "roms_grd_900m_1104_before_riverfix.nc"    # untouched child (input)
child_out  = d * "roms_grd_900m_1104.nc"                    # rebuilt child (output)
h_deep     = 50.0     # child wet cells deeper than this are 'the sea' (connectivity seeds)

for f in (parent_old, parent_new, child_src)
    isfile(f) || error("missing $f")
end

const NB4 = CartesianIndex.([(1, 0), (-1, 0), (0, 1), (0, -1)])
function reach(pass, seeds)          # cells of `pass` connected (4-nbr) to `seeds`
    R = falses(size(pass)); q = [s for s in seeds if pass[s]]; R[q] .= true; k = 1
    while k <= length(q)
        c = q[k]; k += 1
        for dd in NB4
            n = c + dd
            (checkbounds(Bool, pass, n) && pass[n] && !R[n]) || continue
            R[n] = true; push!(q, n)
        end
    end
    R
end

## nearest parent rho cell of every child rho point (local flat-earth km)
plo, pla, pm0 = NCDataset(parent_old) do ds
    mod.(ds["lon_rho"][:, :], 360), ds["lat_rho"][:, :], ds["mask_rho"][:, :] .> 0.5
end
pm1 = NCDataset(ds -> ds["mask_rho"][:, :] .> 0.5, parent_new)
clo, cla, cm, ch = NCDataset(child_src) do ds
    mod.(ds["lon_rho"][:, :], 360), ds["lat_rho"][:, :], ds["mask_rho"][:, :] .> 0.5, ds["h"][:, :]
end
lat0 = (minimum(cla) + maximum(cla)) / 2
xy(lo, la) = [vec(lo)' .* cosd(lat0) .* 111.2; vec(la)' .* 111.2]
idx, dist = nn(KDTree(xy(plo, pla)), xy(clo, cla))
CI = CartesianIndices(pm0)
par_closed = reshape([pm0[CI[i]] && !pm1[CI[i]] for i in idx], size(cm))   # closed by the river fix
println("max child->parent distance: ", round(maximum(dist), digits = 2), " km")

## close the child cells under closed parent cells, then the water they cut off
close1 = cm .& par_closed
wet = cm .& .!close1
seeds = findall(wet .& (ch .>= h_deep))
# only water that was connected to the sea before and is cut off now (ponds that
# were already isolated in the original child mask are left alone)
close2 = wet .& .!reach(wet, seeds) .& reach(cm, findall(cm .& (ch .>= h_deep)))
newmask = Float64.(wet .& .!close2)
println("child wet cells under parent cells closed by the river fix: ", count(close1))
println("child wet cells cut off from the sea by that:             ", count(close2))
println("pre-existing mismatches (child wet & parent land before the fix), left alone: ",
        count(cm .& .!reshape(pm0[CI[idx]], size(cm))))

## write: copy the backup, replace mask_rho, move into place
tmp = child_out * ".tmp"
cp(child_src, tmp; force = true)
NCDataset(tmp, "a") do ds
    ds["mask_rho"][:, :] = newmask
    ds.attrib["river_history"] = "$(today()) child_mask_match.jl: $(count(close1) + count(close2)) wet cells closed to " *
        "follow the river fix of $(basename(parent_new)); original grid: $(basename(child_src))"
end
mv(tmp, child_out; force = true)
println("wrote ", child_out)

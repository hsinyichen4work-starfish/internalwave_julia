# dbry_mat2nc.jl
#
# Rebuild the ROMS dynamic-boundary flux files (roms_dbry_flux_*.nc) from the
# MAT files written by dbry_make.jl, without recomputing the flux. The MAT
# files hold the raw (W/m) Fx/Fy and the same bry_time records as the bry
# file; ROMS reads them divided by rho0, in the layout of write_flux_nc
# (function/dbry_fun.jl):
#   up_west/up_east  (eta_rho, bry_time) from Fx
#   vp_south/vp_north(xi_rho,  bry_time) from Fy
#
#     julia --project=. input_file_make/dbry_mat2nc.jl

using MAT, NCDatasets, Printf

##
proj = "/expanse/lustre/projects/uso102/hchen54/new_design/"
mat_path = proj * "dbry_mat/"
flux_out_path = proj * "bry/"   # next to the bry files (partition_input reads dbry from the bry folder)
mat_re = r"^bryfile_dynamic2700_hycom_(\d{10})\.mat$"
flux_name(cycle) = "roms_dbry_flux_2700m_1104_hycom_$(cycle).nc"
rho0 = 1027.5   # must match dbry_make.jl

const FLUX_VARS = (("west", "up_west", "Fx", "eta_rho"),
                   ("east", "up_east", "Fx", "eta_rho"),
                   ("south", "vp_south", "Fy", "xi_rho"),
                   ("north", "vp_north", "Fy", "xi_rho"))

##
files = sort(filter(f -> occursin(mat_re, f), readdir(mat_path)))
isempty(files) && error("no MAT files in $mat_path")
for f in files
    m = matread(mat_path * f)
    m["rho0"] ≈ rho0 || error("$f: rho0 = $(m["rho0"]) ≠ $rho0")
    t = vec(Float64.(m["time"]))
    out = flux_out_path * flux_name(match(mat_re, f)[1])
    NCDataset(out, "c"; format = :netcdf4_classic) do ds
        defDim(ds, "bry_time", length(t))
        tv = defVar(ds, "bry_time", Float64, ("bry_time",); attrib = [
            "units" => "days since 1900-12-31 00:00:00",
            "long_name" => "time for dynamic boundary flux"])
        tv.var[:] = t
        for (d, vname, src, dim) in FLUX_VARS
            data = Float64.(m["$(d)_$(src)"]) ./ rho0
            size(data, 2) == length(t) || error("$f: $(d)_$(src) has $(size(data, 2)) records, time has $(length(t))")
            haskey(ds.dim, dim) || defDim(ds, dim, size(data, 1))
            v = defVar(ds, vname, Float64, (dim, "bry_time"); attrib = [
                "long_name" => "HF baroclinic energy flux, $d boundary, rho0-normalized",
                "units" => "m4 s-3",
                "rho0" => rho0,
                "note" => "raw flux (W/m) divided by rho0 (kg/m3)"])
            v.var[:, :] = data
        end
    end
end
@printf("wrote %d flux files to %s\n", length(files), flux_out_path)

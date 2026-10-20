# ncom2nc_make.jl
#
# Converts the raw NCOM flat files (/home/mbui/ModelOutput/NCOM/data/<yyyymmddHH>/)
# of one nest into the daily parent NetCDF files the h2r routines read:
#     <par_name>_lthick.nc, _ssh.nc, _ts.nc, _uv.nc   (make_bry_need_nc)
#     <par_name>_flx.nc, _wnd.nc, _pres.nc            (make_frc_need_nc)
# Set up for nest 1 (the "1o" files, 2700 m, grid ohgrd_1.nc); nest 2 is
# the 900 m "2o" data already in NCOM_DATA_NC. Files that already exist
# are left alone unless remake = true.
#
# Run from the Documents/Julia project, e.g.
#     julia --project=/home/hsinyi/Documents/Julia input_file_make/ncom2nc_make.jl

using Dates

include("/home/hsinyi/Documents/Julia/function/input_nc/load_input_nc.jl")
using .InputNC

##
nest = 1                                                # 1 -> 2700 m ("1o"), 2 -> 900 m ("2o")
out_path = "/home/hsinyi/roms_data/NCOM_DATA_NC_2700/"  # NCOM_DATA_NC/ for nest 2
mkpath(out_path)

parent_data_path = "/home/mbui/ModelOutput/NCOM/data/"
grid_process_path = "/home/mbui/ModelOutput/NCOM/grid/"   # ohgrd_1.nc (no ohgrd_1.B/.A or ovgrd_1.*)

dating = Date(2022, 8, 22):Day(1):Date(2022, 11, 30)
remake = false

##
fod = Dates.format.(DateTime.(dating), "yyyymmddHH")
for par_name in fod
    if !isdir(joinpath(parent_data_path, par_name))
        @warn "No NCOM data folder for $par_name, skipped."
        continue
    end
    println("==== ", par_name, " ====")
    t0 = time()
    make_bry_need_nc(par_name, out_path, remake; parent_data_path, grid_process_path, nest)
    make_frc_need_nc(par_name, out_path, remake; parent_data_path, grid_process_path, nest)
    println("done ", par_name, " in ", round(time() - t0; digits = 1), " s")
    flush(stdout)   # stdout is block-buffered when redirected to a log file
end

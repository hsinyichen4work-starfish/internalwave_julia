# make_run_folder.jl
#
# Ported from make_run_folder.m — creates a new ROMS run folder under
# `myrun` by copying the template files from `example_folder` and filling in
# the run settings below.
#
#     julia make_run_folder/make_run_folder.jl

include(joinpath(@__DIR__, "folder_make.jl"))

##
title = "Amazon shelf internal wave simulation 1 month - 900m bry fix "
fold_name = "amazon_900m_3mon"
TAG_USE = "900m_3mon"

NP_XI = 8; NP_ETA = 8; node = 1; cpn = 64; do_dia = true
NP_XI * NP_ETA == node * cpn || error("tiled and node mismatch!!!")

time_stepping = (NTIMES = 94080, dt = 90, NDTFAST = 45,
                 rst = 86400, his = 3600, avg = 3600, dia = 3600)   # dia = his
Scoord = (THETA_S = 6, THETA_B = 3, hc = 250)
grid = (LLm = 684, MMm = 854, N = 128)   # 900m grid

walltime = "48:00:00"
input_filenames = (grd = "roms_grd_900m",
                   ini = "roms_ini_900m2022082400",
                   bry = "roms_bry_900m",
                   dbry = "roms_dbry_flux_900m",
                   frc = "roms_frc_900m")
input_folder = (grd = "grid", ini = "ini", bry = "bry", frc = "frc")
filename = "amazon_1mon_dbry.in"

projectpath = "/expanse/lustre/projects/uso101/hchen54/"
output_fold = projectpath * fold_name

##
example_folder = "/home/hchen54/myrun/example/"
myrun = "/home/hchen54/myrun/"
new_folder = myrun * fold_name * "/"

c = (; title, fold_name, TAG_USE, NP_XI, NP_ETA, node, cpn, do_dia,
       time_stepping, Scoord, grid, walltime, input_filenames, input_folder,
       filename, projectpath, output_fold, example_folder, new_folder)

mkpath(new_folder)
copy_example(f) = cp(example_folder * f, new_folder * f; force = true)

copy_example("amazon_3day.in")
infile_make(c)

copy_example("cppdefs.opt")
cppdef_make(c)

foreach(copy_example, ("submit_joint_mpi.sh", "joint_output_record_mpi.sh", "readme_joint"))
do_joint_make(c)

foreach(copy_example, ("do_partition.sh", "partition_input"))
do_partition_make(c)

copy_example("do_roms_expanse.sh")
do_roms_make(c)

foreach(copy_example, ("flux_frc.opt", "Makefile", "ocean_vars.opt"))
oceanvar_make(c)

copy_example("param.opt")
param_make(c)

foreach(copy_example, ("obc_tune.opt", "extract_data.opt"))

println("new folder make")

# make_run_folder_cmems.jl
#
# make_run_folder.jl for the 2700 m CMEMS test run: same template
# (/home/hchen54/myrun/example/), inputs from input_cmems/ instead of input/:
#   grid  roms_cmems_grd_2700m.nc            ini  roms_ini_2700m2022082400.nc (CMEMS)
#   bry   roms_bry_2700m_<date>.nc (CMEMS)
#   frc   frc_era5/roms_era5_frc_2700m_<date>.nc (ERA5)  or  frc/roms_cmems_frc_2700m_<date>.nc (NCOM/NAVGEM),
#         chosen with `forcing` below; both are partitioned into frc_<TAG_USE>/
#   tide  roms_tide_2700m_TPXO10v2.nc (TPXO10 atlas v2, 10 constituents)
# On top of the template it
#   - turns the tides on: tides.opt (ntides = 10, bry_tides, pot_tides) in the
#     run folder, the tide file first in the forcing list, and a tide group in
#     partition_input (CMEMS boundaries carry no tides, unlike NCOM's)
#   - points extract_data.opt at the 2700 m mooring file (copied into the run
#     folder, where ROMS reads it by bare name); extraction is switched by
#     do_extract in extract_data.opt, not by the EXTRACT_DATA cpp flag
#
#     julia make_run_folder/make_run_folder_cmems.jl

include("/home/hchen54/internalwave_julia/make_run_folder/folder_make.jl")

##
forcing = "era5"   # "era5": frc_era5/roms_era5_frc_2700m_* (era5_frc_make.py); "ncom": frc/roms_cmems_frc_2700m_* (NCOM/NAVGEM)
frc_src = Dict("era5" => (folder = "frc_era5", prefix = "roms_era5_frc_2700m", label = "ERA5"),
               "ncom" => (folder = "frc", prefix = "roms_cmems_frc_2700m", label = "NCOM"))[forcing]
title = "Amazon shelf internal wave simulation - 2700m CMEMS test, $(frc_src.label) forcing, TPXO10 tides, pure orlanski"
fold_name = "amazon_2700m_cmems_test"
TAG_USE = "2700m_cmems"
dyn_bry = false   # no dynamic boundary flux files for CMEMS -> pure Orlanski

NP_XI = 8; NP_ETA = 8; node = 1; cpn = 64; do_dia = true
NP_XI * NP_ETA == node * cpn || error("tiled and node mismatch!!!")

# same barotropic/baroclinic CFL as the 900 m runs (dt 90, NDTFAST 45) at 3x dx;
# 2022-08-24 00:00 -> 2022-09-30 00:00 = 37 days
time_stepping = (NTIMES = 37 * 86400 ÷ 180, dt = 180, NDTFAST = 30,
                 rst = 86400, his = 3600, avg = 3600, dia = 3600)   # dia = his
Scoord = (THETA_S = 6, THETA_B = 3, hc = 250)
grid = (LLm = 376, MMm = 507, N = 128)   # 2700m grid (378 x 509 rho points)

walltime = "24:00:00"
input_dir = "input_cmems"
input_filenames = (grd = "roms_cmems_grd_2700m",
                   ini = "roms_ini_2700m2022082400",
                   bry = "roms_bry_2700m",
                   dbry = "roms_dbry_flux_2700m",   # unused (dyn_bry = false)
                   frc = frc_src.prefix)
input_folder = (grd = "grid", ini = "ini", bry = "bry", frc = frc_src.folder)
tide_folder = "tide"; tide_file = "roms_tide_2700m_TPXO10v2"
ntides = 10
filename = "amazon_2700m_cmems.in"

projectpath = "/expanse/lustre/projects/uso101/hchen54/"
output_fold = projectpath * fold_name

##
example_folder = "/home/hchen54/myrun/example/"
myrun = "/home/hchen54/myrun/"
new_folder = myrun * fold_name * "/"

c = (; title, fold_name, TAG_USE, dyn_bry, NP_XI, NP_ETA, node, cpn, do_dia,
       time_stepping, Scoord, grid, walltime, input_dir, input_filenames, input_folder,
       filename, projectpath, output_fold, example_folder, new_folder)

isfile(projectpath * input_dir * "/" * tide_folder * "/" * tide_file * ".nc") || error("no tide file")

mkpath(new_folder)
copy_example(f) = cp(example_folder * f, new_folder * f; force = true)

copy_example("amazon_3day.in")
infile_make(c)
# tide file first in the forcing list (tides.F looks for the file with "omega")
tide_path = input_base(c) * tide_folder * "_" * TAG_USE * "/" * tide_file * ".nc"
sub_file!(new_folder * filename, "forcing: filename\n" => "forcing: filename\n     " * tide_path * "\n")

copy_example("cppdefs.opt")
cppdef_make(c)

foreach(copy_example, ("submit_joint_mpi.sh", "joint_output_record_mpi.sh", "readme_joint"))
do_joint_make(c)
# serial join fallback (bash submit_joint_serial.sh on the login node)
foreach(copy_example, ("submit_joint_serial.sh", "joint_output_record_serial.sh"))
sub_file!(new_folder * "submit_joint_serial.sh", "BASE=\$EXAMPLE_OUTPUT" => "BASE=" * c.output_fold)
chmod(new_folder * "joint_output_record_serial.sh", 0o755)

foreach(copy_example, ("do_partition.sh", "partition_input"))
do_partition_make(c)
# partition the tide file too, into tide_<TAG>/
sub_file!(new_folder * "partition_input",
    "\necho \"\"\necho \"Partitioning complete" =>
    """
    \n# ---- Tides (TPXO) ----
    : "\${DO_TIDE:=1}"
    if [ "\$DO_TIDE" = "1" ]; then
        tide_files=( "\$INPUT_BASE/$tide_folder"/$tide_file.nc )
        partition_group "tide" "\$INPUT_BASE/$tide_folder" "\$INPUT_BASE/$(tide_folder)_\${TAG}" "\${tide_files[@]}"
    else
        echo "Skipping tide (DO_TIDE=0)."
    fi

    echo ""
    echo "Partitioning complete""")
sub_file!(new_folder * "partition_input",
    "\"\$INPUT_BASE/dynbry_\${TAG}\"; do" => "\"\$INPUT_BASE/dynbry_\${TAG}\" \"\$INPUT_BASE/$(tide_folder)_\${TAG}\"; do")

copy_example("do_roms_expanse.sh")
do_roms_make(c)

foreach(copy_example, ("flux_frc.opt", "Makefile", "ocean_vars.opt"))
oceanvar_make(c)

copy_example("param.opt")
param_make(c)

foreach(copy_example, ("obc_tune.opt", "extract_data.opt"))
obctune_make(c)

# Extraction-point file for the 2700 m grid, read by bare name next to ./roms
edata_file = projectpath * "input/grid/roms_cmems_grd_2700m_mor_edata.nc"
cp(edata_file, new_folder * basename(edata_file); force = true)
sub_file!(new_folder * "extract_data.opt", "roms_grd_900m_mor_edata.nc" => basename(edata_file))

# tides.opt: the run folder copy overrides src/tides.opt at compile time
cp("/home/hchen54/ucla-roms/src/tides.opt", new_folder * "tides.opt"; force = true)
sub_file!(new_folder * "tides.opt",
    r"ntides *= *\d+" => "ntides=$ntides",
    r"bry_tides *= *\.false\." => "bry_tides=.true.",
    r"pot_tides *= *\.false\." => "pot_tides=.true.")

println("new folder made: ", new_folder)

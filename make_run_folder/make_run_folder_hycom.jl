# make_run_folder_hycom.jl
#
# 2700 m HYCOM run on the new design grid (1104), inputs in
# /expanse/lustre/projects/uso102/hchen54/new_design_fix/ (grid with hmin = 10 m + smoothing, inputs remade on it):
#   grid  roms_grd_2700m_1104.nc (sits in new_design_fix/ itself, not in a grid/ subfolder)
#   ini   ini/roms_ini_2700m_1104_hycom_2022082200.nc
#   bry   bry/roms_bry_2700m_1104_hycom_<yyyymmdd>12.nc  (3-hourly, tides already in the bry)
#   frc   frc/roms_frc_2700m_1104_ncom_<yyyymmdd>00.nc   (hourly)
#   edata roms_grd_2700m_1104_edata.nc (moorings + 900m child-grid bry, hourly)
#   dbry  bry/roms_dbry_flux_2700m_1104_hycom_<yyyymmdd>12.nc (HF baroclinic flux, dbry_make.jl)
#   river riverfile/roms_riv_2700m_1104.nc (river_input.jl: 3 rivers, monthly climatology,
#         cycle_length 365.25); river mouths are `river_flux` in the grid file
# Compared with make_run_folder_cmems.jl: no tide file / tides.opt (bry already
# carries the tides); dynamic boundary on (obc_tuning + dbry flux files), as in the 900 m runs.
#
#     julia make_run_folder/make_run_folder_hycom.jl

include("/home/hchen54/internalwave_julia/make_run_folder/folder_make.jl")

##
fold_name = "design_1113_hycom_2700"
# ini is 2022-08-22 00:00 (first available) -> 2022-11-30 00:00 = 100 days
run_days_n = 100
sponge_size = 37   # grid points, ~100 km at 2700 m; V_SPONGE stays as in the .in file
title = "Amazon shelf internal wave simulation - 2700m new design 1104, HYCOM bry (with tides), NCOM forcing, dynamic bry"
TAG_USE = "2700m_hycom1113"
dyn_bry = true   # true: dynamic boundary (obc_tuning + dbry files); false: pure Orlanski

NP_XI = 8; NP_ETA = 8; node = 1; cpn = 64; do_dia = true
NP_XI * NP_ETA == node * cpn || error("tiled and node mismatch!!!")

# same dt/NDTFAST as the 2700m CMEMS run; avg + dia daily averages, rst weekly
time_stepping = (NTIMES = run_days_n * 86400 ÷ 180, dt = 180, NDTFAST = 30,
                 rst = 7 * 86400, his = 3600, avg = 86400, dia = 86400)   # dia: diag_avg = .true. -> daily mean
Scoord = (THETA_S = 6, THETA_B = 3, hc = 250)   # matches ini/bry
grid = (LLm = 524, MMm = 655, N = 128)   # 526 x 657 rho points

# CMEMS 2700m (378x509, 121 d) took 9.1 h; this grid is ~1.8x larger -> ~14 h
walltime = "24:00:00"
input_dir = "new_design_fix"
input_filenames = (grd = "roms_grd_2700m_1104",
                   ini = "roms_ini_2700m_1104_hycom_2022082200",
                   bry = "roms_bry_2700m_1104_hycom",
                   dbry = "roms_dbry_flux_2700m_1104_hycom",   # in bry/
                   frc = "roms_frc_2700m_1104_ncom")
input_folder = (grd = ".", ini = "ini", bry = "bry", frc = "frc")
filename = "design_1113_hycom_2700.in"
riv_folder = "riverfile"; riv_file = "roms_riv_2700m_1104"
nriv = 3   # Amazon_N, Amazon_S, Tocantins (must match nriver in riv_file and river_flux in the grid)

projectpath = "/expanse/lustre/projects/uso102/hchen54/"
output_fold = projectpath * fold_name

##
example_folder = "/home/hchen54/myrun/example/"
myrun = "/home/hchen54/myrun/"
new_folder = myrun * fold_name * "/"

c = (; title, fold_name, TAG_USE, dyn_bry, NP_XI, NP_ETA, node, cpn, do_dia,
       time_stepping, Scoord, grid, walltime, input_dir, input_filenames, input_folder,
       filename, projectpath, output_fold, example_folder, new_folder)

mkpath(new_folder)
copy_example(f) = cp(example_folder * f, new_folder * f; force = true)

copy_example("amazon_3day.in")
infile_make(c)

# bry/dbry files are stamped at 12:00 (each covers day-12h .. day+12h), so the
# 08-21 file is needed to cover the 08-22 00:00 start; run_days() drops it.
bry_dir = input_base(c) * input_folder.bry * "/"
for (prefix, sub) in ((input_filenames.bry, "bry_"), (input_filenames.dbry, "dynbry_"))
    path = input_base(c) * sub * TAG_USE * "/"
    f0 = prefix * "_2022082112.nc"
    f1 = prefix * "_2022082212.nc"
    isfile(bry_dir * f0) || error("missing $f0")
    sub_file!(new_folder * filename, path * f1 => path * f0 * "\n     " * path * f1)
end

# river forcing file first in the forcing list
isfile(input_base(c) * riv_folder * "/" * riv_file * ".nc") || error("no river file")
riv_path = input_base(c) * "riv_" * TAG_USE * "/" * riv_file * ".nc"
sub_file!(new_folder * filename, "forcing: filename\n" => "forcing: filename\n     " * riv_path * "\n")

copy_example("cppdefs.opt")
cppdef_make(c)
# south edge of this grid is all land -> wall, not an open boundary
sub_file!(new_folder * "cppdefs.opt", r"#define OBC_SOUTH\b" => "#undef OBC_SOUTH")

foreach(copy_example, ("submit_joint_mpi.sh", "joint_output_record_mpi.sh", "readme_joint"))
do_joint_make(c)
# serial join fallback (bash submit_joint_serial.sh on the login node)
foreach(copy_example, ("submit_joint_serial.sh", "joint_output_record_serial.sh"))
sub_file!(new_folder * "submit_joint_serial.sh", "BASE=\$EXAMPLE_OUTPUT" => "BASE=" * c.output_fold)
chmod(new_folder * "joint_output_record_serial.sh", 0o755)

foreach(copy_example, ("do_partition.sh", "partition_input"))
do_partition_make(c)
# grid lives in new_design_fix/ next to *_before_riverfix.nc and *_edata.nc, which the
# GRDEX*.nc glob would also pick up -> match the grid file exactly
sub_file!(new_folder * "partition_input",
    "\"\$INPUT_GRID\"/$(input_filenames.grd)*.nc" => "\"\$INPUT_GRID\"/$(input_filenames.grd).nc")
# river file: no xi/eta dims, so partit writes nothing for it; ROMS (PARALLEL_FILES)
# still opens <name>.<rank>.nc on every rank -> one copy per rank, named like partit's
nranks = NP_XI * NP_ETA
ndig = nranks < 10 ? 1 : nranks < 100 ? 2 : nranks < 1000 ? 3 : 4
sub_file!(new_folder * "partition_input",
    "\necho \"\"\necho \"Partitioning complete" =>
    """
    \n# ---- River forcing (copied, one file per rank) ----
    : "\${DO_RIVER:=1}"
    if [ "\$DO_RIVER" = "1" ]; then
        riv_dir="\$INPUT_BASE/riv_\${TAG}"
        mkdir -p "\$riv_dir"; rm -f "\$riv_dir"/*.nc
        for ((r = 0; r < NP_XI * NP_ETA; r++)); do
            cp "\$INPUT_BASE/$riv_folder/$riv_file.nc" "\$riv_dir/$riv_file.\$(printf %0$(ndig)d \$r).nc"
        done
        echo "River file copied into \$riv_dir"
    else
        echo "Skipping river (DO_RIVER=0)."
    fi

    echo ""
    echo "Partitioning complete""")
sub_file!(new_folder * "partition_input",
    "\"\$INPUT_BASE/dynbry_\${TAG}\"; do" => "\"\$INPUT_BASE/dynbry_\${TAG}\" \"\$INPUT_BASE/riv_\${TAG}\"; do")

copy_example("do_roms_expanse.sh")
do_roms_make(c)

foreach(copy_example, ("flux_frc.opt", "Makefile", "ocean_vars.opt"))
oceanvar_make(c)

copy_example("param.opt")
param_make(c)

foreach(copy_example, ("obc_tune.opt", "extract_data.opt"))
obctune_make(c)

# Extraction file for this grid (moorings + 900m child-grid boundaries), read by bare name next to ./roms
edata_file = input_base(c) * "roms_grd_2700m_1104_edata.nc"
cp(edata_file, new_folder * basename(edata_file); force = true)
sub_file!(new_folder * "extract_data.opt", "roms_grd_900m_mor_edata.nc" => basename(edata_file))

# sponge width: local copy, the run folder's .F overrides src/ at compile time
cp("/home/hchen54/ucla-roms/src/set_nudgcof.F", new_folder * "set_nudgcof.F"; force = true)
sub_file!(new_folder * "set_nudgcof.F", r"sponge_size = \d+" => "sponge_size = $sponge_size")

# rivers on: local river_frc.opt overrides src/ at compile time
cp("/home/hchen54/ucla-roms/src/river_frc.opt", new_folder * "river_frc.opt"; force = true)
sub_file!(new_folder * "river_frc.opt",
    r"river_source *= *\.false\." => "river_source = .true.",
    r"nriv *= *\d+" => "nriv = $nriv")

println("new folder made: ", new_folder)

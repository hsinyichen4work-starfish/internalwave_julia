# make_run_folder_cmems_1104.jl
#
# 2700 m CMEMS run on the new design grid (1104, hmin 10 m), inputs in
# /expanse/lustre/projects/uso102/hchen54/new_design_cmems/ (made by the *_1104 scripts):
#   grid  grid/roms_grd_2700m_1104.nc (copy of new_design_fix/)
#   ini   ini/roms_ini_2700m_1104_cmems_2022080100.nc            (cmems_input_make_1104.jl)
#   bry   bry/roms_bry_2700m_1104_cmems_<yyyymmdd>00.nc  (6-hourly 00..24 h, E/N/W only, no tides)
#   frc   frc_era5/roms_era5_frc_2700m_1104_<yyyymmdd>00.nc (hourly, era5_frc_make_1104.py)
#   tide  tide/roms_tide_2700m_1104_TPXO10v2_0801_1130.nc (10 constituents, tpxo_tide_make_1104.py)
#   river riverfile/roms_riv_2700m_1104.nc (copied from new_design_fix/; climatology, 3 rivers)
#   edata new_design_fix/roms_grd_2700m_1104_edata.nc (moorings + 900 m child-grid bry)
# Combines make_run_folder_cmems.jl (tides on, pure Orlanski) with the grid-specific
# parts of make_run_folder_hycom.jl (closed south boundary, rivers, edata, sponge 37).
#
#     julia make_run_folder/make_run_folder_cmems_1104.jl

include("/home/hchen54/internalwave_julia/make_run_folder/folder_make.jl")

##
fold_name = "design_1116_cmems_2700"
# ini 2022-08-01 00:00 (tide spin-up, as the other CMEMS runs) -> 2022-11-30 00:00 = 121 days
run_days_n = 121
sponge_size = 37   # grid points, ~100 km at 2700 m, same as design_1113_hycom_2700
title = "Amazon shelf internal wave simulation - 2700m new design 1104, CMEMS ini/bry, ERA5 forcing, TPXO10 tides, rivers, pure orlanski"
TAG_USE = "2700m_cmems1104"
dyn_bry = false   # no dynamic boundary flux for CMEMS -> pure Orlanski

NP_XI = 8; NP_ETA = 8; node = 1; cpn = 64; do_dia = true
NP_XI * NP_ETA == node * cpn || error("tiled and node mismatch!!!")

# same as design_1113_hycom_2700: his hourly, avg + dia daily means, rst weekly
time_stepping = (NTIMES = run_days_n * 86400 ÷ 180, dt = 180, NDTFAST = 30,
                 rst = 7 * 86400, his = 3600, avg = 86400, dia = 86400)
Scoord = (THETA_S = 6, THETA_B = 3, hc = 250)   # matches ini/bry
grid = (LLm = 524, MMm = 655, N = 128)   # 526 x 657 rho points

# design_1113_hycom_2700 took 21.7 h for 100 days -> ~26 h for 121 days
walltime = "32:00:00"
input_dir = "new_design_cmems"
input_filenames = (grd = "roms_grd_2700m_1104",
                   ini = "roms_ini_2700m_1104_cmems_2022080100",
                   bry = "roms_bry_2700m_1104_cmems",
                   dbry = "roms_dbry_flux_2700m_1104",   # unused (dyn_bry = false)
                   frc = "roms_era5_frc_2700m_1104")
input_folder = (grd = "grid", ini = "ini", bry = "bry", frc = "frc_era5")
tide_folder = "tide"; tide_file = "roms_tide_2700m_1104_TPXO10v2_0801_1130"
ntides = 10
riv_folder = "riverfile"; riv_file = "roms_riv_2700m_1104"
nriv = 3   # Amazon_N, Amazon_S, Tocantins (must match nriver in riv_file and river_flux in the grid)
filename = "design_1116_cmems_2700.in"

projectpath = "/expanse/lustre/projects/uso102/hchen54/"
output_fold = projectpath * fold_name
design_fix = projectpath * "new_design_fix/"   # river + edata source

##
example_folder = "/home/hchen54/myrun/example/"
myrun = "/home/hchen54/myrun/"
new_folder = myrun * fold_name * "/"

c = (; title, fold_name, TAG_USE, dyn_bry, NP_XI, NP_ETA, node, cpn, do_dia,
       time_stepping, Scoord, grid, walltime, input_dir, input_filenames, input_folder,
       filename, projectpath, output_fold, example_folder, new_folder)

isfile(input_base(c) * tide_folder * "/" * tide_file * ".nc") || error("no tide file")
# river file next to the other inputs, so partition_input finds it under $INPUT_BASE
mkpath(input_base(c) * riv_folder)
cp(design_fix * riv_folder * "/" * riv_file * ".nc", input_base(c) * riv_folder * "/" * riv_file * ".nc"; force = true)

mkpath(new_folder)
copy_example(f) = cp(example_folder * f, new_folder * f; force = true)

copy_example("amazon_3day.in")
infile_make(c)
# forcing list: tide file first (tides.F looks for the file with "omega"), then the river file
tide_path = input_base(c) * tide_folder * "_" * TAG_USE * "/" * tide_file * ".nc"
riv_path = input_base(c) * "riv_" * TAG_USE * "/" * riv_file * ".nc"
sub_file!(new_folder * filename, "forcing: filename\n" => "forcing: filename\n     " * tide_path * "\n     " * riv_path * "\n")

copy_example("cppdefs.opt")
cppdef_make(c)
# south edge of this grid is all land -> wall, not an open boundary (bry files have no south)
sub_file!(new_folder * "cppdefs.opt", r"#define OBC_SOUTH\b" => "#undef OBC_SOUTH")

foreach(copy_example, ("submit_joint_mpi.sh", "joint_output_record_mpi.sh", "readme_joint"))
do_joint_make(c)
# serial join fallback (bash submit_joint_serial.sh on the login node)
foreach(copy_example, ("submit_joint_serial.sh", "joint_output_record_serial.sh"))
sub_file!(new_folder * "submit_joint_serial.sh", "BASE=\$EXAMPLE_OUTPUT" => "BASE=" * c.output_fold)
chmod(new_folder * "joint_output_record_serial.sh", 0o755)

foreach(copy_example, ("do_partition.sh", "partition_input"))
do_partition_make(c)
# match the grid file exactly (no stray *_edata / *_before_riverfix copies picked up)
sub_file!(new_folder * "partition_input",
    "\"\$INPUT_GRID\"/$(input_filenames.grd)*.nc" => "\"\$INPUT_GRID\"/$(input_filenames.grd).nc")
# tide file: partitioned like the others, into tide_<TAG>/
# river file: no xi/eta dims, so partit writes nothing for it; ROMS (PARALLEL_FILES)
# still opens <name>.<rank>.nc on every rank -> one copy per rank, named like partit's
nranks = NP_XI * NP_ETA
ndig = nranks < 10 ? 1 : nranks < 100 ? 2 : nranks < 1000 ? 3 : 4
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

    # ---- River forcing (copied, one file per rank) ----
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
    "\"\$INPUT_BASE/dynbry_\${TAG}\"; do" =>
    "\"\$INPUT_BASE/dynbry_\${TAG}\" \"\$INPUT_BASE/$(tide_folder)_\${TAG}\" \"\$INPUT_BASE/riv_\${TAG}\"; do")

copy_example("do_roms_expanse.sh")
do_roms_make(c)

foreach(copy_example, ("flux_frc.opt", "Makefile", "ocean_vars.opt"))
oceanvar_make(c)

copy_example("param.opt")
param_make(c)

foreach(copy_example, ("obc_tune.opt", "extract_data.opt"))
obctune_make(c)

# Extraction file for this grid (moorings + 900m child-grid boundaries), read by bare name next to ./roms
edata_file = design_fix * "roms_grd_2700m_1104_edata.nc"
cp(edata_file, new_folder * basename(edata_file); force = true)
sub_file!(new_folder * "extract_data.opt", "roms_grd_900m_mor_edata.nc" => basename(edata_file))

# tides.opt: the run folder copy overrides src/tides.opt at compile time
cp("/home/hchen54/ucla-roms/src/tides.opt", new_folder * "tides.opt"; force = true)
sub_file!(new_folder * "tides.opt",
    r"ntides *= *\d+" => "ntides=$ntides",
    r"bry_tides *= *\.false\." => "bry_tides=.true.",
    r"pot_tides *= *\.false\." => "pot_tides=.true.")

# sponge width: local copy, the run folder's .F overrides src/ at compile time
cp("/home/hchen54/ucla-roms/src/set_nudgcof.F", new_folder * "set_nudgcof.F"; force = true)
sub_file!(new_folder * "set_nudgcof.F", r"sponge_size = \d+" => "sponge_size = $sponge_size")

# rivers on: local river_frc.opt overrides src/ at compile time
cp("/home/hchen54/ucla-roms/src/river_frc.opt", new_folder * "river_frc.opt"; force = true)
sub_file!(new_folder * "river_frc.opt",
    r"river_source *= *\.false\." => "river_source = .true.",
    r"nriv *= *\d+" => "nriv = $nriv")

println("new folder made: ", new_folder)

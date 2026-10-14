# folder_make.jl
#
# Ported from matlab_funtion/folder_make/*.m — fill in the template files
# copied from the example run folder. Each function corresponds to one of
# the MATLAB scripts (infile_make, cppdef_make, ...), but takes the run
# configuration `c` (a NamedTuple built in make_run_folder.jl) explicitly.

using Printf
using Dates

# Read `file`, apply the replacements in order (like successive strrep calls),
# and write it back.
function sub_file!(file, pairs::Pair...)
    s = read(file, String)
    for p in pairs
        s = replace(s, p)
    end
    write(file, s)
    return nothing
end

# Fortran double-precision literal, e.g. 6 -> "6.0D+00"
fortran_d(x) = replace(uppercase(@sprintf("%.1e", x)), "E" => "D")

# Date stamp (YYYYMMDDHH) at the end of an input file name, e.g.
# roms_bry_900m_2022082400.nc or roms_ini_900m2022082400 -> 2022-08-24T00:00
file_date(f) = DateTime(match(r"(\d{10})(\.nc)?$", f)[1], dateformat"yyyymmddHH")

# Days of daily frc/bry files the run needs: the day of the initial condition
# through the day the run ends, plus one extra day of margin.
function run_days(c)
    t0 = file_date(c.input_filenames.ini)
    t1 = t0 + Second(c.time_stepping.NTIMES * c.time_stepping.dt)
    return Date(t0), Date(t1) + Day(1)
end

# Files in `dir` whose names start with `prefix` (MATLAB dir([dir, prefix, '*'])),
# keeping only those dated within `days`, joined as continuation lines of the
# .in file with paths under `path`.
function file_list(dir, prefix, path, days)
    d0, d1 = days
    fil = sort(filter(f -> startswith(f, prefix) && d0 <= Date(file_date(f)) <= d1,
                      readdir(dir)))
    isempty(fil) && error("no files matching $(joinpath(dir, prefix))* between $d0 and $d1 found")
    return join((path * f for f in fil), "\n     ")
end

function infile_make(c)
    if c.filename != "amazon_3day.in"
        mv(c.new_folder * "amazon_3day.in", c.new_folder * c.filename; force = true)
    end

    grid_path = c.projectpath * "input/grid_" * c.TAG_USE * "/"
    bry_path = c.projectpath * "input/bry_" * c.TAG_USE * "/"
    dbry_path = c.projectpath * "input/dynbry_" * c.TAG_USE * "/"
    frc_path = c.projectpath * "input/frc_" * c.TAG_USE * "/"
    ini_path = c.projectpath * "input/ini_" * c.TAG_USE * "/"
    output_path = c.projectpath * c.fold_name * "/roms"
    mkpath(c.projectpath * c.fold_name)

    input_dir = c.projectpath * "input/"
    days = run_days(c)
    frc_str = file_list(input_dir * c.input_folder.frc, c.input_filenames.frc, frc_path, days)
    bry_str = file_list(input_dir * c.input_folder.bry, c.input_filenames.bry, bry_path, days)

    # Pure Orlanski runs don't read the dynamic boundary flux files
    if c.dyn_bry
        dbry_str = file_list(input_dir * c.input_folder.bry, c.input_filenames.dbry, dbry_path, days)
    else
        sub_file!(c.new_folder * c.filename, r"\n *EXAMPLE_dbry" => "")
        dbry_str = ""
    end

    ts = c.time_stepping
    sub_file!(c.new_folder * c.filename,
        "EXAMPLE TITLE" => c.title,
        "nt_ex" => rpad(string(ts.NTIMES), 5),
        " dt_ex" => rpad(string(ts.dt), 6),
        "ndt_ex" => rpad(string(ts.NDTFAST), 6),
        "the_s_ex" => rpad(fortran_d(c.Scoord.THETA_S), 8),
        "the_b_ex" => rpad(fortran_d(c.Scoord.THETA_B), 8),
        "hc_ex" => rpad(fortran_d(c.Scoord.hc), 5),
        "EXAMPLE_grid" => grid_path * c.input_filenames.grd * ".nc",
        "EXAMPLE_ini" => ini_path * c.input_filenames.ini * ".nc",
        "EXAMPLE_frc" => frc_str,
        "EXAMPLE_bry" => bry_str,
        "EXAMPLE_dbry" => dbry_str,
        "EXAMPLE_output" => output_path)
end

function cppdef_make(c)
    sub_file!(c.new_folder * "cppdefs.opt",
        c.do_dia ? ("#undef DIAGNOSTICS" => "#define DIAGNOSTICS") :
                   ("#define DIAGNOSTICS" => "#undef DIAGNOSTICS"))

    if c.do_dia
        cp(c.example_folder * "diagnostics.opt", c.new_folder * "diagnostics.opt"; force = true)
        sub_file!(c.new_folder * "diagnostics.opt", "DIASTEP" => string(c.time_stepping.dia))
    end
end

function do_joint_make(c)
    # Parallel join (ncjoin_mpi), see readme_joint. Only the submitter needs the
    # output path; joint_output_record_mpi.sh gets its folders as arguments.
    sub_file!(c.new_folder * "submit_joint_mpi.sh",
        "BASE=\$EXAMPLE_OUTPUT" => "BASE=" * c.output_fold)

    # Make the export line in the readme copy-pasteable for this run
    sub_file!(c.new_folder * "readme_joint",
        "/expanse/lustre/projects/.../<run folder>   (absolute path)" => c.output_fold)
end

function do_partition_make(c)
    sub_file!(c.new_folder * "do_partition.sh",
        "EXAMPLE_INPUT" => c.projectpath * "input/",
        "npxi_ex" => string(c.NP_XI),
        "npeta_ex" => string(c.NP_ETA),
        "tag_ex" => c.TAG_USE,
        "DO_DYN_BRY=1" => "DO_DYN_BRY=" * (c.dyn_bry ? "1" : "0"))

    fo, fn = c.input_folder, c.input_filenames
    sub_file!(c.new_folder * "partition_input",
        "FGRDEX" => fo.grd,
        "FINIEX" => fo.ini,
        "FBRYEX" => fo.bry,
        "FFRCEX" => fo.frc,
        "FDBEX" => fo.bry,
        "GRDEX" => fn.grd,
        "INIEX" => fn.ini,
        "BRYEX" => fn.bry,
        "FRCEX" => fn.frc,
        "DBEX" => fn.dbry)
end

function do_roms_make(c)
    sub_file!(c.new_folder * "do_roms_expanse.sh",
        "jobname_ex" => c.fold_name,
        "NODEEX" => string(c.node),
        "CPNEX" => string(c.cpn),
        "NTASKEX" => string(c.node * c.cpn),
        "WALLTIME_EX" => c.walltime,
        "FILE_NAME_EX" => c.filename)
end

function oceanvar_make(c)
    ts = c.time_stepping
    sub_file!(c.new_folder * "ocean_vars.opt",
        "RST_STEP" => string(ts.rst),
        "HIS_STEP" => string(ts.his),
        "AVG_STEP" => string(ts.avg))
end

function param_make(c)
    sub_file!(c.new_folder * "param.opt",
        "llmex" => string(c.grid.LLm),
        "mmmex" => string(c.grid.MMm),
        "nex" => string(c.grid.N),
        "npei_ex" => string(c.NP_XI),
        "npeta_ex" => string(c.NP_ETA))
end

function obctune_make(c)
    c.dyn_bry && return nothing
    sub_file!(c.new_folder * "obc_tune.opt",
        r"obc_tuning *= *\.true\." => "obc_tuning = .false.")
end

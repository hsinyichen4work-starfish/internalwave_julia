# folder_make.jl
#
# Ported from matlab_funtion/folder_make/*.m — fill in the template files
# copied from the example run folder. Each function corresponds to one of
# the MATLAB scripts (infile_make, cppdef_make, ...), but takes the run
# configuration `c` (a NamedTuple built in make_run_folder.jl) explicitly.

using Printf

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

# Files in `dir` whose names start with `prefix` (MATLAB dir([dir, prefix, '*'])),
# joined as continuation lines of the .in file with paths under `path`.
function file_list(dir, prefix, path)
    fil = sort(filter(f -> startswith(f, prefix), readdir(dir)))
    isempty(fil) && error("no files matching $(joinpath(dir, prefix))* found")
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
    frc_str = file_list(input_dir * c.input_folder.frc, c.input_filenames.frc, frc_path)
    bry_str = file_list(input_dir * c.input_folder.bry, c.input_filenames.bry, bry_path)
    dbry_str = file_list(input_dir * c.input_folder.bry, c.input_filenames.dbry, dbry_path)

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
    for f in ("joint_multi_job", "joint_output_record.sh")
        sub_file!(c.new_folder * f, "EXAMPLE_OUTPUT" => c.output_fold)
    end
end

function do_partition_make(c)
    sub_file!(c.new_folder * "do_partition.sh",
        "EXAMPLE_INPUT" => c.projectpath * "input/",
        "npxi_ex" => string(c.NP_XI),
        "npeta_ex" => string(c.NP_ETA),
        "tag_ex" => c.TAG_USE)

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

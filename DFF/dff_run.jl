# Which ROMS run the DFF scripts (DDF_on_ssh.jl, DFF_ploting.jl, DFF_transect_analysis.jl) work on, and where its
# input, DFF output and figures live. Set by the environment variable DFF_RUN (the batch chain
# sbatch_files/submit_dff_chain.sh passes it to every job); without it, the 3-month run.
#   run_dir/after_joint   roms_his.*.nc (input)
#   run_dir/ddf_ssh       raw ssh, DFF fields, transects, envelope/phase speed files (output)
#   fig_dir               figures
dff_run = get(ENV, "DFF_RUN", "amazon_900m_3mon_1008_2")
run_dir = "/expanse/lustre/projects/uso101/hchen54/$dff_run"
isdir(run_dir) || error("DFF_RUN = $dff_run: no run directory $run_dir")
ddf_dir = joinpath(run_dir, "ddf_ssh")
# figure folder name per run; the 3-month run keeps its existing folder (named without the _2)
fig_run = Dict("amazon_900m_3mon_1008_2" => "amazon_900m_3mon_1008")
fig_dir = "/expanse/lustre/projects/uso101/hchen54/figure/$(get(fig_run, dff_run, dff_run))/DFF"

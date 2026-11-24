#= tidal_models_setup.jl
Hsin-Yi Chen, USM DMS
Shared settings for the multi-model SSH tide validation against FES2022b:
  ssh_harmonic_models.jl  ->  ssh_tidalcomp_fes_models.jl  ->  tidal_comp_models_plot.jl
Included by all three. Picks the machine (expanse / aurelia) and defines the sources.

Machine:  TC_HOST = expanse | aurelia   (default: expanse if /expanse/lustre exists, else aurelia)
Sources:  TC_SRC  = comma separated names from SOURCES below (default per machine: the data that lives there)
  ROMS_HYCOMparent   ROMS 2700 m new design, HYCOM boundaries (with tides)   design_1113_hycom_2700   expanse
  ROMS_GLORYSparent  ROMS 2700 m new design, CMEMS boundaries + TPXO10 tides  design_1116_cmems_2700   expanse
  HYCOM              HYCOM_DATA_NC_2700m_1104, 3-hourly ssh                                         aurelia
  NCOM2700           NCOM_DATA_NC_2700, hourly ssh                                                   aurelia
GLORYS / CMEMS GLO12 is not in the list: its zos is de-tided, and the ocean_tide of the merged-sl product is
FES2014 (attribute source = "MERCATOR GLO12,FES2014"), i.e. an atlas, not a model tide.

Time window (same for every model, so tc and the nodal corrections are the same):
  TC_T0, TC_T1   default 2022-08-25T00:00 .. 2022-11-30T00:00 (inclusive)
Test runs:  TC_STRIDE = n  every n-th point (fit: native grid; comparison: 2700 m target grid); files get _s<n>
=#

using Dates

const HOST = get(ENV, "TC_HOST", isdir("/expanse/lustre") ? "expanse" : "aurelia")
const P = HOST == "expanse" ? (
        func   = "/home/hchen54/internalwave_julia/function/",
        base   = "/expanse/lustre/projects/uso102/hchen54/",
        pdata  = "/home/hchen54/aurelia/roms_data/",                     # HYCOM / NCOM over sshfs: slow, fit them on aurelia
        fes    = "/expanse/lustre/projects/uso102/hchen54/input_cmems/fes2022/ocean_tide_extrapolated/",
        grid   = "/expanse/lustre/projects/uso102/hchen54/new_design_fix/roms_grd_2700m_1104.nc",
        out    = "/expanse/lustre/projects/uso102/hchen54/figure/tidal_comp_models/",
        fitdirs = ["/expanse/lustre/projects/uso102/hchen54/figure/tidal_comp_models/",
                   "/home/hchen54/aurelia/roms_data/tidal_comp_models/"],    # where fit / comparison files are looked for
        py     = "/home/hchen54/.venvs/utide/bin/python",                # python with pyTMD
        default = "ROMS_HYCOMparent,ROMS_GLORYSparent") :
    HOST == "aurelia" ? (
        func   = "/home/hsinyi/Documents/Julia/function/",
        base   = "/home/hsinyi/roms_data/",
        pdata  = "/home/hsinyi/roms_data/",
        fes    = "/home/hsinyi/data_notm/fes2022/ocean_tide_extrapolated/",
        grid   = "/home/hsinyi/roms_data/new_design_fix/roms_grd_2700m_1104.nc",
        out    = "/home/hsinyi/roms_data/tidal_comp_models/",
        fitdirs = ["/home/hsinyi/roms_data/tidal_comp_models/"],
        py     = get(ENV, "TC_PY", "/home/hsinyi/.venvs/pytmd/bin/python"),   # needs pyTMD (see the usage notes)
        default = "HYCOM,NCOM2700") :
    error("TC_HOST = $HOST, not expanse / aurelia")

const SOURCES = Dict(
    "ROMS_HYCOMparent"  => (kind = :roms, data = P.base * "design_1113_hycom_2700/after_joint/", grid = P.grid,
                            label = "ROMS 2700 m (HYCOM parent)"),
    "ROMS_GLORYSparent" => (kind = :roms, data = P.base * "design_1116_cmems_2700/after_joint/", grid = P.grid,
                            label = "ROMS 2700 m (GLORYS parent)"),
    "HYCOM"             => (kind = :hycom, data = P.pdata * "HYCOM_DATA_NC_2700m_1104/", prefix = "hycom_", gkind = "grd",
                            label = "HYCOM"),
    "NCOM2700"          => (kind = :ncom, data = P.pdata * "NCOM_DATA_NC_2700/", prefix = "", gkind = "lthick",
                            label = "NCOM"),
)
const SRC_ORDER = ["ROMS_HYCOMparent", "ROMS_GLORYSparent", "HYCOM", "NCOM2700"]   # order in tables / figures

const SRC_RUN = split(get(ENV, "TC_SRC", P.default), ",")
for s in SRC_RUN; haskey(SOURCES, s) || error("TC_SRC: $s not one of $(join(SRC_ORDER, ", "))"); end
const T0     = DateTime(get(ENV, "TC_T0", "2022-08-25T00:00:00"))
const T1     = DateTime(get(ENV, "TC_T1", "2022-11-30T00:00:00"))
const STRIDE = parse(Int, get(ENV, "TC_STRIDE", "1"))
const SFX    = STRIDE > 1 ? "_s$STRIDE" : ""
const T_REF  = DateTime(1994, 1, 1)                     # clock of every fit (ROMS ocean_time reference)
const CONS   = ["M2", "S2", "N2", "K1", "O1"]           # fitted; K2, NU2, P1 inferred with FES ratios in the comparison

fitfile(src)  = "$(src)_sshfit$(SFX).nc"
compfile(src) = "$(src)_FES_sshfit_comp$(SFX).nc"
findfile(name) = (for d in P.fitdirs; isfile(d * name) && return d * name; end; nothing)
lon180(x) = x > 180 ? x - 360 : x

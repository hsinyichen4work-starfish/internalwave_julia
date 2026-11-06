#!/usr/bin/env julia
#=
Download HYCOM GOFS 3.1 (GLBy0.08/expt_93.0) 3-hourly output from HYCOM.org via NCSS.
  https://www.hycom.org/data/glby0pt08/expt-93pt0

Grid      : 0.08 deg lon x 0.04 deg lat, 40 z-levels (0-5000 m), lon in 0..360
Variables : surf_el (SSH), water_temp, salinity, water_u, water_v
Coverage  : roughly 2018-12-04 .. 2024-09-04
Output    : one NetCDF file per variable per 3-hourly snapshot (00Z..21Z), in OUTDIR/<var>/
            (smaller requests are needed: HYCOM's server fails on a full day of 3-D fields)

Usage
  1. Edit the USER SETTINGS block below (region + time + variables)
  2. julia hycom_data_download.jl
     Re-running skips pieces that are already downloaded, so just re-run after failures.

Merge afterwards, e.g.
  cd /home/hsinyi/roms_data/HYCOM_downlaod
  for v in surf_el water_temp salinity water_u water_v; do ncrcat $v/hycom_gofs31_${v}_*.nc hycom_gofs31_${v}.nc; done
  # or in Julia: NCDataset(sort(readdir("water_temp"; join=true)), aggdim="time")
=#

using Dates
using Downloads
using NCDatasets

# ============================== USER SETTINGS ==============================
# (1) Region: taken from a ROMS grid file (lon_rho/lat_rho extent + padding).
#     Set GRID_FILE = "" to use the manual box below instead.
#     Extents of the 1104 grids (lon in 0..360):
#       2700m: lon [305.707, 325.241]  lat [-6.254, 14.038]   <- outermost parent
#        900m: lon [310.068, 319.817]  lat [-1.894,  8.199]
#        300m: lon [311.294, 318.598]  lat [-0.812,  7.114]
#        100m: lon [312.525, 317.355]  lat [ 0.285,  6.020]
const GRID_FILE = "/home/hsinyi/roms_data/grid/new_design/roms_grd_2700m_1104.nc"
const PAD       = 0.5              # deg added on every side (~6 HYCOM cells) for interpolation

# manual box (only used when GRID_FILE = ""); lon in -180..180 or 0..360
const MANUAL_WEST, MANUAL_EAST   = -51.0, -39.0
const MANUAL_SOUTH, MANUAL_NORTH =  -2.0,   9.0

# (2) Time: inclusive date range (all 8 snapshots of each day, 00Z..21Z)
const START_DATE = Date(2022, 8, 1)
const END_DATE   = Date(2022, 11, 30)

# Variables to grab (any subset of ALL_VARS)
const VARS = ["surf_el", "water_temp", "salinity", "water_u", "water_v"]

const STRIDE  = 1                  # horizontal stride (1 = full resolution)
const OUTDIR  = "/home/hsinyi/roms_data/HYCOM_downlaod"
const RETRIES = 5
const TIMEOUT = 1200               # seconds per request (one var, one snapshot: ~2 min seen)
const NWORKERS = 4                 # requests running at the same time. The work is
                                   # server-side, so these are tasks, not CPU cores.
                                   # HYCOM throttles/blocks heavy users: keep <= ~6.
# ===========================================================================

const NCSS_URL = "https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0"  # aggregated ssh+ts3z+uv3z
const ALL_VARS = ["surf_el", "water_temp", "salinity", "water_u", "water_v"]

to360(lon) = mod(lon, 360.0)

# region from the ROMS grid (rounded outward to 0.01 deg), or the manual box
function get_region()
    isempty(GRID_FILE) && return (MANUAL_WEST, MANUAL_EAST, MANUAL_SOUTH, MANUAL_NORTH)
    lon, lat = NCDataset(GRID_FILE) do ds
        to360.(ds["lon_rho"][:, :]), ds["lat_rho"][:, :]
    end
    w = floor((minimum(lon) - PAD) * 100) / 100
    e = ceil((maximum(lon) + PAD) * 100) / 100
    s = floor((minimum(lat) - PAD) * 100) / 100
    n = ceil((maximum(lat) + PAD) * 100) / 100
    return (w, e, s, n)
end

const WEST, EAST, SOUTH, NORTH = get_region()

# one request = one variable at one time step (the HYCOM server rejects bigger
# requests over this domain with "Inconsistent array length read" / read timeouts)
function build_url(var::String, t::DateTime)
    params = [
        "var=$var",
        "north=$(NORTH)",
        "south=$(SOUTH)",
        "west=$(to360(WEST))",
        "east=$(to360(EAST))",
        "horizStride=$(STRIDE)",
        "time=$(Dates.format(t, "yyyy-mm-ddTHH:MM:SS"))Z",
        "accept=netcdf4",
    ]
    return NCSS_URL * "?" * join(params, "&")
end

# print + flush right away, so progress shows up in a nohup log file
logmsg(io::IO, s) = (println(io, s); flush(io))
logmsg(s) = logmsg(stdout, s)

# NCSS can take minutes to subset before sending the first byte; Downloads' default
# aborts after 20 s of <1 byte/s, so stretch that "stall" limit to TIMEOUT.
const DOWNLOADER = let d = Downloads.Downloader()
    d.easy_hook = (easy, info) -> Downloads.Curl.setopt(easy, Downloads.Curl.CURLOPT_LOW_SPEED_TIME, TIMEOUT)
    d
end

function is_netcdf(path)
    magic = open(f -> read(f, 4), path)
    return magic in (UInt8[0x89, 'H', 'D', 'F'], UInt8['C', 'D', 'F', 0x01], UInt8['C', 'D', 'F', 0x02])
end

outfile(var, t) = joinpath(OUTDIR, var, "hycom_gofs31_$(var)_$(Dates.format(t, "yyyymmddTHH")).nc")

# returns :ok, :skip, :missing (HYCOM has no output at t) or :fail
function fetch_one(var::String, t::DateTime)
    fname = outfile(var, t)
    isfile(fname) && return :skip           # files only appear after a complete, checked download

    url = build_url(var, t)
    tmp = fname * ".part"
    for attempt in 1:RETRIES
        try
            # Downloads.download throws on HTTP errors (non-2xx)
            Downloads.download(url, tmp; timeout = TIMEOUT, downloader = DOWNLOADER)
            is_netcdf(tmp) || error("response is not a NetCDF file: " * String(read(tmp))[1:min(end, 300)])
            # NCSS returns the *nearest* time if t is missing in HYCOM -> don't keep a duplicate
            tgot = NCDataset(ds -> ds["time"][1], tmp)
            if abs(tgot - t) > Hour(1)
                rm(tmp)
                logmsg(stderr, "[gap]  $var $t not in HYCOM (nearest is $tgot), skipped")
                return :missing
            end
            mv(tmp, fname; force = true)
            logmsg("[ok]   $(basename(fname))  ($(round(filesize(fname) / 1e6, digits = 1)) MB)")
            return :ok
        catch e
            msg = e isa Downloads.RequestError ? "HTTP $(e.response.status): $(e.message)" : sprint(showerror, e)
            logmsg(stderr, "[fail] $var $t attempt $attempt/$RETRIES: $(first(msg, 300))")
            isfile(tmp) && rm(tmp)
            attempt < RETRIES && sleep(min(60, 10 * attempt))
        end
    end
    return :fail
end

function main()
    # ---- sanity checks ----
    bad = setdiff(VARS, ALL_VARS)
    isempty(bad) || error("unknown variable(s): $bad, choose from $ALL_VARS")
    SOUTH < NORTH || error("SOUTH must be < NORTH")
    START_DATE <= END_DATE || error("START_DATE must be <= END_DATE")
    if to360(WEST) > to360(EAST)
        @warn "Box crosses the 0/360 meridian (west=$(to360(WEST)), east=$(to360(EAST))); " *
              "NCSS may not handle this — split into two boxes if the request fails."
    end

    foreach(v -> mkpath(joinpath(OUTDIR, v)), VARS)
    src = isempty(GRID_FILE) ? "manual box" : "$(basename(GRID_FILE)) + $(PAD) deg pad"
    logmsg("HYCOM GOFS 3.1 expt_93.0  region from $src")
    logmsg("  lon [$WEST, $EAST]  lat [$SOUTH, $NORTH]  $START_DATE .. $END_DATE")
    logmsg("vars: " * join(VARS, ", ") * "   ->  $OUTDIR/<var>/")

    # all (var, time) pieces, ordered by time so data arrive chronologically
    times = collect(DateTime(START_DATE):Hour(3):DateTime(END_DATE) + Hour(21))
    jobs = [(v, t) for t in times for v in VARS]
    logmsg("$(length(jobs)) requests ($(length(times)) snapshots x $(length(VARS)) vars), $NWORKERS at a time\n")

    # NWORKERS tasks pull pieces from a shared queue
    queue = Channel{Tuple{String,DateTime}}(length(jobs))
    foreach(j -> put!(queue, j), jobs)
    close(queue)

    failed, missing_ = Tuple{String,DateTime}[], Tuple{String,DateTime}[]
    ndone = Ref(0)
    lk = ReentrantLock()
    @sync for _ in 1:min(NWORKERS, length(jobs))
        @async for (v, t) in queue
            r = try
                fetch_one(v, t)
            catch e
                logmsg(stderr, "[fail] $v $t: " * sprint(showerror, e))
                :fail
            end
            lock(lk) do
                r == :fail && push!(failed, (v, t))
                r == :missing && push!(missing_, (v, t))
                ndone[] += 1
                ndone[] % 100 == 0 && logmsg("[progress] $(ndone[])/$(length(jobs))")
            end
        end
    end

    if !isempty(missing_)
        logmsg("\nNot available in HYCOM (gaps in the model output):")
        foreach(((v, t),) -> logmsg("   $v $t"), sort(missing_; by = last))
    end
    if !isempty(failed)
        logmsg("\nFailed (re-run the script to retry; finished pieces are skipped):")
        foreach(((v, t),) -> logmsg("   $v $t"), sort(failed; by = last))
        exit(1)
    end
    logmsg("\nAll requests downloaded.")
end

main()

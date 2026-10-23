"""era5_download.py

ERA5 hourly single-level fields for the 2700 m Amazon grid (lon -50.8..-36.2,
lat -2.3..13.0) + margin, 2022-08-21 00:00 .. 2022-12-01 00:00, from the
Copernicus Climate Data Store (credentials in ~/.cdsapirc).

One request per month; each returns a zip with
    data_stream-oper_stepType-avg.nc      avg_* : means over the hour ENDING at valid_time
    data_stream-oper_stepType-instant.nc  msl, u10, v10, t2m, d2m, sst
unpacked into ERA5_DATA/<yyyymm>/. Months already unpacked are skipped.

Flux fields (sign: positive downward / into the ocean):
    avg_iews, avg_inss  eastward / northward turbulent surface stress [N m-2]
    avg_snswrf          net shortwave      [W m-2]
    avg_snlwrf          net longwave       [W m-2]
    avg_slhtf           latent heat        [W m-2]
    avg_ishf            sensible heat      [W m-2]
    avg_ie              evaporation rate   [kg m-2 s-1] (negative = evaporation)
    avg_tprate          precipitation rate [kg m-2 s-1]

usage:
    ~/envs/cmems/bin/python -I era5_download.py
"""
import datetime as dt
import os
import zipfile
import cdsapi

out = "/expanse/lustre/projects/uso101/hchen54/ERA5_DATA/"
t0, t1 = dt.datetime(2022, 8, 21), dt.datetime(2022, 12, 1)
area = [14.5, -52.0, -4.0, -35.0]   # N, W, S, E

variables = [
    # fluxes for the ROMS flux forcing
    "mean_eastward_turbulent_surface_stress", "mean_northward_turbulent_surface_stress",
    "mean_surface_net_short_wave_radiation_flux", "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_latent_heat_flux", "mean_surface_sensible_heat_flux",
    "mean_evaporation_rate", "mean_total_precipitation_rate",
    "mean_sea_level_pressure",
    # state variables (bulk forcing / checks)
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_temperature", "2m_dewpoint_temperature", "sea_surface_temperature",
]

c = cdsapi.Client()
month = dt.datetime(t0.year, t0.month, 1)
while month <= t1:
    nxt = dt.datetime(month.year + month.month // 12, month.month % 12 + 1, 1)
    days = [d for d in range(1, 32)
            if (month + dt.timedelta(days=d - 1)).month == month.month
            and t0 <= month + dt.timedelta(days=d - 1) <= t1]
    tag = month.strftime("%Y%m")
    mdir = os.path.join(out, tag)
    if os.path.isdir(mdir) and len([f for f in os.listdir(mdir) if f.endswith(".nc")]) == 2:
        print(tag, "already there, skipped")
    else:
        os.makedirs(mdir, exist_ok=True)
        zfile = os.path.join(out, f"era5_{tag}.zip")
        print(tag, f"days {days[0]}..{days[-1]}: requesting", flush=True)
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"], "variable": variables,
            "year": [str(month.year)], "month": [f"{month.month:02d}"],
            "day": [f"{d:02d}" for d in days], "time": [f"{h:02d}:00" for h in range(24)],
            "area": area, "data_format": "netcdf", "download_format": "unarchived",
        }, zfile)
        with zipfile.ZipFile(zfile) as z:
            z.extractall(mdir)
        os.remove(zfile)
        print(tag, "->", sorted(os.listdir(mdir)), flush=True)
    month = nxt
print("done")

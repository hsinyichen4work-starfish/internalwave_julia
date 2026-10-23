"""era5_frc_make.py

Daily ROMS surface forcing files for the 2700 m grid from ERA5 hourly fields
(era5_download.py), same layout and conventions as the NCOM ones made by
h2r_frc.jl (roms_frc_*.nc): 25 hourly records 00..24 h, time in days since
1994-01-01, one time dimension per field group.

    sustr, svstr [N m-2]   ERA5 avg_iews / avg_inss, rotated to xi/eta, at u/v points
    shflux [W m-2]         NET heat flux incl. solar = snswrf + snlwrf + slhtf + ishf
    swrad  [W m-2]         net shortwave = snswrf
    swflux [cm day-1]      E - P = (-avg_ie - avg_tprate) / 1000 * 8.64e6
    Pair   [mbar]          msl / 100
ERA5 sign convention (positive downward) = ROMS convention (positive into the ocean).

Time: ERA5 avg_* fields are means over the hour ENDING at valid_time; the value
at hour H is taken as the mean of the two hourly means ending at H and H+1,
so it is centred on H like the instantaneous msl.

Land: ERA5 fluxes over land are land fluxes. Before the bilinear
interpolation, ERA5 land cells (where sst is missing) take the value of the
nearest ERA5 ocean cell, so the coast (Amazon mouth) gets ocean fluxes.

usage (Python with netCDF4 + scipy, e.g. ~/.venvs/utide):
    python -I era5_frc_make.py
"""
import datetime as dt
import glob
import os
import numpy as np
from netCDF4 import Dataset, num2date
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

era5 = "/expanse/lustre/projects/uso101/hchen54/ERA5_DATA/"
grdfile = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_cmems_grd_2700m.nc"
outdir = "/expanse/lustre/projects/uso101/hchen54/input_cmems/frc_era5/"
prefix = "roms_era5_frc_2700m_"
days = [dt.datetime(2022, 8, 22) + dt.timedelta(days=i) for i in range(101)]   # 08-22 .. 11-30
t_epoch = dt.datetime(1994, 1, 1)
remake = False

## ROMS grid (eta, xi)
with Dataset(grdfile) as g:
    lon = g["lon_rho"][:].data % 360.0
    lat = g["lat_rho"][:].data
    ang = g["angle"][:].data
    mask = g["mask_rho"][:].data
umask = mask[:, 1:] * mask[:, :-1]
vmask = mask[1:, :] * mask[:-1, :]
ny, nx = lon.shape
cosa, sina = np.cos(ang), np.sin(ang)

## ERA5: concatenate the monthly files along time
avg_vars = ["avg_iews", "avg_inss", "avg_snswrf", "avg_snlwrf", "avg_slhtf", "avg_ishf", "avg_ie", "avg_tprate"]
inst_vars = ["msl", "sst"]


def load(kind, names):
    files = sorted(glob.glob(os.path.join(era5, "2022??", f"data_stream-oper_stepType-{kind}.nc")))
    if not files:
        raise FileNotFoundError(f"no ERA5 {kind} files in {era5}")
    t, data = [], {v: [] for v in names}
    for f in files:
        with Dataset(f) as d:
            tv = d["valid_time"]
            t += list(num2date(tv[:], tv.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
            for v in names:
                data[v].append(np.ma.filled(d[v][:].astype(float), np.nan))
            elat = d["latitude"][:].data
            elon = d["longitude"][:].data % 360.0
    t = np.array(t)
    order = np.argsort(t)
    return t[order], {v: np.concatenate(data[v])[order] for v in names}, elat, elon


ta, A, elat, elon = load("avg", avg_vars)
ti, I, _, _ = load("instant", inst_vars)
if np.any(np.diff(ta) != dt.timedelta(hours=1)) or np.any(np.diff(ti) != dt.timedelta(hours=1)):
    raise ValueError("ERA5 times are not continuous hourly")
print(f"ERA5 avg {ta[0]} .. {ta[-1]}, instant {ti[0]} .. {ti[-1]}, grid {len(elat)} x {len(elon)}")

# ERA5 latitude is north->south; flip to ascending for the interpolator
flip = elat[0] > elat[-1]
if flip:
    elat = elat[::-1]
    A = {v: a[:, ::-1, :] for v, a in A.items()}
    I = {v: a[:, ::-1, :] for v, a in I.items()}

# ocean = sst present (time-invariant land mask); nearest-ocean index for land cells
ocean = np.isfinite(I["sst"]).any(axis=0)
fill_idx = distance_transform_edt(~ocean, return_distances=False, return_indices=True)
pts = np.column_stack([lat.ravel(), lon.ravel()])
if lon.min() < elon.min() or lon.max() > elon.max() or lat.min() < elat.min() or lat.max() > elat.max():
    raise ValueError("ERA5 area does not cover the ROMS grid")


def to_roms(field2d):
    f = field2d[tuple(fill_idx)]                       # land <- nearest ocean
    if np.isnan(f).any():
        raise ValueError("NaN left in an ERA5 ocean cell")
    return RegularGridInterpolator((elat, elon), f, method="linear")(pts).reshape(ny, nx)


def avg_at(v, t):
    """mean of the hourly means ending at t and t+1h -> centred on t"""
    k = np.searchsorted(ta, t)
    if k + 1 >= len(ta) or ta[k] != t:
        raise ValueError(f"ERA5 {v} does not cover {t} .. {t + dt.timedelta(hours=1)}")
    return 0.5 * (A[v][k] + A[v][k + 1])


def inst_at(v, t):
    k = np.searchsorted(ti, t)
    if k >= len(ti) or ti[k] != t:
        raise ValueError(f"ERA5 {v} missing at {t}")
    return I[v][k]


def create(fname, times):
    with Dataset(fname, "w", format="NETCDF4") as d:
        for nm, n in (("xi_u", nx - 1), ("xi_v", nx), ("xi_rho", nx), ("eta_u", ny), ("eta_v", ny - 1), ("eta_rho", ny)):
            d.createDimension(nm, n)
        tdesc = {"sms_time": "surface momentum stress time", "shf_time": "surface net heat flux time",
                 "swf_time": "surface salt/freshwater flux time", "srf_time": "surface shortwave radiation time",
                 "pair_time": "surface air pressure time"}
        tdays = [(t - t_epoch).total_seconds() / 86400.0 for t in times]
        for tn, ln in tdesc.items():
            d.createDimension(tn, None)
            v = d.createVariable(tn, "f8", (tn,)); v.long_name = ln; v.units = "day"; v[:] = tdays
        spec = [("sustr", ("sms_time", "eta_u", "xi_u"), "surface u-momentum stress", "Newton meter-2"),
                ("svstr", ("sms_time", "eta_v", "xi_v"), "surface v-momentum stress", "Newton meter-2"),
                ("shflux", ("shf_time", "eta_rho", "xi_rho"), "surface net heat flux", "Watt meter-2"),
                ("swflux", ("swf_time", "eta_rho", "xi_rho"), "surface freshwater flux (E-P)", "centimeter day-1"),
                ("swrad", ("srf_time", "eta_rho", "xi_rho"), "solar shortwave radiation", "Watt meter-2"),
                ("Pair", ("pair_time", "eta_rho", "xi_rho"), "surface air pressure", "millibar")]
        for nm, dims, ln, un in spec:
            v = d.createVariable(nm, "f4", dims); v.long_name = ln; v.units = un
        d["swrad"].positive_value = "downward flux, heating"
        d["swrad"].negative_value = "upward flux, cooling"
        d.title = "Surface forcing file produced from ERA5 hourly single levels (era5_frc_make.py)"
        d.date = dt.datetime.now().strftime("%d-%b-%Y")
        d.grd_file = grdfile
        d.type = "FORCING file"
        d.history = "ROMS"
        d.source = "ERA5 reanalysis-era5-single-levels, Copernicus Climate Change Service"


os.makedirs(outdir, exist_ok=True)
for day in days:
    fname = os.path.join(outdir, prefix + day.strftime("%Y%m%d%H") + ".nc")
    if os.path.isfile(fname) and not remake:
        continue
    times = [day + dt.timedelta(hours=h) for h in range(25)]
    tmp = fname + ".tmp"
    create(tmp, times)
    with Dataset(tmp, "a") as d:
        for k, t in enumerate(times):
            ue, vn = to_roms(avg_at("avg_iews", t)), to_roms(avg_at("avg_inss", t))
            ug = ue * cosa + vn * sina
            vg = vn * cosa - ue * sina
            d["sustr"][k] = 0.5 * (ug[:, 1:] + ug[:, :-1]) * umask
            d["svstr"][k] = 0.5 * (vg[1:, :] + vg[:-1, :]) * vmask
            sw = to_roms(avg_at("avg_snswrf", t))
            net = sw + to_roms(avg_at("avg_snlwrf", t) + avg_at("avg_slhtf", t) + avg_at("avg_ishf", t))
            d["shflux"][k] = net * mask
            d["swrad"][k] = sw * mask
            emp = -(avg_at("avg_ie", t) + avg_at("avg_tprate", t)) / 1000.0 * 8.64e6   # kg m-2 s-1 -> cm/day
            d["swflux"][k] = to_roms(emp) * mask
            d["Pair"][k] = to_roms(inst_at("msl", t)) / 100.0
    os.replace(tmp, fname)
    print("wrote", os.path.basename(fname), flush=True)
print("done")

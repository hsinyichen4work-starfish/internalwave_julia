"""tpxo_tide_make.py

UCLA-ROMS tide forcing file (read by src/tides.F) for the 2700 m grid from
TPXO10 atlas v2 (1/30 deg, NetCDF), for the run starting 2022-08-24.

tides.F rebuilds every field as
    F(t) = F_Re cos(omega t) - F_Im sin(omega t) = Re{(F_Re + i F_Im) exp(i omega t)}
with t the model time in seconds. The ini file's ocean_time counts from
1994-01-01 00:00, so t does too. TPXO (OTIS convention, pyTMD corrections='OTIS'):
    eta(t) = f A cos(G(t) + u - phi),   Z = A exp(-i phi) = hRe + i hIm
with G the Greenwich equilibrium argument. Hence
    F = Z f exp(i (G(t_ref) + u)) exp(-i omega t_ref),   t_ref = run start
with omega = dG/dt (taken from pyTMD's G so the two match), and f, u at
the middle of the run (nodal factors barely change over 38 days).

  ssh   : TPXO elevation [mm -> m], land filled from the nearest TPXO ocean
          cell before interpolating so the coast is not damped
  u, v  : TPXO transports [cm^2/s -> m^2/s] interpolated to rho points and
          divided by the ROMS h (keeps the TPXO volume transport on the ROMS
          bathymetry), rotated to xi/eta with the grid angle, averaged to
          u/v points
  pot   : equilibrium tide incl. load Love factor (TPXO/OTIS parameters):
          alpha A f L(lat) cos(G + u + s lon), L = cos^2(lat), sin(2 lat),
          1/2 - 3/2 sin^2(lat) for species s = 2, 1, 0 (prsgrd.F: P -= g ptide)

usage (Python env with pyTMD, e.g. ~/.venvs/utide):
    python -I tpxo_tide_make.py
"""
import datetime as dt
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pyTMD.arguments import arguments
from pyTMD.predict import _constituent_parameters

## settings
cons = ["m2", "s2", "n2", "k2", "k1", "o1", "p1", "q1", "mf", "mm"]   # ntides = 10 in tides.opt
t_start = dt.datetime(2022, 8, 24)     # run start = ini ocean_time
t_end = dt.datetime(2022, 12, 1)       # 37-day test run: 2022-10-01
t_epoch = dt.datetime(1994, 1, 1)      # model time origin (ocean_time / bry_time)

tpxo = "/expanse/lustre/projects/uso101/hchen54/input_cmems/tpxo/TPXO10a_v2/1_30deg/TPXO10_atlas_v2_nc/"
grdfile = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_cmems_grd_2700m.nc"
outfile = "/expanse/lustre/projects/uso101/hchen54/input_cmems/tide/roms_tide_2700m_TPXO10v2_0824_1130.nc"   # 37-day test run: roms_tide_2700m_TPXO10v2.nc
pad = 1.0                              # deg of TPXO around the grid

## ROMS grid (eta, xi); lon_rho is 0..360
with Dataset(grdfile) as g:
    lon = g["lon_rho"][:].data % 360.0
    lat = g["lat_rho"][:].data
    h = g["h"][:].data
    ang = g["angle"][:].data
    mask = g["mask_rho"][:].data
umask = mask[:, 1:] * mask[:, :-1]
vmask = mask[1:, :] * mask[:-1, :]
pts = np.column_stack([lat.ravel(), lon.ravel()])

## TPXO subset
with Dataset(tpxo + "grid_tpxo10atlas_v2.nc") as g:
    lon_z, lat_z = g["lon_z"][:].data, g["lat_z"][:].data
    lon_u, lat_u = g["lon_u"][:].data, g["lat_u"][:].data
    lon_v, lat_v = g["lon_v"][:].data, g["lat_v"][:].data
    ix = np.where((lon_z >= lon.min() - pad) & (lon_z <= lon.max() + pad))[0]
    iy = np.where((lat_z >= lat.min() - pad) & (lat_z <= lat.max() + pad))[0]
    xs, ys = slice(ix[0], ix[-1] + 1), slice(iy[0], iy[-1] + 1)
    wet = g["hz"][xs, ys].data.T > 0                   # (lat, lon)


def interp(x, y, fld):
    """bilinear from the TPXO (lat y, lon x) subset to the ROMS rho points"""
    f = RegularGridInterpolator((y, x), fld, method="linear", bounds_error=True)
    return f(pts).reshape(lon.shape)


def fill_land(fld, wet):
    """land cells take the value of the nearest ocean cell"""
    idx = distance_transform_edt(~wet, return_distances=False, return_indices=True)
    return fld[tuple(idx)]


## time factors (pyTMD, OTIS)
MJD0 = 51544.0                                     # MJD of 2000-01-01
mjd = lambda t: MJD0 + (t - dt.datetime(2000, 1, 1)).total_seconds() / 86400.0
t_mid = t_start + (t_end - t_start) / 2
_, _, G0 = arguments(np.array([mjd(t_start)]), cons, corrections="OTIS")
_, _, G1 = arguments(np.array([mjd(t_start) + 1.0]), cons, corrections="OTIS")
pu, pf, _ = arguments(np.array([mjd(t_mid)]), cons, corrections="OTIS")
G0, G1, pu, pf = G0[0], G1[0], pu[0], pf[0]        # G [deg], pu [rad]
t_ref = (t_start - t_epoch).total_seconds()

ntide = len(cons)
ny, nx = lon.shape
out = {k: np.zeros((ntide, ny, nx)) for k in ("ssh_Re", "ssh_Im", "pot_Re", "pot_Im")}
out.update({k: np.zeros((ntide, ny, nx - 1)) for k in ("u_Re", "u_Im")})
out.update({k: np.zeros((ntide, ny - 1, nx)) for k in ("v_Re", "v_Im")})
omega = np.zeros(ntide)

for k, c in enumerate(cons):
    amp_eq, _, om_nom, alpha, species = _constituent_parameters(c)
    # omega = dG/dt: one day of G advance, unwrapped with the nominal frequency
    dG = (G1[k] - G0[k]) % 360.0
    ncyc = np.round((np.degrees(om_nom) * 86400.0 - dG) / 360.0)
    omega[k] = np.radians(dG + 360.0 * ncyc) / 86400.0
    # complex factor that turns Z into the ROMS amplitude at t = 0 (1994-01-01)
    fac = pf[k] * np.exp(1j * (np.radians(G0[k]) + pu[k] - omega[k] * t_ref))

    with Dataset(tpxo + f"h_{c}_tpxo10_atlas_30_v2.nc") as d:
        z = (d["hRe"][xs, ys].data.T + 1j * d["hIm"][xs, ys].data.T) * 1e-3          # m
    with Dataset(tpxo + f"u_{c}_tpxo10_atlas_30_v2.nc") as d:
        U = (d["uRe"][xs, ys].data.T + 1j * d["uIm"][xs, ys].data.T) * 1e-4          # m^2/s
        V = (d["vRe"][xs, ys].data.T + 1j * d["vIm"][xs, ys].data.T) * 1e-4

    z = fill_land(z, wet)
    zc = interp(lon_z[xs], lat_z[ys], z.real) + 1j * interp(lon_z[xs], lat_z[ys], z.imag)
    Uc = interp(lon_u[xs], lat_u[ys], U.real) + 1j * interp(lon_u[xs], lat_u[ys], U.imag)
    Vc = interp(lon_v[xs], lat_v[ys], V.real) + 1j * interp(lon_v[xs], lat_v[ys], V.imag)
    ue, vn = Uc / h, Vc / h                                                          # m/s, east/north

    # rotate to the grid (as in h2r_make_ini) and move to u/v points
    ug = ue * np.cos(ang) + vn * np.sin(ang)
    vg = vn * np.cos(ang) - ue * np.sin(ang)
    uu = 0.5 * (ug[:, 1:] + ug[:, :-1]) * umask
    vv = 0.5 * (vg[1:, :] + vg[:-1, :]) * vmask

    # equilibrium tide
    phi = np.radians(lat)
    L = {2: np.cos(phi) ** 2, 1: np.sin(2 * phi), 0: 0.5 - 1.5 * np.sin(phi) ** 2}[int(species)]
    pot = alpha * amp_eq * L * np.exp(1j * int(species) * np.radians(lon))

    for name, fld in (("ssh", zc * mask), ("u", uu), ("v", vv), ("pot", pot)):
        F = fld * fac
        out[name + "_Re"][k] = F.real
        out[name + "_Im"][k] = F.imag

    sp = np.abs(ue * mask)
    print(f"{c:3s} omega {omega[k]:.10e} (nominal {om_nom:.6e})  f {pf[k]:.4f}  u {np.degrees(pu[k]):7.2f} deg"
          f"  |ssh| {np.abs(zc[mask == 1]).min():.3f}-{np.abs(zc[mask == 1]).max():.3f} m"
          f"  |u_east| max {sp.max():.2f} m/s  |pot| max {np.abs(pot).max():.3f} m")

## write
import os
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with Dataset(outfile, "w", format="NETCDF4") as d:
    d.createDimension("ntides", ntide)
    d.createDimension("eta_rho", ny); d.createDimension("xi_rho", nx)
    d.createDimension("eta_v", ny - 1); d.createDimension("xi_u", nx - 1)
    d.createDimension("nchar", 4)
    v = d.createVariable("omega", "f8", ("ntides",))
    v.long_name = "tidal angular frequency"; v.units = "radian second-1"; v[:] = omega
    v = d.createVariable("name", "S1", ("ntides", "nchar"))
    v[:] = np.array([list(c.upper().ljust(4)) for c in cons], dtype="S1")
    dims = {"ssh": ("ntides", "eta_rho", "xi_rho"), "pot": ("ntides", "eta_rho", "xi_rho"),
            "u": ("ntides", "eta_rho", "xi_u"), "v": ("ntides", "eta_v", "xi_rho")}
    longn = {"ssh": "tidal elevation", "pot": "equilibrium tidal potential (elevation, incl. load Love factor)",
             "u": "tidal barotropic xi-velocity", "v": "tidal barotropic eta-velocity"}
    units = {"ssh": "meter", "pot": "meter", "u": "meter second-1", "v": "meter second-1"}
    for nm in ("ssh", "u", "v", "pot"):
        for part in ("Re", "Im"):
            v = d.createVariable(f"{nm}_{part}", "f8", dims[nm])
            v.long_name = f"{longn[nm]}, {'real' if part == 'Re' else 'imaginary'} part"
            v.units = units[nm]
            v[:] = out[f"{nm}_{part}"]
    d.data_source = "TPXO10_atlas_v2"
    d.constituents = " ".join(c.upper() for c in cons)
    d.grd_file = grdfile
    d.time_reference = ("F(t) = Re{(F_Re + i F_Im) exp(i omega t)}, t = seconds since "
                        f"{t_epoch:%Y-%m-%d %H:%M} UTC; phases from G at {t_start:%Y-%m-%d %H:%M}, "
                        f"nodal f, u at {t_mid:%Y-%m-%d %H:%M}")
    d.script = "internalwave_julia/input_file_make/tpxo_tide_make.py"
    d.date = dt.datetime.now().strftime("%d-%b-%Y")
print("wrote", outfile)

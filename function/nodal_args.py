"""nodal_args.py
Copy of M. Buijsman's claudecodes/ROMS/nodal_args.py (2026-10-15), import adjusted for pyTMD 2.0.x
(Python 3.8 from the Expanse anaconda3 module; ~/.venvs/utide).

Equilibrium arguments and nodal corrections for a time vector, written to NetCDF for
ROMS_TPXO_tidalcomp.jl. Uses pyTMD with the OTIS/TPXO conventions (corrections='OTIS',
latitude independent), so the harmonic constants are consistent with TPXO:
    eta(t) = sum_k pf_k(t) A_k cos(G_k(t) + pu_k(t) - phi_k),  Z_k = A_k exp(-i phi_k)

usage: python nodal_args.py times.nc out.nc m2,s2,...
  times.nc: variable t = seconds since 2000-01-01 00:00 UTC
  out.nc:   G [deg], pu [deg], pf [-] (time, ncon) at the times, and at the record centre / start
"""
import sys
import numpy as np
from scipy.io import netcdf_file
try:
    from pyTMD.arguments import arguments          # pyTMD 2.0.x
except ImportError:
    from pyTMD.constituents import arguments       # newer pyTMD (Python >= 3.10)

fin, fout, cons = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
with netcdf_file(fin, "r", mmap=False) as f:
    t = f.variables["t"][:].astype(float)
MJD0 = 51544.0                                  # MJD of 2000-01-01 00:00
MJD = MJD0 + t / 86400.0
tc = 0.5 * (t[0] + t[-1])
pu, pf, G = arguments(MJD, cons, corrections="OTIS")
puc, pfc, Gc = arguments(np.array([MJD0 + tc / 86400.0]), cons, corrections="OTIS")
pus, pfs, Gs = arguments(np.array([MJD[0]]), cons, corrections="OTIS")

with netcdf_file(fout, "w") as f:
    f.createDimension("time", len(t)); f.createDimension("con", len(cons)); f.createDimension("one", 1)
    nch = max(4, max(len(c) for c in cons)); f.createDimension("nchar", nch)
    v = f.createVariable("t", "d", ("time",)); v[:] = t
    v = f.createVariable("tc", "d", ("one",)); v[:] = tc
    for nm, a, u in (("G", G, "deg"), ("pu", np.degrees(pu), "deg"), ("pf", pf, "1")):
        v = f.createVariable(nm, "d", ("time", "con")); v[:] = a; v.units = u
    for nm, a in (("G_c", Gc), ("pu_c", np.degrees(puc)), ("pf_c", pfc),
                  ("G_s", Gs), ("pu_s", np.degrees(pus)), ("pf_s", pfs)):
        v = f.createVariable(nm, "d", ("one", "con")); v[:] = a
    v = f.createVariable("con", "c", ("con", "nchar")); v[:] = np.array([list(c.ljust(nch)) for c in cons], dtype="S1")
print(f"nodal_args: {len(t)} times, {len(cons)} constituents -> {fout}")

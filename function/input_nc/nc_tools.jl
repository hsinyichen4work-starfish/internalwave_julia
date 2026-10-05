# nc_tools.jl
#
# Small MATLAB-compatibility helpers used by every file in input_nc:
# ncread / ncwrite / read_nc_fun / standardize_name / datenum / num2str.
#
# Arrays keep NetCDF's native (Fortran) dimension order, i.e. the same
# order MATLAB's ncread returns — a ROMS variable declared
# (eta_rho, xi_rho) in `ncdump` comes back as (xi_rho, eta_rho).

# fill value -> NaN for float variables; integer variables are returned
# exactly as stored (MATLAB's ncread does the same, e.g. NCOM's `kb`
# keeps its int32 fill value -2147483647 on land).
function _clean_ncdata(v, data)
    if eltype(data) <: AbstractFloat
        out = Float64.(data)
        for att in ("_FillValue", "missing_value")
            if haskey(v.attrib, att)
                fv = v.attrib[att]
                out[data .== fv] .= NaN
            end
        end
        return out
    end
    return data
end

"""
    ncread(file, varname)
    ncread(file, varname, start, count)

MATLAB-style ncread. `start`/`count` are 1-based, one entry per
dimension. Floats come back as Float64 with fill values set to NaN.
"""
function ncread(file::AbstractString, varname::AbstractString)
    NCDataset(file, "r") do ds
        v = ds[varname].var
        data = ndims(v) == 0 ? [v[]] : Array(v)
        _clean_ncdata(v, data)
    end
end

function ncread(file::AbstractString, varname::AbstractString, start, count)
    NCDataset(file, "r") do ds
        v = ds[varname].var
        idx = ntuple(d -> start[d]:(start[d] + count[d] - 1), ndims(v))
        _clean_ncdata(v, v[idx...])
    end
end

"""
    ncsize(file, varname) -> Tuple

Size of a variable without reading it (MATLAB: `ncinfo(file, var).Size`).
"""
ncsize(file::AbstractString, varname::AbstractString) =
    NCDataset(ds -> size(ds[varname]), file, "r")

"""
    ncwrite(file, varname, data)
    ncwrite(file, varname, data, start)

MATLAB-style ncwrite into an existing variable. With `start`, `data` is
written as a block beginning at that (1-based) index; trailing
dimensions `data` doesn't have are treated as length 1, so a 2D slab can
be written straight into one time index of a 3D variable.
"""
function ncwrite(file::AbstractString, varname::AbstractString, data, start = nothing)
    NCDataset(file, "a") do ds
        v = ds[varname].var
        nd = ndims(v)
        arr = data isa AbstractArray ? data : [data]
        st = start === nothing ? ones(Int, nd) : collect(Int, start)
        sz = ntuple(d -> size(arr, d), nd)
        prod(sz) == length(arr) ||
            error("ncwrite: data of size $(size(arr)) does not fit variable $varname with $nd dimension(s)")
        idx = ntuple(d -> st[d]:(st[d] + sz[d] - 1), nd)
        v[idx...] = reshape(arr, sz)
    end
    return nothing
end

"""
    read_nc_fun(file) -> Dict{String,Any}

Read every variable of a NetCDF file (port of read_nc_fun.m, which
returned a struct with one field per variable).
"""
function read_nc_fun(file::AbstractString)
    out = Dict{String,Any}()
    NCDataset(file, "r") do ds
        for name in keys(ds)
            v = ds[name].var
            data = ndims(v) == 0 ? [v[]] : Array(v)
            out[name] = _clean_ncdata(v, data)
        end
    end
    return out
end

"""
    standardize_name(s) -> s

Rename lon/lat to lon_rho/lat_rho (port of standardize_name.m). After
this the Dict is guaranteed to hold lon_rho/lat_rho whenever it had
either spelling.
"""
function standardize_name(s::AbstractDict)
    for (short, long) in (("lon", "lon_rho"), ("lat", "lat_rho"))
        if haskey(s, short)
            haskey(s, long) || (s[long] = s[short])
            delete!(s, short)
        end
    end
    return s
end

# MATLAB datenum: days since year 0, with datenum(0,1,1) == 1
datenum(dt::Dates.TimeType) = Dates.value(DateTime(dt) - DateTime(0, 1, 1)) / 86_400_000 + 1
datenum(y::Integer, m::Integer, d::Integer, h::Integer = 0, mi::Integer = 0, s::Integer = 0) =
    datenum(DateTime(y, m, d, h, mi, s))

# MATLAB num2str for a scalar (integers print as integers, otherwise %.Ng
# with N = 5 significant digits past the leading one, like MATLAB)
function num2str(x::Real)
    (x isa Integer || (isfinite(x) && x == round(x))) && return string(round(Int, x))
    dgt = clamp(floor(Int, log10(abs(x))) + 5, 5, 16)
    return Printf.format(Printf.Format("%.$(dgt)g"), x)
end

# MATLAB `date`, e.g. "27-Aug-2026"
matlab_date() = Dates.format(Dates.today(), "dd-u-yyyy")

# MATLAB min/max ignore NaN
nanmin(x) = minimum(v for v in x if !isnan(v))
nanmax(x) = maximum(v for v in x if !isnan(v))

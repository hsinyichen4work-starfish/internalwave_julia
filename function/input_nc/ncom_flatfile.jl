# ncom_flatfile.jl
#
# Readers for the raw NCOM output: the binary flat files and the static
# ohgrd/ovgrd grid files. Ported from read_ncom_flatfile.m,
# extract_ncom_name.m, read_ohgrd.m, read_ovgrdA.m, avg_face_to_center.m
# and vel_rot.m.
#
# Everything here is returned in NATIVE NCOM order, (igrd, jgrd[, nlev]).
# (The MATLAB reader transposed 2D fields to (jgrd, igrd) and every caller
# transposed them straight back; that round trip is dropped.)

"""
    extract_ncom_name(name) -> NamedTuple

Split an NCOM flat-file name such as
`seatmp_mod_000001_000100_2o1244x1334_2022082400_00010000_fcstfld`
into (fldname, igrd, jgrd, nest, datestr_in, timetag, appd, nlev, isface).
"""
function extract_ncom_name(name::AbstractString)
    m = match(r"^([A-Za-z]+)_([A-Za-z]+)_(\d+)_(\d+)_(\d+)o(\d+)x(\d+)_(\d+)_(\d+)_([A-Za-z]+)$", name)
    m === nothing && error("data file name string mismatch: $name")
    fldname, kind, _, lvl, nest, igrd, jgrd, datestr_in, timetag, appd = m.captures

    if lowercase(kind) == "sfc"
        nlev, isface = 1, false
    elseif lowercase(kind) == "mod"
        nlev = 100
        nl = parse(Int, lvl)
        nl == 100 || nl == 101 || error("data file name string mismatch on vertical: $name")
        isface = nl == 101
    else
        error("data file name string mismatch: $name")
    end

    return (fldname = String(fldname), igrd = parse(Int, igrd), jgrd = parse(Int, jgrd),
            nest = parse(Int, nest), datestr_in = String(datestr_in), timetag = String(timetag),
            appd = "_" * appd, nlev = nlev, isface = isface)
end

"""
    read_ncom_flatfile(path_ff, fldname, igrd, jgrd, nest, datestr_in, timetag, appd, nlev, isface)

Read one NCOM/COAMPS-style binary flat file (big-endian, single
precision). Returns Float64 (igrd, jgrd) for a 2D field (`nlev == 1`) or
(igrd, jgrd, nlev) for a 3D one (nlev+1 levels for a face field).
"""
function read_ncom_flatfile(path_ff, fldname, igrd, jgrd, nest, datestr_in, timetag, appd, nlev, isface)
    if nlev == 1
        lvl_str = @sprintf("_sfc_000000_000000_%do", nest)
        dims = (igrd, jgrd)
    else
        lvl_out = isface ? nlev + 1 : nlev
        lvl_str = @sprintf("_mod_000001_%06d_%do", lvl_out, nest)
        dims = (igrd, jgrd, lvl_out)
    end
    grd_str = @sprintf("%04dx%04d", igrd, jgrd)
    file_name = joinpath(path_ff, string(fldname, lvl_str, grd_str, "_", datestr_in, "_", timetag, appd))

    isfile(file_name) || error("read_ncom_flatfile: could not open file: $file_name")
    reclen = prod(dims)
    filesize(file_name) >= 4 * reclen ||
        error("read_ncom_flatfile: $file_name holds $(filesize(file_name) ÷ 4) values, expected $reclen. " *
              "Check igrd/jgrd/nlev/isface against the file name.")
    tmp = Vector{Float32}(undef, reclen)
    open(io -> read!(io, tmp), file_name, "r")
    return reshape(Float64.(ntoh.(tmp)), dims)
end

read_ncom_flatfile(path_ff, s::NamedTuple) =
    read_ncom_flatfile(path_ff, s.fldname, s.igrd, s.jgrd, s.nest, s.datestr_in, s.timetag, s.appd, s.nlev, s.isface)

"""
    ncom_file_list(path_setup, fld; nest = nothing) -> Vector{NamedTuple}

All `fld*` flat files in `path_setup` (only those of `nest` when given,
since one folder holds every nest), parsed and sorted chronologically
by (date tag, time tag) rather than trusting directory order.
"""
function ncom_file_list(path_setup::AbstractString, fld::AbstractString; nest::Union{Nothing,Integer} = nothing)
    names = filter(startswith(fld), readdir(path_setup))
    parsed = extract_ncom_name.(names)
    nest === nothing || filter!(s -> s.nest == nest, parsed)
    isempty(parsed) && error("No $(fld)* files" * (nest === nothing ? "" : " of nest $nest") * " found in $path_setup")
    return sort(parsed; by = s -> s.datestr_in * s.timetag)
end

"""
    ncom_MT(parsed) -> Vector{Float64}

MT (days since 1900-12-31) of each file: analysis date + forecast lead
hours taken from the first four digits of the time tag.
"""
function ncom_MT(parsed::AbstractVector)
    t_ref = datenum(1900, 12, 31, 0, 0, 0)
    return map(parsed) do s
        base_dt = datenum(DateTime(s.datestr_in, dateformat"yyyymmddHH"))
        lead_hrs = parse(Float64, s.timetag[1:4])
        base_dt + lead_hrs / 24 - t_ref
    end
end

"""
    read_ohgrd(opath, nest) -> (igrd, jgrd, lon, lat)

Dimensions and longitude/latitude from ohgrd_[nest].B/.A, lon/lat as
(igrd, jgrd). (read_ohgrd.m also returns dx, dy, h and ang; nothing in
this pipeline uses them.)
"""
function read_ohgrd(opath::AbstractString, nest::Integer)
    dims = parse.(Int, split(read(joinpath(opath, "ohgrd_$(nest).B"), String)))
    igrd, jgrd = dims[1], dims[2]
    lon = Matrix{Float32}(undef, igrd, jgrd)
    lat = Matrix{Float32}(undef, igrd, jgrd)
    open(joinpath(opath, "ohgrd_$(nest).A"), "r") do io
        read!(io, lon)
        read!(io, lat)
    end
    return igrd, jgrd, Float64.(ntoh.(lon)), Float64.(ntoh.(lat))
end

"""
    read_ovgrdA(opath, nest) -> vgrd

Interface depths from ovgrd_[nest].A as (igrd, jgrd, lo+1).
"""
function read_ovgrdA(opath::AbstractString, nest::Integer)
    dims = parse.(Int, split(read(joinpath(opath, "ovgrd_$(nest).B"), String)))
    vgrd = Array{Float32}(undef, dims[1], dims[2], dims[3])
    open(io -> read!(io, vgrd), joinpath(opath, "ovgrd_$(nest).A"), "r")
    return Float64.(ntoh.(vgrd))
end

"""
    read_ohgrd_nc(file) -> (igrd, jgrd, lon, lat)

Same as `read_ohgrd`, from an ohgrd_[nest].nc file (for nests that have
no ohgrd_[nest].B/.A, e.g. nest 1).
"""
function read_ohgrd_nc(file::AbstractString)
    lon = ncread(file, "lon")
    lat = ncread(file, "lat")
    igrd, jgrd = size(lon)
    return igrd, jgrd, lon, lat
end

"""
    ovgrd_from_ohgrd_nc(file) -> vgrd

Rebuild what ovgrd_[nest].A holds, (igrd, jgrd, lo+1) interface depths,
from zw3/kb/h/mask of an ohgrd_[nest].nc file. zw3 is only defined down
to interface kb, so the rest is filled the way ovgrd_*.A does it:
interface kb+1 is the bottom (-h) and every deeper layer keeps the
thickness of layer kb. Land columns are the kb = 40 sigma column
scaled to the unmasked depth hu (0.1 m where hu is not below sea
level), extended the same way. Checked against ovgrd_2.A.
"""
function ovgrd_from_ohgrd_nc(file::AbstractString)
    zw = ncread(file, "zw3")
    kb = ncread(file, "kb")
    h = ncread(file, "h")
    hu = ncread(file, "hu")
    mask = ncread(file, "mask")
    nw = size(zw, 3)

    # sigma fractions z/z_bottom of the shallowest (kb = 40) columns
    ks = 40
    Is = findfirst(I -> mask[I] == 1 && kb[I] == ks, CartesianIndices(mask))
    Is === nothing && error("ovgrd_from_ohgrd_nc: no kb = $ks sea point in $file")
    sig = [zw[Is, 1:ks]; h[Is]] ./ h[Is]

    vgrd = similar(zw)
    for I in CartesianIndices(mask)
        if mask[I] == 1
            k = kb[I]
            vgrd[I, 1:k] .= view(zw, I, 1:k)
            vgrd[I, k+1] = h[I]
        else
            k = ks
            vgrd[I, 1:k+1] .= (hu[I] < 0 ? hu[I] : -0.1) .* sig
        end
        dz = vgrd[I, k] - vgrd[I, k+1]
        for kk in k+2:nw
            vgrd[I, kk] = vgrd[I, kk-1] - dz
        end
    end
    return vgrd
end

"""
    avg_face_to_center(uface, dim) -> uc

Average a staggered (face-point) field onto cell centers along `dim`
(1 for uucurr, 2 for vvcurr). uface(i) sits on the WEST/SOUTH face of
cell i, so center(i) = 0.5*(uface(i) + uface(i+1)); the last row/column
has no east/north face stored and is copied from its west/south face.

Pass RAW face values (land/below-bottom faces = 0 as NCOM stores them),
not NaN-masked ones, and mask the centers afterwards.
"""
function avg_face_to_center(uface::AbstractArray, dim::Integer)
    dim == 1 || dim == 2 || error("avg_face_to_center: dim must be 1 or 2.")
    uc = copy(uface)
    n = size(uface, dim)
    selectdim(uc, dim, 1:n-1) .= 0.5 .* (selectdim(uface, dim, 1:n-1) .+ selectdim(uface, dim, 2:n))
    return uc
end

"""
    vel_rot(u_in, v_in, grdang_deg, direction) -> (u_out, v_out)

Rotate vector components between grid-relative and true east/north
frames. `grdang_deg` is the grid angle in degrees from true east
(counterclockwise), matching the first two dimensions of u_in/v_in.
`direction` is "grid2geo" or "geo2grid".
"""
function vel_rot(u_in::AbstractArray, v_in::AbstractArray, grdang_deg::AbstractMatrix, direction::AbstractString)
    c = cos.(deg2rad.(grdang_deg))
    s = sin.(deg2rad.(grdang_deg))
    if lowercase(direction) == "grid2geo"
        return u_in .* c .- v_in .* s, u_in .* s .+ v_in .* c
    elseif lowercase(direction) == "geo2grid"
        return u_in .* c .+ v_in .* s, v_in .* c .- u_in .* s
    else
        error("vel_rot: direction must be \"grid2geo\" or \"geo2grid\".")
    end
end

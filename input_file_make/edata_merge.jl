# edata_merge.jl
#
# Merges edata files (e.g. the mooring <grid>_mor_edata.nc from edata_make.jl
# and the child-boundary <grid>_bry_edata.nc from edata_make_forbry.jl) into
# one file, since ROMS (extract_data.F) reads a single edata file.
#   - Every variable is an extraction object, copied with its attributes.
#   - Dimensions shared by name (e.g. "three") are defined once; their sizes
#     must agree. Variable names must be unique across the inputs.
#   - Global attributes (<name>_info) are copied; a repeated name must have
#     the same value.
#
# Use as a function:
#     include("input_file_make/edata_merge.jl")
#     merge_edata(["roms_grd_2700m_1104_mor_edata.nc",
#                  "roms_grd_2700m_1104_bry_edata.nc"], "roms_grd_2700m_1104_edata.nc")
# or from the shell (last argument is the output):
#     julia --project=/home/hsinyi/Documents/Julia input_file_make/edata_merge.jl \
#         mor_edata.nc bry_edata.nc out_edata.nc

using NCDatasets

function merge_edata(in_files::AbstractVector{<:AbstractString}, out_file::AbstractString;
                     overwrite = false)
    isfile(out_file) && !overwrite && error("$out_file exists; remove it or set overwrite = true")
    abspath(out_file) in abspath.(in_files) && error("output must differ from the inputs")

    # check everything before writing anything
    dims, vars, gatts = Dict{String,Int}(), Dict{String,String}(), Dict{String,Any}()
    for f in in_files
        NCDataset(f) do ds
            for (d, n) in ds.dim
                haskey(dims, d) && dims[d] != n &&
                    error("dimension $d is $(dims[d]) in one file and $n in $(basename(f))")
                dims[d] = n
            end
            for v in keys(ds)
                haskey(vars, v) && error("object $v is in both $(vars[v]) and $(basename(f))")
                vars[v] = basename(f)
            end
            for (a, val) in ds.attrib
                haskey(gatts, a) && gatts[a] != val &&
                    error("global attribute $a differs between the files")
                gatts[a] = val
            end
        end
    end

    NCDataset(out_file, "c") do out
        for f in in_files
            NCDataset(f) do ds
                for (d, n) in ds.dim
                    haskey(out.dim, d) || defDim(out, d, n)
                end
                for name in keys(ds)
                    v = ds[name].var                     # raw values, no scaling
                    defVar(out, name, Array(v), dimnames(v);
                           attrib = [k => val for (k, val) in v.attrib])
                end
                for (a, val) in ds.attrib
                    haskey(out.attrib, a) || (out.attrib[a] = val)
                end
            end
            println("  copied ", basename(f))
        end
        println("wrote ", out_file, "  (", length(keys(out)), " objects)")
    end
    out_file
end

## run from the shell: julia edata_merge.jl in1.nc in2.nc ... out.nc
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 3 || error("usage: julia edata_merge.jl in1.nc in2.nc [...] out.nc")
    merge_edata(ARGS[1:end-1], ARGS[end])
end

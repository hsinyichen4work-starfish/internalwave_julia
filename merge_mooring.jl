using NCDatasets

src_dir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon/joined_ext"
out_dir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon/mooring_merged"
mkpath(out_dir)

files = filter(f -> endswith(f, ".nc"), readdir(src_dir))
instruments = unique(first.(split.(files, '.')))
filter!(!=("ocean"), instruments)   # ocean.*.nc are empty (0 time records)

for inst in instruments
    # the timestamp in the name is yyyymmddHHMMSS, so sorting by name == sorting by time
    fnames = sort(joinpath.(src_dir, filter(f -> startswith(f, inst * "."), files)))
    outfile = joinpath(out_dir, inst * ".nc")

    NCDataset(fnames; aggdim = "time") do src
        t = src["ocean_time"][:]
        @assert issorted(t) && allunique(t) "$inst: ocean_time is not strictly increasing"

        NCDataset(outfile, "c") do dst
            for name in keys(src)
                v = src[name]
                attrib = Dict(v.attrib)
                # the source files say "Time since 2000", but the values are seconds since 1994-01-01
                name == "ocean_time" && (attrib["long_name"] = "Time since 1994/01/01")
                defVar(dst, name, Array(v), dimnames(v); attrib = attrib)
            end
            dst.attrib["source_files"] = join(basename.(fnames), ", ")
        end
        println(inst, ": ", length(fnames), " files, ", length(t), " records -> ", outfile)
    end
end

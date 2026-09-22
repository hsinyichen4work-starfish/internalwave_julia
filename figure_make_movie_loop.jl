using FFMPEG_jll

base_path = "/home/hsinyi/figure/20260921_output_check2/dbry/bry_match";
movie_path = joinpath(base_path, "movie");

function make_movie(figure_path::String, prefix::String; fps = 4)
    pngs = sort(filter(f -> startswith(basename(f), prefix) && endswith(f, ".png") &&
                            !startswith(basename(f), "."),
                        readdir(figure_path, join = true)))
    if isempty(pngs)
        println("no PNGs found for prefix \"$prefix\", skipping movie")
        return
    end

    listfile = joinpath(figure_path, "$(prefix)_filelist.txt")
    open(listfile, "w") do io
        for p in pngs
            println(io, "file '$(p)'")
        end
    end

    outname = joinpath(movie_path, "$(prefix)_movie.mp4")
    FFMPEG_jll.ffmpeg() do ffmpeg_path
        run(`$ffmpeg_path -y -r $fps -f concat -safe 0 -i $listfile -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p $outname`)
    end
    rm(listfile)
    println("saved movie to ", outname)
end

isdir(movie_path) || mkpath(movie_path)

subfolders = filter(d -> isdir(joinpath(base_path, d)) && d != "movie", readdir(base_path))

for folder in subfolders
    figure_path = joinpath(base_path, folder)
    pngs = filter(f -> endswith(f, ".png") && !startswith(f, "."), readdir(figure_path))
    if isempty(pngs)
        println("no PNGs found in \"$folder\", skipping")
        continue
    end

    # strip trailing _YYYYMMDD.png to recover each file's prefix, so folders
    # containing several series (e.g. flux_timeplot) get one movie per series
    prefixes = unique(replace.(pngs, r"_\d{8}\.png$" => ""))

    println("=== folder: $folder ($(length(prefixes)) prefix(es)) ===")
    for prefix in prefixes
        make_movie(figure_path, prefix)
    end
end

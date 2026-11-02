using FFMPEG_jll

figure_path = "/expanse/lustre/projects/uso101/hchen54/figure/dbry900m/bry_match/flux_timeplot";
movie_path = "/expanse/lustre/projects/uso101/hchen54/figure/dbry900m/bry_match";
start_string = "vp_south"
delete_png = true   # true: delete the PNGs once their movie is saved

cd(figure_path)
readdir(figure_path) 
files = filter(f -> startswith(f, start_string), readdir())

function make_movie(prefix::String; fps = 4)
    pngs = sort(filter(f -> startswith(basename(f), prefix) && endswith(f, ".png"),
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
    # only after the mp4 exists, so a failed ffmpeg run never loses the figures
    if delete_png && isfile(outname)
        foreach(rm, pngs)
        println("deleted ", length(pngs), " PNGs for prefix \"$prefix\"")
    end
end

make_movie(start_string)

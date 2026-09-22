using FFMPEG_jll

figure_path = "/home/hsinyi/figure/20260921_output_check2/dbry/bry_match/temp_east_bry";
movie_path = "/home/hsinyi/figure/20260921_output_check2/dbry/bry_match/movie";
start_string = "temp_east_bry"

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
end

make_movie(start_string)

#= butter_filters.jl
   Maarten Buijsman, USM, 2025-8-7
   This function file includes band, low, and highpass butter filters
=#
"""
    bandpass_butter(y,Tl,Th,dt,N)

Bandpass butterworth filter, returns filtered time series.
filtfilt is always applied along the first dimension of multi-dim. arrays.

# Arguments
y: time series that needs to be filtered.

Tl, Th: low and high cutoff periods. 

dt: delta-t.

N: order of the filter.

Make sure all the time variables have the same time units!   

# Example: 9-15 hour bandpass filter for 1 cph sampling (Δt=1hour)
yf = bandpass_butter(y,9,15,1,4)

# Info
Maarten Buijsman, USM, 2025-8-8
"""
function bandpass_butter(y,Tl,Th,dt,N);
    f1=1/Th  
    f2=1/Tl 
    fs=1/dt
    b = digitalfilter(Bandpass(f1, f2), Butterworth(N); fs=fs)

    return filtfilt(b, y)
end


"""
    lowhighpass_butter(y,Tcut,dt,N,fstring)

Low/high-pass butterworth filter, returns filtered time series.
filtfilt is always applied along the first dimension of multi-dim. arrays.

# Arguments
y: time series that needs to be filtered.

Tcut: low or high-pass cutoff period. 

dt: delta-t.

N: order of the filter.

fstring: "low" or "high"

Make sure all the time variables have the same time units!   

# Example: 9 hour low-bandpass filter for 1 cph sampling (Δt=1hour)
yf = lowhighpass_butter(y,9,1,4,"low")

# Info
Maarten Buijsman, USM, 2025-8-8
"""
function lowhighpass_butter(y,Tcut,dt,N,fstring);
    fcut=1/Tcut 
    fs=1/dt
    #println("fstring=",fstring)

    if fstring=="low"
        b = digitalfilter(Lowpass(fcut), Butterworth(N); fs=fs)
    elseif fstring=="high"
        b = digitalfilter(Highpass(fcut), Butterworth(N); fs=fs)        
    end

    return filtfilt(b, y)
end
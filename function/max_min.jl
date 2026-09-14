# max_min.jl
#
# Ported from max_min.m (useful_tools) — global min/max of an array.
# NaN values are skipped, matching MATLAB's default min(...,[],"all") /
# max(...,[],"all") behavior (Julia's minimum/maximum do NOT skip NaN
# by default, so this filters them out explicitly).

function max_min(variable)
    valid = filter(!isnan, variable)
    return [minimum(valid), maximum(valid)]
end

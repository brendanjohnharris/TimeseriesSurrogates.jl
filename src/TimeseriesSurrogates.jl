module TimeseriesSurrogates

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end TimeseriesSurrogates

using Random
using Distributions
using Distances # Will be used by the LombScargle method
using StatsBase
using AbstractFFTs
using DSP
using Interpolations
using Wavelets
using Requires
using StateSpaceSets
export standardize

include("core/api.jl")
include("core/surrogate_test.jl")

include("utils/testsystems.jl")

# The different surrogate routines
include("methods/randomshuffle.jl")
include("methods/large_shuffle.jl")
include("methods/randomfourier.jl")
include("methods/aaft.jl")
include("methods/iaaft.jl")
include("methods/truncated_fourier.jl")
include("methods/partial_randomization.jl")
include("methods/wavelet_based.jl")
include("methods/pseudoperiodic.jl")
include("methods/pseudoperiodic_twin.jl")
include("methods/multidimensional.jl")
include("methods/ar.jl")
include("methods/trend_based.jl")

# Methods for irregular time series
include("methods/lombscargle.jl")

# Visualization routine for time series + surrogate + periodogram/acf/histogram
using Requires
function __init__()
    @require UncertainData="dcd9ba68-c27b-5cea-ae21-829cd07325bf" begin
        include("utils/uncertaindatasets.jl")
    end
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        include("plotting/surrogate_plot.jl")
    end
end

export surroplot_path
function surroplot_path()
    joinpath(@__DIR__, "plotting", "surrogate_plot.jl")
end

end # module

using Random
export RandomShuffle

"""
    RandomShuffle() <: Surrogate

A random constrained surrogate, generated by shifting values around.

Random shuffle surrogates preserve the mean, variance and amplitude
distribution of the original signal. Properties not preserved are *any
temporal information*, such as the power spectrum and hence linear
correlations.

The null hypothesis this method can test for is whether the data
are uncorrelated noise, possibly measured via a nonlinear function.
Specifically, random shuffle surrogate can test
the null hypothesis that the original signal is produced by independent and
identically distributed random variables[^Theiler1991, ^Lancaster2018].

For a multidimensional input, the `dims` argument can be used to specify which dimensions to shuffle over (for example, `dims=[2, 3]` will randomly shuffle the columns of a 3D array across the second and third dimensions).

*Beware: random shuffle surrogates do not cover the case of correlated noise*[^Lancaster2018].

[^Theiler1991]: J. Theiler, S. Eubank, A. Longtin, B. Galdrikian, J. Farmer, Testing for nonlinearity in time series: The method of surrogate data, Physica D 58 (1–4) (1992) 77–94.
"""
struct RandomShuffle <: Surrogate; dims; end
RandomShuffle() = RandomShuffle(nothing)

function surrogenerator(x::AbstractArray, rf::RandomShuffle, rng = Random.default_rng())
    init = (
        dims = isnothing(rf.dims) ? (1:ndims(x)) : rf.dims,
    )

    return SurrogateGenerator(rf, x, similar(x), init, rng)
end

function (sg::SurrogateGenerator{<:RandomShuffle})()
    x, s, rng = sg.x, sg.s, sg.rng
    s .= deepcopy(x)
    dims  = getfield.(Ref(sg.init), (:dims))
    shuffleslices!(rng, s, dims)
    return s
end

function shuffleslices!(rng, s::AbstractArray, dims) # Is there something like this in base??
    tmp = deepcopy(s) # To remember the original order when iterating
    for d in dims
        ps = randperm(rng, size(s, d))
        if s isa AbstractVector
            s .= s[ps]
        else
            idxs = Base.compute_itspace(s, Val(d))
            for (r, i) in enumerate(idxs)
                s[i...] .= tmp[idxs[ps[r]]...]
            end
            tmp .= s
        end
    end
end
shuffleslices!(s::AbstractArray, args...) = shuffleslices!(Random.GLOBAL_RNG, s, args...)

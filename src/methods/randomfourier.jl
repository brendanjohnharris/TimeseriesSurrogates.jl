export RandomFourier, FT

"""
    RandomFourier(phases = true, dims = nothing)

A surrogate that randomizes the Fourier components
of the signal in some manner. If `phases==true`, the phases are randomized,
otherwise the amplitudes are randomized.
If `dims` is given, only the Fourier components in the corresponding dimensions are randomized.
`FT` is an alias for `RandomFourier`.

Random Fourier phase surrogates[^Theiler1991] preserve the
autocorrelation function, or power spectrum, of the original signal.
Random Fourier amplitude surrogates preserve the mean and autocorrelation
function but do not preserve the variance of the original. Random
amplitude surrogates are not common in the literature, but are provided
for convenience.

Random phase surrogates can be used to test the null hypothesis that
the original signal was produced by a linear Gaussian process [^Theiler1991].

[^Theiler1991]: J. Theiler, S. Eubank, A. Longtin, B. Galdrikian, J. Farmer, Testing for nonlinearity in time series: The method of surrogate data, Physica D 58 (1–4) (1992) 77–94.
"""
struct RandomFourier <: Surrogate
    phases::Bool
    dims
    fix::Bool
end
RandomFourier(phases::Bool=true, dims=nothing; fix=false) = RandomFourier(phases, dims, fix)
const FT = RandomFourier

function surrogenerator(x::AbstractArray, rf::RandomFourier, rng = Random.default_rng())
    dims = isnothing(rf.dims) ? (1:ndims(x)) : rf.dims
    dims = sort([dims...]; rev=true) # In case scalar. Also better to have last dimension (usually time) first
    any(.!in.(dims, (1:ndims(x),))) && error("FFT dimensions exceed array dimensions")
    forward = plan_rfft(x, dims)
    # The rfft discards negative frequencies only for the first dimension. This is no issue (?).
    inverse = plan_irfft(forward*x, size(x)[dims[1]], dims)
    m = nanmean(x; dims=(dims...,))
    if any(isnan, x) # In this case we replace NaN's with zeros in the mean-centered data, which is O.K. because zeroes don't contribute to the FT integral
        _x = deepcopy(x) .- m
        _x[isnan.(x)] .= 0.0
        𝓕 = forward*(_x)
    else
        𝓕 = forward*(x .- m)
    end
    shuffled𝓕 = zero(𝓕)
    s = similar(x)
    n = size(𝓕)
    r = abs.(𝓕)
    ϕ = angle.(𝓕)
    coeffs = zero(r)

    init = (inverse = inverse, m = m, coeffs = coeffs, n = n, r = r,
            ϕ = ϕ, shuffled𝓕 = shuffled𝓕, fix = rf.fix, dims = dims)
    return SurrogateGenerator(rf, x, s, init, rng)
end

function (sg::SurrogateGenerator{<:RandomFourier})()
    inverse, m, coeffs, n, r, ϕ, shuffled𝓕, fix, dims =
        getfield.(Ref(sg.init),
        (:inverse, :m, :coeffs, :n, :r, :ϕ, :shuffled𝓕, :fix, :dims))
    s, rng, phases = sg.s, sg.rng, sg.method.phases

    if fix && sort(dims) != 1:ndims(s) # Then we use the SAME randomised phases for each slice
        sz = size(coeffs)[dims]
        _coeffs = rand(rng, Uniform(0, 2π), sz)
        negdims = (Base._negdims(ndims(s), dims)...,)
        idxs = Base.compute_itspace(coeffs, Val(negdims))
        for i in idxs
            coeffs[i...] .= _coeffs
        end
    else
        coeffs .= rand(rng, Uniform(0, 2π), n)
    end

    if phases
        shuffled𝓕 .= r .* exp.(coeffs .* 1im)
    else
        coeffs .= r .* coeffs
        shuffled𝓕 .= coeffs .* exp.(ϕ .* 1im)
    end
    s .= inverse * shuffled𝓕 .+ m

    # if any(isnan, sg.x)
    #     s[isnan.(sg.x)] .= NaN
    # end

    return s
end

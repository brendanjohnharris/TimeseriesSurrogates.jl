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
end
RandomFourier() = RandomFourier(true, nothing)
RandomFourier(phases::Bool) = RandomFourier(phases, nothing)
const FT = RandomFourier

function surrogenerator(x::AbstractArray, rf::RandomFourier, rng = Random.default_rng())
    dims = isnothing(rf.dims) ? (1:ndims(x)) : rf.dims
    any(.!in.(dims, (1:ndims(x),))) && error("FFT dimensions exceed array dimensions")
    forward = plan_rfft(x, dims)
    inverse = plan_irfft(forward*x, size(x, 1), dims)
    m = mean(x; dims)
    𝓕 = forward*(x .- m)
    shuffled𝓕 = zero(𝓕)
    s = similar(x)
    n = size(𝓕)
    r = abs.(𝓕)
    ϕ = angle.(𝓕)
    coeffs = zero(r)

    init = (inverse = inverse, m = m, coeffs = coeffs, n = n, r = r,
            ϕ = ϕ, shuffled𝓕 = shuffled𝓕)
    return SurrogateGenerator(rf, x, s, init, rng)
end

function (sg::SurrogateGenerator{<:RandomFourier})()
    inverse, m, coeffs, n, r, ϕ, shuffled𝓕 =
        getfield.(Ref(sg.init),
        (:inverse, :m, :coeffs, :n, :r, :ϕ, :shuffled𝓕))
    s, rng, phases = sg.s, sg.rng, sg.method.phases

    if phases
        coeffs .= rand(rng, Uniform(0, 2π), n)
        shuffled𝓕 .= r .* exp.(coeffs .* 1im)
    else
        coeffs .= r .* rand(rng, Uniform(0, 2π), n)
        shuffled𝓕 .= coeffs .* exp.(ϕ .* 1im)
    end
    s .= inverse * shuffled𝓕 .+ m
    return s
end

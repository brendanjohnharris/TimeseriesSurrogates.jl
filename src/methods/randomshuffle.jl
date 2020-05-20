export RandomShuffle
"""
    RandomShuffle() <: Surrogate

A random constrained surrogate, generated by shifting values around.

Random shuffle surrogates preserve the mean, variance and amplitude 
distribution of the original signal. Properties not preserved are *any 
temporal information*, such as the power spectrum[^Lancaster2018] 
and hence linear correlations. 

The null hypothesis this method can test for is whether the data 
are uncorrelated noise, possibly measured via a nonlinear function.
Specifically, random shuffle surrogate can test 
the null hypothesis that the original signal is produced by independent and 
identically distributed random variables[^Theiler1991, ^Lancaster2018]. 

*Beware: random shuffle surrogates do not cover the case of correlated noise*[^Lancaster2018]. 

[^Theiler1991]: J. Theiler, S. Eubank, A. Longtin, B. Galdrikian, J. Farmer, Testing for nonlinearity in time series: The method of surrogate data, Physica D 58 (1–4) (1992) 77–94.
[^Lancaster2018]: Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. (2018). Surrogate data for hypothesis testing of physical systems. Physics Reports, 748, 1–60. doi:10.1016/j.physrep.2018.06.001
"""
struct RandomShuffle <: Surrogate end

function surrogenerator(x::AbstractVector, rf::RandomShuffle)
    return SurrogateGenerator(rf, x, nothing)
end

function (rf::SurrogateGenerator{<:RandomShuffle})()
    n = length(rf.x)
    rf.x[sample(1:n, n, replace = false)]
end

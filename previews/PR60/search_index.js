var documenterSearchIndex = {"docs":
[{"location":"constrained/randomshuffle/#Shuffle-based-1","page":"Shuffle-based","title":"Shuffle-based","text":"","category":"section"},{"location":"constrained/randomshuffle/#Random-shuffle-(RS)-1","page":"Shuffle-based","title":"Random shuffle (RS)","text":"","category":"section"},{"location":"constrained/randomshuffle/#","page":"Shuffle-based","title":"Shuffle-based","text":"Randomly shuffled surrogates are simply permutations of the original time series.","category":"page"},{"location":"constrained/randomshuffle/#","page":"Shuffle-based","title":"Shuffle-based","text":"Thus, they break any correlations in the signal.","category":"page"},{"location":"constrained/randomshuffle/#","page":"Shuffle-based","title":"Shuffle-based","text":"using TimeseriesSurrogates, Plots\nx = AR1() # create a realization of a random AR(1) process\nphases = true\ns = surrogate(x, RandomShuffle())\n\nsurroplot(x, s)","category":"page"},{"location":"constrained/randomshuffle/#Block-shuffle-(BS)-1","page":"Shuffle-based","title":"Block shuffle (BS)","text":"","category":"section"},{"location":"constrained/randomshuffle/#","page":"Shuffle-based","title":"Shuffle-based","text":"Randomly shuffled surrogates are generated by dividing the original signal into blocks, then permuting those blocks. Block positions are randomized, and blocks at the end of the signal gets wrapped around to the start of the time series.","category":"page"},{"location":"constrained/randomshuffle/#","page":"Shuffle-based","title":"Shuffle-based","text":"Thus, they keep short-term correlations within blocks, but destroy any long-term dynamical information in the signal.","category":"page"},{"location":"constrained/randomshuffle/#","page":"Shuffle-based","title":"Shuffle-based","text":"using TimeseriesSurrogates, Plots\nx = NSAR2(n_steps = 300)\n\n# We want to divide the signal into 8 blocks.\ns = surrogate(x, BlockShuffle(8))\np = surroplot(x, s)\nsavefig(p, \"../surroplot.png\") # hide\np","category":"page"},{"location":"constrained/truncated_fourier_transform/#Truncated-FT/AAFT-surrogates-1","page":"Truncated FT/AAFT","title":"Truncated FT/AAFT surrogates","text":"","category":"section"},{"location":"constrained/truncated_fourier_transform/#TFTS-1","page":"Truncated FT/AAFT","title":"TFTS","text":"","category":"section"},{"location":"constrained/truncated_fourier_transform/#","page":"Truncated FT/AAFT","title":"Truncated FT/AAFT","text":"Truncated Fourier transform surrogates preserve some portion of the frequency spectrum of  the original signal. Here, we randomize the 95% highest frequencies, while keeping the  5% lowermost frequencies intact.","category":"page"},{"location":"constrained/truncated_fourier_transform/#","page":"Truncated FT/AAFT","title":"Truncated FT/AAFT","text":"using TimeseriesSurrogates\nn = 300\na = 0.7\nA = 20\nσ = 15\nx = cumsum(randn(n)) .+ [(1 + a*i) .+ A*sin(2π/10*i) for i = 1:n] .+ \n    [A^2*sin(2π/2*i + π) for i = 1:n] .+ σ .* rand(n).^2;\n\n\nfϵ = 0.05\ns_tfts = surrogate(x, TFTS(fϵ))\nsurroplot(x, s_tfts)","category":"page"},{"location":"constrained/truncated_fourier_transform/#","page":"Truncated FT/AAFT","title":"Truncated FT/AAFT","text":"One may also choose to preserve the opposite end of the frequency spectrum. Below,  we randomize the 20% lowermost frequencies, while keeping the 80% highest frequencies intact.","category":"page"},{"location":"constrained/truncated_fourier_transform/#","page":"Truncated FT/AAFT","title":"Truncated FT/AAFT","text":"using TimeseriesSurrogates\nn = 300\na = 0.7\nA = 20\nσ = 15\nx = cumsum(randn(n)) .+ [(1 + a*i) .+ A*sin(2π/10*i) for i = 1:n] .+ \n    [A^2*sin(2π/2*i + π) for i = 1:n] .+ σ .* rand(n).^2;\n\nfϵ = -0.2\ns_tfts = surrogate(x, TFTS(fϵ))\nsurroplot(x, s_tfts)","category":"page"},{"location":"constrained/truncated_fourier_transform/#TAAFT-1","page":"Truncated FT/AAFT","title":"TAAFT","text":"","category":"section"},{"location":"constrained/truncated_fourier_transform/#","page":"Truncated FT/AAFT","title":"Truncated FT/AAFT","text":"Truncated AAFT surrogates are similar to TFTS surrogates, but adds the extra step of rescaling back  to the original values of the signal, so that the original signal and the surrogates consists of  the same values. ","category":"page"},{"location":"constrained/truncated_fourier_transform/#","page":"Truncated FT/AAFT","title":"Truncated FT/AAFT","text":"using TimeseriesSurrogates\nn = 300\na = 0.7\nA = 20\nσ = 15\nx = cumsum(randn(n)) .+ [(1 + a*i) .+ A*sin(2π/10*i) for i = 1:n] .+ \n    [A^2*sin(2π/2*i + π) for i = 1:n] .+ σ .* rand(n).^2;\n\n\nfϵ = 0.05\ns_tfts = surrogate(x, TFTS(fϵ))\nsurroplot(x, s_tfts)","category":"page"},{"location":"constrained/truncated_fourier_transform/#","page":"Truncated FT/AAFT","title":"Truncated FT/AAFT","text":"using TimeseriesSurrogates\nn = 300\na = 0.7\nA = 20\nσ = 15\nx = cumsum(randn(n)) .+ [(1 + a*i) .+ A*sin(2π/10*i) for i = 1:n] .+ \n    [A^2*sin(2π/2*i + π) for i = 1:n] .+ σ .* rand(n).^2;\n\nfϵ = -0.2\ns_tfts = surrogate(x, TFTS(fϵ))\nsurroplot(x, s_tfts)","category":"page"},{"location":"constrained/pps/#Pseudo-periodic-1","page":"Pseudo-periodic","title":"Pseudo-periodic","text":"","category":"section"},{"location":"constrained/pps/#","page":"Pseudo-periodic","title":"Pseudo-periodic","text":"using TimeseriesSurrogates, Plots\nt = 0:0.05:20π\nx = @. 4 + 7cos(t) + 2cos(2t + 5π/4)\nx .+= randn(length(x))*0.2\n\n# Optimal d, τ values deduced using DynamicalSystems.jl\nd, τ = 3, 31\n\n# For ρ you can use `noiseradius`\nρ = 0.11\n\nmethod = PseudoPeriodic(d, τ, ρ, false)\ns = surrogate(x, method)\nsurroplot(x, s)","category":"page"},{"location":"constrained/wls/#Wavelet-surrogates-1","page":"Wavelet-based","title":"Wavelet surrogates","text":"","category":"section"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"Wavelet surrogates are constructed by taking the maximal overlap  discrete wavelet transform (MODWT) of the signal, shuffling detail  coefficients across dyadic scales, then inverting the transform to  obtain the surrogate. ","category":"page"},{"location":"constrained/wls/#IAAFT-shuffling-detail-coefficients-1","page":"Wavelet-based","title":"IAAFT shuffling detail coefficients","text":"","category":"section"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"In Keylock (2006),  IAAAFT shuffling is used, yielding surrogates that preserve the local mean and  variance of the original signal, but randomizes nonlinear properties of the signal. This also preserves nonstationarities in the signal.","category":"page"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"using TimeseriesSurrogates, Random\nRandom.seed!(5040)\nn = 500\nσ = 30\nx = cumsum(randn(n)) .+ \n    [20*sin(2π/30*i) for i = 1:n] .+ \n    [20*cos(2π/90*i) for i = 1:n] .+\n    [50*sin(2π/2*i + π) for i = 1:n] .+ \n    σ .* rand(n).^2 .+ \n    [0.5*t for t = 1:n];\n\n# Rescale surrogate back to original values\nmethod = WLS(IAAFT(), true)\ns = surrogate(x, method);\np = surroplot(x, s)","category":"page"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"Even without rescaling, IAAFT shuffling also yields surrogates with local properties  very similar to the original signal.","category":"page"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"using TimeseriesSurrogates, Random\nRandom.seed!(5040)\nn = 500\nσ = 30\nx = cumsum(randn(n)) .+ \n    [20*sin(2π/30*i) for i = 1:n] .+ \n    [20*cos(2π/90*i) for i = 1:n] .+\n    [50*sin(2π/2*i + π) for i = 1:n] .+ \n    σ .* rand(n).^2 .+ \n    [0.5*t for t = 1:n];\n\n# Don't rescale back to original time series.\nmethod = WLS(IAAFT(), false)\ns = surrogate(x, method);\np = surroplot(x, s)","category":"page"},{"location":"constrained/wls/#Other-shuffling-methods-1","page":"Wavelet-based","title":"Other shuffling methods","text":"","category":"section"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"The choice of coefficient shuffling method determines how well and  which properties of the original signal are retained by the surrogates.  There might be use cases where surrogates do not need to perfectly preserve the  autocorrelation of the original signal, so additional shuffling  methods are provided for convenience.","category":"page"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"Using random shuffling of the detail coefficients does not preserve the  autocorrelation structure of the original signal. ","category":"page"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"using TimeseriesSurrogates, Random\nRandom.seed!(5040)\nn = 500\nσ = 30\nx = cumsum(randn(n)) .+ \n    [20*sin(2π/30*i) for i = 1:n] .+ \n    [20*cos(2π/90*i) for i = 1:n] .+\n    [50*sin(2π/2*i + π) for i = 1:n] .+ \n    σ .* rand(n).^2 .+ \n    [0.5*t for t = 1:n];\n\nmethod = WLS(RandomShuffle())\ns = surrogate(x, method);\np = surroplot(x, s)","category":"page"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"Block shuffling the detail coefficients better preserve local properties because the shuffling is not completely random, but still does not  preserve the autocorrelation of the original signal.","category":"page"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"using TimeseriesSurrogates, Random\nRandom.seed!(5040)\nn = 500\nσ = 30\nx = cumsum(randn(n)) .+ \n    [20*sin(2π/30*i) for i = 1:n] .+ \n    [20*cos(2π/90*i) for i = 1:n] .+\n    [50*sin(2π/2*i + π) for i = 1:n] .+ \n    σ .* rand(n).^2 .+ \n    [0.5*t for t = 1:n];\n\ns = surrogate(x, WLS(BlockShuffle(10)));\np = surroplot(x, s)","category":"page"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"Random Fourier phase shuffling the detail coefficients does a decent job at preserving the autocorrelation.","category":"page"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"using TimeseriesSurrogates, Random\nRandom.seed!(5040)\nn = 500\nσ = 30\nx = cumsum(randn(n)) .+ \n    [20*sin(2π/30*i) for i = 1:n] .+ \n    [20*cos(2π/90*i) for i = 1:n] .+\n    [50*sin(2π/2*i + π) for i = 1:n] .+ \n    σ .* rand(n).^2 .+ \n    [0.5*t for t = 1:n];\n\ns = surrogate(x, WLS(RandomFourier()));\nsurroplot(x, s)","category":"page"},{"location":"constrained/wls/#","page":"Wavelet-based","title":"Wavelet-based","text":"To generate surrogates that preserve linear properties of the original signal, AAFT or IAAFT shuffling is required.","category":"page"},{"location":"man/exampleprocesses/#Example-processes-1","page":"Utility systems","title":"Example processes","text":"","category":"section"},{"location":"man/exampleprocesses/#","page":"Utility systems","title":"Utility systems","text":"SNLST\nrandomwalk\nNSAR2\nAR1","category":"page"},{"location":"man/exampleprocesses/#TimeseriesSurrogates.SNLST","page":"Utility systems","title":"TimeseriesSurrogates.SNLST","text":"SNLST(n_steps, x₀, k)\n\nDynamically linear process transformed by a strongly nonlinear static transformation (SNLST)[1].\n\nEquations\n\nThe system is by the following map:\n\nx(t) = k x(t-1) + a(t)\n\nwith the transformation s(t) = x(t)^3.\n\nReferences\n\n[1]: Lucio et al., Phys. Rev. E 85, 056202 (2012). https://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.056202\n\n\n\n\n\n","category":"function"},{"location":"man/exampleprocesses/#TimeseriesSurrogates.randomwalk","page":"Utility systems","title":"TimeseriesSurrogates.randomwalk","text":"randomwalk(n_steps, x₀)\n\nLinear random walk (AR(1) process with a unit root)[1]. This is an example of a nonstationary linear process.\n\nReferences\n\n[1]: Lucio et al., Phys. Rev. E 85, 056202 (2012). https://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.056202\n\n\n\n\n\n","category":"function"},{"location":"man/exampleprocesses/#TimeseriesSurrogates.NSAR2","page":"Utility systems","title":"TimeseriesSurrogates.NSAR2","text":"NSAR2(n_steps, x₀, x₁)\n\nCyclostationary AR(2) process[1].\n\nReferences\n\n[1]: Lucio et al., Phys. Rev. E 85, 056202 (2012). https://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.056202\n\n\n\n\n\n","category":"function"},{"location":"man/exampleprocesses/#TimeseriesSurrogates.AR1","page":"Utility systems","title":"TimeseriesSurrogates.AR1","text":"AR1(n_steps, x₀, k)\n\nSimple AR(1) model with no static transformation[1].\n\nEquations\n\nThe system is given by the following map:\n\nx(t+1) = k x(t) + a(t)\n\nwhere a(t) is a draw from a normal distribution with zero mean and unit variance. x₀ sets the initial condition and k is the tunable parameter in the map.\n\nReferences\n\n[1]: Lucio et al., Phys. Rev. E 85, 056202 (2012). https://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.056202\n\n\n\n\n\n","category":"function"},{"location":"constrained/fourier_surrogates/#Fourier-based-1","page":"Fourier-based","title":"Fourier-based","text":"","category":"section"},{"location":"constrained/fourier_surrogates/#","page":"Fourier-based","title":"Fourier-based","text":"Fourier based surrogates are a form of constrained surrogates created by taking the Fourier transform of a time series, then shuffling either the phase angles or the amplitudes of the resulting complex numbers. Then, we take the inverse Fourier transform, yielding a surrogate time series.","category":"page"},{"location":"constrained/fourier_surrogates/#Random-phase-1","page":"Fourier-based","title":"Random phase","text":"","category":"section"},{"location":"constrained/fourier_surrogates/#","page":"Fourier-based","title":"Fourier-based","text":"using TimeseriesSurrogates, Plots\nts = AR1() # create a realization of a random AR(1) process\nphases = true\ns = surrogate(ts, RandomFourier(phases))\n\nsurroplot(ts, s)","category":"page"},{"location":"constrained/fourier_surrogates/#Random-amplitude-1","page":"Fourier-based","title":"Random amplitude","text":"","category":"section"},{"location":"constrained/fourier_surrogates/#","page":"Fourier-based","title":"Fourier-based","text":"using TimeseriesSurrogates, Plots\nts = AR1() # create a realization of a random AR(1) process\nphases = false\ns = surrogate(ts, RandomFourier(phases))\n\nsurroplot(ts, s)","category":"page"},{"location":"constrained/amplitude_adjusted/#Amplitude-adjusted-Fourier-transform-surrogates-1","page":"Amplitude-adjusted FT","title":"Amplitude adjusted Fourier transform surrogates","text":"","category":"section"},{"location":"constrained/amplitude_adjusted/#AAFT-1","page":"Amplitude-adjusted FT","title":"AAFT","text":"","category":"section"},{"location":"constrained/amplitude_adjusted/#","page":"Amplitude-adjusted FT","title":"Amplitude-adjusted FT","text":"using TimeseriesSurrogates, Plots\nts = AR1() # create a realization of a random AR(1) process\ns = surrogate(ts, AAFT())\n\nsurroplot(ts, s)","category":"page"},{"location":"constrained/amplitude_adjusted/#Iterative-AAFT-(IAAFT)-1","page":"Amplitude-adjusted FT","title":"Iterative AAFT (IAAFT)","text":"","category":"section"},{"location":"constrained/amplitude_adjusted/#","page":"Amplitude-adjusted FT","title":"Amplitude-adjusted FT","text":"The IAAFT surrogates add an iterative step to the AAFT algorithm improve convergence.","category":"page"},{"location":"constrained/amplitude_adjusted/#","page":"Amplitude-adjusted FT","title":"Amplitude-adjusted FT","text":"using TimeseriesSurrogates, Plots\nts = AR1() # create a realization of a random AR(1) process\ns = surrogate(ts, IAAFT())\n\nsurroplot(ts, s)","category":"page"},{"location":"#","page":"Documentation","title":"Documentation","text":"(Image: )","category":"page"},{"location":"#","page":"Documentation","title":"Documentation","text":"If you are new to this method of surrogate time series, feel free to read the What is a surrogate? page.","category":"page"},{"location":"#API-1","page":"Documentation","title":"API","text":"","category":"section"},{"location":"#","page":"Documentation","title":"Documentation","text":"TimeseriesSurrogates.jl exports two main functions. Both of them dispatch on the chosen method, a subtype of Surrogate.","category":"page"},{"location":"#","page":"Documentation","title":"Documentation","text":"surrogate\nsurrogenerator","category":"page"},{"location":"#TimeseriesSurrogates.surrogate","page":"Documentation","title":"TimeseriesSurrogates.surrogate","text":"surrogate(x, method::Surrogate) → s\n\nCreate a single surrogate timeseries s from x based on the given method. If you want to generate more than one surrogates from x, you should use surrogenerator.\n\n\n\n\n\n","category":"function"},{"location":"#TimeseriesSurrogates.surrogenerator","page":"Documentation","title":"TimeseriesSurrogates.surrogenerator","text":"surrogenerator(x, method::Surrogate) → sg::SurrogateGenerator\n\nInitialize a generator that creates surrogates of x on demand, based on given method. This is efficient, because for most methods some things can be initialized and reused for every surrogate.\n\nTo generate a surrogate, call sg as a function with no arguments, e.g.:\n\nsg = surrogenerator(x, method)\nfor i in 1:1000\n    s = sg()\n    # do stuff with s and or x\n    result[i] = stuff\nend\n\n\n\n\n\n","category":"function"},{"location":"#Surrogate-methods-1","page":"Documentation","title":"Surrogate methods","text":"","category":"section"},{"location":"#","page":"Documentation","title":"Documentation","text":"RandomShuffle\nBlockShuffle\nRandomFourier\nTFTS\nAAFT\nTAAFT\nIAAFT\nPseudoPeriodic\nWLS","category":"page"},{"location":"#TimeseriesSurrogates.RandomShuffle","page":"Documentation","title":"TimeseriesSurrogates.RandomShuffle","text":"RandomShuffle() <: Surrogate\n\nA random constrained surrogate, generated by shifting values around.\n\nRandom shuffle surrogates preserve the mean, variance and amplitude  distribution of the original signal. Properties not preserved are any  temporal information, such as the power spectrum and hence linear  correlations. \n\nThe null hypothesis this method can test for is whether the data  are uncorrelated noise, possibly measured via a nonlinear function. Specifically, random shuffle surrogate can test  the null hypothesis that the original signal is produced by independent and  identically distributed random variables[^Theiler1991, ^Lancaster2018]. \n\nBeware: random shuffle surrogates do not cover the case of correlated noise[Lancaster2018]. \n\n[Theiler1991]: J. Theiler, S. Eubank, A. Longtin, B. Galdrikian, J. Farmer, Testing for nonlinearity in time series: The method of surrogate data, Physica D 58 (1–4) (1992) 77–94.\n\n\n\n\n\n","category":"type"},{"location":"#TimeseriesSurrogates.BlockShuffle","page":"Documentation","title":"TimeseriesSurrogates.BlockShuffle","text":"BlockShuffle(n::Int) <: Surrogate\n\nA block shuffle surrogate constructed by dividing the time series into n blocks of roughly equal width at random indices (end blocks are wrapped around to the start of the time series).\n\nBlock shuffle surrogates roughly preserve short-range temporal properties  in the time series (e.g. correlations at lags less than the block length),  but break any long-term dynamical information (e.g. correlations beyond  the block length).\n\nHence, these surrogates can be used to test any null hypothesis aimed at  comparing short-range dynamical properties versus long-range dynamical  properties of the signal.\n\n\n\n\n\n","category":"type"},{"location":"#TimeseriesSurrogates.RandomFourier","page":"Documentation","title":"TimeseriesSurrogates.RandomFourier","text":"RandomFourier(phases = true) <: Surrogate\n\nA surrogate that randomizes the Fourier components of the signal in some manner. If phases==true, the phases are randomized, otherwise the amplitudes are randomized. \n\nRandom Fourier phase surrogates[Theiler1991] preserve the  autocorrelation function, or power spectrum, of the original signal.  Random Fourier amplitude surrogates preserve the mean and autocorrelation  function but do not preserve the variance of the original. Random  amplitude surrogates are not common in the literature, but are provided  for convenience.\n\nRandom phase surrogates can be used to test the null hypothesis that  the original signal was produced by a linear Gaussian process [Theiler1991]. \n\n[Theiler1991]: J. Theiler, S. Eubank, A. Longtin, B. Galdrikian, J. Farmer, Testing for nonlinearity in time series: The method of surrogate data, Physica D 58 (1–4) (1992) 77–94.\n\n\n\n\n\n","category":"type"},{"location":"#TimeseriesSurrogates.TFTS","page":"Documentation","title":"TimeseriesSurrogates.TFTS","text":"TFTS(fϵ::Real)\n\nA truncated Fourier transform surrogate[Nakamura2006] (TFTS). \n\nTFTS surrogates are generated by leaving some frequencies untouched when performing the  phase shuffling step (as opposed to randomizing all frequencies, like for RandomFourier surrogates). \n\nThese surrogates were designed to deal with data with irregular fluctuations superimposed  over long term trends (by preserving low frequencies)[Nakamura2006]. Hence, TFTS surrogates  can be used to test the null hypothesis that the signal is a stationary linear system  generated the irregular fluctuations part of the signal[Nakamura2006].\n\nControlling the truncation of the spectrum\n\nThe truncation parameter fϵ ∈ [-1, 0) ∪ (0, 1] controls which parts of the spectrum are preserved.\n\nIf fϵ > 0, then fϵ indicates the ratio of high frequency domain to the entire frequency domain.   For example, fϵ = 0.5 preserves 50% of the frequency domain (randomizing the higher    frequencies, leaving low frequencies intact).\nIf fϵ < 0, then fϵ indicates ratio of low frequency domain to the entire frequency domain.   For example, fϵ = -0.2 preserves 20% of the frequency domain (leaving higher frequencies intact,   randomizing the lower frequencies).\nIf fϵ ± 1, then all frequencies are randomized. The method is then equivalent to    RandomFourier.\n\nThe appropriate value of fϵ strongly depends on the data and time series length, and must be  manually determined[Nakamura2006], for example by comparing periodograms for the time series and  the surrogates.\n\n[Nakamura2006]: Nakamura, Tomomichi, Michael Small, and Yoshito Hirata. \"Testing for nonlinearity in irregular fluctuations with long-term trends.\" Physical Review E 74.2 (2006): 026205.\n\n\n\n\n\n","category":"type"},{"location":"#TimeseriesSurrogates.AAFT","page":"Documentation","title":"TimeseriesSurrogates.AAFT","text":"AAFT()\n\nAn amplitude-adjusted-fourier-transform surrogate[Theiler1991].\n\nAAFT have the same linear correlation, or periodogram, and also preserves the amplitude distribution of the original data.\n\nAAFT can be used to test the null hypothesis that the data come from a monotonic nonlinear transformation of a linear Gaussian process (also called integrated white noise)[Theiler1991].\n\n[Theiler1991]: J. Theiler, S. Eubank, A. Longtin, B. Galdrikian, J. Farmer, Testing for nonlinearity in time series: The method of surrogate data, Physica D 58 (1–4) (1992) 77–94.\n\n\n\n\n\n","category":"type"},{"location":"#TimeseriesSurrogates.TAAFT","page":"Documentation","title":"TimeseriesSurrogates.TAAFT","text":"TAAFT(fϵ)\n\nAn truncated version of the amplitude-adjusted-fourier-transform surrogate[Theiler1991][Nakamura2006].\n\nThe truncation parameter and phase randomization procedure is identical to TFTS, but here an  additional step of rescaling back to the original data is performed. This preserves the  amplitude distribution of the original data.\n\nReferences\n\n[Theiler1991]: J. Theiler, S. Eubank, A. Longtin, B. Galdrikian, J. Farmer, Testing for nonlinearity in time series: The method of surrogate data, Physica D 58 (1–4) (1992) 77–94.\n\n[Nakamura2006]: Nakamura, Tomomichi, Michael Small, and Yoshito Hirata. \"Testing for nonlinearity in irregular fluctuations with long-term trends.\" Physical Review E 74.2 (2006): 026205.\n\n\n\n\n\n","category":"type"},{"location":"#TimeseriesSurrogates.IAAFT","page":"Documentation","title":"TimeseriesSurrogates.IAAFT","text":"IAAFT(M = 100, tol = 1e-6, W = 75)\n\nAn iteratively adjusted amplitude-adjusted-fourier-transform surrogate[SchreiberSchmitz1996].\n\nIAAFT surrogate have the same linear correlation, or periodogram, and also preserves the amplitude distribution of the original data, but are improved relative to AAFT through iterative adjustment (which runs for a maximum of M steps). During the iterative adjustment, the periodograms of the original signal and the surrogate are coarse-grained and the powers are averaged over W equal-width frequency bins. The iteration procedure ends when the relative deviation between the periodograms is less than tol (or when M is reached).\n\nIAAFT, just as AAFT, can be used to test the null hypothesis that the data  come from a monotonic nonlinear transformation of a linear Gaussian process.\n\n[SchreiberSchmitz1996]: T. Schreiber; A. Schmitz (1996). \"Improved Surrogate Data for Nonlinearity Tests\". Phys. Rev. Lett. 77 (4)\n\n\n\n\n\n","category":"type"},{"location":"#TimeseriesSurrogates.PseudoPeriodic","page":"Documentation","title":"TimeseriesSurrogates.PseudoPeriodic","text":"PseudoPeriodic(d, τ, ρ, shift=true) <: Surrogate\n\nCreate surrogates suitable for pseudo-periodic signals. They retain the periodic structure of the signal, while inter-cycle dynamics that are either deterministic or correlated noise are destroyed (for appropriate ρ choice). Therefore these surrogates are suitable to test the null hypothesis that the signal is periodic with uncorrelated noise[Small2001].\n\nArguments d, τ, ρ are as in the paper, the embedding dimension, delay time and noise radius. The method works by performing a delay coordinates ambedding via the library DynamicalSystems.jl. See its documentation for choosing appropriate values for d, τ. For ρ, we have implemented the method proposed in the paper in the function noiseradius.\n\nThe argument shift is not discussed in the paper, but it is possible to adjust the algorithm so that there is little phase shift between the periodic component of the original and surrogate data.\n\n[Small2001]: Small et al., Surrogate test for pseudoperiodic time series data, Physical Review Letters, 87(18)\n\n\n\n\n\n","category":"type"},{"location":"#TimeseriesSurrogates.WLS","page":"Documentation","title":"TimeseriesSurrogates.WLS","text":"WLS(surromethod::Surrogate = IAAFT(), \n    rescale::Bool = true,\n    wt::Wavelets.WT.OrthoWaveletClass = Wavelets.WT.Daubechies{16}())\n\nA wavelet surrogate generated by taking the maximal overlap discrete  wavelet transform (MODWT) of the signal, shuffling detail  coefficients at each dyadic scale using the provided surromethod, then taking the inverse transform to obtain a surrogate.\n\nCoefficient shuffling method\n\nIn contrast to the original  implementation where IAAFT is used, you may choose to use any surrogate  method from this package to perform the randomization of the detail  coefficients at each dyadic scale. Note: The iterative procedure after  the rank ordering step (step [v] in [Keylock2006]) is not performed in  this implementation.\n\nIf surromethod == IAAFT(), the wavelet surrogates preserves the local  mean and variance structure of the signal, but randomises nonlinear  properties of the signal (i.e. Hurst exponents)[Keylock2006]. These surrogates can therefore be used to test for changes in nonlinear properties of the  original signal.\n\nIn contrast to IAAFT surrogates, the IAAFT-wavelet surrogates also  preserves nonstationarity. Using other surromethods does not necessarily preserve nonstationarity.\n\nTo deal with nonstationary signals, Keylock (2006) recommends using a  wavelet with a high number of vanishing moments. Thus, the default is to use a Daubechies wavelet with 16 vanishing moments.\n\nRescaling\n\nIf rescale == true, then surrogate values are mapped onto the  values of the original time series, as in the AAFT algorithm. If rescale == false, surrogate values are not constrained to the  original time series values.\n\n[Keylock2006]: C.J. Keylock (2006). \"Constrained surrogate time series with preservation of the mean and variance structure\". Phys. Rev. E. 73: 036707. doi:10.1103/PhysRevE.73.036707.\n\n\n\n\n\n","category":"type"},{"location":"#Utils-1","page":"Documentation","title":"Utils","text":"","category":"section"},{"location":"#","page":"Documentation","title":"Documentation","text":"noiseradius","category":"page"},{"location":"#TimeseriesSurrogates.noiseradius","page":"Documentation","title":"TimeseriesSurrogates.noiseradius","text":"noiseradius(x::AbstractVector, d::Int, τ, ρs, n = 1) → ρ\n\nUse the proposed* algorithm of[Small2001] to estimate optimal ρ value for PseudoPeriodic surrogates, where ρs is a vector of possible ρ values.\n\n*The paper is ambiguous about exactly what to calculate. Here we count how many times we have pairs of length-2 that are identical in x and its surrogate, but are not also part of pairs of length-3.\n\nThis function directly returns the arg-maximum of the evaluated distribution of these counts versus ρ, use TimeseriesSurrogates._noiseradius with same arguments to get the actual distribution. n means to repeat τhe evaluation n times, which increases accuracy.\n\n[Small2001]: Small et al., Surrogate test for pseudoperiodic time series data, Physical Review Letters, 87(18)\n\n\n\n\n\n","category":"function"},{"location":"#Visualization-1","page":"Documentation","title":"Visualization","text":"","category":"section"},{"location":"#","page":"Documentation","title":"Documentation","text":"TimeseriesSurrogates.jl provides the function surroplot(x, s), which comes into scope when using Plots. This function is used in the example applications.","category":"page"},{"location":"man/whatisasurrogate/#What-is-a-surrogate?-1","page":"What is a surrogate?","title":"What is a surrogate?","text":"","category":"section"},{"location":"man/whatisasurrogate/#The-method-of-surrogate-testing-1","page":"What is a surrogate?","title":"The method of surrogate testing","text":"","category":"section"},{"location":"man/whatisasurrogate/#","page":"What is a surrogate?","title":"What is a surrogate?","text":"The method of surrogate testing is a statistical method for testing whether a given input timeseries x satisfies a specific hypothesis or not. Surrogate testing can be used to test, for example, whether a timeseries that appears noisy represents a nonlinear dynamical system, or it instead comes from a purely stochastic and uncorrelated process. For the suitable hypothesis to test for, see the documentation strings of provided <: Surrogate methods.","category":"page"},{"location":"man/whatisasurrogate/#","page":"What is a surrogate?","title":"What is a surrogate?","text":"The actual hypothesis testing is done by computing an appropriate discriminatory statistic for the input timeseries and the surrogates. If the statistic differs greatly between surrogate and input, then the formulated hypothesis can be rejected. For an overview of surrogate methods and the hypotheses they can test, see the review from Lancaster et al. (2018)[Lancaster2018].","category":"page"},{"location":"man/whatisasurrogate/#","page":"What is a surrogate?","title":"What is a surrogate?","text":"Notice that of course another application of surrogate timeseries is to simply generate more timeseries with similar properties as x.","category":"page"},{"location":"man/whatisasurrogate/#What-is-a-surrogate-time-series?-1","page":"What is a surrogate?","title":"What is a surrogate time series?","text":"","category":"section"},{"location":"man/whatisasurrogate/#","page":"What is a surrogate?","title":"What is a surrogate?","text":"Let's say we have a nontrivial timeseries x consisting of n observations. A surrogate time series for x is another timeseries s of n values which (roughly) preserves one or many mathematical/statistical properties of x.","category":"page"},{"location":"man/whatisasurrogate/#","page":"What is a surrogate?","title":"What is a surrogate?","text":"The upper panel in the figure below shows an example of a time series and one surrogate realization that preserves its autocorrelation. The time series \"look alike\", which is due to the fact the surrogate realization almost exactly preserved the power spectrum and autocorrelation of the time series, as shown in the lower panels.","category":"page"},{"location":"man/whatisasurrogate/#","page":"What is a surrogate?","title":"What is a surrogate?","text":"using TimeseriesSurrogates, Plots\nx = LinRange(0, 20π, 300) .+ 0.05 .* rand(300)\nts = sin.(x./rand(20:30, 300) + cos.(x))\ns = surrogate(ts, IAAFT())\n\nsurroplot(ts, s)","category":"page"},{"location":"man/whatisasurrogate/#","page":"What is a surrogate?","title":"What is a surrogate?","text":"[Lancaster2018]: Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. (2018). Surrogate data for hypothesis testing of physical systems. Physics Reports, 748, 1–60. doi:10.1016/j.physrep.2018.06.001","category":"page"}]
}

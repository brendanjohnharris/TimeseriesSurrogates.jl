# Fourier-based

Fourier based surrogates are a form of constrained surrogates created by taking the Fourier
transform of a time series, then shuffling either the phase angles or the amplitudes of the resulting complex numbers. Then, we take the inverse Fourier transform, yielding a surrogate time series.

## Random phase

```@example MAIN
using TimeseriesSurrogates, CairoMakie
ts = AR1() # create a realization of a random AR(1) process
phases = true
s = surrogate(ts, RandomFourier(phases))

surroplot(ts, s)
```

## Random amplitude

```@example MAIN
using TimeseriesSurrogates, CairoMakie
ts = AR1() # create a realization of a random AR(1) process
phases = false
s = surrogate(ts, RandomFourier(phases))

surroplot(ts, s)
```


 ## Partial randomization

 ### Without rescaling

 [`PartialRandomization`](@ref) surrogates are similar to random phase surrogates,
 but allows for tuning the "degree" of phase randomization.

```@example MAIN
using TimeseriesSurrogates, CairoMakie
ts = AR1() # create a realization of a random AR(1) process

# 50 % randomization of the phases
s = surrogate(ts, PartialRandomization(0.5))

surroplot(ts, s)
```

### With rescaling

[`PartialRandomizationAAFT`](@ref) adds a rescaling step to the [`PartialRandomization`](@ref) surrogates to obtain surrogates that contain the same values as the original time
series.

```@example MAIN
using TimeseriesSurrogates, CairoMakie
ts = AR1() # create a realization of a random AR(1) process

# 50 % randomization of the phases
s = surrogate(ts, PartialRandomizationAAFT(0.7))

surroplot(ts, s)
```

## Amplitude adjusted Fourier transform (AAFT)


```@example MAIN
using TimeseriesSurrogates, CairoMakie
ts = AR1() # create a realization of a random AR(1) process
s = surrogate(ts, AAFT())

surroplot(ts, s)
```

## Iterative AAFT (IAAFT)

The IAAFT surrogates add an iterative step to the AAFT algorithm to improve similarity
of the power spectra of the original time series and the surrogates.

```@example MAIN
using TimeseriesSurrogates, CairoMakie
ts = AR1() # create a realization of a random AR(1) process
s = surrogate(ts, IAAFT())

surroplot(ts, s)
```

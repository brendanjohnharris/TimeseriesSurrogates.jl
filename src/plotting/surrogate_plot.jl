using TimeseriesSurrogates.StatsBase
using TimeseriesSurrogates.DSP
# import Makie.KernelDensity.kde as kde

"""
    surroplot(x::AbstractVector, s; kwargs...)

Plot a timeseries `x` along with its surrogate realization `s`, and compare the
periodogram and histogram of the two time series.

## Keyword arguments
- `cx` and `cs`: Colors of the original and the surrogate time series, respectively.
- `nbins`: The number of bins for the histograms.
- `resolution`: A tuple giving the resolution of the figure.
"""
function surroplot(x::T, s::T;
        cx = "#1B1B1B", cs = ("#2DB9C5", 0.9), resolution = (500, 600),
        nbins = 50,
    ) where T <: AbstractVector

    t = 1:length(x)
    fig = Makie.Figure(resolution = resolution)

    # Time series
    ax1, _ = Makie.lines(fig[1,1], t, x; color = cx)
    Makie.lines!(ax1, t, s; color = cs)
    # Autocorrelation
    acx = autocor(x)
    ax2, _ = Makie.lines(fig[2,1], 0:length(acx)-1, acx; color = cx)
    Makie.lines!(ax2, 0:length(acx)-1, autocor(s); color = cs)

    # Binned multitaper periodograms
    p, psurr = DSP.mt_pgram(x), DSP.mt_pgram(s)
    ax3 = Makie.Axis(fig[3,1]; yscale = log10)
    Makie.lines!(ax3, p.freq, p.power; color = cx)
    Makie.lines!(ax3, psurr.freq, psurr.power; color = cs)

    # Histograms
    ax4 = Makie.Axis(fig[4,1])
    Makie.hist!(ax4, x; label = "Original", bins = nbins, color = (cx, 0.5))
    Makie.hist!(ax4, s; label = "Surrogate", bins = nbins, color = (cs, 0.5))
    Makie.axislegend(ax4)

    ax1.xlabel = "time step"
    ax1.ylabel = "value"
    ax2.xlabel = "lag"
    ax2.ylabel = "autocor"
    ax3.xlabel = "binned freq."
    ax3.ylabel = "power"
    ax4.xlabel = "binned value"
    ax4.ylabel = "histogram"
    return fig
end
export surroplot


function _surroplot(X::T, S::T;
    cx = :turbo, cs = :turbo, resolution = (500, 600),
    nbins = 50, cp = Makie.cgrad([Makie.RGBA(1, 1, 1, 0), Makie.RGBA(0, 0, 0, 1)])
) where T <: AbstractArray{D, 3} where D
    x = 1:size(X, 1)
    y = 1:size(X, 2)
    fig = Makie.Figure(resolution = resolution)

    # Time series
    _t = Makie.Observable(1)
    _X = Makie.@lift X[:, :, $_t]
    _S = Makie.@lift S[:, :, $_t]
    ax1, _ = Makie.heatmap(fig[1,1], x, y, _X; colormap = cx)
    sax1, _ = Makie.heatmap(fig[1,2], x, y, _S; colormap = cs)

    # # Autocorrelation
    # acx = autocor(x)
    # ax2, _ = Makie.lines(fig[2,1], 0:length(acx)-1, acx; color = cx)
    # Makie.lines!(ax2, 0:length(acx)-1, autocor(s); color = cs)

    # FFT spectrum for the temporal dimension
    p = abs.(fft(X)).^2
    p = p./sum(p)
    ps = abs.(fft(S)).^2
    ps = ps./sum(ps)
    cmax = max(p..., ps...)
    fs = fftfreq.(size(X))
    # ax2 = Makie.Axis3(fig[2,1])
    # Makie.volume!(ax2, fs..., p, algorithm=:additive, colormap=cp, colorrange=(0, cmax))
    # sax2 = Makie.Axis3(fig[2,2])
    # Makie.volume!(sax2, fs..., ps, algorithm=:additive, colormap=cp, colorrange=(0, cmax))
    # Makie.xlims!(ax2, 0, 1, 0, 1, 0, 1)
    idxs = sample(CartesianIndices(p[:, :, 1]), 100)
    ax2 = Makie.Axis(fig[2,1], xscale=Makie.pseudolog10, yscale=Makie.log10)
    [Makie.lines!(ax2, fs[3], p[idx, :], color=Makie.RGBA(0, 0, 0, 0.5)) for idx in idxs]
    Makie.xlims!(ax2, 0, nothing)
    sax2 = Makie.Axis(fig[2,2], xscale=Makie.pseudolog10, yscale=Makie.log10)
    [Makie.lines!(sax2, fs[3], ps[idx, :], color=Makie.RGBA(0, 0, 0, 0.5)) for idx in idxs]
    Makie.xlims!(sax2, 0, nothing)

    # Distribution
    ax3 = Makie.Axis(fig[3,1])
    Makie.density!(ax3, X[:], color=Makie.RGBA(0, 0, 0, 0.5))
    sax3 = Makie.Axis(fig[3,2])
    Makie.density!(sax3, S[:], color=Makie.RGBA(0, 0, 0, 0.5))

    # # Binned multitaper periodograms
    # p, psurr = DSP.mt_pgram(x), DSP.mt_pgram(s)
    # ax3 = Makie.Axis(fig[3,1]; yscale = log10)
    # Makie.lines!(ax3, p.freq, p.power; color = cx)
    # Makie.lines!(ax3, psurr.freq, psurr.power; color = cs)

    # # Histograms
    # ax4 = Makie.Axis(fig[4,1])
    # Makie.hist!(ax4, x; label = "Original", bins = nbins, color = (cx, 0.5))
    # Makie.hist!(ax4, s; label = "Surrogate", bins = nbins, color = (cs, 0.5))
    # Makie.axislegend(ax4)

    ax1.xlabel="x"; ax1.ylabel="y"; sax1.xlabel="x"; sax1.ylabel="y"
    ax2.xlabel="f"; ax2.ylabel="S"; sax2.xlabel="f"; sax2.ylabel="S"
    ax3.xlabel="Value"; ax3.ylabel="Probability density"; sax3.xlabel="Value"; sax3.ylabel="Probability Density"

    return fig, _t
end

"""
    surroplot(x::T, s::T; kwargs...) where T <: AbstractArray{D, 3} where D

Plot a two-dimensional time series `x` along with its surrogate realization `s`, and compare the
periodogram and histogram of the two time series.
By defaut this returns a static plot showing the first `dims=[1, 2]` slice of the time series (animating along the third dimension).
Tanimate the plot, use `animatesurroplot` instead.

## Keyword arguments
- `cx` and `cs`: Colormaps of the original and the surrogate time series, respectively.
- `cp`: Colormap for volume plots such as the periodogram.
- `nbins`: The number of bins for the histograms.
- `resolution`: A tuple giving the resolution of the figure.
"""
function surroplot(x::T, s::T; kwargs...) where T <: AbstractArray{D, 3} where D
    fig, _ = _surroplot(x, s; kwargs...)
    return fig
end

function animatesurroplot(x, s; filepath=mktemp()[1]*".gif", framerate=24, kwargs...)
    times = 1:size(x)[end]
    @assert maximum(times) == size(s)[end]
    fig, _t = _surroplot(x, s; kwargs...)
    Makie.record(fig, filepath, times; framerate) do t
        _t[] = t
    end
end
export animatesurroplot

using TimeseriesSurrogates
using GLMakie

X = randn(100, 100, 100)
S = surrogate(X, FT())
p = abs.(s).^2


animatesurroplot(X, S)

using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, Setfield
using Flux:throttle
using SumDenseProduct: fit!, mappath, samplepath
using ToyProblems: flower2

using Plots

using DiffEqFlux
using FfjordFlow


plotly()

function ffjord_to_cnf(model)
	_c = []
	for c in model.components
		tmp_mod = Any
		if typeof(c.m) <: Ffjord
			tmp_mod = Cnf(c.m)
		else
			tmp_mod = c.m
		end
		tmp_p = Any
		if typeof(c.p) <: SumNode
			tmp_p = ffjord_to_cnf(c.p)
		else
			tmp_p = c.p
		end
		push!(_c, DenseNode(tmp_mod, tmp_p))
	end
	return SumNode(_c, model.prior)
end

function resflow_to_iresnet(model)
	_c = []
	for c in model.components
		tmp_mod = Any
		if typeof(c.m) <: ResidualFlow
			tmp_mod = iResNet(c.m, 5)
		else
			tmp_mod = c.m
		end
		tmp_p = Any
		if typeof(c.p) <: SumNode
			tmp_p = resflow_to_iresnet(c.p)
		else
			tmp_p = c.p
		end
		push!(_c, DenseNode(tmp_mod, tmp_p))
	end
	return SumNode(_c, model.prior)
end

function plot_contour(m, x)
	levels = quantile(exp.(logpdf(m, x)), 0.01:0.09:0.99)
	xr = range(minimum(x[1,:]) - 1 , maximum(x[1,:])+ 1 , length = 200)
	yr = range(minimum(x[2,:]) - 1 , maximum(x[2,:])+ 1 , length = 200)
	# contour(xr, yr, (x...) ->  logpdf(m, [x[1],x[2]])[1], levels = levels)
	# heatmap(xr, yr, (x...) ->  exp(logpdf(m, [x[1],x[2]])[1]))
	# scatter!(x[1,:], x[2,:])
	contour(xr, yr, (x...) ->  exp(logpdf(m, [x[1],x[2]])[1]))
end

function plot_components(m, x)
	path = hash.(mappath(m, x)[2])
	u = unique(path)
	hash2int = Dict(u[i] => i for i in 1:length(u))
	i = [hash2int[k] for k in path]
	scatter(x[1,:], x[2,:], color = i)
end

function plot_rand(m, n)
	xx = reduce(hcat, rand(m) for i in 1:n)
	scatter(xx[1,:], xx[2,:])
end

function buildmff(n)
	SumNode([DenseNode(Ffjord(Chain(Dense(2, 10, tanh), Dense(10, 2)),
	 					(0.0, 1.0)), MvNormal(2,1f0)) for i in 1:n])
end

function buildmrf(n)
	SumNode([DenseNode(ResidualFlow(Chain(Chain(Dense(2, 10, tanh), Dense(10, 2)))),
			MvNormal(2,1f0)) for i in 1:n])
end

###############################################################################
#			non-normal mixtures
###############################################################################

#FFJORD nodes
x = flower2(1000, npetals = 9)
model = buildmff(9)
history = fit!(model, x, 200, 10000, 100; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM(), check=1)

model_exact = ffjord_to_cnf(model)

mean(logpdf(model_exact, x))

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.5:10, -10.0:0.5:10)])

res = reshape(logpdf(model_exact, xx), 41, 41)
heatmap(exp.(res))

#ResidualFlow nodes - NOT WORKING WITHOUT SPECT NORMALIZATION
x = flower2(1000, npetals = 9)
model = buildmrf(9)
history = fit!(model, x, 200, 10000, 100; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM(), check=1)

model_exact = resflow_to_iresnet(model)

mean(logpdf(model_exact, x))

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.5:10, -10.0:0.5:10)])

res = reshape(logpdf(model_exact, xx), 41, 41)
heatmap(exp.(res))

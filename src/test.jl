include("FfjordFlow.jl")
include("Ffjord.jl")
include("Cnf.jl")
using ToyProblems

x = flower(200)
m = Ffjord(Chain(Dense(2, 10, tanh), Dense(10, 2)), (0.0, 1.0))

ps = Flux.params(m)

function loss_adjoint()
    prob = logpdf(m, x)
    loss = - mean(prob)
    println(loss)
    return loss
end

_data = Iterators.repeated((), 100)

Flux.Optimise.train!(loss_adjoint, ps, _data, ADAM(0.1))

mm = Cnf(m)

using Plots
xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.5:10, -10.0:0.5:10)])

res = reshape(logpdf(m, xx), 41, 41)
heatmap((res))

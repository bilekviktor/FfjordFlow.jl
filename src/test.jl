include("FfjordFlow.jl")
include("Ffjord.jl")
include("Cnf.jl")
using ToyProblems

x = ToyProblems.flower2(200, npetals = 9)
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

loss_adjoint()

mm = Cnf(m)

using Plots
xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.5:10, -10.0:0.5:10)])

res = reshape(logpdf(mm, xx), 41, 41)
heatmap(exp.(res))
#-------------------------------------------------------------
include("iResNet.jl")
include("ResidualFlow.jl")
using ToyProblems
using Plots

x = sixgaussians(20)
lip_swish(x) = swish(x)/1.1
m = iResNet(Chain(Chain(Dense(2, 40, tanh), Dense(40, 40, tanh), Dense(40, 2, tanh))), 5)
y = logpdf(m, x)

function loss_resnet()
    loss = -mean(logpdf(m, x))
    println(loss)
    return loss
end
loss_resnet()

ps = Flux.params(m)
_data = Iterators.repeated((), 1000)

Flux.Optimise.train!(loss_resnet, ps, _data, ADAM(0.1))

mm = iResNet(m, 6)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.5:10, -10.0:0.5:10)])

res = reshape(logpdf(m, xx), 41, 41)
heatmap((res))

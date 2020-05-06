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
include("SpectralNormalization.jl")
using ToyProblems
using Plots, MLDataPattern

x = ToyProblems.flower2(1500)
y = RandomBatches(x, 400, 20)
lip_swish(x) = swish(x)/1.1
n = 20
m = iResNet(f64(Chain(Chain(Dense(2, n, tanh), Dense(n, n, tanh),Dense(n, 2)))), 5)


function loss_resnet(x)
    pred = [logpdf(m, x[:, i]) for i in 1:size(x)[2]]
    loss = -mean(pred)
    println(loss)

    return loss
end
loss_resnet(x)

ps = Flux.params(m)
_data = Iterators.repeated((), 100)

opt = ADAM(0.01)
sopt = SpecNormalization(0.1)
specttrain!(x -> loss_resnet(getobs(x)), ps, y, opt, sopt, 2)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.5:10, -10.0:0.5:10)])

res = reshape([logpdf(m, xx[:, i]) for i in 1:1681], 41, 41)
Plots.heatmap(exp.(res))
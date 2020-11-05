using ToyProblems, Flux
using Statistics
using FfjordFlow
using Plots
using MLDataPattern
using Distributions
using DiffEqFlux
#--------------------Ffjord---------------------------------
#dataset
x = ToyProblems.flower2(1000, npetals = 9)
#ffjord model
m = Ffjord(Chain(Dense(2, 10, tanh), Dense(10, 2)), (0.0, 1.0))

function loss_adjoint()
    prob = logpdf(m, x)
    loss = - mean(prob)
    println(loss)
    return loss
end
loss_adjoint()

ps = Flux.params(m)
_data = Iterators.repeated((), 100)

opt = ADAM(0.1)
Flux.Optimise.train!(loss_adjoint, ps, _data, opt)

#converting ffjord to exact cnf model
mm = Cnf(m)

#heatmap of result
xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(logpdf(mm, xx), 201, 201)
heatmap(exp.(res))
#----------------------------ResidualFlow---------------------------------
#generating dataset
include("../src/iResNet.jl")
include("../src/ResidualFlow.jl")
x = ToyProblems.sixgaussians(20)
#y = RandomBatches(x, 100, 300)

#ResidualFlow model - first argument requires chain of residual blocks - 1 res block here
n = 20
m = ResidualFlow(Chain(Chain(Dense(2, n, tanh), Dense(n, n, tanh),Dense(n, 2))))

function loss_resnet(x)
    pred = logpdf(m, x)
    loss = -mean(pred)
    println(loss)

    return loss
end
loss_resnet(x)


ps = Flux.params(m)
_data = Iterators.repeated((), 100)

opt = ADAM(0.1)
sopt = SpecNormalization(0.1) #optimeser for spectral normalization
oopt = Flux.Optimise.Optimiser(opt, sopt)
#specttrain!(() -> loss_resnet(x), ps, _data, opt, sopt, 1)
Flux.Optimise.train!(() -> loss_resnet(x), ps, _data, oopt)

#converting ResidualFlow to exact iResNet model
mm = iResNet(m, 5)

#heatmap of result
xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.5:10, -10.0:0.5:10)])
res = reshape(logpdf(mm, xx), 41, 41)
Plots.heatmap(exp.(res))

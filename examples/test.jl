using ToyProblems, Flux
using Statistics
using FfjordFlow
using Plots
using MLDataPattern
#--------------------Ffjord---------------------------------
#dataset
x = ToyProblems.flower2(200, npetals = 9)
#ffjord model
m = Ffjord(Chain(Dense(2, 10, tanh), Dense(10, 2)), (0.0, 1.0))

function loss_adjoint()
    prob = logpdf(m, x)
    loss = - mean(prob)
    println(loss)
    return loss
end

ps = Flux.params(m)
_data = Iterators.repeated((), 100)

Flux.Optimise.train!(loss_adjoint, ps, _data, ADAM(0.1))

#converting ffjord to exact cnf model
mm = Cnf(m)

#heatmap of result
xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.5:10, -10.0:0.5:10)])
res = reshape(logpdf(mm, xx), 41, 41)
heatmap(exp.(res))
#----------------------------ResidualFlow---------------------------------
#generating dataset
x = ToyProblems.sixgaussians(150)
y = RandomBatches(x, 100, 300)

#ResidualFlow model - first argument requires chain of residual blocks - 1 res block here
n = 20
m = ResidualFlow(Chain(Chain(Dense(2, n, tanh), Dense(n, n, tanh),Dense(n, 2))))

function loss_resnet(x)
    pred = [logpdf(m, x[:, i]) for i in 1:size(x)[2]]
    loss = -mean(pred)
    println(loss)

    return loss
end
loss_resnet(x)

ps = Flux.params(m)
_data = Iterators.repeated((), 100)

opt = ADAM(0.1)
sopt = SpecNormalization(0.7) #optimeser for spectral normalization
specttrain!(x -> loss_resnet(getobs(x)), ps, y, opt, sopt, 1)

#converting ResidualFlow to exact iResNet model
mm = iResNet(m, 5)

#heatmap of result
xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.5:10, -10.0:0.5:10)])
res = reshape([logpdf(mm, xx[:, i]) for i in 1:1681], 41, 41)
Plots.heatmap(exp.(res))

using DiffEqFlux, Flux, Zygote, DifferentialEquations
using Plots, DiffEqSensitivity, LinearAlgebra
using ToyProblems, Statistics, MLDataPattern
using FiniteDiff

function jacobian(f, x::AbstractVector)
    y::AbstractVector, back = Zygote.pullback(f, x)
    ȳ(i) = [i == j for j = 1:length(y)]
    vcat([transpose(back(ȳ(i))[1]) for i = 1:length(y)]...)
end

m = Chain(Dense(2, 30, tanh), Dense(30, 2))
p, re = Flux.destructure(m)
function cnf(du, u, p, t)
    z = @view u[1:end-1]
    m = re(p)
    _du = m(z)
    du[1] = _du[1]
    du[2] = _du[2]
    #deltalog = -tr(jacobian(x -> m(x), z))
    deltalog = -tr(jacobian(x -> m(x), z))
    du[end] = deltalog
    return du
end
u0 = Float32.([1.0, 2.0])
tspan = Float32.((0.0, 1.0))
prob = ODEProblem(cnf, u0, tspan, p)

function predict_adjoint_cnf(x)
    Array(concrete_solve(prob, Tsit5(), [x; 0.0f0], p, abstol = 1e-8, reltol = 1e-6,
                sensealg = InterpolatingAdjoint(autojacvec = DiffEqSensitivity.ReverseDiffVJP())))
end

sol = predict_adjoint(u0)

function ffjord(du, u, p, t, e)
    z = @view u[1:end-1]
    m = re(p)
    _du, back = Zygote.pullback(x -> m(x), z)
    eJ = back(e)[1]
    eJe = sum(eJ .* e)
    du[1] = _du[1]
    du[2] = _du[2]
    #deltalog = -tr(jacobian(x -> m(x), z))
    du[3] = -eJe
    return du
end

function predict_adjoint_ffjord(x)
    e = Float32.(randn(2))
    _ffjord(du, u, p, t) = ffjord(du, u, p, t, e)
    prob = ODEProblem(_ffjord, [x; 0.0f0], tspan, p)
    Array(concrete_solve(prob, Tsit5(), [x; 0.0f0], p, abstol = 1e-8, reltol = 1e-6,
                sensealg = InterpolatingAdjoint(autojacvec = DiffEqSensitivity.TrackerVJP())))
end

log_normal(x::AbstractVector) = - sum(x.^2) / 2 - length(x)*log(2π) / 2
log_normal(x) = -0.5f0 .* sum(x.^2, dims = 1) .- (size(x,1)*log(Float32(2π)) / 2)

function loss_adjoint(x)
    m, n = size(x)
    pred = [predict_adjoint_cnf(x[:, i])[:, end] for i in 1:n]
    prob = [log_normal(pr[1:2]) for pr in pred]
    delta_log = [pr[end] for pr in pred]
    loss = -mean(prob .- delta_log)
    return loss
end

function loss_adjoint_single(x)
    loss = 0.0
    m, n = size(x)
    for i in 1:n
        pred = predict_adjoint_cnf(x[:, i])[:, end]
        loss = loss - (log_normal(pred[1:2]) - pred[end])
    end
    println(loss/n)
    return loss/n
end
display(loss_adjoint_single(x))
#display(loss_adjoint(x))

#=
cb_single = function ()
    display(loss_adjoint_single())
end
=#

x = Float32.(sixgaussians(100))
y = RandomBatches((x,), 200, 100)
scatter(x[1,:],x[2,:])

opt = ADAM(0.1)
_data = Iterators.repeated((), 20)
ps = Flux.params(p)
Flux.Optimise.train!(x -> loss_adjoint_single(getobs(x)), ps, y, opt)

function view_cnf(x)
    pred = predict_adjoint_cnf(x)[:, end]
    return log_normal(pred[1:end-1]) - pred[end]
end

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-8.0:0.5:8.0, -8.0:0.5:8.0)])

distribution_before = reshape([exp(log_normal(xx[:, i])) for i in 1:size(xx)[2]], 33, 33)
heatmap(distribution_before)

distribution_after = reshape([exp(view_cnf(xx[:, i])) for i in 1:size(xx)[2]], 33, 33)
heatmap((distribution_after))

#-----------------------------------------------------------------------

using DiffEqFlux, Flux, DifferentialEquations, DiffEqSensitivity
using Zygote, ToyProblems, Statistics


log_normal(x::AbstractVector) = - sum(x.^2) / 2 - length(x)*log(2π) / 2
log_normal(x) = -0.5f0 .* sum(x.^2, dims = 1) .- (size(x,1)*log(Float32(2π)) / 2)

u0 = Float64[2.0, 1.1, 0.0]
tspan = (0.0e0,1.0e0)

ann = f64(Chain(Dense(2,10,tanh), Dense(10,2)))

p3,re = Flux.destructure(ann)
ps = Flux.params(p3)
e = randn(2)

function dudt(u,p,t)
    tmp, back = Zygote.pullback(re(p), u[1:2])
    eJ = back(e)[1]
    eJe = sum(eJ .* e)
    return [tmp; -eJe]
end
prob2 = ODEProblem(dudt,u0,tspan,p3)
sol = solve(prob2)

function predict_adjoint()
  Array(concrete_solve(prob2,Tsit5(),u0,p3,saveat=0.0:0.1:1.0,abstol=1e-8,
                 reltol=1e-6,sensealg=InterpolatingAdjoint(autojacvec = ZygoteVJP())))
  # ^ wrapped this in Array as done in the previous example
end
function loss_adjoint()
    pred = predict_adjoint()[:, end]
    return -(log_normal(pred[1:2]) - pred[end])
end

_data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_adjoint())
  #display(plot(solve(remake(prob,p=p3,u0=u0),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

Flux.Optimise.train!(loss_adjoint, ps, _data, opt, cb = cb)

u0 = vcat(sixgaussians(10), zeros(90)')
ann = f64(Chain(Dense(2,10,tanh), Dense(10,2)))

p3,re = Flux.destructure(ann)
ps = Flux.params(p3)
e = randn(2, 90)

function ddudt(u, p, t)
    tmp, back = Zygote.pullback(re(p), u[1:2, :])
    eJ = back(e)[1]
    eJe = sum(eJ .* e, dims = 1)
    return [tmp; -eJe]
end
prob = ODEProblem(ddudt,u0,tspan,p3)
sol = solve(prob)

function predict_adjoint()
  Array(concrete_solve(prob,Tsit5(),u0,p3,saveat=0.0:0.1:1.0,abstol=1e-8,
                 reltol=1e-6,sensealg=InterpolatingAdjoint(autojacvec = ZygoteVJP())))
  # ^ wrapped this in Array as done in the previous example
end
function loss_adjoint()
    pred = predict_adjoint()
    _pred = pred[:, :, end]
    log_z = log_normal(_pred[1:2, :])

    log_x = log_z .- _pred[end, :]
    loss = - mean(log_x)
    return loss
end

_data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_adjoint())
  #display(plot(solve(remake(prob,p=p3,u0=u0),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

Flux.Optimise.train!(loss_adjoint, ps, _data, opt, cb = cb)

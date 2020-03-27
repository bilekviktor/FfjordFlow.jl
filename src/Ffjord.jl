using Flux, LinearAlgebra, MLDataPattern, ToyProblems
using Plots, DiffEqFlux, DifferentialEquations, Statistics
using Zygote, DiffEqSensitivity, Distributions

#calculation of ϵᵀ*J*ϵ
#asfdasdfdf

struct Ffjord{M, T, P}
    m::M
    tspan::Tuple{T, T}
    param::P
end

function Ffjord(_m::M, _tspan::Tuple{T, T}) where {M, T}
    p, re = Flux.destructure(_m)
    return Ffjord(_m, _tspan, p)
end

Flux.params(F::Ffjord) = Flux.params(F.param)

function ffjord(u, m, p, e, re)
    u1, back = Zygote.pullback(re(p), u[1:2, :])
    eJ = back(e)[1]
    eJe = sum(eJ .* e, dims = 1)
    return [u1; -eJe]
end

function jacobian(f, x::AbstractVector)
  y::AbstractVector, back = Zygote.pullback(f, x)
  ȳ(i) = [i == j for j = 1:length(y)]
  vcat([transpose(back(ȳ(i))[1]) for i = 1:length(y)]...)
end

function multi_trace_jacobian(f, x::AbstractArray)
    y::AbstractArray, back = Zygote.pullback(f, x)
    m, n = size(y)
    bool_y(i) = hcat([[i == j for j = 1:m] for k = 1:n]...)
    trace = back(bool_y(1))[1][1, :]
    for i in 2:m
        trace = trace .+ back(bool_y(i))[1][i, :]
    end
    return [y; -trace']
end

function cnf(u, p, rel)
    return multi_trace_jacobian(rel(p), u[1:2, :])
end

#ODE for FFJORD
function diffeq_ffjord(m, x, ps, tspan, e, args...; kwargs...)
    p, re = Flux.destructure(m)
    dudt(u, p, t) = ffjord(u, m, p, e, re) #|> gpu
    prob = ODEProblem(dudt, x, tspan, ps)

    return Array(concrete_solve(prob, Tsit5(), x, ps, abstol = 1e-6, reltol = 1e-3,
                    sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())))
end

#ODE for CNF
function diffeq_cnf(m, x, ps, tspan)
    p, re = Flux.destructure(m)
    dudt(u, p, t) = cnf(u, p, re)
    prob = ODEProblem(dudt, x, tspan, ps)

    return Array(concrete_solve(prob, Tsit5(), x, ps, abstol = 1e-6, reltol = 1e-3,
    sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.TrackerVJP())))
end

#log nomrla densitiy function
log_normal(x::AbstractVector) = - sum(x.^2) / 2 - length(x)*log(2π) / 2
log_normal(x) = -0.5f0 .* sum(x.^2, dims = 1) .- (size(x,1)*log(Float32(2π)) / 2)

#simplification of FFJORD ode with argument for initial condition only
function predict_ffjord(m, x, p, tspan)
    v, w = size(x)
    e = randn(v-1, w)
    diffeq_ffjord(m, x, p, tspan, e)
end

function predict_cnf(x, p)
    diffeq_cnf(m, x, p, tspan)
end

function (F::Ffjord)(xx::Tuple{A, B}) where {A, B}
    x, logdet = xx
    u0 = vcat(x, logdet)
    pred = predict_ffjord(F.m, u0, F.param, F.tspan)[:, :, end]
    x1 = pred[1:size(x)[1], :]
    logdet1 = pred[end, :]
    return (x1, logdet1')
end

function Distributions.logpdf(m::Ffjord, x::AbstractMatrix{T}) where {T}
    x, logdet = m((x, zeros(size(x)[2])'))
    return log_normal(x) - logdet
end


#definition of loss function(maximum likelihood method)
function loss_ffjord(x, p)
    pred = predict_ffjord(x, p)
    _pred = pred[:, :, end]
    log_z = log_normal(_pred[1:size(x)[1]-1, :])

    log_x = log_z .- _pred[end, :]'
    loss = - mean(log_x)
    println(loss)
    return loss
end

function loss_cnf(x, p)
    pred = predict_cnf(x, p)
    _pred = pred[:, :, end]
    log_z = log_normal(_pred[1:size(x)[1]-1, :])

    log_x = log_z .- _pred[end, :]'
    loss = - mean(log_x)
    println(loss)
    return loss
end


#----------------start of training--------------------#
#model definition for later use
n = 20
m = f64(Chain(Dense(2, n, tanh), Dense(n, n, tanh), Dense(n, 2)))
tspan = Float64.((0.0, 1.0))

f = Ffjord(m, tspan)

#generation of training set
l, ρ = 100, 8
x = flower(100)

probab = Distributions.logpdf(f, x)

function loss_adjoint(x)
     l = -mean(logpdf(f, x))
     println(l)
     return l
 end
loss_adjoint() = loss_adjoint(x)

y = RandomBatches((x,), 100, 100)
y2 = RandomBatches((x,), 300, 10)
#scatter(x[1,:],x[2,:])
#loss_cnf(getobs(y)[1][1])
#loss_ffjord(getobs(y)[1][1])
_data = Iterators.repeated((), 100)

#training
opt = ADAM(0.1)
opt2 = ADAM(0.01)
ps = Flux.params(f)

Flux.Optimise.train!(loss_adjoint, ps, _data, opt)
Flux.Optimise.train!(x -> loss_cnf(getobs(x), p), ps, y2, opt2)

#                   heatmap of distribution
#---------------------------------------------------------------
function logpdf_cnf(x, p)
    pred = predict_cnf(x, p)
    _pred = pred[:, :, end]
    log_z = log_normal(_pred[1:size(x)[1]-1, :])

    log_x = log_z .- _pred[end, :]'
end

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-8.0:0.5:8.0, -8.0:0.5:8.0)])

distribution_before = reshape([exp(log_normal(xx[:, i])) for i in 1:size(xx)[2]], 33, 33)
heatmap(distribution_before)

distribution_after = reshape(logpdf_cnf(vcat(xx, zeros(1089)'), f.param)', 33, 33)
heatmap(exp.(distribution_after))

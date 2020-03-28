using Flux, LinearAlgebra, MLDataPattern
using DifferentialEquations
using Zygote
using DiffEqSensitivity:TrackerVJP
using DiffEqSensitivity:ZygoteVJP
using Distributions
using DiffEqFlux:InterpolatingAdjoint

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

Flux.@functor Ffjord
Flux.trainable(m::Ffjord) = (m.param, )

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


function (F::Ffjord)(xx::Tuple{A, Number}) where {A}
    x, logdet_single = xx
    logdet = fill(logdet_single, size(x)[2])'
    u0 = vcat(x, logdet)
    pred = predict_ffjord(F.m, u0, F.param, F.tspan)[:, :, end]
    x1 = pred[1:size(x)[1], :]
    logdet1 = pred[end, :]
    return (x1, logdet1')
end


function (m::Ffjord)(x::AbstractArray)
    y, logdet = m((x, zeros(size(x)[2])'))
    return y
end

function Distributions.logpdf(m::Ffjord, x::AbstractMatrix{T}) where {T}
    y, logdet = m((x, zeros(size(x)[2])'))
    return vec(log_normal(y) - logdet)
end

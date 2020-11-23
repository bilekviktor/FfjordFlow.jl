using Flux, LinearAlgebra, MLDataPattern
using DifferentialEquations
using Zygote
using DiffEqSensitivity: TrackerVJP
using DiffEqSensitivity: ZygoteVJP
using Distributions
using DiffEqFlux:InterpolatingAdjoint

#--------------------------
struct Cnf{M, T, P}
    m::M
    tspan::Tuple{T, T}
    param::P
end

Base.show(io::IO, a::Cnf) = print(io, "Cnf{$(a.m) on $(a.tspan)}")

function Cnf(_m::M, _tspan::Tuple{T, T}) where {M, T}
    p, re = Flux.destructure(_m)
    return Cnf(_m, _tspan, p)
end

Flux.@functor Cnf
Flux.trainable(m::Cnf) = (m.param, )

function multi_trace_jacobian(rel, p, u::AbstractArray)
    y::AbstractArray, back = Zygote.pullback(rel(p), u[1:size(u)[1]-1, :])
    m, n = size(y)
    bool_y(i) = hcat([[i == j for j = 1:m] for k = 1:n]...)
    trace = back(bool_y(1))[1][1, :]
    for i in 2:m
        trace = trace .+ back(bool_y(i))[1][i, :]
    end
    return [y; -trace']
end

function cnf(u, p, rel)
    return multi_trace_jacobian(rel, p, u)
end

function diffeq_cnf(m, x, ps, tspan)
    p, re = Flux.destructure(m)
    dudt(u, p, t) = cnf(u, p, re)
    prob = ODEProblem(dudt, x, tspan, ps)

    return Array(solve(prob, Tsit5(), u0=x, p=ps, abstol = 1e-6, reltol = 1e-3,
    sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP())))
end

function predict_cnf(m, x, p, tspan)
    diffeq_cnf(m, x, p, tspan)
end

function (F::Cnf)(xx::Tuple{A, B}) where {A, B}
    x, logdet = xx
    u0 = vcat(x, -logdet)
    pred = predict_cnf(F.m, u0, F.param, F.tspan)[:, :, end]
    x1 = pred[1:size(x)[1], :]
    logdet1 = pred[end, :]
    return (x1, -logdet1')
end


function (F::Cnf)(xx::Tuple{A, Number}) where {A}
    x, logdet_single = xx
    logdet = fill(logdet_single, size(x)[2])'
    u0 = vcat(x, -logdet)
    pred = predict_cnf(F.m, u0, F.param, F.tspan)[:, :, end]
    x1 = pred[1:size(x)[1], :]
    logdet1 = pred[end, :]
    return (x1, -logdet1')
end

function (m::Cnf)(x::AbstractArray)
    y, logdet = m((x, zeros(size(x)[2])'))
    return y
end

function Distributions.logpdf(m::Cnf, x::AbstractMatrix{T}) where {T}
    y, logdet = m((x, zeros(size(x)[2])'))
    return vec(log_normal(y) + logdet)
end

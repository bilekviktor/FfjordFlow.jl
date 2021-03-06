using Flux, LinearAlgebra, MLDataPattern
using OrdinaryDiffEq
using Zygote
using DiffEqSensitivity: TrackerVJP
using DiffEqSensitivity: ZygoteVJP
using Distributions
using DiffEqFlux:InterpolatingAdjoint

struct Ffjord{M, T, P}
    m::M
    tspan::Tuple{T, T}
    param::P
end

Base.show(io::IO, a::Ffjord) = print(io, "Ffjord{$(a.m) on $(a.tspan)}")

function Ffjord(_m::M, _tspan::Tuple{T, T}) where {M, T}
    p, re = Flux.destructure(_m)
    return Ffjord(_m, _tspan, p)
end

Flux.@functor Ffjord
Flux.trainable(m::Ffjord) = (m.param, )

function ffjord(u, p, e, re)
    u1, back = Zygote.pullback(re(p), u[1:size(u)[1]-1, :])
    eJ = back(e)[1]
    eJe = sum(eJ .* e, dims = 1)
    return [u1; -eJe]
end

#ODE for FFJORD
function diffeq_ffjord(m, x, ps, tspan, e, args...; kwargs...)
    p, re = Flux.destructure(m)
    dudt(u, p, t) = ffjord(u, p, e, re) #|> gpu
    prob = ODEProblem(dudt, x, tspan, p)

    return Array(solve(prob, Tsit5(), u0=x, p=ps, abstol = 1e-3, reltol = 1e-1,
                    sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())))
end

#simplification of FFJORD ode with argument for initial condition only
function predict_ffjord(m, x, p, tspan)
    v, w = size(x)
    e = randn(v-1, w)
    diffeq_ffjord(m, x, p, tspan, e)
end

function (F::Ffjord)(xx::Tuple{A, B}) where {A, B}
    x, logdet = xx
    u0 = vcat(x, -logdet)
    pred = predict_ffjord(F.m, u0, F.param, F.tspan)[:, :, end]
    x1 = pred[1:size(x)[1], :]
    logdet1 = pred[end, :]
    return (x1, -logdet1')
end


function (F::Ffjord)(xx::Tuple{A, Number}) where {A}
    x, logdet_single = xx
    logdet = fill(logdet_single, size(x)[2])'
    u0 = vcat(x, -logdet)
    pred = predict_ffjord(F.m, u0, F.param, F.tspan)[:, :, end]
    x1 = pred[1:size(x)[1], :]
    logdet1 = pred[end, :]
    return (x1, -logdet1')
end


function (m::Ffjord)(x::AbstractArray)
    y, logdet = m((x, zeros(size(x)[2])'))
    return y
end


function Distributions.logpdf(m::Ffjord, x::AbstractMatrix{T}) where {T}
    y, logdet = m((x, zeros(size(x)[2])'))
    return vec(log_normal(y) + logdet)
end

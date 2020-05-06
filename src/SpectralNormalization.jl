using Flux, Zygote
using LinearAlgebra

mutable struct SpecNormalization
    α::Real
    u::IdDict
end

SpecNormalization(alpha = 1.0) = SpecNormalization(alpha, IdDict())

function Flux.Optimise.apply!(o::SpecNormalization, x::Matrix, Δ)
    u = get!(o.u, x, randn(size(x, 1)))
    v = transpose(x) * u
    v = v./norm(v)
    u = x * v
    u = u./norm(u)
    σ = transpose(u) * x * v
    #σ = norm(x)
    o.u[x] = u
    Δ = (o.α) .* (σ-1)/σ .* x
end

function Flux.Optimise.apply!(o::SpecNormalization, x::Vector, Δ)
  #σ = norm(x)
  #Δ = (o.α) .* (σ-1)/σ .* x
  return 0.0
end


function specttrain!(loss, ps, data, opt, sopt, normnumber = 5)
    ps = Zygote.Params(ps)
    for d in data
        for p in ps
          for i in 1:normnumber
              Δ = Flux.Optimise.apply!(sopt, p, 0.0)
              p .-= Δ
              #println("norm: ", norm(p))
          end
        end
        #println("norm_before: ", norm(first(ps)))
        gs = gradient(() -> loss(d), ps)
        Flux.Optimise.update!(opt, ps, gs)
        #println("norm_after: ", norm(first(ps)))
    end
    for p in ps
      for i in 1:normnumber
          Δ = Flux.Optimise.apply!(sopt, p, 0.0)
          p .-= Δ
          #println("norm: ", norm(p))
      end
    end
end
#-----------------------------------------------------------

mutable struct SpectralADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
end
SpectralADAM(η = 0.001, β = (0.9, 0.999)) = SpectralADAM(η, β, IdDict())

function apply!(o::SpectralADAM, x::Matrix, Δ)
  η, β = o.eta, o.beta
  mt, vt, βp, u = get!(o.state, x, (zero(x), zero(x), β, randn(size(x, 1))))
  println("zk")
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  v = transpose(x) * u
  v = v./norm(v)
  u = x * v
  u = u./norm(u)
  σ = transpose(u) * x * v
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  Δ = Δ .+ (σ-1)/σ * (x .- Δ)
  o.state[x] = (mt, vt, βp .* β, u)
  return Δ
end

function apply!(o::SpectralADAM, x::Vector, Δ)
  η, β = o.eta, o.beta
  mt, vt, βp = get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  o.state[x] = (mt, vt, βp .* β)
  return Δ
end
#---------------------------------------------------
struct SpectralDense{F,S,T}
  W::S
  b::T
  σ::F
  dict::IdDict
end

SpectralDense(W, b, σ) = SpectralDense(W, b, σ, IdDict())
SpectralDense(in::Int, out::Int, σ) = SpectralDense(Dense(in, out).W, Dense(in, out).b, σ)
SpectralDense(in::Int, out::Int) = SpectralDense(in, out, identity)

Flux.@functor SpectralDense
Flux.trainable(S::SpectralDense) = (S.W, S.b,)
outdims(S::SpectralDense, isize) = (size(S.m.W)[1],)

Zygote.@nograd get_u(S::SpectralDense) = get!(S.dict, S.W, randn(size(S.W, 1)))
Zygote.@nograd function spectralcoef(S::SpectralDense)
  W = S.W
  u = get_u(S)
  v = transpose(W) * u
  v = v./norm(v)
  u = W * v
  u = u./norm(u)
  S.dict[W] = u
  spec = transpose(u) * W * v
end


function (S::SpectralDense)(x::AbstractArray)
  W, b, σ = S.W, S.b, S.σ
  #u = get!(s.dict, W, randn(size(W, 1)))
  spec = norm(W)
  #println("norm: ", norm(W./spec))
  return σ.((W./Zygote.nograd(spec))*x .+ b)
end

(a::SpectralDense{<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::SpectralDense{<:Any,W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))
#=
m = Chain(SpectralDense(2, 10, tanh), SpectralDense(10, 2))
x = rand(2, 100)
function loss()
  l = sum(abs2, m(x))
  println(l)
  return l
end

ps = Flux.params(m)
_data = Iterators.repeated((), 1000)
opt = ADAM(0.1)
Flux.Optimise.train!(loss, ps, _data, opt)
=#

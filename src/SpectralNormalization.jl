using Flux, Zygote
using LinearAlgebra

mutable struct SpecNormalization
    α::Real
    u::IdDict
end

SpecNormalization(alpha = 1.0) = SpecNormalization(alpha, IdDict())

function Flux.Optimise.apply!(o::SpecNormalization, x::Matrix, Δ)
    u = get!(o.u, x, randn(size(x, 1)))
    x .-= Δ
    v = transpose(x) * u
    v = v./norm(v)
    u = x * v
    u = u./norm(u)
    σ = transpose(u) * x * v
    #println("sigma: ", σ)
    if (o.α)/σ < 1.01
        σ = σ/o.α
    else
        σ = 1.0
    end
    o.u[x] = u
    #λ = maximum([one(σ), σ/o.α])
    x .*= 1/σ
    Δ = zero(Δ)
    return Δ
end

function Flux.Optimise.apply!(o::SpecNormalization, x, Δ)
  #σ = norm(x)
  #Δ = (o.α) .* (σ-1)/σ .* x
  #x .*= 1/norm(x)
  x .-= Δ
  return zero(Δ)
end


function specttrain!(loss, ps, data, opt, sopt, normnumber = 1)
    ps = Zygote.Params(ps)
    for d in data
        for p in ps
          for i in 1:normnumber
              Δ = Flux.Optimise.apply!(sopt, p, zero(p))
              p .*= Δ
              #println("norm: ", norm(p))
          end
        end
        #println("norm_before: ", norm(first(ps)))
        gs = gradient(() -> loss(), ps)
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
  return 0.1 * spec
end


function (S::SpectralDense)(x::AbstractArray)
  W, b, σ = S.W, S.b, S.σ
  u = get_u(S)
  v = transpose(W) * u
  v = v./norm(v)
  u = W * v
  u = u./norm(u)
  S.dict[W] = u
  spec = transpose(u) * W * v
  #u = get!(s.dict, W, randn(size(W, 1)))
  #println("norm: ", norm(W./spec))
  return σ.((W./spec)*x .+ b)
end

(a::SpectralDense{<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::SpectralDense{<:Any,W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

##############################333

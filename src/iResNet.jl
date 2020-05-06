using Flux, Distributions, LinearAlgebra
using Zygote

include("ResidualFlow.jl")
include("SpectralNormalization.jl")

log_normal(x::AbstractVector) = - sum(x.^2) / 2 - length(x)*log(2π) / 2
log_normal(x) = -0.5f0 .* sum(x.^2, dims = 1) .- (size(x,1)*log(Float32(2π)) / 2)
lip_swish(x) = swish(x)/1.1

#this structure is not yet used in upcoming code
struct iResNet{M, I}
    m::M
    n::I
end

Base.show(io::IO, a::iResNet) = print(io, "iResNet{$(a.m)}")

iResNet(R::ResidualFlow, n) = iResNet(R.m, n)

Flux.@functor iResNet
Flux.trainable(R::iResNet) = (R.m, )
#---------------------------------------------------
function jacobian2(f, x)
    m = length(x)
    bf = Zygote.Buffer(x,m, m)
    for i in 1:m
        bf[i, :] = gradient(x -> f(x)[i], x)[1]
    end
    copy(bf)
end

Zygote.∇getindex(x::AbstractArray, inds) = dy -> begin
  if inds isa  NTuple{<:Any,Integer}
    dx = Zygote.Buffer(zero(x), false)
    dx[inds...] = dy
  else
    dx = Zygote.Buffer(zero(x), false)
    @views dx[inds...] .+= dy
  end
  (copy(dx), map(_->nothing, inds)...)
end

#function for computation of det of probability transformation
Zygote.@nograd irezidual_coef(k) = ((-1)^(k+1))/k
#=
function rezidual_block(m, x)
    l, _n = size(x)
    n = 3
    J = [jacobian(m, x[:, i]) for i in 1:_n]

    Jk = J
    sum_Jac = [tr(J[i]) for i in 1:_n]
    for k in 2:n
        Jk = [Jk[i] * J[i] for i in 1:_n]
        new_tr = [tr(rezidual_coef(k).*Jk[i]) for i in 1:_n]
        sum_Jac = sum_Jac .+ new_tr
    end
    return sum_Jac
end
=#
function single_block(m, x, n)
    _n = length(x)
    J = jacobian2(m ,x)
    Jk = J
    sum_Jac = tr(J)
    for k in 2:n
        sum_Jac = sum_Jac + irezidual_coef(k) * tr(Jk * J)
    end
    sum_Jac
end
#=
function single_flow(m, x)
    z = copy(x)
    log_prob = 0.0
    for i in 1:length(m)
        log_prob = log_prob + single_block(m[i], z)
        z = identity(z) .+ m[i](z)
    end
    return log_normal(z) + log_prob
end

#function for computation of probability of given data
function rezidual_flow(m, x)
    z = copy(x)
    log_prob = zeros(size(x)[2])
    for i in 1:length(m)
        log_prob = log_prob .+ rezidual_block(m[i], z)
        #here is computed residuality of neural network
        z = z .+ m[i](z)
    end
    return (z, reshape(log_prob, 1, :))
end
=#
function (R::iResNet)(xx::Tuple{A, B}) where {A, B}
    x, logdet = xx
    z = copy(x)
    m = R.m
    log_det = copy(logdet')
    for i in 1:length(m)
        log_det = log_det .+ single_block(m[i], z, R.n)
        z = z .+ m[i](z)
    end
    return (z, log_det')
end


(R::iResNet)(x::AbstractArray) = x .+ R.m(x)

function Distributions.logpdf(R::iResNet, x::AbstractMatrix{T}) where {T}
    y, logdet = m((x, 0.0))
    return vec(log_normal(y) + logdet)
end


function Distributions.logpdf(R::iResNet, x::Vector) where {T}
    y, logdet = m((x, 0.0))
    return log_normal(y) + logdet
end

using Flux, Distributions, LinearAlgebra
using Zygote

log_normal(x::AbstractVector) = - sum(x.^2) / 2 - length(x)*log(2π) / 2
log_normal(x) = -0.5f0 .* sum(x.^2, dims = 1) .- (size(x,1)*log(Float32(2π)) / 2)
lip_swish(x) = swish(x)/1.1

#this structure is not yet used in upcoming code
struct ResNet{F,S,T}
    W::S
    b::T
    σ::F
end

(m::ResNet)(x) = Dense(m.W, m.b, m.σ)(x) .+ x

function ResNet(n::Integer, σ = identity)
    m = Dense(n, n, σ)
    return ResNet(m.W, m.b, m.σ)
end
#---------------------------------------------------
function jacobian(f, x)
    m = length(x)
    bf = Zygote.Buffer(x,m, m)
    for i in 1:m
        bf[i, :] = gradient(x -> f(x)[i], x)[1]
    end
    copy(bf)
end

function _jacobian(f, x)
    m = length(x)
    gs = [gradient(x -> f(x)[i], x)[1] for i in 1:m]
    hcat(gs...)
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
Zygote.@nograd rezidual_coef(k) = (-1)^(k+1)/k
Zygote.@nograd function part_trace(J, m)
    k = size(J)[1]
    trace = zeros(Int16(k/m))
    for i in 1:m:k
        _tr = 1.0
        for j in 0:m-1
            _tr = _tr*J[i+j, i+j]
        end
        trace[Int16((i+1)/m)] = _tr
    end
    return trace
end

function rezidual_block(m, x)
    l, _n = size(x)
    #d = Distributions.Geometric(0.5)
    #@show n = rand(d, 1)[1] + 1
    n = 3
    #ϵ = randn(_n)
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

function single_block(m, x)
    _n = length(x)
    n = 4
    J = jacobian(m ,x)
    Jk = J
    Jkk = J
    sum_Jac = tr(J)
    for k in 2:n
        Jk = Jk * J
        Jkk = Jkk .+ rezidual_coef(k).*Jk
    end
    return tr(Jkk)
end

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

function rezflow_probability(m, x)
    z, log_prob = rezidual_flow(m, x)
    return log_prob .+ log_normal(z)
end
rezflow_probability(x) = rezflow_probability(m, x)

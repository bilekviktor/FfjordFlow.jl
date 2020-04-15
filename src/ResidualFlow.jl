using Flux, Distributions, LinearAlgebra
using Zygote

log_normal(x::AbstractVector) = - sum(x.^2) / 2 - length(x)*log(2π) / 2
log_normal(x) = -0.5f0 .* sum(x.^2, dims = 1) .- (size(x,1)*log(Float32(2π)) / 2)
lip_swish(x) = swish(x)/1.1

struct ResidualFlow{M}
    m::M
end

Base.show(io::IO, a::ResidualFlow) = print(io, "ResidualFlow{$(a.m)}")

Flux.@functor ResidualFlow
Flux.trainable(R::ResidualFlow) = (R.m, )
#---------------------------------------------------

function jacobian(f, x::AbstractVector)
  y::AbstractVector, back = Zygote.pullback(f, x)
  ȳ(i) = [i == j for j = 1:length(y)]
  vcat([transpose(back(ȳ(i))[1]) for i = 1:length(y)]...)
end

#function for computation of det of probability transformation
d = Geometric(0.5)
Zygote.@nograd sumnumber() = rand(d) + 1
Zygote.@nograd rezidual_coef(k) = (-1)^(k+1)/(k*ccdf(d, k-2))
#=
function rezidual_block(m, x)
    l, _n = size(x)
    n =
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
function residual_block(m, x)
    e = randn(size(x))
    _n = length(x)
    n = sumnumber()
    J = jacobian(m ,x)
    Jk = J
    sum_Jac = tr(J)
    for k in 2:n
        sum_Jac = sum_Jac + rezidual_coef(k) * (e' *(J*e))
    end
    sum_Jac
end
#=
function residual_flow(m, x)
    z = copy(x)
    log_prob = 0.0
    for i in 1:length(m)
        log_prob = log_prob + single_block(m[i], z)
        z = identity(z) .+ m[i](z)
    end
    return log_normal(z) + log_prob
end
=#
#function for computation of probability of given data
#=
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
function (R::ResidualFlow)(xx::Tuple{A, B}) where {A, B}
    x, logdet = xx
    z = copy(x)
    m = R.m
    log_det = copy(logdet')
    for i in 1:length(m)
        log_det = log_det .+ [residual_block(m[i], z[:, j]) for j in 1:size(z)[2]]
        z = z .+ m[i](z)
    end
    return (z, log_det')
end


(R::ResidualFlow)(x::AbstractArray) = x .+ R.m(x)

function Distributions.logpdf(R::ResidualFlow, x::AbstractMatrix{T}) where {T}
    y, logdet = m((x, 0.0))
    return vec(log_normal(y) + logdet)
end

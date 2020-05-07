using Flux, Distributions, LinearAlgebra
using Zygote

struct ResidualFlow{M}
    m::M
end

Base.show(io::IO, a::ResidualFlow) = print(io, "ResidualFlow{$(a.m)}")

Flux.@functor ResidualFlow
Flux.trainable(R::ResidualFlow) = (R.m, )
#---------------------------------------------------
function jacobian2(f, x)
    m = length(x)
    bf = Zygote.Buffer(x,m, m)
    for i in 1:m
        bf[i, :] = gradient(x -> f(x)[i], x)[1]
    end
    copy(bf)
end

d = Geometric(0.5)
Zygote.@nograd sumnumber() = rand(d) + 1
Zygote.@nograd rezidual_coef(k) = ((-1)^(k+1))/(k*ccdf(d, k-2))

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

function (R::ResidualFlow)(xx::Tuple{A, B}) where {A, B}
    x, logdet = xx
    z = copy(x)
    m = R.m
    log_det = copy(logdet')
    for i in 1:length(m)
        log_det = log_det .+ residual_block(m[i], z)
        z = z .+ m[i](z)
    end
    return (z, log_det')
end

function (R::ResidualFlow)(x::AbstractArray)
    m = R.m
    z = x
    for i in 1:length(m)
        z = z .+ m[i](z)
    end
    return z
end

function Distributions.logpdf(R::ResidualFlow, x::AbstractMatrix{T}) where {T}
    y, logdet = R((x, 0.0))
    return vec(log_normal(y) + logdet)
end

function Distributions.logpdf(R::ResidualFlow, x::Vector)
    y, logdet = R((x, 0.0))
    return log_normal(y) + logdet
end

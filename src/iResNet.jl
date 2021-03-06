using Flux, Distributions, LinearAlgebra
using Zygote

#this structure is not yet used in upcoming code
struct iResNet{M, I}
    m::M
    n::I
end

Base.show(io::IO, a::iResNet) = print(io, "iResNet{$(a.m)}")

Flux.@functor iResNet
Flux.trainable(R::iResNet) = (R.m, )
#---------------------------------------------------

Zygote.@nograd irezidual_coef(k) = ((-1)^(k+1))/k

function single_block(m, x, n)
    _n = length(x)
    J = jacobian(m ,x)
    Jk = J
    sum_Jac = tr(Jk)
    for k in 2:n
        Jk = Jk * J
        sum_Jac = sum_Jac + irezidual_coef(k) * tr(Jk)
    end
    sum_Jac
end

function (R::iResNet)(xx::Tuple{A, B}) where {A, B}
    x, logdet = xx
    z = copy(x)
    m = R.m
    log_det = copy(logdet')
    n = size(z, 2)
    for i in 1:length(m)
        mm = m[i][1]
        d = m[i][2]
        log_det = log_det .+ [single_block(mm, z[:, j], R.n) for j in 1:n]
        z = z .+ mm(z)

        log_det = log_det .+ (1 * log(abs(prod(d.α))))
        z = d(z)
    end
    return (z, log_det')
end


function (R::iResNet)(x::AbstractArray)
    m = R.m
    z = x
    for i in 1:length(m)
        z = z .+ m[i](z)
        z = 0.2 .* z
    end
    return 0.0
end

function Distributions.logpdf(R::iResNet, x::AbstractMatrix{T}) where {T}
    y, logdet = R((x, 0.0))
    return vec(log_normal(y) .+ logdet)
end


function Distributions.logpdf(R::iResNet, x::Vector) where {T}
    y, logdet = R((x', 0.0))
    return log_normal(y) + logdet
end

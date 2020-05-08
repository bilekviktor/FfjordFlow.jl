using FfjordFlow
using Test, Flux, Distributions, LinearAlgebra

@testset "ODE solver in Ffjord 1-D" begin
    A = hcat(0.5)
    b = vcat(0.0)
    f = Dense(A, b)
    x = [0.0 1.0]
    m = Ffjord(f, (0.0, 1.0))
    y = m(x)
    _y = exp(0.5) * x
    @test isapprox(y, _y, atol=1e-3)
end

@testset "ODE solver in Ffjord 2-D" begin
    A = [0.5 0.0; 0.0 0.3]
    b = [0.0; 0.0]
    ff = Dense(A, b)
    x = [1.0 2.0; 1.0 2.0]
    m = Ffjord(ff, (0.0, 1.0))
    y = m(x)
    _y = exp.(A) * x - x
    @test isapprox(y, _y, atol=1e-3)
end

@testset "Trivial check of calculated likelihood function via Ffjord 1-D" begin
    A = hcat(0.0)
    b = vcat(0.0)
    f = Dense(A, b)
    x = [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0]
    m = Ffjord(f, (0.0, 1.0))
    d = Distributions.Normal()
    lkh = mean(logpdf(m, x))
    _lkh = mean(logpdf.(d, x))
    @test isapprox(lkh, _lkh, atol=1e-3)
end

@testset "Trivial check of calculated likelihood function via Ffjord 2-D" begin
    A = zeros(2, 2)
    b = zeros(2)
    f = Dense(A, b)
    x = [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0;
         0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0]
    m = Ffjord(f, (0.0, 1.0))
    d = Distributions.MvNormal(2, 1.0)
    lkh = mean(logpdf(m, x))
    _lkh = mean(logpdf(d, x))
    @test isapprox(lkh, _lkh, atol=1e-3)
end

@testset "Likelihood training via Ffjord 2-D" begin
    mu = Float32[30.0; 40.0]
    sigma = Float32[1.0 0.0; 0.0 1.0]
    d = MvNormal(mu, sigma)
    x = rand(d, 100)
    m = Ffjord(Dense(2,2), (0.0, 1.0))
    loss() = -mean(logpdf(m, x))
    l1 = loss()
    ps = Flux.params(m)
    opt = ADAM(0.1)
    Flux.Optimise.train!(loss, ps, Iterators.repeated((), 30), opt)
    l2 = loss()
    @test 10*l2 < l1
end

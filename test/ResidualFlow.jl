using FfjordFlow
using Test, Flux, Distributions, LinearAlgebra
using FfjordFlow: rezidual_coef
using MLDataPattern, Zygote

@testset "Test of coef of iResNet" begin
    c1 = rezidual_coef(1)
    c2 = rezidual_coef(2)
    c3 = rezidual_coef(3)
    c4 = rezidual_coef(4)
    c5 = rezidual_coef(5)
    @test c1 ≈ 1
    @test c2 ≈ -1
    @test c3 ≈ 4/3
    @test c4 ≈ -2
    @test c5 ≈ 3.2
end

@testset "Testing structure of iResNet" begin
    A = [2.0 0.0; 0.0 2.0]
    b = [0.0; 0.0]
    x = [1.0; 2.0]
    m = ResidualFlow(Chain(Dense(A, b), Dense(A, b), Dense(A, b)))
    y = m(x)
    _y = x
    for i in 1:3
        _y = _y + m.m[i](_y)
    end
    @test y ≈ _y
end

@testset "Trivial check of calculated likelihood function via iResNet 1-D" begin
    A = hcat(0.0)
    b = vcat(0.0)
    f = Dense(A, b)
    x = [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0]
    m = ResidualFlow(Chain(Dense(A, b)))
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
    m = ResidualFlow(Chain(Dense(A, b)))
    d = Distributions.MvNormal(2, 1.0)
    lkh = mean(logpdf(m, x))
    _lkh = mean(logpdf(d, x))
    @test isapprox(lkh, _lkh, atol=1e-3)
end

@testset "Likelihood training via iResNet 2-D" begin
    mu = [30.0; 40.0]
    sigma = [1.0 0.0; 0.0 1.0]
    d = MvNormal(mu, sigma)
    x = rand(d, 100)
    y = RandomBatches(x, 100, 60)
    m = ResidualFlow(Chain(Chain(Dense(2,2))))
    loss(x) = -mean(logpdf(m, x))
    l1 = loss(x)
    ps = Flux.params(m)
    opt = ADAM(0.01)
    Flux.Optimise.train!(x -> loss(getobs(x)), ps, y, opt)
    l2 = loss(x)
    @test 5*l2 < l1
end

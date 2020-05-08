using FfjordFlow
using Test, Flux, Distributions, LinearAlgebra
using MLDataPattern, Zygote

@testset "Test of spectral normalization" begin
    sopt = SpecNormalization(1.0)
    A = f32([1000 -3000; 200 5000])
    B = copy(A)
    for i in 1:10
        Δ = Flux.Optimise.apply!(sopt, B, 0.0)
        B .-= Δ
    end
    @test norm(B) < norm(A)
    @test norm(B) < 2
end

@testset "Test of specttrain!" begin
    m = Chain(Dense(2, 10, tanh), Dense(10, 2))
    x = 20 .* randn(2, 30)
    loss(x) = sum(abs2, m(x))
    l1 = loss(x)
    opt = ADAM(0.1)
    sopt = SpecNormalization(0.2)
    ps = Flux.params(m)
    specttrain!(x -> loss(x), ps, Iterators.repeated(x, 20), opt, sopt, 1)
    l2 = loss(x)
    @test 5*l2 < l1
end

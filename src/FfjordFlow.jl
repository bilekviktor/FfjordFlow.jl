module FfjordFlow

include("utils.jl")
include("Cnf.jl")
include("Ffjord.jl")
include("iResNet.jl")
include("ResidualFlow.jl")
include("SpectralNormalization.jl")


Cnf(F::Ffjord) = Cnf(F.m, F.tspan, F.param)
iResNet(R::ResidualFlow, n) = iResNet(R.m, n)

export Ffjord, Cnf, iResNet
export ResidualFlow, SpecNormalization, specttrain!
end

module FfjordFlow

include("Cnf.jl")
include("Ffjord.jl")
include("iResNet.jl")
include("ResidualFlow.jl")
include("SpectralNormalization.jl")

export Ffjord, Cnf, iResNet
export ResidualFlow, SpecNormalization, specttrain!
end

"""Pure Julia DDA engine with CPU and optional CUDA backends."""
module NativeDDA

using Distributed
using LinearAlgebra
using Statistics
using ..DDADefaults
using ..OptionalDeps
using ..ModelEncoding: generate_monomials

include("NativeDDA/Types.jl")
include("NativeDDA/Preparation.jl")
include("NativeDDA/Problems.jl")
include("NativeDDA/CPU.jl")
include("NativeDDA/GPU.jl")
include("NativeDDA/Engine.jl")
include("NativeDDA/Parallel.jl")

end

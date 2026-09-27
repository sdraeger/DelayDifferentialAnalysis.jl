using Test
using DelayDifferentialAnalysis

# Core module tests
include("test_variants.jl")
include("test_runner.jl")

# New module tests
include("test_results.jl")
include("test_model_encoding.jl")
include("test_native_dda.jl")
include("test_native_parallel.jl")
include("test_structure_selection.jl")
include("test_claudia_contract.jl")
include("test_batch.jl")
include("test_stats.jl")
include("test_api.jl")
include("test_plotting.jl")

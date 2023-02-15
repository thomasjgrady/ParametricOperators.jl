module ParametricOperators

import Base: *, ∘
import Base: adjoint, kron

using ChainRulesCore
using CUDA
using DataStructures: DefaultDict
using Distributions
using UUIDs

if CUDA.functional()
    @info "ParametricOperators.jl succesfully loaded CUDA.jl. Methods for CUDA types will be defined."
end

include("ParCommon.jl")
include("ParOperator.jl")

include("ParAdjoint.jl")
include("ParParameterized.jl")

include("ParCompose.jl")
include("ParKron.jl")

include("ParMatrix.jl")

end
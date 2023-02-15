export ParCompose

"""
Composition operator.
"""
struct ParCompose{D,R,L,P,E,F,N} <: ParOperator{D,R,L,P,Internal,E}
    ops::F
    function ParCompose(ops...)
        ops = collect(ops)
        N = length(ops)
        if N == 1
            return ops[1]
        end
        @ignore_derivatives begin
            for i in 1:N-1
                @assert Domain(ops[i]) == Range(ops[i+1])
                @assert DDT(ops[i]) == RDT(ops[i+1])
            end
        end

        D = DDT(ops[N])
        R = RDT(ops[1])
        L = foldl(promote_linearity, map(linearity, ops))
        P = foldl(promote_parametricity, map(parametricity, ops))

        @assert allequal(device(x) for x in ops)
        E = device(ops[1])

        return new{D,R,L,P,E,typeof(ops),length(ops)}(ops)
    end
end

∘(ops::ParOperator...) = ParCompose(ops...)
∘(A::ParCompose, op::ParOperator) = ParCompose(A.ops..., op)
∘(op::ParOperator, A::ParCompose) = ParCompose(op, A.ops...)
∘(A::ParCompose, B::ParCompose) = ParCompose(A.ops..., B.ops...)
*(ops::ParLinearOperator...) = ∘(ops...)

Domain(A::ParCompose{D,R,L,P,E,F,N}) where {D,R,L,P,E,F,N} = Domain(A.ops[N])
Range(A::ParCompose{D,R,L,P,E,F,N}) where {D,R,L,P,E,F,N} = Range(A.ops[1])
children(A::ParCompose) = A.ops
rebuild(::ParCompose, cs) = ParCompose(cs...)

adjoint(A::ParCompose{D,R,Linear,P,E,F,N}) where {D,R,P,E,F,N} = ParCompose(reverse(map(adjoint, A.ops))...)

function (A::ParCompose{D,R,L,<:Applicable,E,F,N})(x::X) where {D,R,L,E,F,N,X<:AbstractVector{D}}
    for i in 1:N
        x = A.ops[N-i+1](x)
    end
    return x
end

function (A::ParCompose{D,R,L,<:Applicable,E,F,N})(x::X) where {D,R,L,E,F,N,X<:AbstractMatrix{D}}
    for i in 1:N
        x = A.ops[N-i+1](x)
    end
    return x
end

function *(x::X, A::ParCompose{D,R,Linear,<:Applicable,E,F,N}) where {D,R,E,F,N,X<:AbstractMatrix{R}}
    for i in 1:N
        x = x*A.ops[i]
    end
    return x
end
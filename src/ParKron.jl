export ParSeparableOperator, ParKron, ⊗
export reorder

abstract type ParSeparableOperator{D,R,P,T,E} <: ParLinearOperator{D,R,P,T,E} end

order(::ParSeparableOperator) = throw(ParException("Unimplemented"))

struct ParKron{D,R,P,E,F,N} <: ParSeparableOperator{D,R,P,Internal,E}
    ops::F
    order::Vector{Int}
    function ParKron(ops...)

        # Collect operators into a vector
        ops = collect(ops)
        N = length(ops)

        # We can't Kronecker only a single operator
        if N == 1
            return ops[1]
        end

        # Find the domain type which is the most "sub" type
        DDTs = map(DDT, ops)
        RDTs = map(RDT, ops)
        T = foldl(subset_type, DDTs)

        # Compute operator application order
        order = zeros(Int, N)
        @ignore_derivatives begin
            for i in 1:N

                # Find all operator indices with the current domain type
                candidates = filter(j -> DDTs[j] == T && j ∉ order, 1:N)

                # From the candidates, find the range type which is the most
                # "sub" type and filter candidates
                R = mapreduce(j -> RDTs[j], subset_type, candidates)
                candidates = filter(j -> RDTs[j] == R, candidates)

                # Filter the candidates to those which minimize the array size
                # on output
                differences = map(j -> Domain(ops[j]) - Range(ops[j]), candidates)
                max_difference = maximum(differences)
                candidates = filter(j -> (Domain(ops[j]) - Range(ops[j])) == max_difference, candidates)

                # Finally, if there is more than one potential candidate, select
                # in right-to-left order
                order[i] = candidates[end]
                T = RDTs[order[i]]
            end
        end
        
        D = DDT(ops[order[1]])
        R = RDT(ops[order[N]])
        P = foldl(promote_parametricity, map(parametricity, ops))

        @assert allequal(device(op) for op in ops)
        E = device(ops[1])

        return new{D,R,P,E,typeof(ops),N}(ops, order)
    end

    function ParKron(D,R,P,E,ops,order)
        return new{D,R,P,E,typeof(ops),length(ops)}(ops, order)
    end
end

kron(A::ParLinearOperator, B::ParLinearOperator) = ParKron(A, B)
kron(A::ParKron, B::ParLinearOperator) = ParKron(A.ops..., B)
kron(A::ParLinearOperator, B::ParKron) = ParKron(A, B.ops...)
kron(A::ParKron, B::ParKron) = ParKron(A.ops..., B.ops...)
⊗(A::ParLinearOperator, B::ParLinearOperator) = kron(A, B)

Domain(A::ParSeparableOperator) = prod(map(Domain, children(A)))
Range(A::ParSeparableOperator) = prod(map(Range, children(A)))

children(A::ParKron) = A.ops
rebuild(::ParKron, cs) = ParKron(cs...)
adjoint(A::ParKron{D,R,P,F,E,N}) where {D,R,P,F,E,N} = ParKron(R,D,P,E,collect(map(adjoint, A.ops)), reverse(A.order))
order(A::ParKron) = A.order

function _kron_impl(A::ParKron{D,R,<:Applicable,E,F,N}, x::X) where {D,R,E,F,N,X<:AbstractMatrix}
    
    # Reshape to input shape
    b = size(x)[2]
    s = reverse(collect(map(Domain, A.ops)))
    x = reshape(x, s..., b)

    # Apply operators in order, permuting to enforce leading dim of x to
    # align with current operator
    x = rotate_dims_batched(x, -(N-A.order[1]))

    for i in 1:N
        o = A.order[i]
        s = size(x)
        x = as_matrix(x)
        Ai = A.ops[o]
        x = Ai*x
        x = reshape(x, Range(Ai), s[2:end]...)
        if i < N
            x = rotate_dims_batched(x, N-o-(N-A.order[i+1]))
        else
            x = rotate_dims_batched(x, o)
        end
    end

    nelem = prod(size(x))
    return reshape(x, nelem÷b, b)
end

(A::ParKron{D,R,<:Applicable,CPU,F,N})(x::X) where {D,R,F,N,X<:AbstractMatrix{D}} = _kron_impl(A, x)
(A::ParKron{D,R,<:Applicable,GPU,F,N})(x::CuMatrix{D}) where {D,R,F,N} = _kron_impl(A, x)

(A::ParKron{D,R,<:Applicable,E,F,N})(x::X) where {D,R,E,F,N,X<:AbstractVector{D}} =
    vec(A(reshape(x, length(x), 1)))
export ParParameterized

struct ParParameterized{D,R,L,E,F,V} <: ParOperator{D,R,L,Parameterized,Internal,E}
    op::F
    params::V
    ParParameterized(op, params) = new{DDT(op),RDT(op),linearity(op),device(op),typeof(op),typeof(params)}(op, params)
end

Domain(A::ParParameterized) = Domain(A.op)
Range(A::ParParameterized) = Range(A.op)

parameterize(A::ParOperator{D,R,L,Parametric,External,E}, params::AbstractDict) where {D,R,L,E} = ParParameterized(A, params[A])
parameterize(A::ParOperator{D,R,L,Parametric,External,E}, params) where {D,R,L,E} = ParParameterized(A, params)
children(A::ParParameterized) = [A.op]
rebuild(A::ParParameterized, cs) = ParParameterized(cs[0], A.params)
adjoint(A::ParParameterized) = ParParameterized(adjoint(A.op), A.params)
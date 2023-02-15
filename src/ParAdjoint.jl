export ParAdjoint

struct ParAdjoint{D,R,P,E,F} <: ParLinearOperator{R,D,P,Internal,E}
    op::F
    ParAdjoint(op) = new{DDT(op),RDT(op),parametricity(op),device(op),typeof(op)}(op)
end

Domain(A::ParAdjoint) = Range(A.op)
Range(A::ParAdjoint) = Domain(A.op)
children(A::ParAdjoint) = [A.op]

adjoint(A::ParLinearOperator) = ParAdjoint(A)
adjoint(A::ParAdjoint) = A.op
(A::ParAdjoint{D,R,Parametric,E,F})(params) where {D,R,E,F} = adjoint(A.op(params))
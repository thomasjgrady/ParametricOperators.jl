export Linearity, Linear, NonLinear
export Parametricity, Parametric, NonParametric, Parameterized, Applicable
export ASTLocation, Internal, External
export Device, CPU, GPU

# ==== Type Definitions ====

abstract type Linearity end
struct Linear <: Linearity end
struct NonLinear <: Linearity end

abstract type Parametricity end
struct Parametric <: Parametricity end
struct NonParametric <: Parametricity end
struct Parameterized <: Parametricity end

const Applicable = Union{NonParametric, Parameterized}

abstract type ASTLocation end
struct Internal <: ASTLocation end
struct External <: ASTLocation end

abstract type Device end
struct CPU <: Device end
struct GPU <: Device end

abstract type ParOperator{
    D,
    R,
    L <: Linearity,
    P <: Parametricity,
    T <: ASTLocation,
    E <: Device
} end

const ParLinearOperator{D,R,P,T,E} = ParOperator{D,R,Linear,P,T,E}

# ==== Type Helper Functions ====

export promote_linearity, promote_parametricity

promote_linearity(::Type{Linear}, ::Type{Linear}) = Linear
promote_linearity(::Type{<:Linearity}, ::Type{<:Linearity}) = NonLinear

promote_parametricity(::Type{NonParametric}, ::Type{NonParametric}) = NonParametric
promote_parametricity(::Type{NonParametric}, ::Type{Parameterized}) = Parameterized
promote_parametricity(::Type{Parameterized}, ::Type{NonParametric}) = Parameterized
promote_parametricity(::Type{Parameterized}, ::Type{Parameterized}) = Parameterized
promote_parametricity(::Type{<:Parametricity}, ::Type{<:Parametricity}) = Parametric

# ==== Operator Traits ====

export DDT, RDT, linearity, parametricity, ast_location, device
export Domain, Range
export children, init

DDT(::ParOperator{D,R,L,P,T,E}) where {D,R,L,P,T,E} = D
RDT(::ParOperator{D,R,L,P,T,E}) where {D,R,L,P,T,E} = R
linearity(::ParOperator{D,R,L,P,T,E}) where {D,R,L,P,T,E} = L
parametricity(::ParOperator{D,R,L,P,T,E}) where {D,R,L,P,T,E} = P
ast_location(::ParOperator{D,R,L,P,T,E}) where {D,R,L,P,T,E} = T
device(::ParOperator{D,R,L,P,T,E}) where {D,R,L,P,T,E} = E
id(::ParOperator) = throw(ParException("Unimplemented"))

Domain(::ParOperator) = throw(ParException("Unimplemented"))
Range(::ParOperator) = throw(ParException("Unimplemented"))

children(::ParOperator{D,R,L,P,External,E}) where {D,R,L,P,E} = []
children(::ParOperator{D,R,L,P,Internal,E}) where {D,R,L,P,E} = throw(ParException("Unimplemented"))

# ==== Operator Parameterization ====

init!(::ParOperator{D,R,L,<:Applicable,T,E}, params) where {D,R,L,T,E} = identity
init!(::ParOperator{D,R,L,<:Applicable,T,E}, params, init_dict) where {D,R,L,T,E} = identity
init!(::ParOperator{D,R,L,Parametric,External,E}, params) where {D,R,L,E} = throw(ParException("Unimplemented"))
init!(::ParOperator{D,R,L,Parametric,External,E}, params, init_dict) where {D,R,L,E} = throw(ParException("Unimplemented"))

function init!(A::ParOperator{D,R,L,Parametric,Internal,E}, params) where {D,R,L,E}
    for c in children(A)
        init!(c, params)
    end
end

function init!(A::ParOperator{D,R,L,Parametric,Internal,E}, params, init_dict) where {D,R,L,E}
    for c in children(A)
        init!(c, params, init_dict)
    end
end

function init(A::ParOperator)
    params = Dict()
    init!(A, params)
    return params
end

function init(A::ParOperator, init_dict)
    params = Dict()
    init!(A, params, init_dict)
    return params
end

rebuild(A::ParOperator{D,R,L,P,External,E}, _) where {D,R,L,P,E} = A
rebuild(A::ParOperator{D,R,L,P,Internal,E}, cs) where {D,R,L,P,E} = throw(ParException("Unimplemented"))

parameterize(A::ParOperator{D,R,L,<:Applicable,T,E}, _) where {D,R,L,T,E} = A
parameterize(A::ParOperator{D,R,L,Parametric,Internal,E}, params) where {D,R,L,E} = A(params)

(A::ParOperator{D,R,L,Parametric,External,E})(params) where {D,R,L,E} = parameterize(A, params)
(A::ParOperator{D,R,L,Parametric,Internal,E})(params) where {D,R,L,E} =
    rebuild(A, [parameterize(op, params) for op in children(A)])

# ==== Operator Functionality ====

(A::ParOperator{D,R,L,<:Applicable,T,E})(x::AbstractVector) where {D,R,L,T,E} = throw(ParException("Unimplemented"))
(A::ParOperator{D,R,L,<:Applicable,T,E})(x::AbstractMatrix) where {D,R,L,T,E} = mapreduce(col -> A(col), hcat, eachcol(x))

*(A::ParLinearOperator, x::AbstractVector) = A(x)
*(A::ParLinearOperator, x::AbstractMatrix) = A(x)
*(x::AbstractMatrix, A::ParLinearOperator) = (A'*x')'
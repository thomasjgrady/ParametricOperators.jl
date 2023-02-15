export ParMatrix

struct ParMatrix{T,E} <: ParLinearOperator{T,T,Parametric,External,E}
    m::Int
    n::Int
    id::UUID
    ParMatrix(T, m, n; device=CPU, id=uuid4()) = new{T,device}(m, n, id)
    ParMatrix(m, n; device=CPU, id=uuid4()) = ParMatrix(Float64, m, n; device=device, id=id)
end

Domain(A::ParMatrix) = A.n
Range(A::ParMatrix) = A.m
id(A::ParMatrix) = A.id

function init!(A::ParMatrix{T,CPU}, params) where {T}
    params[A] = rand(T, A.m, A.n)/sqrt(A.m*A.n)
end

function init!(A::ParMatrix{T,GPU}, params) where {T}
    params[A] = CUDA.rand(T, A.m, A.n)/sqrt(A.m*A.n)
end

(A::ParParameterized{T,T,Linear,CPU,ParMatrix{T,CPU},V})(x::X) where {T,V<:AbstractMatrix{T},X<:AbstractVector{T}} = A.params*x
(A::ParParameterized{T,T,Linear,CPU,ParMatrix{T,CPU},V})(x::X) where {T,V<:AbstractMatrix{T},X<:AbstractMatrix{T}} = A.params*x
(A::ParParameterized{T,T,Linear,GPU,ParMatrix{T,GPU},CuMatrix{T}})(x::CuVector{T}) where {T} = A.params*x
(A::ParParameterized{T,T,Linear,GPU,ParMatrix{T,GPU},CuMatrix{T}})(x::CuMatrix{T}) where {T} = A.params*x

(A::ParParameterized{T,T,Linear,CPU,ParAdjoint{T,T,Parametric,CPU,ParMatrix{T,CPU}},V})(x::X) where {T,V<:AbstractMatrix{T},X<:AbstractVector{T}} = A.params'*x
(A::ParParameterized{T,T,Linear,CPU,ParAdjoint{T,T,Parametric,CPU,ParMatrix{T,CPU}},V})(x::X) where {T,V<:AbstractMatrix{T},X<:AbstractMatrix{T}} = A.params'*x
(A::ParParameterized{T,T,Linear,GPU,ParAdjoint{T,T,Parametric,GPU,ParMatrix{T,GPU}},CuMatrix{T}})(x::CuVector{T}) where {T} = A.params'*x
(A::ParParameterized{T,T,Linear,GPU,ParAdjoint{T,T,Parametric,GPU,ParMatrix{T,GPU}},CuMatrix{T}})(x::CuMatrix{T}) where {T} = A.params'*x

*(x::X, A::ParParameterized{T,T,Linear,CPU,ParMatrix{T,CPU},V}) where {T,V<:AbstractMatrix{T},X<:AbstractMatrix{T}} = x*A.params
*(x::CuMatrix{T}, A::ParParameterized{T,T,Linear,GPU,ParMatrix{T,GPU},CuMatrix{T}}) where {T} = x*A.params

*(x::X, A::ParParameterized{T,T,Linear,CPU,ParAdjoint{T,T,Parametric,CPU,ParMatrix{T,CPU}},V}) where {T,V<:AbstractMatrix{T},X<:AbstractMatrix{T}} = x*A.params'
*(x::CuMatrix{T}, A::ParParameterized{T,T,Linear,GPU,ParAdjoint{T,T,Parametric,GPU,ParMatrix{T,GPU}},CuMatrix{T}}) where {T} = x*A.params'
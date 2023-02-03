export ParDFT

"""
Fouier transform operator. Dispatchs real-valued fft if applicable.
"""
struct ParDFT{D,R} <: ParLinearOperator{D,R,NonParametric,External}
    m::Int
    n::Int
    function ParDFT(T, n)
        if T <: Real
            return new{T,Complex{T}}(n÷2+1, n)
        elseif T <: Complex
            return new{T,T}(n, n)
        else
            throw(ParException("Invalid type $T for DFT"))
        end
    end
    ParDFT(n) = ParDFT(ComplexF64, n)
end

Domain(A::ParDFT) = A.n
Range(A::ParDFT) = A.m

(A::ParDFT{D,R})(x::X) where {D<:Complex,R,X<:AbstractMatrix{D}} = fft(x, 1)./sqrt(A.n)
(A::ParDFT{D,R})(x::X) where {D<:Real,R,X<:AbstractMatrix{D}} = rfft(x, 1)./sqrt(A.n)
(A::ParDFT{D,R})(x::X) where {D,R,X<:AbstractVector{D}} = vec(A(reshape(x, length(x), 1)))

(A::ParAdjoint{D,R,NonParametric,ParDFT{D,R}})(x::X) where {D<:Complex,R,X<:AbstractMatrix{R}} = ifft(x, 1).*sqrt(A.op.n)
(A::ParAdjoint{D,R,NonParametric,ParDFT{D,R}})(x::X) where {D<:Real,R,X<:AbstractMatrix{R}} = irfft(x, A.op.n, 1).*sqrt(A.op.n)
(A::ParAdjoint{D,R,NonParametric,ParDFT{D,R}})(x::X) where {D,R,X<:AbstractVector{R}} = vec(A(reshape(x, length(x), 1)))

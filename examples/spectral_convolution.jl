using ParametricOperators

m = SomeMachineModel()

# Number of channels, spatial, and time dimensions
(nc, nx, ny, nt) = (20, 64, 64, 50)

# Number of Fourier modes to keep in each dimension
(mx, my, mt) = (4, 4, 4)

# Base element type
T = Float64

# Identity matrix on channels
Ic = ParIdentity(Complex{T}, nc)

# Declare Fourier transform through Kronecker products
Fx = ParDFT(nx)
Fy = ParDFT(ny)
Ft = ParDFT(nt)
F = Ic ⊗ Ft ⊗ Fy ⊗ Fx

# Declare restriction through Kronecker products
Rx = ParRestriction(Complex{T}, nx, [1:mx, nx-mx+1:nx])
Ry = ParRestriction(Complex{T}, ny, [1:my, ny-my+1:ny])
Rt = ParRestriction(Complex{T}, nt, [1:mt])
R = Ic ⊗ Rt ⊗ Ry ⊗ Rx

# Elementwise weighting and channel mixing operator
D = ParMatrix(Complex{T}, nc, nc) ⊗ ParDiagonal(Complex{T}, Range(R)÷nc)

# Full spectral convolution
S = F'*R'*D*R*F

# Initialize parameters and apply to a random vector
θ = init(S)
x = rand(DDT(S), Domain(S))
y = S(θ)*x

Sc = complexity(S, m)
println("complexity is $Sc for F'*R'*D*R*F.")

RF = (Ic*Ic) ⊗ (Rt*Ft) ⊗ (Ry*Fy) ⊗ (Rx*Fx)
S2 = RF'*D*RF

S2c = complexity(S2, m)
println("complexity is $S2c after interleaving restrictions")

y2 = S2(θ)*x

maxerror = maximum(abs.(y-y2))
println("max error is $maxerror")

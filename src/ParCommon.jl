struct ParException <: Exception
    msg
end

FLOAT_TYPES = [:Float16, :Float32, :Float64]
for i in 1:length(FLOAT_TYPES)
    for j in i:length(FLOAT_TYPES)
        @eval begin
            subset_type(::Type{$(FLOAT_TYPES[i])}, ::Type{$(FLOAT_TYPES[j])}) = $(FLOAT_TYPES[i])
            subset_type(::Type{Complex{$(FLOAT_TYPES[i])}}, ::Type{Complex{$(FLOAT_TYPES[j])}}) = Complex{$(FLOAT_TYPES[i])}
            subset_type(::Type{$(FLOAT_TYPES[i])}, ::Type{Complex{$(FLOAT_TYPES[j])}}) = $(FLOAT_TYPES[i])
            subset_type(::Type{Complex{$(FLOAT_TYPES[j])}}, ::Type{$(FLOAT_TYPES[i])}) = $(FLOAT_TYPES[i])
            superset_type(::Type{$(FLOAT_TYPES[i])}, ::Type{$(FLOAT_TYPES[j])}) = $(FLOAT_TYPES[j])
            superset_type(::Type{Complex{$(FLOAT_TYPES[i])}}, ::Type{Complex{$(FLOAT_TYPES[j])}}) = Complex{$(FLOAT_TYPES[i])}
            superset_type(::Type{$(FLOAT_TYPES[i])}, ::Type{Complex{$(FLOAT_TYPES[j])}}) = Complex{$(FLOAT_TYPES[i])}
            superset_type(::Type{Complex{$(FLOAT_TYPES[j])}}, ::Type{$(FLOAT_TYPES[i])}) = Complex{$(FLOAT_TYPES[i])}
        end
    end
end

function rotate_dims_batched(x, rot)
    n = length(size(x))
    perm = [circshift(collect(1:n-1), rot)..., n]
    return permutedims(x, perm)
end

function as_matrix(x)
    s = size(x)
    return reshape(x, s[1], prod(s[2:end]))
end
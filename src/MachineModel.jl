export MachineModel
export MultCost, AddCost
export SomeMachineModel

"""
A fake machine for estimating execution costs.
"""
struct MachineModel
    elementwise_multiplication_costs::Dict{Type,Float64}
    elementwise_addition_costs::Dict{Type,Float64}
    # TODO: maybe memory footprint?
    # TODO: maybe network transfer costs?
    # TODO: maybe GPU computation and memory?
    # TODO: maybe power consumption?
end

function MultCost(m::MachineModel, t::Type)::Float64
    if haskey(m.elementwise_multiplication_costs, t) == false
        throw(ErrorException("I don't know how much it costs to multiply two $t together"))
    end
    return m.elementwise_multiplication_costs[t]
end

function AddCost(m::MachineModel, t::Type)::Float64
    if haskey(m.elementwise_addition_costs, t) == false
        throw(ErrorException("I don't know how much it costs to add two $t together"))
    end
    return m.elementwise_addition_costs[t]
end

function SomeMachineModel()::MachineModel
    # TODO: get some accurate numbers for this
    a = MachineModel(
        # elementwise multiplication costs
        Dict([
            # made-up costs for multiplying floating point types, assume perfect vectorization
            (Float16, 0.5),
            (Float32, 1.0),
            (Float64, 2.0),
            (ComplexF16, 1.0),
            (ComplexF32, 2.0),
            (ComplexF64, 4.0),
            # made-up costs for multiplying digital/integer types, assume perfect vectorization
            (Bool, 0.0078125),
            (Int8, 0.0625),
            (Int16, 0.125),
            (Int32, 0.25),
            (Int64, 0.5),
            (Int128, 1.0),
            (UInt8, 0.0625),
            (UInt16, 0.125),
            (UInt32, 0.25),
            (UInt64, 0.5),
            (UInt128, 1.0),
        ]),
        # elementwise addition costs
        Dict([
            # made-up costs for adding floating point types, assume perfect vectorization
            (Float16, 0.25),
            (Float32, 0.5),
            (Float64, 1.0),
            (ComplexF16, 0.5),
            (ComplexF32, 1.0),
            (ComplexF64, 2.0),
            # made-up costs for adding digital/integer types, assume perfect vectorization
            (Bool, 0.00390625),
            (Int8, 0.03125),
            (Int16, 0.0625),
            (Int32, 0.125),
            (Int64, 0.25),
            (Int128, 0.5),
            (UInt8, 0.03125),
            (UInt16, 0.0625),
            (UInt32, 0.125),
            (UInt64, 0.25),
            (UInt128, 0.5),
        ]),
    )
    a
end

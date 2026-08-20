function admission_sparse_vector(coordinates)
    data = zeros(Float64, maximum(coordinates))
    for (position, coordinate) in pairs(coordinates)
        data[coordinate] = isodd(position) ? Float64(position) : -Float64(position)
    end
    tensor = Tensor(SparseList(Element(0.0)), data)
    @test tensor.lvl.idx == coordinates
    return tensor, data
end

function admission_description(tensor)
    structure = Regularity.Structure((
        Regularity.CompressedLevel(tensor.lvl.ptr, tensor.lvl.idx),
    ))
    description = Regularity.mine(
        structure;
        families=(Regularity.PeriodicAffineConfig(; pmax=RX.SPECIALIZE_PMAX[]),),
        min_run=1,
        leaf_min_run=1,
        max_nodes=typemax(Int),
    )
    return description, structure
end

function admission_copy_code(A, output; specialize, report=nothing)
    if specialize
        return string(@finch_code specialize = true report = report begin
            output .= 0.0
            for i in _
                output[i] = A[i]
            end
        end)
    end
    return string(@finch_code begin
        output .= 0.0
        for i in _
            output[i] = A[i]
        end
    end)
end

function admission_copy_result(A, size; specialize)
    output = Tensor(Dense(Element(0.0)), zeros(size))
    if specialize
        @finch specialize = true begin
            output .= 0.0
            for i in _
                output[i] = A[i]
            end
        end
    else
        @finch begin
            output .= 0.0
            for i in _
                output[i] = A[i]
            end
        end
    end
    return Array(output)
end

function with_admission_limit(f, limit)
    old_limit = RX.SPECIALIZE_MAX_EMITTED_PHASES_AND_CASES[]
    try
        RX.SPECIALIZE_MAX_EMITTED_PHASES_AND_CASES[] = limit
        return f()
    finally
        RX.SPECIALIZE_MAX_EMITTED_PHASES_AND_CASES[] = old_limit
    end
end

function separated_short_claims(count)
    coordinates = Int[]
    first_coordinate = 1
    for claim in 1:count
        append!(coordinates, first_coordinate:(first_coordinate + 2))
        first_coordinate += 4 + mod(37claim, 101)
    end
    return coordinates
end

@testset "admission depends only on emitted code size" begin
    # Three claimed cells among two hundred opaque cells are intentionally far
    # below half coverage. The claim is also shorter than the former leaf
    # profitability threshold, yet it is exact evidence and must be realized.
    low_coordinates = vcat(collect(1:3), [13 + i * i for i in 1:200])
    low_A, low_data = admission_sparse_vector(low_coordinates)
    low_description, low_structure = admission_description(low_A)
    low_tallies = Regularity.coverage(low_description, low_structure)
    @test 2 * sum(low_tallies.claimed) < sum(low_tallies.total)
    @test any(
        node -> node isa Regularity.Segment && node.count isa Int && node.count < 4,
        low_description.nodes,
    )

    low_output = Tensor(Dense(Element(0.0)), zeros(length(low_data)))
    low_generic = admission_copy_code(low_A, low_output; specialize=false)
    low_report = Ref{F.SpecializeReport}()
    low_specialized = with_admission_limit(typemax(Int)) do
        admission_copy_code(low_A, low_output; specialize=true, report=low_report)
    end
    @test low_specialized != low_generic
    @test low_report[].realized == 1
    @test !low_report[].declined

    generic_result = admission_copy_result(low_A, length(low_data); specialize=false)
    specialized_result = admission_copy_result(low_A, length(low_data); specialize=true)
    @test reinterpret(UInt64, specialized_result) == reinterpret(UInt64, generic_result)
    @test reinterpret(UInt64, specialized_result) == reinterpret(UInt64, low_data)
    @test findall(!iszero, specialized_result) == low_coordinates

    # 103 independent three-cell claims carry 515 description units. Their
    # emitted program is still within the sole code-size budget.
    many_coordinates = separated_short_claims(103)
    many_A, many_data = admission_sparse_vector(many_coordinates)
    many_description, _ = admission_description(many_A)
    @test Regularity.description_units(many_description) > 512
    @test count(node -> node isa Regularity.Segment, many_description.nodes) == 103
    @test all(
        node -> !(node isa Regularity.Segment) || node.count == 3,
        many_description.nodes,
    )

    many_output = Tensor(Dense(Element(0.0)), zeros(length(many_data)))
    many_generic = admission_copy_code(many_A, many_output; specialize=false)
    high_report = Ref{F.SpecializeReport}()
    high_code = with_admission_limit(typemax(Int)) do
        admission_copy_code(many_A, many_output; specialize=true, report=high_report)
    end
    @test high_code != many_generic
    @test high_report[].realized == 1
    @test !high_report[].declined
    emitted_total = high_report[].emitted_sequence_phases +
                    high_report[].emitted_switch_cases
    @test emitted_total > 1

    boundary_report = Ref{F.SpecializeReport}()
    boundary_code = with_admission_limit(emitted_total) do
        admission_copy_code(
            many_A, many_output; specialize=true, report=boundary_report)
    end
    @test boundary_code == high_code
    @test !boundary_report[].declined
    @test boundary_report[].emitted_sequence_phases +
          boundary_report[].emitted_switch_cases == emitted_total

    overflow_report = Ref{F.SpecializeReport}()
    overflow_code = with_admission_limit(emitted_total - 1) do
        admission_copy_code(
            many_A, many_output; specialize=true, report=overflow_report)
    end
    @test overflow_code == many_generic
    @test overflow_report[].declined
    @test overflow_report[].reason === :emitted_phases_and_cases

    # A dense-only reader still has no structural indirection to specialize.
    dense_A = Tensor(Dense(Element(0.0)), low_data)
    dense_output = Tensor(Dense(Element(0.0)), zeros(length(low_data)))
    dense_generic = admission_copy_code(dense_A, dense_output; specialize=false)
    dense_report = Ref{F.SpecializeReport}()
    dense_specialized = admission_copy_code(
        dense_A, dense_output; specialize=true, report=dense_report)
    @test dense_specialized == dense_generic
    @test dense_report[].realized == 0
    @test !dense_report[].declined
end

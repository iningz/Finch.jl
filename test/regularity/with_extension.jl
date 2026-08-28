function banded_data(n, halfwidth)
    data = zeros(Float64, n, n)
    for j in 1:n, i in max(1, j - halfwidth):min(n, j + halfwidth)
        data[i, j] = (17i + 31j) / 1024
    end
    return data
end

function unclipped_banded_data(n, halfwidth)
    data = zeros(Float64, n + 2halfwidth, n)
    for j in 1:n, i in j:(j + 2halfwidth)
        data[i, j] = (17i + 31j) / 1024
    end
    return data
end

function hidden_span_data(spans)
    coordinates = vcat(collect(1:16), [100, 200, 301], collect(400:415))
    data = zeros(Float64, maximum(spans), last(coordinates), length(spans))
    for (k, span) in pairs(spans), (slot, j) in pairs(coordinates)
        width = slot in 17:19 ? span : maximum(spans)
        for i in 1:width
            data[i, j, k] = isodd(i + j + k) ? 1.0e12 + i : -1.0e12 + j
        end
    end
    return data
end

function periodic_count_data(ncolumns)
    counts = [2 + (isodd(j) ? (j + 1) ÷ 2 : j ÷ 2 + 1) for j in 1:ncolumns]
    data = zeros(Float64, maximum(counts), ncolumns)
    for j in 1:ncolumns, i in 1:counts[j]
        data[i, j] = (13i + 29j) / 2048
    end
    return data
end

function parent_dependent_data(nouter, nmiddle)
    data = zeros(Float64, nouter + nmiddle + 1, nmiddle, nouter)
    for k in 1:nouter, j in 1:nmiddle, i in 1:(k + j + 1)
        data[i, j, k] = isodd(i + j + k) ? 1.0e12 + i : -1.0e12 + j
    end
    return data
end

function cancellation_fibers(n)
    # The storage-ordered sum is 2.0. Moving either large term across a small
    # one changes the rounded result, so bitwise equality fingerprints the
    # visit order without exposing any emitter state.
    values = (2.0^54, 1.0, -(2.0^54), -1.0, 3.0)
    data = zeros(Float64, n + length(values) - 1, n)
    for j in 1:n, (offset, value) in enumerate(values)
        data[j + offset - 1, j] = value
    end
    return data
end

function ordered_bits(x::Float64)
    bits = reinterpret(UInt64, x)
    sign = UInt64(1) << 63
    return bits & sign == 0 ? bits | sign : ~bits
end

function ulps_apart(xs, ys)
    axes(xs) == axes(ys) || return typemax(UInt64)
    distance = UInt64(0)
    for (x, y) in zip(xs, ys)
        isnan(x) && isnan(y) && continue
        ox, oy = ordered_bits(x), ordered_bits(y)
        distance = max(distance, max(ox, oy) - min(ox, oy))
    end
    return distance
end

source_lines(code) = count(==('\n'), code) + 1
ptr_reads(code) = length(collect(eachmatch(r"ptr\w*\s*\[", code)))
idx_reads(code) = length(collect(eachmatch(r"idx\w*\s*\[", code)))

function spmv_results(format, data)
    A = Tensor(format, data)
    x = Tensor(Dense(Element(0.0)), [1.0 + j / 32 for j in axes(data, 2)])
    generic = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    specialized = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))

    @finch begin
        generic .= 0.0
        for j in _, i in _
            generic[i] += A[i, j] * x[j]
        end
    end
    @finch specialize = true begin
        specialized .= 0.0
        for j in _, i in _
            specialized[i] += A[i, j] * x[j]
        end
    end
    return Array(generic), Array(specialized)
end

function sum3_results(format, data)
    A = Tensor(format, data)
    generic = Tensor(Dense(Element(0.0)), zeros(size(data, 3)))
    specialized = Tensor(Dense(Element(0.0)), zeros(size(data, 3)))

    @finch begin
        generic .= 0.0
        for k in _, j in _, i in _
            generic[k] += A[i, j, k]
        end
    end
    @finch specialize = true begin
        specialized .= 0.0
        for k in _, j in _, i in _
            specialized[k] += A[i, j, k]
        end
    end
    return Array(generic), Array(specialized)
end

function fiber_sum_results(format, data)
    A = Tensor(format, data)
    generic = Tensor(Dense(Element(0.0)), zeros(size(data, 2)))
    specialized = Tensor(Dense(Element(0.0)), zeros(size(data, 2)))

    @finch begin
        generic .= 0.0
        for j in _, i in _
            generic[j] += A[i, j]
        end
    end
    @finch specialize = true begin
        specialized .= 0.0
        for j in _, i in _
            specialized[j] += A[i, j]
        end
    end
    return Array(generic), Array(specialized)
end

function spmv_code(A, x, y; specialize)
    if specialize
        return string(@finch_code specialize = true begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end)
    end
    return string(@finch_code begin
        y .= 0.0
        for j in _, i in _
            y[i] += A[i, j] * x[j]
        end
    end)
end

function sum3_code(A, y; specialize)
    if specialize
        return string(@finch_code specialize = true begin
            y .= 0.0
            for k in _, j in _, i in _
                y[k] += A[i, j, k]
            end
        end)
    end
    return string(@finch_code begin
        y .= 0.0
        for k in _, j in _, i in _
            y[k] += A[i, j, k]
        end
    end)
end

@testset "Finch extension" begin
    @test RX !== nothing
    @test !isdefined(Finch, :Regularity)
    @test isdefined(Finch, :execute_specialized)

    @testset "dense and sparse roots realize exactly" begin
        data = unclipped_banded_data(96, 2)
        visit_data = cancellation_fibers(18)
        formats = (
            Dense(SparseList(Element(0.0))),
            SparseList(SparseList(Element(0.0))),
        )
        for format in formats
            generic_visits, specialized_visits = fiber_sum_results(format, visit_data)
            @test reinterpret(UInt64, generic_visits) ==
                reinterpret(UInt64, specialized_visits)

            generic, specialized = spmv_results(format, data)
            @test ulps_apart(generic, specialized) <= 4

            A = Tensor(format, data)
            x = Tensor(Dense(Element(0.0)), ones(size(data, 2)))
            y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
            generic_code = spmv_code(A, x, y; specialize=false)
            specialized_code = spmv_code(A, x, y; specialize=true)
            @test specialized_code != generic_code
            @test idx_reads(specialized_code) == 0
            @test ptr_reads(specialized_code) == 0
        end
    end

    @testset "rank-3 chains preserve visit order" begin
        data = parent_dependent_data(18, 6)
        formats = (
            Dense(SparseList(SparseList(Element(0.0)))),
            Dense(Dense(SparseList(Element(0.0)))),
            SparseList(Dense(SparseList(Element(0.0)))),
            SparseList(SparseList(SparseList(Element(0.0)))),
        )
        for format in formats
            generic, specialized = sum3_results(format, data)
            @test reinterpret(UInt64, generic) == reinterpret(UInt64, specialized)

            A = Tensor(format, data)
            y = Tensor(Dense(Element(0.0)), zeros(size(data, 3)))
            generic_code = sum3_code(A, y; specialize=false)
            specialized_code = sum3_code(A, y; specialize=true)
            @test specialized_code != generic_code
            @test idx_reads(specialized_code) == 0
            @test ptr_reads(specialized_code) == 0
        end
    end

    @testset "claimed code is structural-read-free and bounded by the description" begin
        function code_for(n)
            data = unclipped_banded_data(n, 2)
            A = Tensor(Dense(SparseList(Element(0.0))), data)
            x = Tensor(Dense(Element(0.0)), ones(n))
            y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
            return (
                spmv_code(A, x, y; specialize=false),
                spmv_code(A, x, y; specialize=true),
            )
        end

        generic, small = code_for(64)
        _, large = code_for(4096)
        @test small != generic
        @test ptr_reads(generic) > 0
        @test idx_reads(generic) > 0
        @test idx_reads(small) == 0
        @test ptr_reads(small) == 0
        @test source_lines(small) == source_lines(large)
    end

    @testset "FinchExprOps agrees with checked formula words" begin
        data = periodic_count_data(64)
        format = Dense(SparseList(Element(0.0)))
        generic, specialized = spmv_results(format, data)
        @test ulps_apart(generic, specialized) <= 4

        A = Tensor(format, data)
        x = Tensor(Dense(Element(0.0)), ones(size(data, 2)))
        y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
        generic_code = spmv_code(A, x, y; specialize=false)
        specialized_code = spmv_code(A, x, y; specialize=true)
        @test specialized_code != generic_code
        @test idx_reads(specialized_code) == 0
        @test ptr_reads(specialized_code) == 0

        for selector in -1:3
            staged_select = RX.select(RX.STAGE, (11, 22, 33), :selector)
            if selector in 0:2
                actual = Core.eval(@__MODULE__, :(
                    let selector = $selector
                        $staged_select
                    end
                ))
                @test actual == (11, 22, 33)[selector + 1]
            else
                @test_throws BoundsError Core.eval(
                    @__MODULE__, :(
                        let selector = $selector
                            $staged_select
                        end
                    ))
            end
        end
    end

    @testset "description-local capability and value checks decline wholly" begin
        function codes(data)
            A = Tensor(Dense(SparseList(Element(0.0))), data)
            x = Tensor(Dense(Element(0.0)), ones(size(data, 2)))
            y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
            return (
                spmv_code(A, x, y; specialize=false),
                spmv_code(A, x, y; specialize=true),
            )
        end

        no_claims = zeros(32, 32)
        for j in 1:32
            no_claims[1, j] = j
            no_claims[2 + mod(7j, 29), j] = -j
        end
        generic, specialized = codes(no_claims)
        @test specialized == generic

        non_unit_coordinates = zeros(32, 32)
        non_unit_coordinates[1:2:31, :] .= 1.0
        generic, specialized = codes(non_unit_coordinates)
        @test specialized == generic
    end

    @testset "Segment, Opaque, Segment threads every sparse position" begin
        for spans in ((3, 30, 300), (300, 3, 30))
            data = hidden_span_data(spans)
            generic, specialized = sum3_results(
                Dense(SparseList(SparseList(Element(0.0)))), data)
            @test reinterpret(UInt64, generic) == reinterpret(UInt64, specialized)

            A = Tensor(Dense(SparseList(SparseList(Element(0.0)))), data)
            y = Tensor(Dense(Element(0.0)), zeros(size(data, 3)))
            generic_code = sum3_code(A, y; specialize=false)
            specialized_code = sum3_code(A, y; specialize=true)
            @test specialized_code != generic_code
            @test idx_reads(specialized_code) > 0
            @test ptr_reads(specialized_code) > 0
        end
    end

    @testset "use-site refusals leave byte-identical generic code" begin
        data_a = banded_data(64, 2)
        data_b = banded_data(64, 3)
        A = Tensor(Dense(SparseList(Element(0.0))), data_a)
        B = Tensor(Dense(SparseList(Element(0.0))), data_b)
        C = Tensor(Dense(Dense(Element(0.0))), zeros(64, 64))

        two_generic = string(@finch_code begin
            C .= 0.0
            for j in _, i in _
                C[i, j] = A[i, j] * B[i, j]
            end
        end)
        admitted_report = Ref{F.SpecializeReport}()
        admitted = string(@finch_code specialize = true report = admitted_report begin
            C .= 0.0
            for j in _, i in _
                C[i, j] = A[i, j] * B[i, j]
            end
        end)
        @test admitted != two_generic
        @test admitted_report[].realized >= 2
        @test admitted_report[].emitted_sequence_phases +
              admitted_report[].emitted_switch_cases > 0

        write_generic = string(@finch_code begin
            for j in _, i in _
                A[i, j] += 1.0
            end
        end)
        write_specialized = string(@finch_code specialize = true begin
            for j in _, i in _
                A[i, j] += 1.0
            end
        end)
        @test write_specialized == write_generic

        x = Tensor(Dense(Element(0.0)), ones(64))
        y = Tensor(Dense(Element(0.0)), zeros(64))
        follow_generic = string(@finch_code begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, follow(j)] * x[j]
            end
        end)
        follow_specialized = string(@finch_code specialize = true begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, follow(j)] * x[j]
            end
        end)
        @test follow_specialized == follow_generic

        reordered_generic = string(@finch_code mode = :fast begin
            y .= 0.0
            for i in _, j in _
                y[i] += A[i, j] * x[j]
            end
        end)
        reordered_specialized = string(@finch_code mode = :fast specialize = true begin
            y .= 0.0
            for i in _, j in _
                y[i] += A[i, j] * x[j]
            end
        end)
        @test reordered_specialized == reordered_generic
    end

    @testset "reusable kernels refuse stale structure" begin
        token = F.StructuralToken(0)
        F.touch_structure!(token)
        @test token.generation == 1
        @test_throws ArgumentError F.StructuralToken(-1)
        @test_throws ArgumentError F.StructuralToken(BigInt(typemax(UInt64)) + 1)
        exhausted = F.StructuralToken(typemax(UInt64))
        @test_throws OverflowError F.touch_structure!(exhausted)
        @test !exhausted.valid

        data = banded_data(64, 2)
        A = Tensor(Dense(SparseList(Element(0.0))), data)
        x = Tensor(Dense(Element(0.0)), ones(64))
        y = Tensor(Dense(Element(0.0)), zeros(64))
        eval(@finch_kernel specialize = true function guarded_spmv(y, A, x)
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end)
        guarded_spmv(y, A, x)

        identical_copy = Tensor(Dense(SparseList(Element(0.0))), copy(data))
        @test_throws Exception guarded_spmv(y, identical_copy, x)

        replacement_data = banded_data(64, 3)
        replacement = Tensor(Dense(SparseList(Element(0.0))), replacement_data)
        @finch begin
            A .= 0.0
            for j in _, i in _
                A[i, j] = replacement[i, j]
            end
        end
        @test_throws Exception guarded_spmv(y, A, x)

        fresh = Tensor(Dense(Element(0.0)), zeros(64))
        generic = Tensor(Dense(Element(0.0)), zeros(64))
        @finch begin
            generic .= 0.0
            for j in _, i in _
                generic[i] += A[i, j] * x[j]
            end
        end
        @finch specialize = true begin
            fresh .= 0.0
            for j in _, i in _
                fresh[i] += A[i, j] * x[j]
            end
        end
        @test ulps_apart(Array(fresh), Array(generic)) <= 4
    end

    @testset "aliases share one structural token" begin
        key = [1, 2]
        equal_but_distinct = copy(key)
        token = F.structural_token(key)
        @test F.structural_token(equal_but_distinct) !== token
        key[1] = 9
        @test F.structural_token(key) === token

        data = banded_data(64, 2)
        A = Tensor(Dense(SparseList(Element(0.0))), data)
        alias = Tensor(A.lvl)
        replacement = Tensor(
            Dense(SparseList(Element(0.0))), banded_data(64, 3))
        x = Tensor(Dense(Element(0.0)), ones(64))
        y = Tensor(Dense(Element(0.0)), zeros(64))
        eval(@finch_kernel specialize = true function guarded_spmv_alias(y, A, x)
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end)
        guarded_spmv_alias(y, A, x)

        @finch begin
            alias .= 0.0
            for j in _, i in _
                alias[i, j] = replacement[i, j]
            end
        end
        @test_throws Exception guarded_spmv_alias(y, A, x)
    end
end

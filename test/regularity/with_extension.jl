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

# Rank 3 with a middle coordinate stream [1..16, 100, 200, 301, 400..415].
# The child under 200 is two separated runs, a leaf skeleton no neighbor
# shares, so 100 and 200 cannot join a middle claim and stay Opaque between
# claimed middle segments; the children under 100 and 301 are `span` rows.
function hidden_span_data(spans)
    coordinates = vcat(collect(1:16), [100, 200, 301], collect(400:415))
    data = zeros(Float64, maximum(spans), last(coordinates), length(spans))
    for (k, span) in pairs(spans), (slot, j) in pairs(coordinates)
        rows = slot == 18 ? vcat(1:2, 10:11) :
            1:(slot in (17, 19) ? span : maximum(spans))
        for i in rows
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

function strided_column_data(ncolumns, stride; nrows=4)
    # Every stored column holds the same rows, so only the root coordinate
    # stream (1, 1 + stride, 1 + 2stride, ...) distinguishes the fibers.
    data = zeros(Float64, nrows, 1 + stride * (ncolumns - 1))
    for (slot, j) in enumerate(1:stride:size(data, 2)), i in 1:nrows
        data[i, j] = (17i + 31slot) / 1024
    end
    return data
end

# Stencil matrices whose interior rows hold the same offsets: the 5-point
# Laplacian on an n x n grid (offsets -n, -1, 0, 1, n) and the 7-point one on
# an n x n x n grid (offsets -n^2, -n, -1, 0, 1, n, n^2). The diagonal and
# off-diagonal values differ in magnitude by 2^54 with sign changes, so the
# storage-ordered sum fingerprints the visit order.
function laplacian_data(n)
    N = n * n
    data = zeros(Float64, N, N)
    for jj in 1:n, ii in 1:n
        r = (jj - 1) * n + ii
        data[r, r] = 2.0^54 + r
        ii > 1 && (data[r, r - 1] = -1.0 - r / 11)
        ii < n && (data[r, r + 1] = -(2.0^54) + r / 13)
        jj > 1 && (data[r, r - n] = 1.0 + r / 17)
        jj < n && (data[r, r + n] = 3.125 - r / 19)
    end
    data
end

function laplacian3_data(n)
    N = n * n * n
    data = zeros(Float64, N, N)
    for kk in 1:n, jj in 1:n, ii in 1:n
        r = ((kk - 1) * n + (jj - 1)) * n + ii
        data[r, r] = 2.0^54 + r
        ii > 1 && (data[r, r - 1] = -1.0 - r / 11)
        ii < n && (data[r, r + 1] = -(2.0^54) + r / 13)
        jj > 1 && (data[r, r - n] = 1.0 + r / 17)
        jj < n && (data[r, r + n] = 3.125 - r / 19)
        kk > 1 && (data[r, r - n * n] = 5.0 + r / 23)
        kk < n && (data[r, r + n * n] = -7.125 + r / 29)
    end
    data
end

# The stencil restricted to its interior sites: one stored column per interior
# grid point of an n^dims grid, holding the 2dims + 1 offsets. Every column is
# the same template, so the whole tensor lowers as template walks.
function interior_stencil_data(n, dims)
    N = n^dims
    strides = [n^(d - 1) for d in 1:dims]
    interior = [site for site in 1:N if all(
        1 < mod(div(site - 1, strides[d]), n) + 1 < n for d in 1:dims)]
    data = zeros(Float64, N, length(interior))
    for (j, center) in pairs(interior)
        data[center, j] = 2.0^54 + center
        for (d, stride) in pairs(strides)
            data[center - stride, j] = -1.0 - center / (7 + d)
            data[center + stride, j] = d == 1 ? -(2.0^54) + center / 13 : 3.125 - center / (17 + d)
        end
    end
    data
end

function series_data(reps)
    data = zeros(Float64, 4096, 7reps)
    column = 1
    for repetition in 1:reps
        base = 200repetition
        for local_column in 0:3
            for i in (base + 5local_column):(base + 5local_column + 3)
                data[i, column] = (17i + 31column) / 4096
            end
            column += 1
        end
        for gap in 1:3
            for i in (base + 50 + gap, base + 91 + 3gap, base + 143 + 7gap)
                data[i, column] = (13i + 29column) / 2048
            end
            column += 1
        end
    end
    data
end

function growth_word_data(reps=10)
    coordinates = Int[]
    for j in 0:(reps - 1)
        append!(coordinates, (100j + 1, 100j + 2, 100j + 3))
        append!(coordinates, (
            100j + 31 + j^2,
            100j + 41 + mod(7^(j + 1), 23),
        ))
    end
    data = zeros(Float64, maximum(coordinates), 3)
    for column in axes(data, 2), (slot, i) in pairs(coordinates)
        data[i, column] = (17i + 31slot + 7column) / 4096
    end
    data
end

function varying_descendant_series_data(reps=6)
    data = zeros(Float64, reps + 3, 4reps)
    for repetition in 0:(reps - 1)
        for column in (4repetition + 1):(4repetition + 2)
            for i in 1:(2 + repetition)
                data[i, column] = (17i + 31column) / 4096
            end
        end
    end
    data
end

function series_structure(data; dense_root)
    ptr = Int[1]
    idx = Int[]
    for j in axes(data, 2)
        append!(idx, findall(!iszero, view(data, :, j)))
        push!(ptr, length(idx) + 1)
    end
    outer = if dense_root
        Regularity.DenseLevel(size(data, 2))
    else
        Regularity.CompressedLevel([1, size(data, 2) + 1], collect(axes(data, 2)))
    end
    Regularity.Structure((outer, Regularity.CompressedLevel(ptr, idx)))
end

function contains_series(nodes)
    any(nodes) do node
        node isa Regularity.Series ||
            (
                node isa Regularity.Segment &&
                node.body !== nothing &&
                contains_series(node.body)
            )
    end
end

function first_series(nodes)
    for node in nodes
        node isa Regularity.Series && return node
        if node isa Regularity.Segment && node.body !== nothing
            found = first_series(node.body)
            found === nothing || return found
        end
    end
    nothing
end

function has_series_at_level(nodes, target_level, level=1)
    any(nodes) do node
        if node isa Regularity.Series
            level == target_level ||
                has_series_at_level(node.body, target_level, level)
        elseif node isa Regularity.Segment && node.body !== nothing
            has_series_at_level(node.body, target_level, level + 1)
        else
            false
        end
    end
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

function spmv_results(
    format, data; policy=SpecializePolicy(), report=nothing
)
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
    @finch specialize = true policy = policy report = report begin
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

    @testset "clipped band stops at the level extent" begin
        data = banded_data(64, 4)
        report = Ref{F.SpecializeReport}()
        generic, specialized = spmv_results(
            Dense(SparseList(Element(0.0))),
            data;
            policy=SpecializePolicy(; pmax=16),
            report,
        )
        @test report[].realized == 1
        @test reinterpret(UInt64, specialized) == reinterpret(UInt64, generic)
    end

    @testset "non-unit root coordinates keep their logical positions" begin
        # Regression (P0-1): a SparseList{SparseList} whose root coordinates
        # advance by two. Storage ordinals are contiguous, so a claimed root
        # segment fed to a Lookup would pair the right stored values with the
        # wrong x[j]. The claim is a stride formula, so it lowers as a
        # template walk whose coordinates come from the formula.
        for stride in (2, 3)
            data = strided_column_data(48, stride)
            report = Ref{F.SpecializeReport}()
            generic, specialized = spmv_results(
                SparseList(SparseList(Element(0.0))), data; report)
            @test report[].realized == 1
            @test ulps_apart(generic, specialized) <= 4
        end
    end

    @testset "template coordinate claims walk without indirection" begin
        # Every interior row of a stencil holds the same offsets: a template.
        # Values carry cancellation, so bitwise equality fingerprints the
        # visit order as well as the visited set. The full Laplacians keep
        # their boundary rows (some stay Opaque, so their code still walks
        # natively); the interior-only fixtures are templates throughout, and
        # their code reads neither `idx` nor `ptr`.
        formats = (
            Dense(SparseList(Element(0.0))),
            SparseList(SparseList(Element(0.0))),
        )
        for data in (laplacian_data(12), laplacian3_data(6)), format in formats
            report = Ref{F.SpecializeReport}()
            generic, specialized = spmv_results(
                format, data; policy=SpecializePolicy(; pmax=8), report)
            @test report[].realized == 1
            @test reinterpret(UInt64, specialized) == reinterpret(UInt64, generic)
        end
        for data in (interior_stencil_data(12, 2), interior_stencil_data(6, 3)),
            format in formats

            report = Ref{F.SpecializeReport}()
            generic, specialized = spmv_results(
                format, data; policy=SpecializePolicy(; pmax=8), report)
            @test report[].realized == 1
            @test reinterpret(UInt64, specialized) == reinterpret(UInt64, generic)

            A = Tensor(format, data)
            x = Tensor(Dense(Element(0.0)), ones(size(data, 2)))
            y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
            code = spmv_code(A, x, y; specialize=true)
            @test idx_reads(code) == 0
            @test ptr_reads(code) == 0
            @test occursin("while", code)               # The template Stepper.
        end
    end

    @testset "unit-step claims lower byte-identically to the reference" begin
        # `reference/unit_step_spmv.jl` is the code the emitter produced before
        # template walks existed: a unit-step claim still lowers as a Lookup
        # over its contiguous coordinate range, byte for byte.
        data = unclipped_banded_data(64, 2)
        A = Tensor(Dense(SparseList(Element(0.0))), data)
        x = Tensor(Dense(Element(0.0)), ones(size(data, 2)))
        y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
        reference = read(joinpath(@__DIR__, "reference", "unit_step_spmv.jl"), String)
        @test spmv_code(A, x, y; specialize=true) == reference
    end

    @testset "Series lowers exactly through sparse and dense storage levels" begin
        data = series_data(4)
        formats = (
            (Dense(SparseList(Element(0.0))), true),
            (SparseList(SparseList(Element(0.0))), false),
        )
        for (format, dense_root) in formats
            structure = series_structure(data; dense_root)
            description = Regularity.mine(
                structure;
                schemas=(Regularity.PeriodicAffineConfig(1, 1),),
            )
            @test contains_series(description.spans)
            @test has_series_at_level(description.spans, 1)
            @test Regularity.check_description(description, structure)

            generic, specialized = spmv_results(format, data)
            @test ulps_apart(generic, specialized) <= 4
            A = Tensor(format, data)
            x = Tensor(Dense(Element(0.0)), ones(size(data, 2)))
            y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
            report = Ref{F.SpecializeReport}()
            @finch_code specialize = true report = report begin
                y .= 0.0
                for j in _, i in _
                    y[i] += A[i, j] * x[j]
                end
            end
            @test report[].realized == 1
        end
    end

    @testset "growth-style Segment/Opaque Series lowers exactly" begin
        data = growth_word_data()
        structure = series_structure(data; dense_root=true)
        description = Regularity.mine(
            structure;
            schemas=(Regularity.PeriodicAffineConfig(1, 1),),
        )
        @test Regularity.check_description(description, structure)
        root = only(description.spans)
        @test root isa Regularity.Segment
        series = only(root.body)
        @test series isa Regularity.Series
        @test Regularity.bind_count(series.reps, (0,)) == 10
        @test map(typeof, series.body) == [Regularity.Segment, Regularity.Opaque]
        @test Regularity.bind_count(series.body[1].count, (0, 0)) == 3
        @test Regularity.bind_count(series.body[2].count, (0, 0)) == 2

        format = Dense(SparseList(Element(0.0)))
        generic, specialized = fiber_sum_results(format, data)
        @test reinterpret(UInt64, specialized) == reinterpret(UInt64, generic)

        A = Tensor(format, data)
        y = Tensor(Dense(Element(0.0)), zeros(size(data, 2)))
        report = Ref{F.SpecializeReport}()
        @finch_code specialize = true report = report begin
            y .= 0.0
            for j in _, i in _
                y[j] += A[i, j]
            end
        end
        @test report[].realized == 1
    end

    @testset "Series permits j-dependent descendant counts and lowers exactly" begin
        data = varying_descendant_series_data()
        structure = series_structure(data; dense_root=true)
        description = Regularity.mine(
            structure;
            schemas=(Regularity.PeriodicAffineConfig(2),),
            series_min_reps=2,
            series_max_word=4,
        )
        @test Regularity.check_description(description, structure)
        series = first_series(description.spans)
        @test series isa Regularity.Series
        @test series.reps isa Int
        repetitions = series.reps::Int
        direct = only(
            filter(
                node ->
                    node isa Regularity.Segment &&
                        node.body !== nothing && !isempty(node.body),
                series.body,
            ),
        )
        descendant = only(direct.body)
        @test descendant isa Regularity.Segment
        @test Regularity.bind_count(direct.count, (0,)) ==
            Regularity.bind_count(direct.count, (repetitions - 1,))
        @test Regularity.bind_count(descendant.count, (0, 0)) !=
            Regularity.bind_count(descendant.count, (repetitions - 1, 0))

        format = Dense(SparseList(Element(0.0)))
        generic, specialized = fiber_sum_results(format, data)
        @test reinterpret(UInt64, specialized) == reinterpret(UInt64, generic)

        A = Tensor(format, data)
        y = Tensor(Dense(Element(0.0)), zeros(size(data, 2)))
        report = Ref{F.SpecializeReport}()
        @finch_code specialize = true report = report begin
            y .= 0.0
            for j in _, i in _
                y[j] += A[i, j]
            end
        end
        @test report[].realized == 1
    end

    @testset "Series coiterates with SparseList operands" begin
        data = series_data(4)
        other = zeros(size(data))
        for j in axes(data, 2)
            coordinates = findall(!iszero, view(data, :, j))
            other[first(coordinates), j] = data[first(coordinates), j] / 2
            other[last(coordinates), j] = data[last(coordinates), j] / 2
        end
        A = Tensor(Dense(SparseList(Element(0.0))), data)
        B = Tensor(Dense(SparseList(Element(0.0))), other)
        generic = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
        specialized = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))

        @finch begin
            generic .= 0.0
            for j in _, i in _
                generic[i] += A[i, j] * B[i, j]
            end
        end
        @finch specialize = true begin
            specialized .= 0.0
            for j in _, i in _
                specialized[i] += A[i, j] * B[i, j]
            end
        end
        report = Ref{F.SpecializeReport}()
        @finch_code specialize = true report = report begin
            specialized .= 0.0
            for j in _, i in _
                specialized[i] += A[i, j] * B[i, j]
            end
        end
        @test reinterpret(UInt64, Array(specialized)) ==
            reinterpret(UInt64, Array(generic))
        @test report[].realized == 2
    end

    @testset "Series declines normalized-AST-identically and has bounded phase counts" begin
        data = series_data(4)
        A = Tensor(Dense(SparseList(Element(0.0))), data)
        x = Tensor(Dense(Element(0.0)), ones(size(data, 2)))
        y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
        generic = spmv_code(A, x, y; specialize=false)
        declined_report = Ref{F.SpecializeReport}()
        declined = string(
            @finch_code specialize = true policy = SpecializePolicy(; code_density=10^9) report =
                declined_report begin
                y .= 0.0
                for j in _, i in _
                    y[i] += A[i, j] * x[j]
                end
            end
        )
        @test declined_report[].realized == 0
        @test declined == generic

        reports = F.SpecializeReport[]
        for reps in (4, 8, 16)
            ladder_data = series_data(reps)
            ladder_structure = series_structure(ladder_data; dense_root=true)
            ladder_description = Regularity.mine(
                ladder_structure;
                schemas=(Regularity.PeriodicAffineConfig(1, 1),),
            )
            @test contains_series(ladder_description.spans)

            ladder_A = Tensor(Dense(SparseList(Element(0.0))), ladder_data)
            ladder_x = Tensor(Dense(Element(0.0)), ones(size(ladder_data, 2)))
            ladder_y = Tensor(Dense(Element(0.0)), zeros(size(ladder_data, 1)))
            ladder_report = Ref{F.SpecializeReport}()
            @finch_code specialize = true report = ladder_report begin
                ladder_y .= 0.0
                for j in _, i in _
                    ladder_y[i] += ladder_A[i, j] * ladder_x[j]
                end
            end
            push!(reports, ladder_report[])
        end
        @test all(report -> report.realized == 1, reports)
        @test length(unique(report.emitted_sequence_phases for report in reports)) == 1
        @test length(unique(report.emitted_switch_cases for report in reports)) == 1
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

        # Count-only descriptions now realize: constant child counts derive
        # origins and bounded Opaque walks read no ptr.
        no_claims = zeros(32, 32)
        for j in 1:32
            no_claims[1, j] = j
            no_claims[2 + mod(7j, 29), j] = -j
        end
        generic, specialized = codes(no_claims)
        @test specialized != generic

        # Dense non-unit coordinates are valid claims; sparse coordinate
        # claims are still constrained to unit drift by the host policy.
        non_unit_coordinates = zeros(32, 32)
        non_unit_coordinates[1:2:31, :] .= 1.0
        generic, specialized = codes(non_unit_coordinates)
        @test specialized != generic
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

    @testset "use-site refusals leave normalized-AST-identical generic code" begin
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
        token = F.StructuralToken(UInt64(0))
        F.touch_structure!(token)
        @test token.generation == 1

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

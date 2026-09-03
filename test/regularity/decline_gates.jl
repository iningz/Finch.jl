@testset "unrepresentable structural storage declines to generic" begin
    @test RX._int_fits(Int32)
    @test RX._int_fits(Int)
    @test !RX._int_fits(UInt64)
    @test !RX._int_fits(Int128)
    @test !RX._int_fits(BigInt)

    n = 8
    width = 3
    huge = UInt64(typemax(Int)) + UInt64(1)
    ptr = Int[1 + width * j for j in 0:n]
    idx = UInt64[huge + UInt64(i) for j in 1:n for i in 0:(width - 1)]
    vals = [Float64(p) / 1024 for p in eachindex(idx)]
    shape = huge + UInt64(width)
    sparse = F.SparseListLevel{UInt64}(Element(0.0, vals), shape, ptr, idx)
    @test RX._regularity_level(sparse) === nothing
    @test RX._regularity_level(F.DenseLevel{UInt64}(Element(0.0), shape)) === nothing
    @test RX._regularity_level(F.DenseLevel{Int}(sparse, n)) isa Regularity.DenseLevel

    A = Tensor(F.DenseLevel{Int}(sparse, n))
    x = Tensor(Dense(Element(0.0)), ones(n))
    y = Tensor(Dense(Element(0.0)), zeros(n))
    generic = spmv_code(A, x, y; specialize=false)
    report = Ref{F.SpecializeReport}()
    specialized = string(@finch_code specialize = true report = report begin
        y .= 0.0
        for j in _, i in _
            y[i] += A[i, j] * x[j]
        end
    end)
    @test report[].realized == 0
    @test specialized == generic
end

@testset "structural arrays sharing memory through a view decline" begin
    data = banded_data(64, 2)
    A = Tensor(Dense(SparseList(Element(0.0))), data)
    a_sparse = A.lvl.lvl
    rebuilt(idx) = Tensor(
        F.DenseLevel{Int}(
            F.SparseListLevel{Int}(
                Element(0.0, copy(a_sparse.lvl.val)),
                a_sparse.shape,
                copy(a_sparse.ptr),
                idx,
            ),
            A.lvl.shape,
        ),
    )
    C = Tensor(Dense(Dense(Element(0.0))), zeros(64, 64))

    function product_codes(B)
        generic = string(@finch_code begin
            C .= 0.0
            for j in _, i in _
                C[i, j] = A[i, j] * B[i, j]
            end
        end)
        report = Ref{F.SpecializeReport}()
        specialized = string(@finch_code specialize = true report = report begin
            C .= 0.0
            for j in _, i in _
                C[i, j] = A[i, j] * B[i, j]
            end
        end)
        generic, specialized, report[]
    end

    # A view over A's coordinate memory is a distinct object, so identity
    # comparison would miss it; memory overlap must decline both operands.
    viewed = rebuilt(view(a_sparse.idx, :))
    @test viewed.lvl.lvl.idx !== a_sparse.idx
    @test Base.mightalias(viewed.lvl.lvl.idx, a_sparse.idx)
    generic, specialized, report = product_codes(viewed)
    @test report.realized == 0
    @test specialized == generic

    # Equal content in separately allocated arrays never collides.
    separate = rebuilt(copy(a_sparse.idx))
    @test separate.lvl.lvl.idx == a_sparse.idx
    @test !Base.mightalias(separate.lvl.lvl.idx, a_sparse.idx)
    generic, specialized, report = product_codes(separate)
    @test report.realized >= 2
    @test specialized != generic
end

# Fixtures whose leaf rows never lift: neighboring columns alternate between
# five and six entries, so no two share a leaf skeleton and every provisional
# template demotes. What remains is the row's count pattern (period two).
function visit_gate_data(kind; n=64)
    if kind === :short
        data = zeros(Float64, n, n)
        for j in 1:n
            left = 1 + mod(7j^2 + 3j, 20)
            right = 33 + mod(11j^2 + 5j, 32)
            for i in (left, 25, 26, 27, right)
                data[i, j] = (17i + 31j) / 4096
            end
            iseven(j) && (data[30, j] = (17 * 30 + 31j) / 4096)
        end
        return data
    elseif kind === :band
        data = zeros(Float64, n + 8, n)
        for j in 1:n, i in j:(j + 8)
            data[i, j] = (17i + 31j) / 4096
        end
        return data
    elseif kind === :count_only
        data = zeros(Float64, n, n)
        for j in 1:n
            coordinates = (
                2 + mod(3j, 5),
                14 + mod(5j, 7),
                29 + mod(7j, 5),
                43 + mod(2j, 7),
                57 + mod(11j, 7),
            )
            for i in coordinates
                data[i, j] = (17i + 31j) / 4096
            end
            iseven(j) && (data[9, j] = (17 * 9 + 31j) / 4096)
        end
        return data
    end
    throw(ArgumentError("unknown visit-gate fixture: $kind"))
end

function visit_gate_code(A, x, y; specialize, report=nothing, code_density=0)
    if specialize
        return string(
            @finch_code specialize = true policy =
                SpecializePolicy(; pmax=16, visit_density=4, code_density=code_density) report = report begin
                y .= 0.0
                for j in _, i in _
                    y[i] += A[i, j] * x[j]
                end
            end
        )
    end
    string(@finch_code begin
        y .= 0.0
        for j in _, i in _
            y[i] += A[i, j] * x[j]
        end
    end)
end

@testset "visit-density release gates" begin
    short = visit_gate_data(:short)
    short_A = Tensor(Dense(SparseList(Element(0.0))), short)
    short_x = Tensor(Dense(Element(0.0)), ones(size(short, 2)))
    short_y = Tensor(Dense(Element(0.0)), zeros(size(short, 1)))
    short_generic = visit_gate_code(short_A, short_x, short_y; specialize=false)
    # No coordinate claim survives (the unlifted templates demote); the row
    # stands on its derived origin alone: count-only, no idx claims.
    short_report = Ref{F.SpecializeReport}()
    short_specialized = visit_gate_code(
        short_A, short_x, short_y; specialize=true, report=short_report)
    @test short_report[].realized == 1
    @test short_report[].structural_reads == 2 * 64
    @test short_specialized != short_generic
    @test !occursin(".ptr", short_specialized)
    # Under the code rule the 128 reads cannot pay for the description:
    # byte-identical generic.
    dense_report = Ref{F.SpecializeReport}()
    dense_specialized = visit_gate_code(
        short_A, short_x, short_y; specialize=true, report=dense_report, code_density=3000)
    @test dense_report[].realized == 0
    @test dense_specialized == short_generic

    band = visit_gate_data(:band)
    band_A = Tensor(Dense(SparseList(Element(0.0))), band)
    band_x = Tensor(Dense(Element(0.0)), ones(size(band, 2)))
    band_y = Tensor(Dense(Element(0.0)), zeros(size(band, 1)))
    band_generic = visit_gate_code(band_A, band_x, band_y; specialize=false)
    band_report = Ref{F.SpecializeReport}()
    band_specialized = visit_gate_code(
        band_A, band_x, band_y; specialize=true, report=band_report)
    @test band_report[].realized == 1
    @test band_specialized != band_generic

    count_only = visit_gate_data(:count_only)
    count_A = Tensor(Dense(SparseList(Element(0.0))), count_only)
    count_x = Tensor(Dense(Element(0.0)), ones(size(count_only, 2)))
    count_y = Tensor(Dense(Element(0.0)), zeros(size(count_only, 1)))
    count_generic = visit_gate_code(count_A, count_x, count_y; specialize=false)
    count_report = Ref{F.SpecializeReport}()
    count_specialized = visit_gate_code(
        count_A, count_x, count_y; specialize=true, report=count_report)
    @test count_report[].realized == 1
    @test count_specialized != count_generic
    @test ptr_reads(count_specialized) == 0
    @test idx_reads(count_specialized) > 0

    count_result = Tensor(Dense(Element(0.0)), zeros(size(count_only, 1)))
    @finch specialize = true policy = SpecializePolicy(; pmax=16, visit_density=4) begin
        count_result .= 0.0
        for j in _, i in _
            count_result[i] += count_A[i, j] * count_x[j]
        end
    end
    @test reinterpret(UInt64, Array(count_result)) ==
        reinterpret(UInt64, count_only * Array(count_x))
end

@testset "reusable freshness tracks shared sparse indices" begin
    n = 64
    original_data = zeros(Float64, n + 5, n)
    replacement_data = zeros(Float64, size(original_data))
    for j in 1:n
        for i in (j + 1):(j + 5)
            original_data[i, j] = (17i + 31j) / 1024
        end
        for i in j:(j + 4)
            replacement_data[i, j] = (19i + 29j) / 1024
        end
    end

    A = Tensor(Dense(SparseList(Element(0.0))), original_data)
    a_sparse = A.lvl.lvl
    shared_idx = a_sparse.idx
    b_ptr = copy(a_sparse.ptr)
    B = Tensor(
        F.DenseLevel{Int}(
            F.SparseListLevel{Int}(
                Element(0.0, copy(a_sparse.lvl.val)),
                a_sparse.shape,
                b_ptr,
                shared_idx,
            ),
            A.lvl.shape,
        ),
    )
    replacement = Tensor(
        Dense(SparseList(Element(0.0))), replacement_data)

    @test A.lvl.lvl.ptr !== B.lvl.lvl.ptr
    @test A.lvl.lvl.idx === B.lvl.lvl.idx

    x = Tensor(Dense(Element(0.0)), ones(n))
    y = Tensor(Dense(Element(0.0)), zeros(size(original_data, 1)))
    eval(@finch_kernel specialize = true function guarded_spmv_shared_idx(y, A, x)
        y .= 0.0
        for j in _, i in _
            y[i] += A[i, j] * x[j]
        end
    end)
    guarded_spmv_shared_idx(y, A, x)

    a_token = F.structural_token(A.lvl.lvl.ptr)
    b_token = F.structural_token(B.lvl.lvl.ptr)
    a_generation = a_token.generation
    b_generation = b_token.generation
    idx_before = copy(shared_idx)
    idx_length = length(shared_idx)

    @finch begin
        B .= 0.0
        for j in _, i in _
            B[i, j] = replacement[i, j]
        end
    end

    @test A.lvl.lvl.ptr !== B.lvl.lvl.ptr
    @test A.lvl.lvl.idx === B.lvl.lvl.idx === shared_idx
    @test A.lvl.lvl.ptr == B.lvl.lvl.ptr
    @test length(shared_idx) == idx_length
    @test shared_idx != idx_before
    @test a_token.generation == a_generation
    @test b_token.generation > b_generation

    @test_throws Exception guarded_spmv_shared_idx(y, A, x)
end

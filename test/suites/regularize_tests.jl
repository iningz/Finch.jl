@testset "regularize" begin
    # ═════════════════════════════════════════════════════════════════════════
    # Helper: capture specialized and plain codegen strings for SpMV kernel
    # ═════════════════════════════════════════════════════════════════════════

    function spmv_codegen(A, x, y)
        spec = string(@finch_code specialize = true begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end)
        plain = string(@finch_code begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end)
        return (; spec, plain)
    end

    function spmm_codegen(A, B, C)
        spec = string(@finch_code specialize = true begin
            C .= 0.0
            for k in _, j in _, i in _
                C[i, k] += A[i, j] * B[j, k]
            end
        end)
        plain = string(@finch_code begin
            C .= 0.0
            for k in _, j in _, i in _
                C[i, k] += A[i, j] * B[j, k]
            end
        end)
        return (; spec, plain)
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 1. Miner unit tests: one per signature + region_threshold
    # ═════════════════════════════════════════════════════════════════════════

    @testset "mine_nd: empty fibers" begin
        A = Tensor(
            Dense(SparseList(Element(0.0))),
            [
                1.0 0.0 0.0 0.0 2.0 0.0
                0.0 0.0 0.0 0.0 0.0 0.0
                3.0 0.0 0.0 0.0 4.0 0.0
                0.0 0.0 0.0 0.0 0.0 0.0
                5.0 0.0 0.0 0.0 6.0 0.0
            ],
        )
        child = A.lvl.lvl
        patterns = Finch.Regularity.mine_nd(child.ptr, child.idx, [6])
        # Total miner: col 1 → Opaque, cols 2-4 → Empty, cols 5-6 → Opaque
        @test length(patterns) == 3
        @test patterns[1] isa Finch.OpaquePattern
        @test patterns[1].region.ranges[1] == 1:1
        @test patterns[2] isa Finch.EmptyPattern
        @test patterns[2].region.ranges[1] == 2:4
        @test patterns[3] isa Finch.OpaquePattern
        @test patterns[3].region.ranges[1] == 5:6
    end

    @testset "mine_nd: all empty" begin
        A = Tensor(Dense(SparseList(Element(0.0))), zeros(3, 5))
        child = A.lvl.lvl
        patterns = Finch.Regularity.mine_nd(child.ptr, child.idx, [5])
        @test length(patterns) == 1
        @test patterns[1].region.ranges[1] == 1:5
        @test patterns[1] isa Finch.EmptyPattern
    end

    @testset "mine_nd: fully dense → contiguous" begin
        A = Tensor(Dense(SparseList(Element(0.0))), Float64[1 2 3; 4 5 6])
        child = A.lvl.lvl
        patterns = Finch.Regularity.mine_nd(child.ptr, child.idx, [3])
        @test length(patterns) >= 1
        @test patterns[1] isa Finch.ContiguousPattern
        @test patterns[1].nnz == 2
        @test patterns[1].first_idx == 1
    end

    @testset "mine_nd: region_threshold" begin
        A = Tensor(
            Dense(SparseList(Element(0.0))),
            [1.0 0.0 0.0 2.0; 3.0 0.0 0.0 4.0],
        )
        child = A.lvl.lvl

        # Default threshold (2): the 2-column empty run is detected
        patterns2 = Finch.Regularity.mine_nd(child.ptr, child.idx, [4])
        empty_patterns = filter(g -> g isa Finch.EmptyPattern, patterns2)
        @test length(empty_patterns) == 1
        @test empty_patterns[1].region.ranges[1] == 2:3

        # Raise EmptyMiner threshold to 3: the 2-column run is too short
        passes3 = Finch.AbstractMiningPass[
            Finch.EmptyMiner(; region_threshold=3),
            Finch.AffineMiner(),
            Finch.ContiguousMiner(),
            Finch.IdenticalRelativeMiner(),
        ]
        patterns3 = Finch.Regularity.mine_nd(child.ptr, child.idx, [4]; passes=passes3)
        @test isempty(filter(g -> g isa Finch.EmptyPattern, patterns3))
    end

    @testset "mine_nd: identical_relative" begin
        A_data = [
            1.0 2.0 3.0 4.0
            0.0 0.0 0.0 0.0
            5.0 6.0 7.0 8.0
        ]
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        child = A.lvl.lvl
        patterns = Finch.Regularity.mine_nd(child.ptr, child.idx, [4])
        id_patterns = filter(g -> g isa Finch.IdenticalRelativePattern, patterns)
        @test length(id_patterns) == 1
        @test id_patterns[1].region.ranges[1] == 1:4
        @test id_patterns[1].indices == [1, 3]
    end

    @testset "mine_nd: affine (banded)" begin
        A_data = zeros(5, 4)
        for j in 1:4
            A_data[j, j] = 1.0
            A_data[j + 1, j] = 1.0
        end
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        child = A.lvl.lvl
        patterns = Finch.Regularity.mine_nd(child.ptr, child.idx, [4])
        aff_patterns = filter(g -> g isa Finch.AffinePattern, patterns)
        @test length(aff_patterns) == 1
        g = aff_patterns[1]
        @test g.region.ranges[1] == 1:4
        @test g.nnz == 2
        @test g.delta == 1
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 2. Integration: mine_regular_structure! and mine_nd
    # ═════════════════════════════════════════════════════════════════════════

    @testset "mine_regular_structure! on Dense(SparseList)" begin
        A_data = zeros(5, 6)
        A_data[1, 1] = 1.0
        A_data[3, 1] = 3.0
        A_data[5, 1] = 5.0
        A_data[1, 5] = 2.0
        A_data[3, 5] = 4.0
        A_data[5, 5] = 6.0
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)

        ctx = Finch.JuliaContext()
        vfib = Finch.virtualize_concrete(ctx, :A, A, :A)
        Finch.mine_regular_structure!(vfib)

        sparse_lvl = vfib.lvl.lvl
        patterns = sparse_lvl.regularity_map
        @test patterns !== nothing
        @test length(patterns) >= 1
        @test length(filter(g -> g isa Finch.EmptyPattern, patterns)) >= 1
    end

    @testset "mine_nd: 1D" begin
        ptr = ones(Int, 6)
        idx = Int[]
        patterns = Finch.mine_nd(ptr, idx, [5])
        @test length(patterns) >= 1
        @test patterns[1] isa Finch.EmptyPattern
        @test patterns[1].region.ranges[1] == 1:5
    end

    @testset "mine_nd returns empty for multi-D" begin
        n = 3 * 4
        ptr = ones(Int, n + 1)
        idx = Int[]
        patterns = Finch.mine_nd(ptr, idx, [3, 4])
        @test patterns == Finch.AbstractPattern[]
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 3. SpMV: mixed patterns (contiguous + empty + identical_relative)
    #    Codegen: contiguous → _qc, identical_relative → _qi, loop splitting
    # ═════════════════════════════════════════════════════════════════════════

    @testset "SpMV: mixed patterns (contiguous + empty + identical_relative)" begin
        # cols 1-2: contiguous (all 4 rows), cols 3-5: empty, cols 6-8: identical_relative (rows 1,3)
        A_data = zeros(4, 8)
        A_data[:, 1] = [1.0, 2.0, 3.0, 4.0]
        A_data[:, 2] = [5.0, 6.0, 7.0, 8.0]
        A_data[1, 6] = 9.0
        A_data[3, 6] = 10.0
        A_data[1, 7] = 11.0
        A_data[3, 7] = 12.0
        A_data[1, 8] = 13.0
        A_data[3, 8] = 14.0

        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        x = Tensor(Dense(Element(0.0)), ones(8))
        y = Tensor(Dense(Element(0.0)), zeros(4))

        # Codegen verification
        cg = spmv_codegen(A, x, y)
        # Specialized code must differ from plain
        @test cg.spec != cg.plain
        # Contiguous group emits arithmetic qos (_qc)
        @test contains(cg.spec, "_qc")
        # Identical-relative group emits unrolled spike positions (_qi)
        @test contains(cg.spec, "_qi")
        # The outer j-loop is split: no single for j = 1:8 loop
        @test !contains(cg.spec, "1:8")
        # Plain code uses scansearch; specialized hot-path avoids it in the
        # contiguous region (the for-loop with _qc has no scansearch)
        @test contains(cg.plain, "scansearch")

        # Numerical correctness
        @finch specialize = true begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end

        @test Array(y) ≈ A_data * ones(8)
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 4. SpMV: affine / banded pattern
    #    Codegen: affine → _ablo / _abhi block bounds
    # ═════════════════════════════════════════════════════════════════════════

    @testset "SpMV: affine / banded pattern" begin
        A_data = zeros(5, 4)
        for j in 1:4
            A_data[j, j] = Float64(j)
            A_data[j + 1, j] = Float64(j + 10)
        end
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        x = Tensor(Dense(Element(0.0)), [1.0, 2.0, 3.0, 4.0])
        y = Tensor(Dense(Element(0.0)), zeros(5))

        # Codegen verification
        cg = spmv_codegen(A, x, y)
        @test cg.spec != cg.plain
        # Affine group emits block-bound variables
        @test contains(cg.spec, "_ablo")
        @test contains(cg.spec, "_abhi")
        # Affine group emits arithmetic qos using _qa
        @test contains(cg.spec, "_qa")
        # The hot-path uses a dense for-loop, not a while/scansearch stepper
        @test contains(cg.plain, "scansearch")

        # Numerical correctness
        @finch specialize = true begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end

        @test Array(y) ≈ A_data * Array(x)
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 5. SpMV: diagonal (affine nnz=1, delta=1)
    #    Codegen: still affine → _ablo, single-element block
    # ═════════════════════════════════════════════════════════════════════════

    @testset "SpMV: diagonal (affine nnz=1, delta=1)" begin
        m = 12
        A_data = zeros(m, m)
        for i in 1:m
            A_data[i, i] = Float64(i)
        end
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        x = Tensor(Dense(Element(0.0)), collect(1.0:m))
        y = Tensor(Dense(Element(0.0)), zeros(m))

        # Codegen verification
        cg = spmv_codegen(A, x, y)
        @test cg.spec != cg.plain
        # Affine group (diagonal is a special case of affine with nnz=1)
        @test contains(cg.spec, "_ablo")
        @test contains(cg.spec, "_qa")
        @test contains(cg.plain, "scansearch")

        # Numerical correctness
        @finch specialize = true begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end

        @test Array(y) ≈ A_data * collect(1.0:m)
        for i in 1:m
            @test Array(y)[i] ≈ Float64(i * i)
        end
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 6. SpMV: empty group eliminates code
    #    Codegen: outer loop is split, no single 1:N traversal
    # ═════════════════════════════════════════════════════════════════════════

    @testset "SpMV: empty group eliminates code" begin
        A_data = [
            1.0 0.0 0.0 0.0 2.0
            0.0 0.0 0.0 0.0 0.0
            3.0 0.0 0.0 0.0 4.0
        ]
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        x = Tensor(Dense(Element(0.0)), ones(5))
        y = Tensor(Dense(Element(0.0)), zeros(3))

        # Codegen verification
        cg = spmv_codegen(A, x, y)
        @test cg.spec != cg.plain
        # Plain uses a single for j = 1:N loop; specialized splits and skips empties
        @test contains(cg.plain, "scansearch")
        # Specialized code should NOT have a single 1:5 loop, empty cols are eliminated
        @test !contains(cg.spec, "1:5")

        # Numerical correctness
        @finch specialize = true begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end

        @test Array(y) ≈ A_data * ones(5)
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 7. SpMV: contiguous (fully dense) uses arithmetic indexing
    #    Codegen: contiguous → _qc, dense inner loop, no scansearch in hot path
    # ═════════════════════════════════════════════════════════════════════════

    @testset "SpMV: contiguous (fully dense) uses arithmetic indexing" begin
        A_data = Float64[1 2 3; 4 5 6]
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        x = Tensor(Dense(Element(0.0)), [10.0, 20.0, 30.0])
        y = Tensor(Dense(Element(0.0)), zeros(2))

        # Codegen verification
        cg = spmv_codegen(A, x, y)
        @test cg.spec != cg.plain
        # Non-specialized always uses scansearch; specialized uses arithmetic qos
        @test contains(cg.plain, "scansearch")
        @test contains(cg.spec, r"A_lvl_2_ptr\[A_lvl_q")
        # Contiguous emitter uses _qc variable
        @test contains(cg.spec, "_qc")
        # Hot-path loop should have a dense inner loop (for i_N), not a while/stepper
        # Find the first specialized j-loop body and verify it has a for-i and no scansearch
        first_for_j = findfirst(r"for j_\d+ = 1:", cg.spec)
        if first_for_j !== nothing
            # Extract until the next outer-level for-j
            rest = cg.spec[first(first_for_j):end]
            next_j = findnext(r"\n    for j_\d+", rest, 2)
            hot_path = next_j !== nothing ? rest[1:(first(next_j) - 1)] : rest
            @test contains(hot_path, r"for i_\d+")
            @test !contains(hot_path, "scansearch")
        end

        # Numerical correctness
        @finch specialize = true begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end

        @test Array(y) ≈ A_data * Array(x)
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 8. SpMM: contiguous (fully dense A)
    #    Codegen: contiguous → _qc, differs from plain
    # ═════════════════════════════════════════════════════════════════════════

    @testset "SpMM: contiguous (fully dense A)" begin
        A_data = Float64[1 2 3; 4 5 6; 7 8 9]
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        B_data = Float64[1 0; 0 1; 1 1]
        B = Tensor(Dense(Dense(Element(0.0))), B_data)
        C = Tensor(Dense(Dense(Element(0.0))), zeros(3, 2))

        # Codegen verification
        cg = spmm_codegen(A, B, C)
        @test cg.spec != cg.plain
        # Fully-dense A → contiguous emitter with arithmetic qos
        @test contains(cg.spec, "_qc")
        @test contains(cg.plain, "scansearch")

        # Reference with plain Finch
        C_ref = Tensor(Dense(Dense(Element(0.0))), zeros(3, 2))
        @finch begin
            C_ref .= 0.0
            for k in _, j in _, i in _
                C_ref[i, k] += A[i, j] * B[j, k]
            end
        end

        # Numerical correctness
        @finch specialize = true begin
            C .= 0.0
            for k in _, j in _, i in _
                C[i, k] += A[i, j] * B[j, k]
            end
        end

        @test Array(C) ≈ Array(C_ref)
        @test Array(C) ≈ A_data * B_data
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 9. SpMM: mixed patterns
    #    Codegen: contiguous + identical patterns present, loop splitting
    # ═════════════════════════════════════════════════════════════════════════

    @testset "SpMM: mixed patterns" begin
        A_data = zeros(4, 8)
        A_data[:, 1] = [1.0, 2.0, 3.0, 4.0]
        A_data[:, 2] = [5.0, 6.0, 7.0, 8.0]
        A_data[1, 6] = 9.0
        A_data[3, 6] = 10.0
        A_data[1, 7] = 11.0
        A_data[3, 7] = 12.0
        A_data[1, 8] = 13.0
        A_data[3, 8] = 14.0
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        B_data = ones(8, 4)
        B = Tensor(Dense(Dense(Element(0.0))), B_data)
        C = Tensor(Dense(Dense(Element(0.0))), zeros(4, 4))

        # Codegen verification
        cg = spmm_codegen(A, B, C)
        @test cg.spec != cg.plain
        # Must have contiguous (_qc) and identical_relative (_qi) emitters
        @test contains(cg.spec, "_qc")
        @test contains(cg.spec, "_qi")
        # Loop is split: no single 1:8 loop
        @test !contains(cg.spec, "1:8")

        # Reference with plain Finch
        C_ref = Tensor(Dense(Dense(Element(0.0))), zeros(4, 4))
        @finch begin
            C_ref .= 0.0
            for k in _, j in _, i in _
                C_ref[i, k] += A[i, j] * B[j, k]
            end
        end

        # Numerical correctness
        @finch specialize = true begin
            C .= 0.0
            for k in _, j in _, i in _
                C[i, k] += A[i, j] * B[j, k]
            end
        end

        @test Array(C) ≈ Array(C_ref)
        @test Array(C) ≈ A_data * B_data
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 10. Element-wise copy with specialize=true
    #     Codegen: fully dense A → contiguous _qc, differs from plain
    # ═════════════════════════════════════════════════════════════════════════

    @testset "element-wise copy with specialize=true" begin
        A_data = Float64[1 2; 3 4; 5 6]
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        B = Tensor(Dense(Dense(Element(0.0))), zeros(3, 2))

        # Codegen verification
        spec_str = string(@finch_code specialize = true begin
            B .= 0.0
            for j in _, i in _
                B[i, j] = A[i, j]
            end
        end)
        plain_str = string(@finch_code begin
            B .= 0.0
            for j in _, i in _
                B[i, j] = A[i, j]
            end
        end)
        @test spec_str != plain_str
        # Fully dense 3×2 → contiguous emitter
        @test contains(spec_str, "_qc")

        # Numerical correctness
        @finch specialize = true begin
            B .= 0.0
            for j in _, i in _
                B[i, j] = A[i, j]
            end
        end

        @test Array(B) == A_data
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 11. specialize=false: no regularization, generic scansearch code
    # ═════════════════════════════════════════════════════════════════════════

    @testset "specialize=false does not trigger regularization" begin
        A_data = [1.0 0.0 0.0; 0.0 0.0 2.0]
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        x = Tensor(Dense(Element(0.0)), [1.0, 2.0, 3.0])
        y = Tensor(Dense(Element(0.0)), zeros(2))

        # Codegen verification: plain code must NOT contain specialization markers
        plain_str = string(@finch_code begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end)
        @test contains(plain_str, "scansearch")
        @test !contains(plain_str, "_qc")
        @test !contains(plain_str, "_qi")
        @test !contains(plain_str, "_ablo")
        @test !contains(plain_str, "_qa")

        # Numerical correctness
        @finch begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end

        @test Array(y) ≈ A_data * Array(x)
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 12. 1×N edge case: single row: few columns, below region_threshold
    #     Codegen: specialization still runs (code differs from plain)
    # ═════════════════════════════════════════════════════════════════════════

    @testset "1×N edge case: single row" begin
        A_data = reshape([0.0, 0.0, 0.0, 5.0, 0.0], 1, 5)
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        x = Tensor(Dense(Element(0.0)), [1.0, 2.0, 3.0, 4.0, 5.0])
        y = Tensor(Dense(Element(0.0)), zeros(1))

        # Codegen verification: specialization should still produce different code
        cg = spmv_codegen(A, x, y)
        # Even for edge cases the framework must produce code that differs
        # (the miner will find empty patterns for the zero columns)
        @test cg.spec != cg.plain
        # The empty columns (1-3 and 5) are eliminated, so no single 1:5 loop
        @test !contains(cg.spec, "1:5")

        # Numerical correctness
        @finch specialize = true begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end

        @test Array(y) ≈ A_data * Array(x)
    end

    # ═════════════════════════════════════════════════════════════════════════
    # 13. Random safety net: specialize=true matches specialize=false
    #     Codegen: specialized code differs from plain (loop splitting at minimum)
    # ═════════════════════════════════════════════════════════════════════════

    @testset "specialize=true matches specialize=false for random sparse" begin
        m, n = 8, 12
        A_data = zeros(m, n)
        for (i, j) in
            [(1, 1), (3, 1), (1, 4), (5, 4), (7, 4), (2, 7), (4, 7), (6, 7), (8, 7),
            (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10)]
            A_data[i, j] = Float64(i + j)
        end
        A = Tensor(Dense(SparseList(Element(0.0))), A_data)
        x = Tensor(Dense(Element(0.0)), collect(1.0:n))

        y_ref = Tensor(Dense(Element(0.0)), zeros(m))
        y_spec = Tensor(Dense(Element(0.0)), zeros(m))

        # Codegen verification: specialized must differ from plain
        cg = spmv_codegen(A, x, y_spec)
        @test cg.spec != cg.plain
        # The miner finds empty patterns (cols 2-3, 5-6, 8-9, 11-12 are all-zero)
        # so loop splitting must occur, no single 1:12 traversal
        @test !contains(cg.spec, "1:12")

        # Numerical correctness
        @finch begin
            y_ref .= 0.0
            for j in _, i in _
                y_ref[i] += A[i, j] * x[j]
            end
        end

        @finch specialize = true begin
            y_spec .= 0.0
            for j in _, i in _
                y_spec[i] += A[i, j] * x[j]
            end
        end

        @test Array(y_spec) ≈ Array(y_ref)
        @test Array(y_spec) ≈ A_data * collect(1.0:n)
    end
end

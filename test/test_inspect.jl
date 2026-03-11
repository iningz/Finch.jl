using Finch
using Finch: virtualize_with_data, virtualize, VirtualSparseListLevel, VirtualDenseLevel, VirtualFiber, JuliaContext
using SparseArrays

"""
Test that `virtualize_with_data` attaches concrete ptr/idx arrays
to VirtualSparseListLevel during virtualization.
"""

function test_sparse_list_has_data()
    println("  Test: SparseList level gets ptr_data and idx_data")

    A = Tensor(SparseList(Element(0.0)), sprand(10, 0.5))
    lvl = A.lvl

    ctx = JuliaContext()
    vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

    @assert vlvl isa VirtualSparseListLevel
    @assert vlvl.ptr_data !== nothing "ptr_data should not be nothing"
    @assert vlvl.idx_data !== nothing "idx_data should not be nothing"
    @assert vlvl.ptr_data === lvl.ptr "ptr_data should be the same object as lvl.ptr"
    @assert vlvl.idx_data === lvl.idx "idx_data should be the same object as lvl.idx"

    println("    ✓ ptr_data and idx_data are attached")
end

function test_dense_sparse_list_has_data()
    println("  Test: Dense(SparseList(Element)) — data propagates through Dense")

    A = Tensor(Dense(SparseList(Element(0.0))), sprand(5, 4, 0.5))
    lvl = A.lvl

    ctx = JuliaContext()
    vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

    @assert vlvl isa VirtualDenseLevel "outer should be VirtualDenseLevel"
    inner = vlvl.lvl
    @assert inner isa VirtualSparseListLevel "inner should be VirtualSparseListLevel"
    @assert inner.ptr_data !== nothing "inner ptr_data should not be nothing"
    @assert inner.idx_data !== nothing "inner idx_data should not be nothing"
    @assert inner.ptr_data === lvl.lvl.ptr "inner ptr_data should be the same object"
    @assert inner.idx_data === lvl.lvl.idx "inner idx_data should be the same object"

    println("    ✓ data propagates through Dense to inner SparseList")
end

function test_tensor_virtualize_with_data()
    println("  Test: virtualize_with_data on Tensor propagates to levels")

    A = Tensor(Dense(SparseList(Element(0.0))), sprand(5, 4, 0.5))

    ctx = JuliaContext()
    vfbr = virtualize_with_data(ctx, :A, A, :tns)

    @assert vfbr isa VirtualFiber
    outer = vfbr.lvl
    @assert outer isa VirtualDenseLevel
    inner = outer.lvl
    @assert inner isa VirtualSparseListLevel
    @assert inner.ptr_data === A.lvl.lvl.ptr
    @assert inner.idx_data === A.lvl.lvl.idx

    println("    ✓ virtualize_with_data on Tensor works end-to-end")
end

function test_standard_virtualize_has_no_data()
    println("  Test: standard virtualize leaves ptr_data/idx_data as nothing")

    A = Tensor(Dense(SparseList(Element(0.0))), sprand(5, 4, 0.5))

    ctx = JuliaContext()
    vfbr = virtualize(ctx, :A, typeof(A), :tns)

    outer = vfbr.lvl
    inner = outer.lvl
    @assert inner isa VirtualSparseListLevel
    @assert inner.ptr_data === nothing "standard virtualize should leave ptr_data as nothing"
    @assert inner.idx_data === nothing "standard virtualize should leave idx_data as nothing"

    println("    ✓ standard virtualize does not attach data")
end

function test_can_read_structure()
    println("  Test: unfurl-time inspection — can read nnz per column from ptr_data")

    A = Tensor(Dense(SparseList(Element(0.0))), sprand(8, 3, 0.5))
    lvl = A.lvl.lvl  # the SparseList level

    ctx = JuliaContext()
    vlvl = virtualize_with_data(ctx, :A_lvl_lvl, lvl, :tns_lvl)

    ptr = vlvl.ptr_data
    ncols = length(ptr) - 1
    println("    ptr_data = $ptr")
    for col in 1:ncols
        nnz = ptr[col + 1] - ptr[col]
        println("    column $col: nnz = $nnz")
    end

    # Verify the ptr_data values match what we'd compute from the level
    for col in 1:ncols
        expected_nnz = lvl.ptr[col + 1] - lvl.ptr[col]
        actual_nnz = ptr[col + 1] - ptr[col]
        @assert expected_nnz == actual_nnz "nnz mismatch at column $col"
    end

    println("    ✓ structural inspection reads correct nnz per column")
end

function test_can_read_indices()
    println("  Test: can read actual index values from idx_data")

    S = sparse([1, 3, 5], [1, 1, 1], [1.0, 2.0, 3.0], 5, 1)
    A = Tensor(Dense(SparseList(Element(0.0))), S)
    lvl = A.lvl.lvl

    ctx = JuliaContext()
    vlvl = virtualize_with_data(ctx, :A_lvl_lvl, lvl, :tns_lvl)

    ptr = vlvl.ptr_data
    idx = vlvl.idx_data

    q_start = ptr[1]
    q_stop = ptr[2]
    indices = idx[q_start:q_stop-1]
    println("    column 1 indices: $indices")

    @assert length(indices) == 3 "should have 3 nonzeros"
    @assert sort(collect(indices)) == [1, 3, 5] "indices should be [1, 3, 5]"

    println("    ✓ index values are readable and correct")
end

function main()
    println()
    println("=" ^ 60)
    println("  virtualize_with_data: concrete array inspection tests")
    println("=" ^ 60)
    println()

    test_sparse_list_has_data()
    test_dense_sparse_list_has_data()
    test_tensor_virtualize_with_data()
    test_standard_virtualize_has_no_data()
    test_can_read_structure()
    test_can_read_indices()

    println()
    println("=" ^ 60)
    println("  ALL TESTS PASSED ✓")
    println("=" ^ 60)
end

main()

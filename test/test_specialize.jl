using Finch
using Finch: virtualize_with_data, VirtualSparseListLevel, VirtualDenseLevel,
    resolve_fiber_data, SparseListFiberData, is_contiguous,
    VirtualSubFiber, VirtualExtent, unfurl, Run, Thunk, FillLeaf
using Finch.FinchNotation: literal, value, isliteral, getval, reader
using SparseArrays
using Test

@testset "resolve_fiber_data" begin

    @testset "returns nothing when ptr_data is missing" begin
        A = Tensor(Dense(SparseList(Element(0.0))), sprand(5, 3, 0.5))
        ctx = Finch.JuliaContext()
        # Standard virtualize — no concrete data attached
        vfbr = Finch.virtualize(ctx, :A, typeof(A), :tns)
        inner = vfbr.lvl.lvl
        @test inner isa VirtualSparseListLevel
        @test inner.ptr_data === nothing
        @test resolve_fiber_data(inner, literal(1)) === nothing
    end

    @testset "returns nothing when pos is not literal" begin
        A = Tensor(Dense(SparseList(Element(0.0))), sprand(5, 3, 0.5))
        ctx = Finch.JuliaContext()
        vfbr = virtualize_with_data(ctx, :A, A, :tns)
        inner = vfbr.lvl.lvl
        @test inner.ptr_data !== nothing
        # value(...) is a runtime symbol, not a literal
        @test resolve_fiber_data(inner, value(:q, Int)) === nothing
    end

    @testset "returns nothing for out-of-bounds pos" begin
        A = Tensor(Dense(SparseList(Element(0.0))), sprand(5, 3, 0.5))
        ctx = Finch.JuliaContext()
        vfbr = virtualize_with_data(ctx, :A, A, :tns)
        inner = vfbr.lvl.lvl
        # pos = 0 is below valid range
        @test resolve_fiber_data(inner, literal(0)) === nothing
        # pos = length(ptr) is at or past the end (ptr has ncols+1 entries)
        @test resolve_fiber_data(inner, literal(length(inner.ptr_data))) === nothing
    end

    @testset "empty fiber (nnz=0)" begin
        # Construct a matrix where column 2 is entirely zero
        S = sparse([1, 3], [1, 1], [1.0, 2.0], 4, 3)
        A = Tensor(Dense(SparseList(Element(0.0))), S)
        ctx = Finch.JuliaContext()
        vfbr = virtualize_with_data(ctx, :A, A, :tns)
        inner = vfbr.lvl.lvl

        s = resolve_fiber_data(inner, literal(2))  # column 2 — empty
        @test s !== nothing
        @test s isa SparseListFiberData
        @test length(s) == 0
        @test is_contiguous(s) == true  # vacuously contiguous
    end

    @testset "singleton fiber (nnz=1)" begin
        S = sparse([3], [2], [7.0], 5, 3)
        A = Tensor(Dense(SparseList(Element(0.0))), S)
        ctx = Finch.JuliaContext()
        vfbr = virtualize_with_data(ctx, :A, A, :tns)
        inner = vfbr.lvl.lvl

        s = resolve_fiber_data(inner, literal(2))  # column 2 has one entry at row 3
        @test s !== nothing
        @test length(s) == 1
        @test s.indices[1] == 3
        @test is_contiguous(s) == true
    end

    @testset "contiguous indices" begin
        # Column 1 has rows 2,3,4 — contiguous
        S = sparse([2, 3, 4], [1, 1, 1], [1.0, 2.0, 3.0], 6, 2)
        A = Tensor(Dense(SparseList(Element(0.0))), S)
        ctx = Finch.JuliaContext()
        vfbr = virtualize_with_data(ctx, :A, A, :tns)
        inner = vfbr.lvl.lvl

        s = resolve_fiber_data(inner, literal(1))
        @test s !== nothing
        @test length(s) == 3
        @test collect(s.indices) == [2, 3, 4]
        @test is_contiguous(s) == true
    end

    @testset "non-contiguous indices" begin
        # Column 1 has rows 1,3,5 — gaps
        S = sparse([1, 3, 5], [1, 1, 1], [1.0, 2.0, 3.0], 6, 2)
        A = Tensor(Dense(SparseList(Element(0.0))), S)
        ctx = Finch.JuliaContext()
        vfbr = virtualize_with_data(ctx, :A, A, :tns)
        inner = vfbr.lvl.lvl

        s = resolve_fiber_data(inner, literal(1))
        @test s !== nothing
        @test length(s) == 3
        @test collect(s.indices) == [1, 3, 5]
        @test is_contiguous(s) == false
    end

    @testset "start and stop match ptr_data" begin
        S = sparse([1, 2, 4], [1, 1, 2], [10.0, 20.0, 30.0], 5, 3)
        A = Tensor(Dense(SparseList(Element(0.0))), S)
        ctx = Finch.JuliaContext()
        vfbr = virtualize_with_data(ctx, :A, A, :tns)
        inner = vfbr.lvl.lvl
        ptr = inner.ptr_data

        for col in 1:(length(ptr)-1)
            s = resolve_fiber_data(inner, literal(col))
            @test s !== nothing
            @test s.start == ptr[col]
            @test s.stop == ptr[col + 1]
            @test length(s) == ptr[col + 1] - ptr[col]
        end
    end

    @testset "SparseList as outermost level gets pos=literal(1)" begin
        # SparseList(SparseList(Element)) — the outer level gets pos=literal(1)
        S = sparse([1, 3], [1, 2], [5.0, 6.0], 4, 3)
        A = Tensor(SparseList(SparseList(Element(0.0))), S)
        ctx = Finch.JuliaContext()
        vfbr = virtualize_with_data(ctx, :A, A, :tns)
        outer = vfbr.lvl
        @test outer isa VirtualSparseListLevel
        @test outer.ptr_data !== nothing

        # The outermost level always starts at pos=1
        s = resolve_fiber_data(outer, literal(1))
        @test s !== nothing
        # Should reflect which columns are non-empty
        @test length(s) == 2  # columns 1 and 2 have data
        @test collect(s.indices) == [1, 2]
    end

    @testset "indices are a view (no copy)" begin
        S = sparse([1, 2, 3], [1, 1, 1], [1.0, 2.0, 3.0], 5, 2)
        A = Tensor(Dense(SparseList(Element(0.0))), S)
        ctx = Finch.JuliaContext()
        vfbr = virtualize_with_data(ctx, :A, A, :tns)
        inner = vfbr.lvl.lvl

        s = resolve_fiber_data(inner, literal(1))
        @test s !== nothing
        # The indices field should be a view into idx_data, not a copy
        @test s.indices isa SubArray
    end
end

@testset "unfurl specialization" begin

    @testset "empty fiber emits Run instead of Thunk+Stepper" begin
        # Build a 1D SparseList(Element) with no nonzeros directly
        lvl = SparseListLevel{Int}(ElementLevel(0.0, Float64[]), 5, [1, 1], Int[])
        ctx = Finch.JuliaContext()
        vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

        # Confirm the fiber is empty
        fdata = resolve_fiber_data(vlvl, literal(1))
        @test fdata !== nothing
        @test length(fdata) == 0

        # Call unfurl at pos=literal(1) — should get Run, not the generic Thunk
        ext = VirtualExtent(literal(1), literal(5))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader, Finch.defaultread)
        @test looplet isa Run
    end

    @testset "non-empty fiber still emits Thunk (generic path)" begin
        # Build a 1D SparseList(Element) with 2 nonzeros at indices 2 and 4
        lvl = SparseListLevel{Int}(ElementLevel(0.0, [1.0, 2.0]), 5, [1, 3], [2, 4])
        ctx = Finch.JuliaContext()
        vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

        fdata = resolve_fiber_data(vlvl, literal(1))
        @test length(fdata) == 2

        ext = VirtualExtent(literal(1), literal(5))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader, Finch.defaultread)
        @test looplet isa Thunk
    end

    @testset "without concrete data, generic path is used" begin
        lvl = SparseListLevel{Int}(ElementLevel(0.0, Float64[]), 5, [1, 1], Int[])
        ctx = Finch.JuliaContext()
        # Standard virtualize — no data attached
        vlvl = Finch.virtualize(ctx, :A_lvl, typeof(lvl), :tns_lvl)

        @test vlvl.ptr_data === nothing

        ext = VirtualExtent(literal(1), literal(5))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader, Finch.defaultread)
        @test looplet isa Thunk  # no specialization possible
    end
end

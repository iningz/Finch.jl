using Finch
using Finch: virtualize_with_data, VirtualSparseListLevel, VirtualDenseLevel,
    resolve_fiber_data, SparseListFiberData, is_contiguous,
    VirtualSubFiber, VirtualExtent, unfurl, Run, Thunk, FillLeaf,
    Sequence, Phase, Spike, Lookup, SPARSE_LIST_UNROLL_MAX
using Finch.FinchNotation: literal, value, isliteral, getval, reader
using Finch: defaultread
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
        looplet = unfurl(ctx, fbr, ext, reader(), defaultread)
        @test looplet isa Run
    end

    @testset "singleton fiber emits Sequence(Phase(Spike), Phase(Run))" begin
        # Build a 1D SparseList(Element) with 1 nonzero at index 3
        lvl = SparseListLevel{Int}(ElementLevel(0.0, [7.0]), 5, [1, 2], [3])
        ctx = Finch.JuliaContext()
        vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

        fdata = resolve_fiber_data(vlvl, literal(1))
        @test length(fdata) == 1

        ext = VirtualExtent(literal(1), literal(5))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader(), defaultread)
        @test looplet isa Sequence
        @test length(looplet.phases) == 2
        # First phase body should produce a Spike
        phase1_body = looplet.phases[1].body(ctx, ext)
        @test phase1_body isa Spike
        # Second phase body should produce a Run (trailing fill)
        phase2_body = looplet.phases[2].body(ctx, ext)
        @test phase2_body isa Run
    end

    @testset "2-element fiber emits unrolled Sequence with 3 phases" begin
        # Build a 1D SparseList(Element) with 2 nonzeros at indices 2 and 4
        lvl = SparseListLevel{Int}(ElementLevel(0.0, [1.0, 2.0]), 5, [1, 3], [2, 4])
        ctx = Finch.JuliaContext()
        vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

        fdata = resolve_fiber_data(vlvl, literal(1))
        @test length(fdata) == 2

        ext = VirtualExtent(literal(1), literal(5))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader(), defaultread)
        @test looplet isa Sequence
        @test length(looplet.phases) == 3  # 2 Spikes + 1 trailing Run

        # Phase 1: Spike at index 2
        p1_stop = looplet.phases[1].stop(ctx, ext)
        @test p1_stop == literal(2)
        p1_body = looplet.phases[1].body(ctx, ext)
        @test p1_body isa Spike

        # Phase 2: Spike at index 4
        p2_stop = looplet.phases[2].stop(ctx, ext)
        @test p2_stop == literal(4)
        p2_body = looplet.phases[2].body(ctx, ext)
        @test p2_body isa Spike

        # Phase 3: trailing Run(fill)
        p3_body = looplet.phases[3].body(ctx, ext)
        @test p3_body isa Run
    end

    @testset "3-element fiber emits unrolled Sequence with 4 phases" begin
        lvl = SparseListLevel{Int}(ElementLevel(0.0, [1.0, 2.0, 3.0]), 7, [1, 4], [1, 4, 6])
        ctx = Finch.JuliaContext()
        vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

        fdata = resolve_fiber_data(vlvl, literal(1))
        @test length(fdata) == 3

        ext = VirtualExtent(literal(1), literal(7))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader(), defaultread)
        @test looplet isa Sequence
        @test length(looplet.phases) == 4  # 3 Spikes + trailing Run

        # Check stops match the indices
        @test looplet.phases[1].stop(ctx, ext) == literal(1)
        @test looplet.phases[2].stop(ctx, ext) == literal(4)
        @test looplet.phases[3].stop(ctx, ext) == literal(6)

        # All first 3 phases are Spikes
        for k in 1:3
            @test looplet.phases[k].body(ctx, ext) isa Spike
        end

        # Last phase is trailing Run
        @test looplet.phases[4].body(ctx, ext) isa Run
    end

    @testset "4-element fiber (at threshold) emits unrolled Sequence with 5 phases" begin
        lvl = SparseListLevel{Int}(ElementLevel(0.0, [1.0, 2.0, 3.0, 4.0]), 10, [1, 5], [2, 4, 7, 9])
        ctx = Finch.JuliaContext()
        vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

        fdata = resolve_fiber_data(vlvl, literal(1))
        @test length(fdata) == 4
        @test length(fdata) == SPARSE_LIST_UNROLL_MAX  # at the threshold

        ext = VirtualExtent(literal(1), literal(10))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader(), defaultread)
        @test looplet isa Sequence
        @test length(looplet.phases) == 5  # 4 Spikes + trailing Run

        @test looplet.phases[1].stop(ctx, ext) == literal(2)
        @test looplet.phases[2].stop(ctx, ext) == literal(4)
        @test looplet.phases[3].stop(ctx, ext) == literal(7)
        @test looplet.phases[4].stop(ctx, ext) == literal(9)

        for k in 1:4
            @test looplet.phases[k].body(ctx, ext) isa Spike
        end
        @test looplet.phases[5].body(ctx, ext) isa Run
    end

    @testset "5-element fiber (above threshold) emits Thunk (generic path)" begin
        lvl = SparseListLevel{Int}(ElementLevel(0.0, [1.0, 2.0, 3.0, 4.0, 5.0]), 10, [1, 6], [1, 3, 5, 7, 9])
        ctx = Finch.JuliaContext()
        vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

        fdata = resolve_fiber_data(vlvl, literal(1))
        @test length(fdata) == 5
        @test length(fdata) > SPARSE_LIST_UNROLL_MAX  # above the threshold

        ext = VirtualExtent(literal(1), literal(10))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader(), defaultread)
        @test looplet isa Thunk  # too many nonzeros, falls back to generic
    end

    @testset "contiguous 5-element fiber emits Sequence(Run, Lookup, Run)" begin
        # 5 contiguous nonzeros at indices 3,4,5,6,7 — above unroll threshold, contiguous
        lvl = SparseListLevel{Int}(
            ElementLevel(0.0, [10.0, 20.0, 30.0, 40.0, 50.0]),
            10, [1, 6], [3, 4, 5, 6, 7])
        ctx = Finch.JuliaContext()
        vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

        fdata = resolve_fiber_data(vlvl, literal(1))
        @test length(fdata) == 5
        @test length(fdata) > SPARSE_LIST_UNROLL_MAX
        @test is_contiguous(fdata) == true

        ext = VirtualExtent(literal(1), literal(10))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader(), defaultread)
        @test looplet isa Sequence
        @test length(looplet.phases) == 3

        # Phase 1: leading Run(fill) up to index 2
        p1_stop = looplet.phases[1].stop(ctx, ext)
        @test p1_stop == literal(2)   # a - 1 = 3 - 1 = 2
        p1_body = looplet.phases[1].body(ctx, ext)
        @test p1_body isa Run

        # Phase 2: Lookup over the dense block [3..7]
        p2_stop = looplet.phases[2].stop(ctx, ext)
        @test p2_stop == literal(7)
        p2_body = looplet.phases[2].body(ctx, ext)
        @test p2_body isa Lookup

        # Phase 3: trailing Run(fill)
        p3_body = looplet.phases[3].body(ctx, ext)
        @test p3_body isa Run
    end

    @testset "contiguous 6-element fiber emits Lookup, not unrolled Spikes" begin
        # 6 contiguous nonzeros at indices 1..6 — well above threshold
        lvl = SparseListLevel{Int}(
            ElementLevel(0.0, collect(1.0:6.0)),
            8, [1, 7], [1, 2, 3, 4, 5, 6])
        ctx = Finch.JuliaContext()
        vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

        fdata = resolve_fiber_data(vlvl, literal(1))
        @test length(fdata) == 6
        @test is_contiguous(fdata) == true

        ext = VirtualExtent(literal(1), literal(8))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader(), defaultread)
        @test looplet isa Sequence
        @test length(looplet.phases) == 3

        # Leading fill phase stop = a - 1 = 0
        p1_stop = looplet.phases[1].stop(ctx, ext)
        @test p1_stop == literal(0)

        # Dense block phase stop = 6
        p2_stop = looplet.phases[2].stop(ctx, ext)
        @test p2_stop == literal(6)
        @test looplet.phases[2].body(ctx, ext) isa Lookup

        # Trailing fill
        @test looplet.phases[3].body(ctx, ext) isa Run
    end

    @testset "non-contiguous 5-element fiber emits Thunk (generic)" begin
        # 5 non-contiguous nonzeros — above threshold, NOT contiguous
        lvl = SparseListLevel{Int}(
            ElementLevel(0.0, [1.0, 2.0, 3.0, 4.0, 5.0]),
            10, [1, 6], [1, 3, 5, 7, 9])
        ctx = Finch.JuliaContext()
        vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

        fdata = resolve_fiber_data(vlvl, literal(1))
        @test length(fdata) == 5
        @test is_contiguous(fdata) == false

        ext = VirtualExtent(literal(1), literal(10))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader(), defaultread)
        @test looplet isa Thunk  # falls through to generic path
    end

    @testset "contiguous 3-element fiber prefers unrolling over Lookup" begin
        # 3 contiguous nonzeros at indices 2,3,4 — under threshold, contiguous
        # Should use unrolling (Spikes with literal child pos) not Lookup
        lvl = SparseListLevel{Int}(
            ElementLevel(0.0, [1.0, 2.0, 3.0]),
            6, [1, 4], [2, 3, 4])
        ctx = Finch.JuliaContext()
        vlvl = virtualize_with_data(ctx, :A_lvl, lvl, :tns_lvl)

        fdata = resolve_fiber_data(vlvl, literal(1))
        @test length(fdata) == 3
        @test length(fdata) <= SPARSE_LIST_UNROLL_MAX
        @test is_contiguous(fdata) == true

        ext = VirtualExtent(literal(1), literal(6))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader(), defaultread)
        # Should be unrolled Sequence of Spikes, not Lookup
        @test looplet isa Sequence
        @test length(looplet.phases) == 4  # 3 Spikes + trailing Run
        for k in 1:3
            @test looplet.phases[k].body(ctx, ext) isa Spike
        end
        @test looplet.phases[4].body(ctx, ext) isa Run
    end

    @testset "without concrete data, generic path is used" begin
        lvl = SparseListLevel{Int}(ElementLevel(0.0, Float64[]), 5, [1, 1], Int[])
        ctx = Finch.JuliaContext()
        # Standard virtualize — no data attached
        vlvl = Finch.virtualize(ctx, :A_lvl, typeof(lvl), :tns_lvl)

        @test vlvl.ptr_data === nothing

        ext = VirtualExtent(literal(1), literal(5))
        fbr = VirtualSubFiber(vlvl, literal(1))
        looplet = unfurl(ctx, fbr, ext, reader(), defaultread)
        @test looplet isa Thunk  # no specialization possible
    end
end

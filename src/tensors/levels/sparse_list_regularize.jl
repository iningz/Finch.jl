# ═══════════════════════════════════════════════════════════════════════════════
# SparseList Regularization: mining and emission for VirtualSparseListLevel
# ═══════════════════════════════════════════════════════════════════════════════
#
# SparseList-specific implementations of the regularization protocol:
#   - regularize_sparse_data: exposes concrete ptr/idx for mining
#   - emit_looplet: dispatched by (VirtualSparseListLevel, PatternType) pairs
#
# Each pattern type gets its own method.
# Adding a new pattern: adding a new emit_looplet method.
#
# These methods are added to the generic functions defined in
# transforms/regularize.jl.
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# SparseList protocol methods
# ═══════════════════════════════════════════════════════════════════════════════

"""
    regularize_sparse_data(lvl::VirtualSparseListLevel)

Return `(ptr, idx)` if the level has concrete sparsity arrays stashed from
`virtualize_concrete`, otherwise `nothing`.
"""
function regularize_sparse_data(lvl::VirtualSparseListLevel)
    if lvl.ptr_data !== nothing && lvl.idx_data !== nothing
        return (lvl.ptr_data, lvl.idx_data)
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Emitters: one method per pattern type
# ═══════════════════════════════════════════════════════════════════════════════

# ── EmptyPattern ─────────────────────────────────────────────────────────────

"""
All fibers have nnz==0.  Emit a single Run of fill values.
"""
function emit_looplet(ctx, child_lvl::VirtualSparseListLevel,
    pattern::EmptyPattern, mode, parent_pos, Ti)
    Run(FillLeaf(virtual_level_fill_value(child_lvl)))
end

# ── ContiguousPattern ────────────────────────────────────────────────────────
#
# Every fiber stores exactly [first_idx .. first_idx+nnz-1].
# Emit for the i-dimension:
#   Sequence(
#     Phase(stop=first_idx-1,        body=Run(fill)),
#     Phase(stop=first_idx+nnz-1,    body=Lookup(arithmetic qos)),
#     Phase(                          body=Run(fill)),
#   )
# qos = ptr[parent_pos] + (i - first_idx)

function emit_looplet(ctx, child_lvl::VirtualSparseListLevel,
    pattern::ContiguousPattern, mode, parent_pos, Ti)
    fill_leaf = FillLeaf(virtual_level_fill_value(child_lvl))
    first_idx = pattern.first_idx
    nnz = pattern.nnz
    last_idx = first_idx + nnz - 1
    tag = child_lvl.tag
    Tp = postype(child_lvl)

    q_contig = freshen(ctx, tag, :_qc)

    phases = Phase[]

    # before the dense block
    if first_idx > 1
        push!(
            phases,
            Phase(;
                stop=(ctx, ext) -> literal(Ti(first_idx - 1)),
                body=(ctx, ext) -> Run(fill_leaf),
            ),
        )
    end

    # the dense block
    push!(
        phases,
        Phase(;
            stop=(ctx, ext) -> literal(Ti(last_idx)),
            body=(ctx, ext) -> Lookup(;
                body=(ctx, i) -> Thunk(;
                    preamble=quote
                        $q_contig =
                            $(child_lvl.ptr)[$(ctx(parent_pos))] +
                            $(ctx(i)) - $(Ti(first_idx))
                    end,
                    body=(ctx) -> Simplify(
                        instantiate(
                            ctx,
                            VirtualSubFiber(child_lvl.lvl, value(q_contig, Tp)),
                            mode,
                        ),
                    ),
                ),
            ),
        ),
    )

    # after the dense block
    push!(phases, Phase(;
        body=(ctx, ext) -> Run(fill_leaf)
    ))

    return Sequence(phases)
end

# ── IdenticalRelativePattern ─────────────────────────────────────────────────
#
# Every fiber stores the same sparse index set `indices`.  Consecutive
# stored indices are coalesced into contiguous Lookup phases (like
# ContiguousPattern) to reduce the total phase count.  Non-consecutive
# gaps get a Run(fill) phase.
#
# Example: indices = [1,2,3,4,5,6,8,9,10,11,12,13,15,18,20]
# Runs:    [1..6], [8..13], [15], [18], [20]
# Emitted: 5 Lookup phases + up to 5 gap phases + 1 trailing = ~11 phases
# (vs. 30 phases with the naive one-point-per-index approach)

function emit_looplet(ctx, child_lvl::VirtualSparseListLevel,
    pattern::IdenticalRelativePattern, mode, parent_pos, Ti)
    fill_leaf = FillLeaf(virtual_level_fill_value(child_lvl))
    indices = pattern.indices
    tag = child_lvl.tag
    Tp = postype(child_lvl)

    # Split indices into maximal contiguous runs.
    # Each run is (first_offset, first_idx, last_idx) where offset is
    # the 0-based position in the ptr/idx array.
    runs = Tuple{Int,Int,Int}[]
    run_start_offset = 0
    run_start_idx = indices[1]
    for k in 2:length(indices)
        if indices[k] == indices[k - 1] + 1
            continue  # extend current run
        else
            push!(runs, (run_start_offset, run_start_idx, indices[k - 1]))
            run_start_offset = k - 1
            run_start_idx = indices[k]
        end
    end
    push!(runs, (run_start_offset, run_start_idx, indices[end]))

    phases = Phase[]
    prev_idx = 0

    for (run_offset, first_idx, last_idx) in runs
        let run_offset = run_offset, first_idx = first_idx, last_idx = last_idx
            q_run = freshen(ctx, tag, :_qi)

            # Gap phase covering fill positions before this run
            if first_idx > prev_idx + 1
                push!(
                    phases,
                    Phase(;
                        stop=(ctx, ext) -> literal(Ti(first_idx - 1)),
                        body=(ctx, ext) -> Run(fill_leaf),
                    ),
                )
            end

            # Contiguous Lookup phase for this run
            push!(
                phases,
                Phase(;
                    stop=(ctx, ext) -> literal(Ti(last_idx)),
                    body=(ctx, ext) -> Lookup(;
                        body=(ctx, i) -> Thunk(;
                            preamble=quote
                                $q_run =
                                    $(child_lvl.ptr)[$(ctx(parent_pos))] +
                                    $(run_offset) +
                                    $(ctx(i)) - $(Ti(first_idx))
                            end,
                            body=(ctx) -> Simplify(
                                instantiate(
                                    ctx,
                                    VirtualSubFiber(child_lvl.lvl, value(q_run, Tp)),
                                    mode,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        end
        prev_idx = last_idx
    end

    # trailing fill
    push!(phases, Phase(;
        body=(ctx, ext) -> Run(fill_leaf)
    ))

    return Sequence(phases)
end

# ── AffinePattern ────────────────────────────────────────────────────────────
#
# Fiber at parent position p stores a contiguous block of width `nnz` starting
# at  block_start(p) = base_first + (p - outer_start) * delta.
#
# block_start is a runtime expression depending on the outer loop variable.

function emit_looplet(ctx, child_lvl::VirtualSparseListLevel,
    pattern::AffinePattern, mode, parent_pos, Ti)
    fill_leaf = FillLeaf(virtual_level_fill_value(child_lvl))
    nnz = pattern.nnz
    base_first = pattern.base_first
    delta = pattern.delta
    tag = child_lvl.tag
    Tp = postype(child_lvl)
    outer_start = first(region(pattern)[1])

    q_affine = freshen(ctx, tag, :_qa)
    blk_lo = freshen(ctx, tag, :_ablo)
    blk_hi = freshen(ctx, tag, :_abhi)

    phases = Phase[]

    # before the block (stop = blk_lo - 1)
    push!(
        phases,
        Phase(;
            stop=(ctx, ext) -> value(:($blk_lo - $(Ti(1))), Ti),
            body=(ctx, ext) -> Run(fill_leaf),
        ),
    )

    # the dense block (stop = blk_hi)
    push!(
        phases,
        Phase(;
            stop=(ctx, ext) -> value(blk_hi, Ti),
            body=(ctx, ext) -> Lookup(;
                body=(ctx, i) -> Thunk(;
                    preamble=quote
                        $q_affine =
                            $(child_lvl.ptr)[$(ctx(parent_pos))] +
                            $(ctx(i)) - $blk_lo
                    end,
                    body=(ctx) -> Simplify(
                        instantiate(
                            ctx,
                            VirtualSubFiber(child_lvl.lvl, value(q_affine, Tp)),
                            mode,
                        ),
                    ),
                ),
            ),
        ),
    )

    # after the block
    push!(phases, Phase(;
        body=(ctx, ext) -> Run(fill_leaf)
    ))

    seq = Sequence(phases)

    return Thunk(;
        preamble=quote
            $blk_lo =
                $(Ti(base_first)) +
                ($(ctx(parent_pos)) - $(Ti(outer_start))) * $(Ti(delta))
            $blk_hi = $blk_lo + $(Ti(nnz - 1))
        end,
        body=(ctx) -> seq,
    )
end

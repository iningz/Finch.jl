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
# Every fiber stores the same sparse index set `indices`.  Emit an unrolled
# Sequence of Spikes at the known positions.

function emit_looplet(ctx, child_lvl::VirtualSparseListLevel,
    pattern::IdenticalRelativePattern, mode, parent_pos, Ti)
    fill_leaf = FillLeaf(virtual_level_fill_value(child_lvl))
    indices = pattern.indices
    tag = child_lvl.tag
    Tp = postype(child_lvl)

    q_ident = freshen(ctx, tag, :_qi)

    phases = Phase[]

    for (offset, row_idx) in enumerate(indices)
        let offset = offset, row_idx = row_idx
            push!(
                phases,
                Phase(;
                    stop=(ctx, ext) -> literal(Ti(row_idx)),
                    body=(ctx, ext) -> Thunk(;
                        preamble=quote
                            $q_ident =
                                $(child_lvl.ptr)[$(ctx(parent_pos))] +
                                $(offset - 1)
                        end,
                        body=(ctx) -> Spike(;
                            body=fill_leaf,
                            tail=Simplify(
                                instantiate(
                                    ctx,
                                    VirtualSubFiber(child_lvl.lvl, value(q_ident, Tp)),
                                    mode,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        end
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

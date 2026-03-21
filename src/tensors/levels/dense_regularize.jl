# ═══════════════════════════════════════════════════════════════════════════════
# Dense Regularization
# ═══════════════════════════════════════════════════════════════════════════════
#
# Dense-level implementations of the generic regularization protocol defined
# in transforms/regularize.jl.  Contains:
#
#   1. Protocol methods: regularize_level_kind, regularize_child_info for
#      VirtualDenseLevel
#   2. emit_looplet: type-dispatched Phase emission for (VirtualDenseLevel, Pattern)
#
# This file depends on transforms/regularize.jl, Regularity module, and
# dense_levels.jl.
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Protocol Methods
# ═══════════════════════════════════════════════════════════════════════════════

regularize_level_kind(::VirtualDenseLevel) = :dense

"""
    regularize_child_info(lvl::VirtualDenseLevel)

Dense levels with a compile-time-known shape participate in regularization as
addressable parents. The returned NamedTuple provides everything the generic
builder and miner need.
"""
function regularize_child_info(lvl::VirtualDenseLevel)
    isliteral(lvl.shape) || return nothing
    (; shape=Int(lvl.shape.val::Integer),
        child=lvl.lvl,
        tag=lvl.tag,
        Ti=lvl.Ti,
        shape_node=lvl.shape)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  emit_looplet: Dense-parent Phase emission
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each method returns a Phase looplet for the given pattern's region within the
# outer (dense) dimension.  Dispatched on (VirtualDenseLevel, PatternType).

# ── EmptyPattern ─────────────────────────────────────────────────────────────
#
# All fibers in the region have nnz == 0.  Emit a Phase whose body is a
# Run of fill values -- no child-level access at all.

function emit_looplet(ctx, lvl::VirtualDenseLevel, pattern::EmptyPattern,
    mode, parent_pos, Ti)
    dim1_range = pattern.region.ranges[1]
    Phase(;
        stop=(ctx, ext) -> literal(Ti(last(dim1_range))),
        body=(ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl))),
    )
end

# ── OpaquePattern ────────────────────────────────────────────────────────────
#
# Positions not claimed by any specialized miner.  Emit a Phase whose body is
# the generic dense Lookup, same code path as non-regularized access.

function emit_looplet(ctx, lvl::VirtualDenseLevel, pattern::OpaquePattern,
    mode, parent_pos, Ti)
    dim1_range = pattern.region.ranges[1]
    tag = lvl.tag
    q = freshen(ctx, tag, :_q)
    Phase(;
        stop=(ctx, ext) -> literal(Ti(last(dim1_range))),
        body=(ctx, ext) -> Lookup(;
            body=(ctx, i) -> Thunk(;
                preamble=quote
                    $q = ($(ctx(parent_pos)) - $(Ti(1))) * $(ctx(lvl.shape)) + $(ctx(i))
                end,
                body=(ctx) ->
                    instantiate(ctx, VirtualSubFiber(lvl.lvl, value(q, Ti)), mode),
            ),
        ),
    )
end

# ── AbstractPattern (fallback) ───────────────────────────────────────────────
#
# Structured patterns (Contiguous, Affine, IdenticalRelative, ...).  Emit a
# Phase whose body is a Lookup that creates a ChildSubFiber. The sparse
# child's emit_looplet will handle the inner specialization.

function emit_looplet(ctx, lvl::VirtualDenseLevel, pattern::AbstractPattern,
    mode, parent_pos, Ti)
    dim1_range = pattern.region.ranges[1]
    sparse_lvl = lvl.lvl
    Phase(;
        stop=(ctx, ext) -> literal(Ti(last(dim1_range))),
        body=(ctx, ext) -> Lookup(;
            body=(ctx, i) -> begin
                q = freshen(ctx, lvl.tag, :_q)
                preamble = quote
                    $q = ($(ctx(parent_pos)) - $(Ti(1))) * $(ctx(lvl.shape)) + $(ctx(i))
                end
                Thunk(;
                    preamble=preamble,
                    body=(ctx) -> ChildSubFiber(
                        sparse_lvl, value(q, Ti), pattern, mode, Ti
                    ),
                )
            end,
        ),
    )
end

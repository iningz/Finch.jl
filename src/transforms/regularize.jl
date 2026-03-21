# ═══════════════════════════════════════════════════════════════════════════════
# Regularize: compiler integration for regularity mining
# ═══════════════════════════════════════════════════════════════════════════════
#
# This file wires the compiler-independent Regularity module into Finch's
# lowering pipeline. It contains:
#
#   1. Level protocol functions: three extensible functions with safe defaults
#      that levels implement to opt in to regularization
#   2. Mining entry point: walks the virtual level tree, finds dense chains
#      above sparse levels with concrete data, mines regularity patterns, and
#      stores results on the sparse levels' `regularity_map` field
#   3. Emission protocol: `emit_looplet`, an extensible function with a safe
#      default for specialized looplet emission given an AbstractPattern
#   4. regularize_unfurl: generic framework function that builds a regularized
#      Sequence(Phase(...), ...) from a regularity patterns by calling emit_looplet
#   5. ChildSubFiber: wrapper for the sparse child that dispatches to
#      pattern-type-specific emitters during lowering
#
# Level-specific implementations live in their own files:
#   - tensors/levels/dense_regularize.jl       (Dense protocol + emit_looplet)
#   - tensors/levels/sparse_list_regularize.jl  (SparseList protocol + emit_looplet)
#
# The pure mining algorithms and pattern types live in the Regularity module
# (src/Regularity/).
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Level Protocol Functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    regularize_level_kind(lvl::AbstractVirtualLevel) -> Symbol

Return the structural role of this level for regularization purposes:
- `:dense`: statically addressable parent (Dense-like): maps (pos, i) -> child_pos directly
- `:sparse`: minable child: has concrete ptr/idx data that can be analyzed
- `:opaque`: neither (default)
"""
regularize_level_kind(::AbstractVirtualLevel) = :opaque

"""
    regularize_child_info(lvl::AbstractVirtualLevel)

For a dense-like parent level, return a NamedTuple describing its structure:
    (; shape::Int, child::AbstractVirtualLevel, tag::Symbol, Ti::Type, shape_node)
For non-dense levels, return `nothing`.
"""
regularize_child_info(::AbstractVirtualLevel) = nothing

"""
    regularize_sparse_data(lvl::AbstractVirtualLevel)

If `lvl` has concrete sparsity data that can be mined at compile time,
return `(ptr, idx)`. Otherwise return `nothing`.
"""
regularize_sparse_data(::AbstractVirtualLevel) = nothing

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Mining Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

"""
    mine_regular_structure!(fiber)

Pre-lowering pass: walk the virtual level tree, find dense chains above sparse
levels with concrete data, mine regularity patterns, and store results on the
sparse levels' `regularity_map` field.
"""
function mine_regular_structure!(fiber::VirtualFiber)
    _mine_level!(fiber.lvl, Int[], 1)
    return nothing
end

"""
    _mine_level!(lvl, dense_chain_shapes, root_pos)

Recursive walker. Collects Dense shapes in `dense_chain_shapes` (outermost
first). When hitting a sparse level with concrete data, calls the miner.
"""
function _mine_level!(
    lvl::AbstractVirtualLevel, dense_chain_shapes::Vector{Int}, root_pos::Int
)
    # Dense-like level with literal shape: collect shape and recurse into child
    info = regularize_child_info(lvl)
    if info !== nothing
        _mine_level!(info.child, vcat(dense_chain_shapes, [info.shape]), root_pos)
        return nothing
    end

    # Minable sparse level: run the miner if we have a Dense chain above
    sparse_data = regularize_sparse_data(lvl)
    if sparse_data !== nothing
        if !isempty(dense_chain_shapes)
            (ptr, idx_data) = sparse_data
            patterns = mine_nd(ptr, idx_data, dense_chain_shapes)

            if !isempty(patterns)
                if hasproperty(lvl, :regularity_map)
                    lvl.regularity_map = patterns
                end
            end
        end
    end

    # Default: try to recurse into child if the level has one.
    if hasproperty(lvl, :lvl)
        _mine_level!(lvl.lvl, Int[], 1)
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Emission Protocol
# ═══════════════════════════════════════════════════════════════════════════════

"""
    emit_looplet(ctx, lvl, pattern::AbstractPattern, mode, pos, Ti)

Given an `AbstractPattern`, emit a specialized looplet tree. Default returns
`nothing` (fall through to generic unfurl).

Implementations dispatch on both level type and pattern type.

**Dense parent** (emits a `Phase` per pattern region):

    emit_looplet(ctx, ::VirtualDenseLevel, ::EmptyPattern, ...)
    emit_looplet(ctx, ::VirtualDenseLevel, ::OpaquePattern, ...)
    emit_looplet(ctx, ::VirtualDenseLevel, ::AbstractPattern, ...)

**Sparse child** (emits the inner looplet for a fiber access pattern):

    emit_looplet(ctx, ::VirtualSparseListLevel, ::ContiguousPattern, ...)
    emit_looplet(ctx, ::VirtualSparseListLevel, ::AffinePattern, ...)
"""
emit_looplet(
    ctx, ::AbstractVirtualLevel, pattern::AbstractPattern, mode, parent_pos, Ti
) = nothing

# ═══════════════════════════════════════════════════════════════════════════════
# 4. regularize_unfurl: generic framework function
# ═══════════════════════════════════════════════════════════════════════════════
#
# Any parent level that participates in regularization calls this from its
# unfurl method.  It reads the regularity patterns from the child, sorts the
# patterns, and builds a Sequence(Phase(...), ...) by calling emit_looplet
# for each pattern.  Returns `nothing` when regularization is not applicable.

"""
    _get_regularity_patterns(lvl) -> Union{Nothing, Vector{AbstractPattern}}

Retrieve the pre-computed regularity patterns stored on a sparse level, or `nothing`.
"""
function _get_regularity_patterns(lvl)
    hasproperty(lvl, :regularity_map) || return nothing
    return lvl.regularity_map
end

"""
    regularize_unfurl(ctx, lvl, mode, pos) -> Union{Nothing, Sequence}

Attempt to build a regularized `Sequence(Phase(...), ...)` looplet for a
parent level whose direct child carries a pre-computed regularity patterns.

Parent levels call this from their `unfurl` method. The function:
1. Retrieves the regularity patterns from the child level
2. Sorts patterns by region start
3. Calls `emit_looplet(ctx, lvl, pattern, mode, pos, Ti)` for each pattern
4. Returns `Sequence(phases)`, or `nothing` if regularization is not applicable

For now only handles the 1D case where the parent's immediate child is the
minable sparse level.
"""
function regularize_unfurl(ctx, lvl, mode, pos)
    info = regularize_child_info(lvl)
    info === nothing && return nothing
    Ti = info.Ti

    child = info.child
    patterns = _get_regularity_patterns(child)
    patterns === nothing && return nothing
    isempty(patterns) && return nothing

    # The miner is total: every position is covered by some pattern
    # (specialized passes + OpaquePattern for the rest).
    sorted_patterns = sort(patterns; by=p -> first(p.region.ranges[1]))
    phases = Phase[]

    for pattern in sorted_patterns
        push!(phases, emit_looplet(ctx, lvl, pattern, mode, pos, Ti))
    end

    return Sequence(phases)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ChildSubFiber
# ═══════════════════════════════════════════════════════════════════════════════
#
# When the parent's emit_looplet processes a structured pattern, the Lookup
# body returns a ChildSubFiber instead of a plain VirtualSubFiber.  When the
# lowerer then processes the inner loop it calls `unfurl` on this wrapper,
# which dispatches to the child level's emit_looplet methods.

struct ChildSubFiber
    child_lvl::AbstractVirtualLevel  # the minable sparse level
    pos                              # FinchNode for this fiber's position
    pattern::AbstractPattern         # mined pattern
    mode                             # reader / updater FinchNode
    Ti::Type                         # index type
end

FinchNotation.finch_leaf(csf::ChildSubFiber) = virtual(csf)

virtual_fill_value(_, csf::ChildSubFiber) = virtual_level_fill_value(csf.child_lvl)

function virtual_size(ctx, csf::ChildSubFiber)
    return virtual_size(ctx, VirtualSubFiber(csf.child_lvl, csf.pos))
end

instantiate(_, csf::ChildSubFiber, _) = csf

function unfurl(ctx, csf::ChildSubFiber, ext, mode,
    ::Union{typeof(defaultread),typeof(walk),typeof(follow)})
    result = emit_looplet(
        ctx, csf.child_lvl, csf.pattern, csf.mode, csf.pos, csf.Ti
    )
    if result !== nothing
        return result
    end
    return unfurl(ctx, VirtualSubFiber(csf.child_lvl, csf.pos), ext, mode, defaultread)
end

function unfurl(ctx, csf::ChildSubFiber, ext, mode, proto)
    unfurl(ctx, VirtualSubFiber(csf.child_lvl, csf.pos), ext, mode, proto)
end

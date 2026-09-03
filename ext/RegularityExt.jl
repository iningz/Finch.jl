# This extension connects Regularity descriptions to Finch looplets:
#
#     Finch storage -> Structure -> Description -> Finch looplets
#          |                                          |
#          +-------------- freshness <---------------+
#
# The host supplies four things: a projection of Finch storage into a
# Regularity Structure, a staging interpreter for Regularity's arithmetic
# (`FinchExprOps`), a realization of Descriptions as looplets, and a
# freshness check for reusable kernels. Regularity describes counts and
# coordinates; Finch still owns loops, phases, storage positions, and ordinary
# sparse traversal. A Segment becomes Finch `Sequence`, `Phase`, `Lookup`, and
# `Thunk` nodes; a Series becomes one parameterized `Stepper` whose per-step
# bounds are staged formulas of the repetition ordinal; an Opaque stretch
# delegates to Finch's native traversal.
#
# A mined Description lowers as it is. The miner produces an exact ordered
# partition of strictly increasing coordinates, folds a Series only when its
# direct member counts are repetition-invariant and its direct coordinates
# advance affinely with the repetition (`_series_winners`), and lift
# reproduces every fiber it was fitted across. Every quantity the Stepper
# relies on - a positive repetition count, a strictly increasing
# repetition-end sequence, ordered disjoint members - is therefore a mined
# fact, re-verified by `check_description` in the evaluation, never re-proved
# here. The only host-side rewrite is pruning, a profitability policy whose
# demotions keep the partition exact.
#
# At each sparse level, every realized fiber derives its own origin, one of
# two ways:
#
#     origin  = a stage-derived pure formula of the enclosing ordinals
#               (Regularity's `child_origin`: claimed counts summed in closed
#               form), or
#     origin  = ptr[fiber position]          (one read per fiber, fallback)
#     position = origin + earlier children + (coordinate - first coordinate)
#
# Claimed entries never read `idx`; Opaque stretches walk their staged
# position range; `ptr` is read at most once per realized fiber and never
# when the origin derives. Either origin is a pure function of the fiber's
# position and ordinals with no runtime state crossing fibers or phases, so
# emitted bodies stay correct under Sequence composition with other
# multi-phase operands (re-emission, truncation, simplification).

module RegularityExt

using Finch
using Finch:
    VirtualFiber, VirtualSubFiber, AbstractVirtualLevel,
    VirtualDenseLevel, VirtualSparseListLevel, VirtualElementLevel,
    DenseLevel, SparseListLevel, ElementLevel,
    ConcreteStash, concrete_stash, structural_token,
    specialization_attempt,
    Phase, Sequence, Run, Lookup, Thunk, Stepper, Spike, FillLeaf, Simplify,
    instantiate, unfurl, unfurl_sparse_list_walk,
    literal, isliteral, value, freshen, virtual, getstart, getstop,
    virtual_fill_value, virtual_level_fill_value, virtual_size, postype,
    defaultread, walk, follow, gallop,
    VirtualExtent,
    FinchNotation
using Finch.FinchNotation: reader, updater, variable
import Finch: mine_regular_structure!, regularize_unfurl

using Regularity
const R = Regularity
import Regularity: add, sub, mul, fld, mod, select

# -- Staging interpreter -------------------------------------------------------
#
# `FinchExprOps` implements Regularity's arithmetic interface with Julia
# expressions. Literal operands are folded while expressions are built. Pattern
# families therefore use the same formula implementation during mining and
# Finch code generation.

"A staged integer: a literal, a host variable, or host code."
const Staged = Union{Int,Symbol,Expr}

struct FinchExprOps <: R.Ops{Any} end
const STAGE = FinchExprOps()

_both_int(a::Staged, b::Staged)::Bool = a isa Int && b isa Int

function add(::FinchExprOps, a::Staged, b::Staged)::Staged
    _both_int(a, b) && return Base.Checked.checked_add(a, b)
    a === 0 && return b
    b === 0 && return a
    :($a + $b)
end

function sub(::FinchExprOps, a::Staged, b::Staged)::Staged
    _both_int(a, b) && return Base.Checked.checked_sub(a, b)
    b === 0 && return a
    :($a - $b)
end

function mul(::FinchExprOps, a::Staged, b::Staged)::Staged
    _both_int(a, b) && return Base.Checked.checked_mul(a, b)
    (a === 0 || b === 0) && return 0
    a === 1 && return b
    b === 1 && return a
    :($a * $b)
end

function fld(::FinchExprOps, a::Staged, b::Staged)::Staged
    _both_int(a, b) && return Base.fld(a, b)
    b === 1 && return a
    :(fld($a, $b))
end

function mod(::FinchExprOps, a::Staged, b::Staged)::Staged
    _both_int(a, b) && return Base.mod(a, b)
    b === 1 && return 0
    :(mod($a, $b))
end

function select(::FinchExprOps, alternatives::Tuple, selector::Staged)::Staged
    selector isa Int && return alternatives[selector + 1]
    # The tuple length bounds this conditional chain. The final error branch
    # preserves the checked integer interpretation for an invalid selector.
    chain::Staged = :(throw(BoundsError()))
    for arm in (length(alternatives) - 1):-1:0
        chain = :($selector == $arm ? $(alternatives[arm + 1]) : $chain)
    end
    chain
end

# Bind a count with staged enclosing ordinals.
_staged_count(count::Int, ordinals::Tuple)::Staged = count
_staged_count(count::R.Pattern, ordinals::Tuple)::Staged = R.bind(count, ordinals, STAGE)

# The staged count of one node: the direct children it owns at its own level.
# A Segment or Opaque owns its count; a Series owns `reps * word_count`.
_staged_count(node::Union{R.Segment,R.Opaque}, ordinals::Tuple)::Staged =
    _staged_count(node.count, ordinals)
_staged_count(node::R.Series, ordinals::Tuple)::Staged =
    mul(STAGE, _staged_count(node.reps, ordinals), _staged_word_count(node, ordinals))

# The word count of a Series: the direct members' counts summed at repetition
# 0, repetition-invariant by the series law.
function _staged_word_count(node::R.Series, ordinals::Tuple)::Staged
    word_count::Staged = 0
    for member in node.body
        word_count = add(STAGE, word_count, _staged_count(member, (ordinals..., 0)))
    end
    word_count
end

"The stretch's first coordinate: the coordinate pattern at child ordinal 0."
_staged_first_coordinate(coordinate::R.Pattern, ordinals::Tuple)::Staged =
    R.bind(coordinate, (ordinals..., 0), STAGE)

# -- Projection ----------------------------------------------------------------

"A mined description, the freshness it was mined at (reusable kernels only), and the structural reads it avoids."
struct Candidate
    description::R.Description
    freshness::Any               # `Freshness` for reuse; otherwise `nothing`.
    reads::Int
end

Candidate(description::R.Description, freshness) = Candidate(description, freshness, 0)
"Marks a tensor that was examined and not selected, preventing duplicate work."
struct Declined end

struct Projection
    virtuals::Vector{AbstractVirtualLevel}   # Outermost first; index levels only.
    structure::R.Structure
    reusable::Bool
end

# Regularity reads coordinates and extents as `Int`; a storage integer type
# that may not fit is out of scope.
function _int_fits(::Type{T}) where {T}
    T <: Base.BitInteger && typemin(Int) <= typemin(T) && typemax(T) <= typemax(Int)
end

function _regularity_level(lvl::DenseLevel)
    _int_fits(typeof(lvl.shape)) ? R.DenseLevel(Int(lvl.shape)) : nothing
end
function _regularity_level(lvl::SparseListLevel)
    _int_fits(eltype(lvl.idx)) ? R.CompressedLevel(lvl.ptr, lvl.idx) : nothing
end

# Project Finch's virtual and concrete level chains into a Regularity Structure.
# Unsupported levels or missing concrete data return `nothing`, leaving the
# tensor on Finch's ordinary path.
function project_structure(root::AbstractVirtualLevel)::Union{Projection,Nothing}
    virtuals = AbstractVirtualLevel[]
    levels = R.Level[]
    reusable = nothing
    virtual_level = root
    while !(virtual_level isa VirtualElementLevel)
        virtual_level isa Union{VirtualDenseLevel,VirtualSparseListLevel} || return nothing
        stash = concrete_stash(virtual_level)
        stash isa ConcreteStash || return nothing
        level = _regularity_level(stash.lvl)
        level === nothing && return nothing
        push!(virtuals, virtual_level)
        push!(levels, level)
        reusable === nothing || reusable == stash.reusable || return nothing
        reusable = stash.reusable
        virtual_level = virtual_level.lvl
    end
    isempty(levels) && return nothing
    Projection(virtuals, R.Structure(levels...), something(reusable))
end

# -- Freshness -----------------------------------------------------------------

# Values checked before a reusable kernel runs. Each compressed level records
# a token and generation for EACH structural array - ptr and idx carry their
# own tokens, so a lifecycle rewrite reached through any other level that
# shares one of them is still observed, and a replaced array carries a new
# token. Each dense level records its extent.
struct Freshness
    ptr_token_ids::Vector{UInt}
    ptr_generations::Vector{UInt64}
    idx_token_ids::Vector{UInt}
    idx_generations::Vector{UInt64}
    dense_extents::Vector{Int}           # One per dense level, in chain order.
end

function freshness_snapshot(projection::Projection)::Freshness
    freshness = Freshness(UInt[], UInt64[], UInt[], UInt64[], Int[])
    for level in projection.structure.levels
        if level isa R.CompressedLevel
            ptr_token = structural_token(level.ptr)
            idx_token = structural_token(level.idx)
            push!(freshness.ptr_token_ids, objectid(ptr_token))
            push!(freshness.ptr_generations, ptr_token.generation)
            push!(freshness.idx_token_ids, objectid(idx_token))
            push!(freshness.idx_generations, idx_token.generation)
        else
            push!(freshness.dense_extents, (level::R.DenseLevel).extent)
        end
    end
    freshness
end

function _runtime_shape(virtual_level::VirtualDenseLevel)
    virtual_level.shape.kind === FinchNotation.value ? virtual_level.shape.val : nothing
end

function freshness_preamble(root::AbstractVirtualLevel, freshness::Freshness)::Expr
    checks = Expr[]
    compressed_index = 0
    dense_index = 0
    virtual_level = root
    while virtual_level isa Union{VirtualDenseLevel,VirtualSparseListLevel}
        if virtual_level isa VirtualSparseListLevel
            compressed_index += 1
            ptr = virtual_level.ptr
            idx = virtual_level.idx
            push!(
                checks,
                quote
                    (
                        Finch.structure_current(
                            $ptr,
                            $(freshness.ptr_token_ids[compressed_index]),
                            $(freshness.ptr_generations[compressed_index]),
                        ) && Finch.structure_current(
                            $idx,
                            $(freshness.idx_token_ids[compressed_index]),
                            $(freshness.idx_generations[compressed_index]),
                        )
                    ) || error(
                        "Regularity: stale specialization; a structural token does " *
                        "not match the mined snapshot (structure was rewritten since mining)",
                    )
                end,
            )
        else
            dense_index += 1
            shape_variable = _runtime_shape(virtual_level)
            shape_variable === nothing || push!(
                checks,
                quote
                    $shape_variable == $(freshness.dense_extents[dense_index]) ||
                        error(
                            "Regularity: stale specialization; dense extent changed")
                end,
            )
        end
        virtual_level = virtual_level.lvl
    end
    Expr(:block, checks...)
end

# -- Pruning -------------------------------------------------------------------
#
# Pruning is a profitability rewrite over a mined description. It runs
# bottom-up by storage level; within one span list the order is
#
#   1. rebuild: every span's body is replaced by its pruned form (a body that
#      could not prune demotes its owner);
#   2. coalesce: adjacent Segments whose bodies continue each other merge;
#   3. fold: the miner's Series folder is re-run (root list only: the folder
#      binds the repetition ordinal outermost, which is the correct position
#      only where no enclosing ordinal exists yet);
#   4. judge: the code rule per span, then the visit rule per list;
#   5. adjacent integer-counted Opaques merge.
#
# The numerator everywhere is STRUCTURAL READS AVOIDED: a leaf Segment at a
# compressed level avoids one `idx` read per entry; a Segment whose children
# sit at a compressed level avoids two `ptr` reads per child whose origin
# derives (`child_origin` succeeds along the child's trail). Dense levels
# store nothing, so their counts contribute nothing. Reads aggregate over
# every parent visit that shares a list.
#
#   code rule   reads(subtree) >= code_density * abstract_spans(subtree)
#   visit rule  reads(list)    >= visit_density * (concrete_spans(list) - visits)
#
# both summed over the list's visits. A failing visit rule demotes the kept
# span with the fewest reads and re-checks. Demotion replaces a span by an
# Opaque of the same count and is always sound: the partition stays exact.

# A count that is provably one integer under every ordinal assignment.
_constant_count(count::R.Count)::Union{Nothing,Int} = R.constant_value(count)

# The constant `value` as a count with `binders` binders: an `Int` at the
# root, otherwise a Lift chain of constant scalars.
function _constant_pattern(value::Int, binders::Int)::R.Count
    binders == 0 && return value
    pattern = R.PeriodicAffine((value,), 0)
    for _ in 2:binders
        pattern = R.Lift(pattern,
            Tuple(R.PeriodicAffine((n,), 0) for n in R.numbers(pattern)))
    end
    pattern
end

# -- Continuation ---------------------------------------------------------------
#
# `right` continues `left` at binder `shifted` when, for every assignment of
# the other binders, `right` at ordinal `j` equals `left` at ordinal
# `left_count + j` over `j in 0:right_count-1`. Binders outside `shifted` are
# admitted only when constant (the enclosing ones) or determined by the
# compared scalars (the deeper ones, through Lift's refill), so the check is
# the scalar comparison `_series_compatible` makes, taken at the shifted
# binder.
function _scalars_at(pattern::R.Pattern, shifted::Int)::Union{Nothing,Vector{R.Scalar}}
    for _ in 2:shifted
        pattern isa R.Scaled && (pattern = pattern.pattern)
        pattern isa R.Lift || return nothing
        contents = Int[]
        for param in pattern.params
            value = R.constant_value(param)
            value === nothing && return nothing
            push!(contents, value)
        end
        pattern = R.refill(pattern.child, contents, R.Value)
    end
    pattern isa R.Scaled && (pattern = pattern.pattern)
    pattern isa R.Scalar && return R.Scalar[pattern]
    pattern isa R.Lift && return R.Scalar[pattern.params...]
    nothing
end

function _continues(left::R.Pattern, right::R.Pattern, shifted::Int,
    left_count::Int, right_count::Int)::Bool
    R.skeleton(left) == R.skeleton(right) || return false
    left_scalars = _scalars_at(left, shifted)
    right_scalars = _scalars_at(right, shifted)
    (left_scalars === nothing || right_scalars === nothing) && return false
    length(left_scalars) == length(right_scalars) || return false
    for (l, r) in zip(left_scalars, right_scalars), j in 0:(right_count - 1)
        R.evaluate(l, Base.Checked.checked_add(left_count, j)) == R.evaluate(r, j) ||
            return false
    end
    true
end
_continues(left::Int, right::Int, ::Int, ::Int, ::Int)::Bool = left == right
_continues(::R.Count, ::R.Count, ::Int, ::Int, ::Int)::Bool = false

function _opaque_total(spans::Vector{R.Span})::Union{Nothing,Int}
    total = 0
    for span in spans
        span isa R.Opaque || return nothing
        count = R.constant_value(span.count)
        count === nothing && return nothing
        total = Base.Checked.checked_add(total, count)
    end
    total
end

function _continues(left::Vector{R.Span}, right::Vector{R.Span}, shifted::Int,
    left_count::Int, right_count::Int)::Bool
    left_opaque = _opaque_total(left)
    right_opaque = _opaque_total(right)
    left_opaque === nothing || right_opaque === nothing ||
        return left_opaque == right_opaque
    length(left) == length(right) || return false
    all(zip(left, right)) do (l, r)
        _continues(l, r, shifted, left_count, right_count)
    end
end
function _continues(left::R.Segment, right::R.Segment, shifted::Int,
    left_count::Int, right_count::Int)::Bool
    _continues(left.count, right.count, shifted, left_count, right_count) &&
        _continues(left.coord, right.coord, shifted, left_count, right_count) &&
        (left.body === nothing) == (right.body === nothing) &&
        (left.body === nothing ||
         _continues(left.body, right.body, shifted, left_count, right_count))
end
function _continues(left::R.Opaque, right::R.Opaque, shifted::Int,
    left_count::Int, right_count::Int)::Bool
    _continues(left.count, right.count, shifted, left_count, right_count)
end
function _continues(left::R.Series, right::R.Series, shifted::Int,
    left_count::Int, right_count::Int)::Bool
    _continues(left.reps, right.reps, shifted, left_count, right_count) &&
        _continues(left.body, right.body, shifted, left_count, right_count)
end
_continues(::R.Span, ::R.Span, ::Int, ::Int, ::Int)::Bool = false

# Merge adjacent Segments of a list whose patterns take `binders` binders,
# when their coordinates and bodies continue at the list's child ordinal
# (binder `binders + 1`), to a fixed point. Only integer-constant counts
# merge: the merged count is their sum, restated with the list's binder
# count. An Opaque between two Segments is never absorbed.
function _coalesce(spans::Vector{R.Span}, binders::Int)::Vector{R.Span}
    shifted = binders + 1
    out = R.Span[]
    for span in spans
        while span isa R.Segment && !isempty(out) && out[end] isa R.Segment
            left = out[end]::R.Segment
            left_count = _constant_count(left.count)
            right_count = _constant_count(span.count)
            (left_count === nothing || right_count === nothing) && break
            _continues(left.coord, span.coord, shifted, left_count, right_count) || break
            (left.body === nothing) == (span.body === nothing) || break
            left.body === nothing ||
                _continues(left.body, span.body, shifted, left_count, right_count) ||
                break
            pop!(out)
            span = R.Segment(
                _constant_pattern(Base.Checked.checked_add(left_count, right_count),
                    binders),
                left.coord, left.body)
        end
        push!(out, span)
    end
    out
end

# -- Counting -------------------------------------------------------------------

_abstract(span::R.Opaque)::Int = 1
_abstract(span::R.Segment)::Int =
    span.body === nothing ? 1 : 1 + _abstract(span.body)
_abstract(span::R.Series)::Int = 1 + _abstract(span.body)
_abstract(spans::Vector{R.Span})::Int = sum(_abstract, spans; init=0)

# Concrete spans one visit of a list executes: a Series runs its word once
# per repetition.
_concrete(::Union{R.Segment,R.Opaque}, ::Tuple{Vararg{Int}})::Int = 1
function _concrete(span::R.Series, ordinals::Tuple{Vararg{Int}})::Int
    word = sum(member -> _concrete(member, (ordinals..., 0)), span.body; init=0)
    R.bind_count(span.reps, ordinals) * word
end

# Symbolic ordinals for the static trails derivability is judged on: the
# emitter binds every ordinal to a host expression, so the derivation must
# succeed with non-literal ordinals exactly as it will at realization.
_ordinal_symbol(depth::Int)::Symbol = Symbol(:k, depth)

"""
    ReadTally

Aggregated reads and concrete spans per span of one list, plus the number of
parent visits. Keyed by list identity in `_prune_description`.
"""
struct ReadTally
    reads::Vector{Int}
    concrete::Vector{Int}
    visits::Base.RefValue{Int}
    visit_reads::Vector{Vector{Int}}
    visit_concrete::Vector{Vector{Int}}
    visit_ordinals::Vector{Tuple}
    visit_trails::Vector{Tuple}
end
ReadTally(n::Int) = ReadTally(
    zeros(Int, n), zeros(Int, n), Ref(0), Vector{Int}[], Vector{Int}[],
    Tuple[], Tuple[])
# (original list identity -> pruned list), and the per-list tallies of the
# level under judgment.
mutable struct Pruner
    const levels::Tuple{Vararg{R.Level}}
    const pruned::IdDict{Vector{R.Span},Union{Nothing,Vector{R.Span}}}
    const tallies::IdDict{Vector{R.Span},ReadTally}
    const derives::IdDict{Vector{R.Span},Vector{Union{Nothing,Bool}}}
end

_pruned_body(pruner::Pruner, body::Vector{R.Span}) = get(pruner.pruned, body, body)

# Whether the children of `spans[i]` (a Segment with a body) derive their
# origin along `trail`, cached per list and index.
function _derives(pruner::Pruner, spans::Vector{R.Span}, i::Int,
    trail::Tuple{Vararg{R.TrailStep}})::Bool
    cache = get!(pruner.derives, spans) do
        Vector{Union{Nothing,Bool}}(nothing, length(spans))
    end
    known = cache[i]
    known === nothing || return known
    segment = spans[i]::R.Segment
    range = R.ordinal_range(segment.count, map(s -> s.range, trail))
    step = R.Descent(spans, i, _ordinal_symbol(length(trail) + 1), range)
    derived = R.child_origin((trail..., step), STAGE) !== nothing
    cache[i] = derived
    derived
end

# Reads avoided by one visit of `span` at `level` under `ordinals`, walking
# its (pruned) subtree. `trail` is the static trail to the list holding
# `span` (symbolic ordinals), used for derivability.
_reads_of(::Pruner, ::R.Opaque, ::Vector{R.Span}, ::Int, ::Int,
    ::Tuple{Vararg{Int}}, ::Tuple{Vararg{R.TrailStep}})::Int = 0
function _reads_of(pruner::Pruner, span::R.Segment, spans::Vector{R.Span}, i::Int,
    level::Int, ordinals::Tuple{Vararg{Int}}, trail::Tuple{Vararg{R.TrailStep}})::Int
    n = R.bind_count(span.count, ordinals)
    if span.body === nothing
        return pruner.levels[level] isa R.CompressedLevel ? n : 0
    end
    body = _pruned_body(pruner, span.body)
    total = 0
    # A realized child list reads no end pointer: its extent is its staged
    # count. With a derived origin both ptr reads go; otherwise the origin is
    # one ptr read and one is still avoided.
    child_compressed = pruner.levels[level + 1] isa R.CompressedLevel
    if child_compressed && n > 0
        total += _derives(pruner, spans, i, trail) ? 2n : n
    end
    isempty(body) && return total
    range = R.ordinal_range(span.count, map(s -> s.range, trail))
    child_trail = (trail..., R.Descent(spans, i, _ordinal_symbol(length(trail) + 1), range))
    for k in 0:(n - 1)
        total += _reads_of_list(pruner, body, level + 1, (ordinals..., k), child_trail)
    end
    total
end
function _reads_of(pruner::Pruner, span::R.Series, spans::Vector{R.Span}, i::Int,
    level::Int, ordinals::Tuple{Vararg{Int}}, trail::Tuple{Vararg{R.TrailStep}})::Int
    reps = R.bind_count(span.reps, ordinals)
    reps > 0 || return 0
    range = R.ordinal_range(span.reps, map(s -> s.range, trail))
    body_trail = (trail..., R.Along(spans, i, _ordinal_symbol(length(trail) + 1), range))
    total = 0
    for j in 0:(reps - 1)
        total += _reads_of_list(pruner, span.body, level, (ordinals..., j), body_trail)
    end
    total
end
function _reads_of_list(pruner::Pruner, spans::Vector{R.Span}, level::Int,
    ordinals::Tuple{Vararg{Int}}, trail::Tuple{Vararg{R.TrailStep}})::Int
    # An unanchored Series makes sparse lowering use one bounded native walk
    # for the whole list. That walk avoids no idx or descendant ptr reads;
    # the enclosing Segment separately credits this fiber's derived origin.
    pruner.levels[level] isa R.CompressedLevel &&
        _has_unanchored_series(spans) && return 0
    total = 0
    for (i, span) in enumerate(spans)
        total += _reads_of(pruner, span, spans, i, level, ordinals, trail)
    end
    total
end

# Tally one visit of the normalized list `spans` (and, through Series, of
# every repetition of their words).
function _tally_visit!(pruner::Pruner, spans::Vector{R.Span}, level::Int,
    ordinals::Tuple{Vararg{Int}}, trail::Tuple{Vararg{R.TrailStep}})::Nothing
    tally = get!(() -> ReadTally(length(spans)), pruner.tallies, spans)
    tally.visits[] += 1
    visit_reads = zeros(Int, length(spans))
    visit_concrete = zeros(Int, length(spans))
    for (i, span) in enumerate(spans)
        visit_reads[i] = _reads_of(pruner, span, spans, i, level, ordinals, trail)
        visit_concrete[i] = _concrete(span, ordinals)
        tally.reads[i] += visit_reads[i]
        tally.concrete[i] += visit_concrete[i]
        if span isa R.Series
            reps = R.bind_count(span.reps, ordinals)
            range = R.ordinal_range(span.reps, map(s -> s.range, trail))
            body_trail = (trail..., R.Along(spans, i, _ordinal_symbol(length(trail) + 1), range))
            for j in 0:(reps - 1)
                _tally_visit!(pruner, span.body, level, (ordinals..., j), body_trail)
            end
        end
    end
    push!(tally.visit_reads, visit_reads)
    push!(tally.visit_concrete, visit_concrete)
    push!(tally.visit_ordinals, ordinals)
    push!(tally.visit_trails, trail)
    nothing
end

# Walk the original tree to every visit of a list at `target` and tally its
# normalized form (`normalized[list]`).
function _tally_level!(pruner::Pruner, spans::Vector{R.Span}, level::Int, target::Int,
    ordinals::Tuple{Vararg{Int}}, trail::Tuple{Vararg{R.TrailStep}},
    normalized::IdDict{Vector{R.Span},Vector{R.Span}})::Nothing
    if level == target
        _tally_visit!(pruner, normalized[spans], level, ordinals, trail)
        return nothing
    end
    for (i, span) in enumerate(spans)
        if span isa R.Series
            reps = R.bind_count(span.reps, ordinals)
            range = R.ordinal_range(span.reps, map(s -> s.range, trail))
            body_trail = (trail..., R.Along(spans, i, _ordinal_symbol(length(trail) + 1), range))
            for j in 0:(reps - 1)
                _tally_level!(pruner, span.body, level, target, (ordinals..., j),
                    body_trail, normalized)
            end
        elseif span isa R.Segment && span.body !== nothing
            n = R.bind_count(span.count, ordinals)
            n > 0 || continue
            range = R.ordinal_range(span.count, map(s -> s.range, trail))
            child_trail = (trail..., R.Descent(spans, i, _ordinal_symbol(length(trail) + 1), range))
            for k in 0:(n - 1)
                _tally_level!(pruner, span.body, level + 1, target, (ordinals..., k),
                    child_trail, normalized)
            end
        end
    end
    nothing
end

# Every distinct list (by identity) at `target`, in first-visit order, with
# the number of binders its patterns take: one per enclosing level plus one
# per enclosing Series.
function _lists_at!(found::Vector{Vector{R.Span}}, seen::IdDict{Vector{R.Span},Int},
    spans::Vector{R.Span}, level::Int, binders::Int, target::Int)::Nothing
    haskey(seen, spans) && return nothing
    seen[spans] = binders
    if level == target
        push!(found, spans)
        return nothing
    end
    for span in spans
        if span isa R.Series
            _lists_at!(found, seen, span.body, level, binders + 1, target)
        elseif span isa R.Segment && span.body !== nothing
            _lists_at!(found, seen, span.body, level + 1, binders + 1, target)
        end
    end
    nothing
end

# -- Rebuild, judge -------------------------------------------------------------

# The exact count of a Series as a `Count`, or `nothing` where it has no
# closed form here: every direct member count must be one integer under
# every ordinal assignment, so the word count is a constant, and the Series
# count is `reps * word_count` - an integer, or a `Scaled` formula when `reps`
# is one. A word count that shifts with an enclosing ordinal is the one
# inexpressible case.
function _series_count(span::R.Series)::Union{Nothing,R.Count}
    word_count = 0
    for member in span.body
        member_count =
            member isa R.Series ? _series_count(member) :
            R.constant_value(member.count)
        member_count isa Int || return nothing
        word_count += member_count
    end
    reps = R.constant_value(span.reps)
    reps === nothing ? R.Scaled(span.reps, word_count) : reps * word_count
end

_exact_count(span::R.Series) = _series_count(span)
_exact_count(span::Union{R.Segment,R.Opaque}) = span.count

# Reduce a sparse list that must lower as one native walk to the Opaque
# parts the emitter walks: one exact-count Opaque per part, adjacent
# constants merged. The walk's staged range is the sum of the parts, so
# formula-counted parts need no closed-form total. `nothing` when a part
# has no exact count.
function _native_fallback(spans::Vector{R.Span}, binders::Int)::Union{Nothing,Vector{R.Span}}
    out = R.Span[]
    pending = 0
    for span in spans
        exact = _exact_count(span)
        exact === nothing && return nothing
        pending = _demote!(out, R.Opaque(exact), pending, binders)
    end
    _flush_pending_opaque!(out, pending, binders)
    out
end

# Replace bodies by their pruned forms. A Segment whose body could not prune
# demotes; a Series carries the rebuilt members (their bodies pruned, the
# members themselves judged later with the list).
function _rebuild(pruner::Pruner, span::R.Segment)::R.Span
    span.body === nothing && return span
    body = pruner.pruned[span.body]
    body === nothing ? R.Opaque(span.count) : R.Segment(span.count, span.coord, body)
end
_rebuild(::Pruner, span::R.Opaque)::R.Span = span
function _rebuild(pruner::Pruner, span::R.Series)::R.Span
    R.Series(span.reps, R.Span[_rebuild(pruner, member) for member in span.body])
end

# -- Unfold ---------------------------------------------------------------------
#
# The root list is re-normalized from its concrete spans: every Series with
# an integer repetition count is expanded into its members at each
# repetition (the repetition ordinal is the outermost binder of a root
# Series word, so peeling it restores root-style spans), then coalescing
# and the folder run over the expanded list. A fold that still pays is
# rediscovered; one that only stood in the way of a merge is not.
function _peel(pattern::R.Lift, j::Int)
    R.refill(pattern.child, Int[R.evaluate(param, j) for param in pattern.params], R.Value)
end
_peel(pattern::R.Scalar, j::Int)::Int = R.evaluate(pattern, j)
function _peel(pattern::R.Scaled, j::Int)
    child = _peel(pattern.pattern, j)
    child isa Int ? Base.Checked.checked_mul(pattern.k, child) : R.Scaled(child, pattern.k)
end

_peel(span::R.Opaque, j::Int)::R.Span = R.Opaque(_peel(span.count, j))
function _peel(span::R.Segment, j::Int)::R.Span
    body = span.body === nothing ? nothing : R.Span[_peel(member, j) for member in span.body]
    R.Segment(_peel(span.count, j), _peel(span.coord, j), body)
end
_peel(span::R.Series, j::Int)::R.Span =
    R.Series(_peel(span.reps, j), R.Span[_peel(member, j) for member in span.body])

function _unfold(spans::Vector{R.Span})::Vector{R.Span}
    any(span -> span isa R.Series && span.reps isa Int, spans) || return spans
    out = R.Span[]
    for span in spans
        if span isa R.Series && span.reps isa Int
            for j in 0:(span.reps - 1), member in span.body
                push!(out, _peel(member, j))
            end
        else
            push!(out, span)
        end
    end
    _unfold(out)
end

# Flush a pending integer Opaque stated with `binders` binders (an `Int` at
# the root, a constant pattern below it, so it keeps the list's skeleton).
function _flush_pending_opaque!(pruned::Vector{R.Span}, pending_opaque::Int,
    binders::Int)::Int
    pending_opaque > 0 &&
        push!(pruned, R.Opaque(_constant_pattern(pending_opaque, binders)))
    0
end

# Append `span` demoted to an Opaque of its exact count, coalescing integer
# counts into `pending`. Returns the new pending count, or `nothing` when the
# span has no exact count (the owner must demote instead).
function _demote!(out::Vector{R.Span}, span::R.Span, pending::Int,
    binders::Int)::Union{Nothing,Int}
    exact = _exact_count(span)
    exact === nothing && return nothing
    value = R.constant_value(exact)
    value === nothing || return pending + value
    pending = _flush_pending_opaque!(out, pending, binders)
    push!(out, R.Opaque(exact))
    pending
end

# The code rule on one span, recursing into Series words. Returns the kept
# span and its reads, or `(nothing, 0)` to demote.
function _judge_code(pruner::Pruner, span::R.Span, reads::Int,
    code_density::Int, level::Int, binders::Int)::Tuple{Union{Nothing,R.Span},Int}
    if span isa R.Series
        tally = pruner.tallies[span.body]
        body = R.Span[]
        body_reads = 0
        pending = 0
        for (k, member) in enumerate(span.body)
            kept, member_reads =
                _judge_code(pruner, member, tally.reads[k], code_density, level, binders + 1)
            if kept === nothing || kept isa R.Opaque
                pending = _demote!(body, member, pending, binders + 1)
                pending === nothing && return (nothing, 0)
            else
                pending = _flush_pending_opaque!(body, pending, binders + 1)
                push!(body, kept)
                body_reads += member_reads
            end
        end
        _flush_pending_opaque!(body, pending, binders + 1)
        # The Stepper needs a Segment member. A word that pays keeps its
        # pruned form; so does one without an exact count, which cannot
        # demote without breaking the partition - never the original word,
        # whose demoted members would resurrect claims that failed the rule.
        any(m -> m isa R.Segment, body) || return (nothing, 0)
        rebuilt = R.Series(span.reps, body)
        if body_reads >= code_density * _abstract(rebuilt) || _series_count(span) === nothing
            return (rebuilt, body_reads)
        end
        return (nothing, 0)
    end
    span isa R.Opaque && return (span, 0)
    reads >= code_density * _abstract(span) ? (span, reads) : (nothing, 0)
end

# Judge one normalized list: the code rule per span, then the visit rule
# over the list, then integer Opaques merge. `nothing` when a span must
# demote but has no exact count.
function _judge(pruner::Pruner, spans::Vector{R.Span}, level::Int, binders::Int,
    policy)::Union{Nothing,Vector{R.Span}}
    # `_realize_sparse_fiber` emits an unanchored list as one bounded native
    # walk. Judge that emitted representation, not its unused mined claims.
    if pruner.levels[level] isa R.CompressedLevel && _has_unanchored_series(spans)
        return _native_fallback(spans, binders)
    end

    tally = pruner.tallies[spans]
    visits = tally.visits[]
    kept = Vector{Union{Nothing,R.Span}}(nothing, length(spans))
    reads = zeros(Int, length(spans))
    concrete = copy(tally.concrete)
    for (i, span) in enumerate(spans)
        kept[i], reads[i] = _judge_code(pruner, span, tally.reads[i],
            policy.code_density, level, binders)
        kept[i] === nothing && (concrete[i] = visits)
    end

    function materialize()
        out = R.Span[]
        pending = 0
        for (i, span) in enumerate(spans)
            if kept[i] === nothing || kept[i] isa R.Opaque
                pending = _demote!(out, span, pending, binders)
                pending === nothing && return nothing
            else
                pending = _flush_pending_opaque!(out, pending, binders)
                push!(out, kept[i])
            end
        end
        _flush_pending_opaque!(out, pending, binders)
        out
    end

    code_out = materialize()
    code_out === nothing && return nothing
    if pruner.levels[level] isa R.CompressedLevel && _has_unanchored_series(code_out)
        return _native_fallback(code_out, binders)
    end

    # A visit that fails demotes its weakest claim and is re-judged; when
    # every coordinate claim is gone the list is one exact-count Opaque and
    # stands on its derived origin alone (count-only realization).
    if policy.visit_density > 0
        while true
            failing = 0
            for visit in eachindex(tally.visit_reads)
                ordinals = tally.visit_ordinals[visit]
                trail = tally.visit_trails[visit]
                live_reads = zeros(Int, length(spans))
                for i in eachindex(spans)
                    kept_span = kept[i]
                    kept_span isa Union{R.Segment,R.Series} || continue
                    live_reads[i] =
                        _reads_of(pruner, kept_span, spans, i, level, ordinals, trail)
                end
                visit_reads = sum(live_reads)
                visit_spans = 0
                pending_opaque = false
                for i in eachindex(spans)
                    kept_span = kept[i]
                    if kept_span isa Union{R.Segment,R.Series}
                        visit_spans += pending_opaque
                        pending_opaque = false
                        visit_spans += _concrete(kept_span, ordinals)
                    else
                        exact = _exact_count(spans[i])
                        if exact !== nothing && R.constant_value(exact) !== nothing
                            pending_opaque = true
                        else
                            visit_spans += pending_opaque
                            pending_opaque = false
                            visit_spans += 1
                        end
                    end
                end
                visit_spans += pending_opaque
                if visit_reads < policy.visit_density * (visit_spans - 1)
                    failing = visit
                    break
                end
            end
            failing == 0 && break

            weakest = 0
            weakest_reads = 0
            for i in eachindex(spans)
                kept_span = kept[i]
                kept_span isa Union{R.Segment,R.Series} || continue
                claim_reads = _reads_of(pruner, kept_span, spans, i, level,
                    tally.visit_ordinals[failing], tally.visit_trails[failing])
                if weakest == 0 || claim_reads < weakest_reads
                    weakest = i
                    weakest_reads = claim_reads
                end
            end
            weakest == 0 && break
            kept[weakest] = nothing
            reads[weakest] = 0
            concrete[weakest] = visits

            visit_out = materialize()
            visit_out === nothing && return nothing
            if pruner.levels[level] isa R.CompressedLevel &&
                _has_unanchored_series(visit_out)
                return _native_fallback(visit_out, binders)
            end
        end
    end
    materialize()
end

# A nested Series lowers to an inner Stepper, except when its word is
# unanchored (a native stretch follows a member with no staged end): no
# inner phase boundary exists there, so the Series demotes to an exact-count
# Opaque. Without an exact count the enclosing list is `nothing`, so its
# owner demotes as any other unprunable body does.
function _demote_unanchored(spans::Vector{R.Span}, binders::Int,
    inside_series::Bool)::Union{Nothing,Vector{R.Span}}
    out = R.Span[]
    for span in spans
        if span isa R.Series
            body = _demote_unanchored(span.body, binders + 1, true)
            body === nothing && return nothing
            if inside_series && _has_unanchored_series(body)
                count = _exact_count(span)
                count === nothing && return nothing
                push!(out, R.Opaque(count isa Int ? _constant_pattern(count, binders) : count))
            else
                push!(out, R.Series(span.reps, body))
            end
        elseif span isa R.Segment && span.body !== nothing
            body = _demote_unanchored(span.body, binders + 1, false)
            push!(out, body === nothing ? R.Opaque(span.count) :
                R.Segment(span.count, span.coord, body))
        else
            push!(out, span)
        end
    end
    out
end

# The pruned description, or `nothing` when a root span must demote without
# an exact count. With both rules disabled the description is returned as
# mined.
function _prune_description(description::R.Description, structure::R.Structure,
    policy, mining::R.MiningPolicy)::Union{Nothing,R.Description}
    spans = _demote_unanchored(description.spans, 0, false)
    spans === nothing && return nothing
    description = R.Description(spans)
    (policy.code_density > 0 || policy.visit_density > 0) || return description
    isempty(description.spans) && return description
    pruner = Pruner(structure.levels, IdDict{Vector{R.Span},Union{Nothing,Vector{R.Span}}}(),
        IdDict{Vector{R.Span},ReadTally}(), IdDict{Vector{R.Span},Vector{Union{Nothing,Bool}}}())
    root = description.spans
    for level in length(structure.levels):-1:1
        lists = Vector{R.Span}[]
        binders = IdDict{Vector{R.Span},Int}()
        _lists_at!(lists, binders, root, 1, 0, level)
        normalized = IdDict{Vector{R.Span},Vector{R.Span}}()
        for spans in lists
            rebuilt = R.Span[_rebuild(pruner, span) for span in spans]
            if level == 1
                normalized[spans] = R.fold(_coalesce(_unfold(rebuilt), 0), mining)
            else
                normalized[spans] = _coalesce(rebuilt, binders[spans])
            end
        end
        empty!(pruner.tallies)
        empty!(pruner.derives)
        _tally_level!(pruner, root, 1, level, (), (), normalized)
        for spans in lists
            pruner.pruned[spans] =
                _judge(pruner, normalized[spans], level, binders[spans], policy)
        end
    end
    pruned = pruner.pruned[root]
    pruned === nothing ? nothing : R.Description(pruned)
end

"""
    structural_reads_avoided(description, structure) -> Int

The structural reads the description avoids, summed over every parent visit:
one `idx` read per entry of a leaf Segment at a compressed level and two
`ptr` reads per child whose origin derives. A tensor is in scope iff this is
at least one.
"""
function structural_reads_avoided(description::R.Description, structure::R.Structure)::Int
    pruner = Pruner(structure.levels, IdDict{Vector{R.Span},Union{Nothing,Vector{R.Span}}}(),
        IdDict{Vector{R.Span},ReadTally}(), IdDict{Vector{R.Span},Vector{Union{Nothing,Bool}}}())
    _reads_of_list(pruner, description.spans, 1, (), ())
end
# -- Mine before lowering ------------------------------------------------------

# Before lowering, concrete storage is visible but traversal choices are not.
# Project and mine the structure, prune under the attempt's policy, and keep
# a candidate on the root virtual level. A declined tensor does not modify
# the compiler context.
function Finch.mine_regular_structure!(ctx, fbr::VirtualFiber)
    root = fbr.lvl
    root isa Union{VirtualDenseLevel,VirtualSparseListLevel} || return nothing
    root.regularity === nothing || return nothing            # Mine once per tensor.
    projection = project_structure(root)
    projection === nothing && return nothing
    # Mining claims whatever the recognizers confirm under the attempt's
    # policy - pattern search depth (pmax) is the compiler's only mining
    # knob; evidence floors live in the schemas. A claimed coordinate stretch
    # lowers either as a unit-step run (Finch phases over a contiguous
    # coordinate range) or as a template walk over its coordinate formula,
    # so coordinate recognition is unconstrained. Concrete storage is stashed
    # only under a specialization attempt, so the attempt exists here.
    policy = specialization_attempt(ctx).policy
    mining = R.MiningPolicy((R.PeriodicAffineConfig(; pmax=policy.pmax),))
    description = R.mine(projection.structure, mining)
    # Keep only claims that pay under the code and visit rules; everything
    # demoted returns to the ordinary iterator. A description that avoids no
    # structural read declines this tensor - normalized-AST-identical
    # generic code, per tensor.
    description = _prune_description(description, projection.structure, policy, mining)
    reads = description === nothing ? 0 :
        structural_reads_avoided(description, projection.structure)
    if reads < 1
        root.regularity = Declined()
        return nothing
    end
    freshness = projection.reusable ? freshness_snapshot(projection) : nothing
    root.regularity = Candidate(description, freshness, reads)
end

# -- Decide at the unfurl site -------------------------------------------------

# At unfurl time, traversal mode, protocol, entry position, and aliases are
# known. Run every use-site check before allocating symbols or emitting a
# preamble. Returning `nothing` leaves Finch's ordinary unfurl unchanged.
function Finch.regularize_unfurl(ctx, fbr::VirtualSubFiber, ext, mode, proto)
    root = fbr.lvl
    root isa Union{VirtualDenseLevel,VirtualSparseListLevel} || return nothing
    candidate = root.regularity
    candidate isa Candidate || return nothing        # Only the root carries a candidate.
    mode.kind === reader || return nothing
    proto isa Union{typeof(defaultread),typeof(walk)} || return nothing
    # `follow` and `gallop` are not sequential.
    isliteral(fbr.pos) && fbr.pos.val == 1 || return nothing
    # Enter at the root fiber's first position.
    _canonical_traversal(ext, root) || return nothing
    # A windowed loop lacks this entry point.
    _outermost_traversal(ctx) || return nothing      # Preserve storage visit order.
    _alias_free(ctx, root) || return nothing         # Exclude mid-call structural mutation.
    # Independent multi-sparse operands are admitted: every realized sparse
    # fiber derives its own origin, so emitted bodies carry no cross-phase
    # state and compose under (re-)emission.
    realization = Realization(ctx, root, mode)
    body = _realize_fiber(ctx, realization, 1, candidate.description.spans, (),
        fbr.pos)
    preamble = if candidate.freshness === nothing
        Expr(:block)
    else
        freshness_preamble(root, candidate.freshness)
    end
    looplet = Thunk(;
        preamble=preamble,
        body=(ctx_2) -> body,
    )
    # The realization is counted only once the whole looplet exists.
    attempt = specialization_attempt(ctx)
    if attempt !== nothing
        attempt.realized += 1
        attempt.abstract_spans += R.abstract_spans(candidate.description)
        attempt.concrete_spans += R.concrete_spans(candidate.description)
        attempt.structural_reads += candidate.reads
    end
    looplet
end

# Phase boundaries are staged from the description's partition of the FULL
# dimension, so realization requires the loop to enter at coordinate 1; a
# window such as `17:48` uses the ordinary path. Ending early is safe because
# phases clip to the loop extent and fiber positions are derived per
# coordinate from each fiber's own origin.
function _canonical_traversal(ext, root::AbstractVirtualLevel)::Bool
    ext isa VirtualExtent || return false
    ext.start == literal(1)
end

# Realization owns the storage-ordered traversal from its root. If a loop index
# is already bound, this tensor may be re-entered from an enclosing loop, so the
# extension conservatively uses the ordinary path.
function _outermost_traversal(ctx)::Bool
    for bound_variable in keys(ctx.scope.bindings)
        bound_variable.kind === FinchNotation.index && return false
    end
    true
end

# Return the compressed structural arrays (ptr and idx) of one stashed chain.
function _stashed_arrays(root::AbstractVirtualLevel)::Vector{AbstractVector}
    arrays = AbstractVector[]
    virtual_level = root
    while virtual_level isa Union{VirtualDenseLevel,VirtualSparseListLevel}
        stash = concrete_stash(virtual_level)
        if stash isa ConcreteStash && virtual_level isa VirtualSparseListLevel
            push!(arrays, stash.lvl.ptr, stash.lvl.idx)
        end
        virtual_level = virtual_level.lvl
    end
    arrays
end

# Decline when another bound tensor's structural array may share memory with
# one of ours - the same array, or a view or wrapper over it. The compiler
# cannot rule out a write through that alias during the kernel.
function _alias_free(ctx, root::AbstractVirtualLevel)::Bool
    ours = _stashed_arrays(root)
    for (_, bound) in ctx.scope.bindings
        bound.kind === FinchNotation.virtual || continue
        bound.val isa VirtualFiber || continue
        other = bound.val.lvl
        other === root && continue
        for theirs in _stashed_arrays(other), mine in ours
            Base.mightalias(mine, theirs) && return false
        end
    end
    true
end

# -- Realization ---------------------------------------------------------------

"A staged bound clipped to the active loop extent as a looplet phase stop."
function _phase_stop(bound::Staged, ctx, ext, Ti)
    stop = ctx(getstop(ext))
    value(:(min($bound, $stop)), Ti)
end

# State shared while one description is realized.
#
# The one positional invariant: every realized sparse fiber DERIVES ITS OWN
# ORIGIN. Its Thunk preamble binds the fiber's origin - the stage-derived
# position formula where Regularity proves one, otherwise one `ptr` read at
# the fiber's own position - and every in-fiber position is pure arithmetic
# on that origin: origin + staged earlier-children offset + (coordinate -
# first coordinate). No runtime state crosses fibers or phases - the Series
# repetition ordinal included: it is recomputed from the window start
# whenever its Stepper is entered - so each emitted body is a pure function
# of (position, ordinals) and stays correct when Sequence composition against
# another multi-phase operand re-emits, truncates, or simplifies bodies away.
# Cost: at most one `ptr` read per realized fiber - never per nonzero;
# claimed entries read no `idx`.
struct Realization
    levels::Vector{AbstractVirtualLevel}           # Outermost-first level chain.
    origin_symbols::Vector{Union{Nothing,Symbol}}  # Per-fiber origin symbols.
    mode::Any
end

function Realization(ctx, root::AbstractVirtualLevel, mode)
    levels = AbstractVirtualLevel[]
    virtual_level = root
    while virtual_level isa Union{VirtualDenseLevel,VirtualSparseListLevel}
        push!(levels, virtual_level)
        virtual_level = virtual_level.lvl
    end
    origin_symbols = Union{Nothing,Symbol}[
        level isa VirtualSparseListLevel ? freshen(ctx, level.tag, :_origin) : nothing
        for level in levels
    ]
    Realization(levels, origin_symbols, mode)
end

# A described child fiber passed between Finch loop levels. Sequential
# protocols realize its description. Other protocols fall back to Finch's
# native SubFiber for the whole subtree; origin-deriving descendants need no
# bookkeeping for that. The trail records the nodes walked through and their
# staged ordinals - Descent steps through Segment bodies (one storage level
# down) and Along steps through Series bodies (same level, next repetition
# ordinal) - both the ordinals every formula consumes and the static path
# Regularity's fiber-origin derivation sums over.
struct DescribedSubFiber
    realization::Realization
    depth::Int
    nodes::Vector{R.Span}
    trail::Tuple{Vararg{R.TrailStep}}
    pos::Any
end

_level(fiber::DescribedSubFiber) = fiber.realization.levels[fiber.depth]

FinchNotation.finch_leaf(fiber::DescribedSubFiber) = virtual(fiber)
function Finch.virtual_fill_value(ctx, fiber::DescribedSubFiber)
    virtual_level_fill_value(_level(fiber))
end
function Finch.virtual_size(ctx, fiber::DescribedSubFiber)
    virtual_size(ctx, VirtualSubFiber(_level(fiber), fiber.pos))
end
Finch.instantiate(ctx, fiber::DescribedSubFiber, mode) = fiber

function Finch.unfurl(ctx, fiber::DescribedSubFiber, ext, mode,
    ::Union{typeof(defaultread),typeof(walk)})
    _realize_fiber(ctx, fiber.realization, fiber.depth, fiber.nodes, fiber.trail,
        fiber.pos)
end
function Finch.unfurl(ctx, fiber::DescribedSubFiber, ext, mode, proto)
    # Non-sequential protocol: this subtree falls back to Finch's native
    # SubFiber entirely; origin-deriving fibers need no bookkeeping for it.
    unfurl(ctx, VirtualSubFiber(_level(fiber), fiber.pos), ext, mode, proto)
end

# Descend from one described child. At a leaf, instantiate the element at its
# derived position. Otherwise pass the shared body to the next level with the
# current child's trail step appended.
function _descend(ctx, realization::Realization, depth::Int, nodes::Vector{R.Span},
    node_index::Int, trail::Tuple{Vararg{R.TrailStep}}, child_ordinal::Staged,
    position_variable::Symbol, Tpos)
    segment = nodes[node_index]::R.Segment
    if segment.body === nothing
        Simplify(
            instantiate(ctx,
                VirtualSubFiber(realization.levels[depth].lvl,
                    value(position_variable, Tpos)),
                realization.mode),
        )
    else
        child_range = R.ordinal_range(segment.count, map(s -> s.range, trail))
        instantiate(ctx,
            DescribedSubFiber(realization, depth + 1, segment.body,
                (trail..., R.Descent(nodes, node_index, child_ordinal, child_range)),
                value(position_variable, Tpos)),
            realization.mode)
    end
end

# Build the per-child looplet shared by dense and sparse levels: bind the
# derived position and descend. No epilogue: descendant fibers derive their
# own origins, so nothing must survive across children or phases.
function _claimed_lookup(realization::Realization, depth::Int, nodes::Vector{R.Span},
    node_index::Int, trail::Tuple{Vararg{R.TrailStep}}, first_coordinate::Staged,
    position_of::Function, Tpos)
    tag = realization.levels[depth].tag
    Lookup(;
        body=(ctx_2, i) -> begin
            position_variable = freshen(ctx_2, tag, :_q)
            child_ordinal = sub(STAGE, ctx_2(i), first_coordinate)
            Thunk(;
                preamble=:($position_variable = $(position_of(ctx_2, i))),
                body=(ctx_3) -> _descend(ctx_3, realization, depth, nodes,
                    node_index, trail, child_ordinal, position_variable, Tpos),
            )
        end,
    )
end

# The drift of a coordinate pattern as a pattern of its enclosing ordinals:
# the `drift` field of a period-one PeriodicAffine, lifted through every
# enclosing `Lift` by the parameters that fill that slot (the last slots, as
# `numbers` flattens `base` before `drift`). `nothing` for any other shape.
_drift_pattern(pattern::R.PeriodicAffine{1})::Union{Nothing,Int,R.Pattern} = pattern.drift
_drift_pattern(::R.Pattern)::Union{Nothing,Int,R.Pattern} = nothing
function _drift_pattern(pattern::R.Lift)::Union{Nothing,Int,R.Pattern}
    inner = _drift_pattern(pattern.child)
    inner === nothing && return nothing
    inner isa Int && return pattern.params[end]
    R.Lift(inner, pattern.params[(end - R.slot_count(inner) + 1):end])
end

# Is `coord` the unit-step run `first + k` under every binding of its
# enclosing ordinals? Such a Segment covers a contiguous coordinate range and
# lowers as a Lookup over it; every other coordinate formula (a period above
# one, or a stride other than one) lowers as a template walk.
function _unit_step(coord::R.Pattern)::Bool
    drift = _drift_pattern(coord)
    drift !== nothing && R.constant_value(drift) == 1
end

# Walk a claimed Segment whose coordinates are a staged formula rather than a
# unit-step run: the Stepper shape of `unfurl_sparse_list_walk`, with every
# `idx[q]` read replaced by the coordinate formula at child ordinal `q - q0`.
# `q_start` is the Segment's first storage position and `count` its stored
# children, both pure arithmetic on the fiber origin; seek is a linear scan
# over at most `count` positions (the period is bounded by `pmax`). The Spike
# supplies the fill before each coordinate, so no gap phase precedes the
# walk. No `ptr`, no `idx`.
function _template_walk(ctx, realization::Realization, depth::Int, nodes::Vector{R.Span},
    node_index::Int, trail::Tuple{Vararg{R.TrailStep}}, ordinals::Tuple,
    q_start::Staged, count::Staged, fill_leaf, Ti, Tpos)
    segment = nodes[node_index]::R.Segment
    tag = realization.levels[depth].tag
    q0 = freshen(ctx, tag, :_q0)
    q = freshen(ctx, tag, :_q)
    q_stop = freshen(ctx, tag, :_q_stop)
    coordinate = freshen(ctx, tag, :_i)
    child_ordinal = sub(STAGE, q, q0)
    coordinate_at_q = R.bind(segment.coord, (ordinals..., child_ordinal), STAGE)
    Thunk(;
        preamble=quote
            $q0 = $q_start
            $q = $q0
            $q_stop = $(add(STAGE, q0, count))
        end,
        body=(ctx_2) -> Stepper(;
            seek=(ctx_3, ext_3) -> quote
                while $q < $q_stop && $coordinate_at_q < $(ctx_3(getstart(ext_3)))
                    $q += $(Tpos(1))
                end
            end,
            preamble=:($coordinate = $coordinate_at_q),
            stop=(ctx_3, ext_3) -> value(coordinate, Ti),
            chunk=Spike(;
                body=fill_leaf,
                tail=_descend(ctx_2, realization, depth, nodes, node_index, trail,
                    child_ordinal, q, Tpos),
            ),
            next=(ctx_3, ext_3) -> :($q += $(Tpos(1))),
        ),
    )
end

function _realize_fiber(ctx, realization::Realization, depth::Int,
    nodes::Vector{R.Span}, trail::Tuple{Vararg{R.TrailStep}}, pos)
    if realization.levels[depth] isa VirtualDenseLevel
        _realize_dense_fiber(ctx, realization, depth, nodes, trail, pos)
    else
        _realize_sparse_fiber(ctx, realization, depth, nodes, trail, pos)
    end
end

# -- Series steppers -----------------------------------------------------------
#
# A Series lowers to ONE Phase holding ONE Stepper, independent of `reps`. The
# repetition ordinal `j` is ordinary emitted state with three disciplines that
# keep it pure across composition:
#
#   - seek RECOMPUTES j from the window start (never trusts a previous value),
#     inverting the affine repetition-end sequence `end(j) = end(0) + drift*j`
#     (affine by the series law: direct coordinates advance affinely with j;
#     `drift > 0` because the mined repetitions are ordered and disjoint),
#     clamped to the repetition range: j = max(0, min(reps - 1,
#     cld(start - end(0), drift)));
#   - the Stepper preamble stages every claimed-member boundary as a formula
#     of j;
#   - the Stepper's stop is the repetition END and `next` merely advances j,
#     so a window ending mid-repetition re-enters the same j after re-seek.
#
# The chunk is a Sequence of per-member phases. A Segment member gets a fill
# Run up to its first coordinate (the first one doubles as the gap before the
# repetition) and a claimed Lookup up to its last coordinate. An Opaque member
# CONSUMES stored children with the host's native walk - never a fill - up to
# the next Segment member's first coordinate; a trailing Opaque runs to the
# repetition end. Repetition ends follow the trailing member: a trailing
# Segment ends its own repetition; with a trailing Opaque the repetition owns
# everything before the next repetition's first Segment coordinate, and the
# LAST repetition runs to the enclosing window's end (the caller's outer stop
# clips it), because no formula bounds the final opaque tail.
#
# `member_position_of(k, earlier, first_symbol)` supplies the position law for
# claimed member `k`: sparse positions add `earlier` (repetitions passed plus
# members earlier in the repetition, opaque children included) to the fiber
# origin; dense positions ignore it (dense position is a function of the
# coordinate alone). `member_start_of(k, earlier)` supplies the first storage
# position of a template member `k` (dense coordinates are always unit-step,
# so a dense level never needs it). `opaque_member_body(k, earlier, count)`
# supplies the native looplet for opaque member `k` over its staged position
# range.
# `outer_stop` is the caller's phase stop for the
# whole Series (or `nothing` when it is open); the helper returns the stop the
# caller must use - the staged last repetition end for a trailing Segment,
# `outer_stop` itself for a trailing Opaque.
function _series_stepper!(ctx, realization::Realization, depth::Int,
    nodes::Vector{R.Span}, node_index::Int, trail::Tuple{Vararg{R.TrailStep}},
    ordinals::Tuple, preamble::Vector{Expr}, fill_leaf, Ti,
    member_position_of::Function, member_start_of::Function, Tpos,
    opaque_member_body::Function, outer_stop::Union{Nothing,Staged})
    series = nodes[node_index]::R.Series
    tag = realization.levels[depth].tag
    repetition_ordinal = freshen(ctx, tag, :_j)
    push!(preamble, :($repetition_ordinal = 0))
    along = R.Along(nodes, node_index, repetition_ordinal,
        R.ordinal_range(series.reps, map(s -> s.range, trail)))
    repetition_ordinals = (ordinals..., repetition_ordinal)
    members = series.body
    member_counts = Staged[_staged_count(member, repetition_ordinals) for member in members]
    word_count = _staged_word_count(series, ordinals)
    trailing_staged = _node_has_staged_end(members[end])
    first_claim = members[findfirst(m -> !(m isa R.Opaque), members)]

    # Claimed-member boundaries, bound once per step in the Stepper preamble:
    # every non-Opaque member's first coordinate, and the last coordinate of
    # every member with a staged end (a Segment, or a Series ending in one).
    member_first = Union{Nothing,Symbol}[
        member isa R.Opaque ? nothing : freshen(ctx, tag, Symbol(:_c, k))
        for (k, member) in enumerate(members)]
    member_last = Union{Nothing,Symbol}[
        _node_has_staged_end(member) ? freshen(ctx, tag, Symbol(:_e, k)) : nothing
        for (k, member) in enumerate(members)]
    boundaries = Expr(:block)
    for (k, member) in enumerate(members)
        member isa R.Opaque && continue
        push!(boundaries.args,
            :($(member_first[k]) = $(_node_first_coordinate(member, repetition_ordinals))))
        member_last[k] === nothing && continue
        push!(boundaries.args,
            :($(member_last[k]) = $(_node_last_coordinate(member, repetition_ordinals))))
    end
    # A trailing-Opaque repetition's end: the next repetition's first claimed
    # coordinate, minus one. Extrapolating one repetition past the end is
    # harmless - the last repetition's stop is the window end instead.
    repetition_end_symbol = if trailing_staged
        member_last[end]::Symbol
    else
        end_symbol = freshen(ctx, tag, :_end)
        push!(
            boundaries.args,
            :(
                $end_symbol =
                    $(sub(STAGE,
                        _node_first_coordinate(first_claim,
                            (ordinals..., :($repetition_ordinal + 1))), 1))
            ),
        )
        end_symbol
    end

    # The affine-in-j repetition-end sequence, for seek and for the series stop.
    function repetition_end(j)
        if trailing_staged
            _node_last_coordinate(members[end], (ordinals..., j))
        else
            sub(STAGE, _node_first_coordinate(first_claim, (ordinals..., j + 1)), 1)
        end
    end
    first_end = repetition_end(0)
    end_drift = sub(STAGE, repetition_end(1), first_end)
    reps = _staged_count(series.reps, ordinals)
    last_ordinal = sub(STAGE, reps, 1)
    series_stop =
        if trailing_staged
            add(STAGE, first_end, mul(STAGE, end_drift, last_ordinal))
        else
            outer_stop
        end

    chunk_phases = Phase[]
    for (k, member) in enumerate(members)
        earlier = add(STAGE, mul(STAGE, repetition_ordinal, word_count),
            foldl((a, b) -> add(STAGE, a, b), member_counts[1:(k - 1)]; init=0))
        # A native stretch (an Opaque, or a nested Series ending in one) stops
        # where the next claimed member begins; a trailing one runs to the
        # repetition end.
        following = findnext(m -> !(m isa R.Opaque), members, k + 1)
        native_stop = following === nothing ? nothing :
            sub(STAGE, member_first[following]::Symbol, 1)
        if member isa R.Segment
            first_symbol = member_first[k]::Symbol
            last_symbol = member_last[k]::Symbol
            if _unit_step(member.coord)
                position_of = member_position_of(k, earlier, first_symbol)
                push!(
                    chunk_phases,
                    Phase(;
                        stop=(ctx_2, ext_2) -> _phase_stop(
                            sub(STAGE, first_symbol, 1), ctx_2, ext_2, Ti),
                        body=(ctx_2, ext_2) -> Run(fill_leaf)),
                )
                push!(
                    chunk_phases,
                    Phase(;
                        stop=(ctx_2, ext_2) ->
                            _phase_stop(last_symbol, ctx_2, ext_2, Ti),
                        body=(ctx_2, ext_2) -> _claimed_lookup(realization, depth,
                            members, k, (trail..., along), first_symbol,
                            position_of, Tpos)),
                )
            else
                # A template member walks from its first storage position;
                # its Spike supplies the fill before each coordinate.
                q_start = member_start_of(k, earlier)
                push!(
                    chunk_phases,
                    Phase(;
                        stop=(ctx_2, ext_2) ->
                            _phase_stop(last_symbol, ctx_2, ext_2, Ti),
                        body=(ctx_2, ext_2) -> _template_walk(ctx_2, realization,
                            depth, members, k, (trail..., along), repetition_ordinals,
                            q_start, member_counts[k], fill_leaf, Ti, Tpos)),
                )
            end
        elseif member isa R.Series
            # A nested Series is one inner Stepper whose positions and
            # native walks are offset by everything earlier in this
            # repetition; its own seek recomputes the inner ordinal.
            inner_position_of = (k2, inner_earlier, first_symbol) ->
                member_position_of(k, add(STAGE, earlier, inner_earlier), first_symbol)
            inner_start_of = (k2, inner_earlier) ->
                member_start_of(k, add(STAGE, earlier, inner_earlier))
            inner_opaque_body = (k2, inner_earlier, count) ->
                opaque_member_body(k, add(STAGE, earlier, inner_earlier), count)
            inner_stop, inner_looplet = _series_stepper!(ctx, realization, depth,
                members, k, (trail..., along), repetition_ordinals, preamble,
                fill_leaf, Ti, inner_position_of, inner_start_of, Tpos,
                inner_opaque_body, native_stop)
            push!(
                chunk_phases,
                Phase(;
                    stop=(ctx_2, ext_2) -> inner_stop === nothing ? nothing :
                        _phase_stop(inner_stop, ctx_2, ext_2, Ti),
                    body=(ctx_2, ext_2) -> inner_looplet),
            )
        else
            push!(
                chunk_phases,
                Phase(;
                    stop=(ctx_2, ext_2) -> native_stop === nothing ? nothing :
                        _phase_stop(native_stop, ctx_2, ext_2, Ti),
                    body=opaque_member_body(k, earlier, member_counts[k])),
            )
        end
    end

    # Seek inverts the affine repetition-end sequence through the staged Ops
    # (`cld(start - end(0), drift)` written as `fld(start - end(0) + drift - 1,
    # drift)`); the clamp to the repetition range is order arithmetic outside
    # the Ops contract and cannot overflow.
    function seek_ordinal(start)
        start isa Staged || (start = Int(start))   # a literal window start of type Ti
        fld(STAGE, sub(STAGE, add(STAGE, start, end_drift), add(STAGE, first_end, 1)),
            end_drift)
    end
    looplet = Stepper(;
        seek=(ctx_2, ext_2) -> quote
            $repetition_ordinal = max(
                0, min($last_ordinal, $(seek_ordinal(ctx_2(getstart(ext_2)))))
            )
        end,
        preamble=boundaries,
        stop=(ctx_2, ext_2) -> begin
            stop = if trailing_staged
                repetition_end_symbol
            else
                :(
                    if $repetition_ordinal >= $last_ordinal
                        $(ctx_2(getstop(ext_2)))
                    else
                        $repetition_end_symbol
                    end
                )
            end
            _phase_stop(stop, ctx_2, ext_2, Ti)
        end,
        chunk=Sequence(chunk_phases),
        next=(ctx_2, ext_2) -> :($repetition_ordinal += 1),
    )
    (series_stop, looplet)
end

# A node's first claimed coordinate as a staged formula, or `nothing` for an
# Opaque. A Series starts where repetition 0's first claimed member starts -
# a leading Opaque member is walked natively inside the Stepper chunk, but a
# preceding Opaque phase must still stop where the Series' claims begin.
function _node_first_coordinate(node::R.Segment, ordinals::Tuple)
    _staged_first_coordinate(node.coord, ordinals)
end
function _node_first_coordinate(node::R.Series, ordinals::Tuple)
    first_claim = findfirst(m -> !(m isa R.Opaque), node.body)
    _node_first_coordinate(node.body[first_claim], (ordinals..., 0))
end
_node_first_coordinate(::R.Opaque, ordinals::Tuple) = nothing

# A node's last coordinate as a staged formula, defined where the node has a
# staged end: a Segment, or a Series whose last member has one (its last
# repetition's end). A unit-step Segment ends `count - 1` past its first
# coordinate; a template Segment ends where its formula puts its last child.
function _node_last_coordinate(node::R.Segment, ordinals::Tuple)
    count = _staged_count(node.count, ordinals)
    if _unit_step(node.coord)
        add(STAGE, _staged_first_coordinate(node.coord, ordinals), sub(STAGE, count, 1))
    else
        R.bind(node.coord, (ordinals..., sub(STAGE, count, 1)), STAGE)
    end
end
function _node_last_coordinate(node::R.Series, ordinals::Tuple)
    last_ordinal = sub(STAGE, _staged_count(node.reps, ordinals), 1)
    _node_last_coordinate(last(node.body), (ordinals..., last_ordinal))
end

# A node that begins with a native walk has no staged first coordinate. It
# can follow a node with a staged end (the next phase starts at end + 1), but
# it cannot follow an Opaque or a trailing-Opaque Series: no static phase
# boundary separates their native walks.
_node_leads_native(::R.Opaque)::Bool = true
_node_leads_native(::R.Segment)::Bool = false
_node_leads_native(node::R.Series)::Bool = _node_leads_native(first(node.body))
_node_has_staged_end(::R.Segment)::Bool = true
_node_has_staged_end(node::R.Series)::Bool = _node_has_staged_end(last(node.body))
_node_has_staged_end(::R.Opaque)::Bool = false

function _has_unanchored_series(nodes::Vector{R.Span})::Bool
    any(2:length(nodes)) do i
        _node_leads_native(nodes[i]) && !_node_has_staged_end(nodes[i - 1])
    end
end

# -- Dense fibers --------------------------------------------------------------
#
# Dense child position is `(parent_position - 1) * extent + coordinate`, so no
# structural array is needed and no origin symbol either: the position law is
# a function of the coordinate alone, for Series members included. An Opaque
# stretch uses the native dense body; realized descendants elsewhere are
# unaffected, since each derives its own origin.
function _realize_dense_fiber(ctx, realization::Realization, depth::Int,
    nodes::Vector{R.Span}, trail::Tuple{Vararg{R.TrailStep}}, pos)
    level = realization.levels[depth]::VirtualDenseLevel
    ordinals = R.trail_ordinals(trail)
    Ti = level.Ti
    tag = level.tag
    position_of =
        (ctx_2, i) ->
            :(($(ctx_2(pos)) - $(Ti(1))) * $(ctx_2(level.shape)) + $(ctx_2(i)))
    native_lookup =
        (ctx_2, ext_2) -> Lookup(;
            body=(ctx_3, i) -> begin
                position_variable = freshen(ctx_3, tag, :_q)
                Thunk(;
                    preamble=:($position_variable = $(position_of(ctx_3, i))),
                    body=(ctx_4) -> instantiate(ctx_4,
                        VirtualSubFiber(level.lvl,
                            value(position_variable, Ti)),
                        realization.mode),
                )
            end,
        )
    preamble = Expr[]
    phases = Phase[]
    next_first::Staged = 1                   # First coordinate of the next node.
    for (node_index, node) in enumerate(nodes)
        node_count = _staged_count(node, ordinals)
        last_coordinate = sub(STAGE, add(STAGE, next_first, node_count), 1)
        if node isa R.Opaque
            # An Opaque stretch uses the native dense body over its count.
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) ->
                        _phase_stop(last_coordinate, ctx_2, ext_2, Ti),
                    body=native_lookup,
                ),
            )
        elseif node isa R.Series
            series_stop, looplet = _series_stepper!(ctx, realization, depth, nodes,
                node_index, trail, ordinals, preamble,
                FillLeaf(virtual_level_fill_value(level)), Ti,
                (k, earlier, first_symbol) -> position_of,
                (k, earlier) -> error("Regularity: a dense coordinate is unit-step"),
                Ti, (k, earlier, count) -> native_lookup, last_coordinate)
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) ->
                        _phase_stop(series_stop, ctx_2, ext_2, Ti),
                    body=(ctx_2, ext_2) -> looplet,
                ),
            )
        else
            segment = node::R.Segment
            first_coordinate = _staged_first_coordinate(segment.coord, ordinals)
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) ->
                        _phase_stop(last_coordinate, ctx_2, ext_2, Ti),
                    body=(ctx_2, ext_2) -> _claimed_lookup(realization, depth, nodes,
                        node_index, trail, first_coordinate, position_of, Ti),
                ),
            )
        end
        next_first = add(STAGE, next_first, node_count)
    end
    push!(
        phases,
        Phase(;
            body=(ctx_2, ext_2) -> Run(FillLeaf(virtual_level_fill_value(level)))),
    )
    body = Sequence(phases)
    if isempty(preamble)
        body
    else
        Thunk(; preamble=Expr(:block, preamble...), body=(ctx_2) -> body)
    end
end

# -- Sparse fibers -------------------------------------------------------------
#
# A described sparse position is
# `origin + earlier_children + (coordinate - first_coordinate)`: the origin is
# this fiber's first stored position - Regularity's stage-derived formula of
# the trail's ordinals where the position law is expressible, otherwise the
# fiber's own `ptr` read - and staged node COUNTS provide `earlier_children`
# (a Series contributes `reps * word_count`, and positions inside one add the
# repetitions already passed). Opaque nodes use Finch's native sparse walk.
function _realize_sparse_fiber(ctx, realization::Realization, depth::Int,
    nodes::Vector{R.Span}, trail::Tuple{Vararg{R.TrailStep}}, pos)
    level = realization.levels[depth]::VirtualSparseListLevel
    ordinals = R.trail_ordinals(trail)
    Ti = level.Ti
    Tpos = postype(level)
    fill_leaf = FillLeaf(virtual_level_fill_value(level))
    origin = realization.origin_symbols[depth]::Symbol

    # Bind the fiber's origin once in the preamble - a pure formula when
    # derivable, else one ptr read. Every position in this fiber is then pure
    # arithmetic on the origin, so the emitted body is safe under re-emission,
    # truncation, and simplification.
    origin_formula = R.child_origin(trail, STAGE)
    origin_binding = if origin_formula === nothing
        :($origin = $(level.ptr)[$(ctx(pos))])
    else
        :($origin = $origin_formula)
    end

    isempty(nodes) &&
        return Thunk(; preamble=origin_binding, body=(ctx_2) -> Run(fill_leaf))

    # Without a staged boundary, two adjacent native walks cannot be split
    # soundly. Lower this fiber as one bounded ordinary walk; its derived
    # origin still removes both ptr reads, while every coordinate is visited.
    if _has_unanchored_series(nodes)
        total_count::Staged = 0
        for node in nodes
            total_count = add(STAGE, total_count, _staged_count(node, ordinals))
        end
        fiber_bounds = (origin, add(STAGE, origin, total_count))
        body = Sequence([
            Phase(; body=(ctx_2, ext_2) ->
                unfurl_sparse_list_walk(ctx_2, VirtualSubFiber(level, pos),
                    ext_2, realization.mode, defaultread; bounds=fiber_bounds)),
            Phase(; body=(ctx_2, ext_2) -> Run(fill_leaf)),
        ])
        return Thunk(; preamble=origin_binding, body=(ctx_2) -> body)
    end

    first_coordinates = Union{Staged,Nothing}[
        _node_first_coordinate(node, ordinals) for node in nodes
    ]

    preamble = Expr[]
    phases = Phase[]
    earlier::Staged = 0                      # Stored children before this node.
    for (node_index, node) in enumerate(nodes)
        node_count = _staged_count(node, ordinals)
        if node isa R.Opaque
            # An Opaque stretch uses the native sparse walk up to the next
            # node with a derivable first coordinate, bounded by the
            # stretch's staged position range `[origin + earlier, origin +
            # earlier + count)`, so it reads no `ptr`.
            following = findnext(!isnothing, first_coordinates, node_index + 1)
            generic_stop = if following === nothing
                nothing
            else
                sub(STAGE, first_coordinates[following], 1)
            end
            q_start = add(STAGE, origin, earlier)
            node_bounds = (q_start, add(STAGE, q_start, node_count))
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) -> generic_stop === nothing ? nothing :
                        _phase_stop(generic_stop, ctx_2, ext_2, Ti),
                    body=(ctx_2, ext_2) ->
                        unfurl_sparse_list_walk(ctx_2, VirtualSubFiber(level, pos),
                            ext_2, realization.mode, defaultread; bounds=node_bounds),
                ),
            )
        elseif node isa R.Series
            base = add(STAGE, origin, earlier)
            member_position_of =
                (k, member_earlier, first_symbol) ->
                    (ctx_2, i) -> add(STAGE, add(STAGE, base, member_earlier),
                        sub(STAGE, ctx_2(i), first_symbol))
            # An Opaque member walks its own staged position range within
            # the repetition; no `ptr` read.
            opaque_member_body =
                (k, member_earlier, member_count) -> begin
                    q_start = add(STAGE, base, member_earlier)
                    bounds = (q_start, add(STAGE, q_start, member_count))
                    (ctx_2, ext_2) ->
                        unfurl_sparse_list_walk(ctx_2, VirtualSubFiber(level, pos),
                            ext_2, realization.mode, defaultread; bounds=bounds)
                end
            # A trailing-Opaque series has no staged end; it stops where the
            # next node with a derivable first coordinate begins.
            following = findnext(!isnothing, first_coordinates, node_index + 1)
            outer_stop = if following === nothing
                nothing
            else
                sub(STAGE, first_coordinates[following], 1)
            end
            member_start_of = (k, member_earlier) -> add(STAGE, base, member_earlier)
            series_stop, looplet = _series_stepper!(ctx, realization, depth, nodes,
                node_index, trail, ordinals, preamble, fill_leaf, Ti,
                member_position_of, member_start_of, Tpos, opaque_member_body,
                outer_stop)
            # No leading gap phase: the chunk's own fill phase (or a leading
            # opaque member's native walk) runs from the window start to the
            # repetition start.
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) -> series_stop === nothing ? nothing :
                        _phase_stop(series_stop, ctx_2, ext_2, Ti),
                    body=(ctx_2, ext_2) -> looplet,
                ),
            )
        elseif _unit_step((node::R.Segment).coord)
            first_coordinate = first_coordinates[node_index]
            last_coordinate = sub(STAGE, add(STAGE, first_coordinate, node_count), 1)
            gap_stop = sub(STAGE, first_coordinate, 1)
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) ->
                        _phase_stop(gap_stop, ctx_2, ext_2, Ti),
                    body=(ctx_2, ext_2) -> Run(fill_leaf)),
            )
            base = add(STAGE, origin, earlier)
            position_of =
                (ctx_2, i) ->
                    add(STAGE, base, sub(STAGE, ctx_2(i), first_coordinate))
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) ->
                        _phase_stop(last_coordinate, ctx_2, ext_2, Ti),
                    body=(ctx_2, ext_2) -> _claimed_lookup(realization, depth, nodes,
                        node_index, trail, first_coordinate, position_of, Tpos),
                ),
            )
        else
            # A template Segment walks from `origin + earlier` over its
            # coordinate formula; its Spike supplies the fill before each
            # coordinate, so no gap phase precedes it.
            last_coordinate = _node_last_coordinate(node, ordinals)
            q_start = add(STAGE, origin, earlier)
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) ->
                        _phase_stop(last_coordinate, ctx_2, ext_2, Ti),
                    body=(ctx_2, ext_2) -> _template_walk(ctx_2, realization, depth,
                        nodes, node_index, trail, ordinals, q_start, node_count,
                        fill_leaf, Ti, Tpos),
                ),
            )
        end
        earlier = add(STAGE, earlier, node_count)
    end
    push!(phases, Phase(; body=(ctx_2, ext_2) -> Run(fill_leaf)))

    Thunk(;
        preamble=Expr(:block, origin_binding, preamble...),
        body=(ctx_2) -> Sequence(phases),
    )
end

end # module RegularityExt

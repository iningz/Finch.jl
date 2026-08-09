# This extension connects Regularity descriptions to Finch looplets:
#
#     Finch storage -> Structure -> Description -> Finch looplets
#          |                                          |
#          +-------------- freshness guard <----------+
#
# Regularity describes counts and coordinates. Finch still owns loops, phases,
# storage positions, and ordinary sparse traversal. A formula-backed stretch
# becomes Finch `Sequence`, `Phase`, `Lookup`, and `Thunk` nodes. An Opaque
# stretch delegates to Finch's native traversal.
#
# At each sparse level, a position cursor derives formula-backed positions:
#
#     cursor starts at 1
#     position = cursor + earlier children + (coordinate - first coordinate)
#     cursor += completed fiber width
#
# This path reads neither `ptr` nor `idx`. If an opaque parent hides descendant
# widths, the next formula-backed descendant reseeds its cursor with the same
# pointer read the ordinary traversal would use.

module RegularityExt

using Finch
using Finch:
    VirtualFiber, VirtualSubFiber, AbstractVirtualLevel,
    VirtualDenseLevel, VirtualSparseListLevel, VirtualElementLevel,
    DenseLevel, SparseListLevel, ElementLevel,
    ConcreteStash, concrete_stash, structural_token,
    Phase, Sequence, Run, Lookup, Thunk, FillLeaf, Simplify,
    instantiate, unfurl, unfurl_sparse_list_walk,
    literal, isliteral, value, freshen, virtual,
    virtual_fill_value, virtual_level_fill_value, virtual_size, postype,
    defaultread, walk, follow, gallop,
    VirtualExtent,
    FinchNotation
using Finch.FinchNotation: reader, updater, variable
import Finch: mine_regular_structure!, regularize_unfurl

using Regularity
const R = Regularity
import Regularity: add, sub, mul, fld, mod, select

# -- Host policy ---------------------------------------------------------------
#
# Capability checks decide whether Finch can realize a description exactly.
# Policy checks decide whether that exact realization is likely worth emitting.
# The constants below affect recognition and policy, never formula meaning.

const SPECIALIZE_PMAX = Ref(8)           # Largest PeriodicAffine period to search.
const SPECIALIZE_MIN_RUN = Ref(16)       # Shorter outer claims become opaque.
const SPECIALIZE_LEAF_MIN_RUN = Ref(4)
const SPECIALIZE_MAX_NODES = Ref(128)    # Maximum nodes in one level description.
const SPECIALIZE_MAX_UNITS = Ref(512)    # Maximum units in the whole description.
const SPECIALIZE_MIN_COVERAGE = Ref(0.5) # Minimum fraction covered by Segments.

"Set and return the PeriodicAffine period limit used by Finch specialization."
function bind_specialize_pmax!(pmax::Integer)::Int
    value = Int(pmax)
    R.PeriodicAffineConfig(value)         # Validate the same bound used by mining.
    SPECIALIZE_PMAX[] = value
    value
end

# -- Staged arithmetic ---------------------------------------------------------
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

# Resolve a description field with staged enclosing ordinals.

_staged_count(count::Int, binders::Tuple)::Staged = count
_staged_count(count::R.Pattern, binders::Tuple)::Staged =
    R.resolve(count, binders, STAGE)

"The stretch's first coordinate: the coordinate pattern at child ordinal 0."
_staged_first_coordinate(coordinate::R.Pattern, binders::Tuple)::Staged =
    R.resolve(coordinate, (binders..., 0), STAGE)

# -- Extract concrete structure ------------------------------------------------

"A mined description and the data needed to decide whether it can be reused."
struct Candidate
    description::R.Description
    snapshot::Any                # `GuardSnapshot` for reuse; otherwise `nothing`.
    reusable::Bool
end

"Marks a tensor that was mined but not selected, preventing duplicate work."
struct Declined end

struct ExtractedChain
    virtuals::Vector{AbstractVirtualLevel}   # Outermost first; index levels only.
    structure::R.Structure
    reusable::Bool
end

# Project Finch's virtual and concrete level chains into a Regularity Structure.
# Unsupported levels, missing concrete data, or invalid structural arrays return
# `nothing`, leaving the tensor on Finch's ordinary path.
function extract_structure(root::AbstractVirtualLevel)::Union{ExtractedChain,Nothing}
    virtuals = AbstractVirtualLevel[]
    levels = R.Level[]
    reusable = nothing
    virtual_level = root
    while true
        if virtual_level isa VirtualElementLevel
            break
        elseif virtual_level isa VirtualDenseLevel
            stash = concrete_stash(virtual_level)
            stash isa ConcreteStash || return nothing
            push!(virtuals, virtual_level)
            push!(levels, R.DenseLevel(Int(stash.lvl.shape)))
            reusable === nothing || reusable == stash.reusable || return nothing
            reusable = stash.reusable
            virtual_level = virtual_level.lvl
        elseif virtual_level isa VirtualSparseListLevel
            stash = concrete_stash(virtual_level)
            stash isa ConcreteStash || return nothing
            push!(virtuals, virtual_level)
            push!(levels, R.CompressedLevel(stash.lvl.ptr, stash.lvl.idx))
            reusable === nothing || reusable == stash.reusable || return nothing
            reusable = stash.reusable
            virtual_level = virtual_level.lvl
        else
            return nothing
        end
    end
    isempty(levels) && return nothing
    structure = try
        R.Structure(levels...)
    catch failure
        failure isa ArgumentError && return nothing      # Invalid storage stays ordinary.
        rethrow()
    end
    ExtractedChain(virtuals, structure, something(reusable))
end

# -- Capability checks ---------------------------------------------------------

# Trial-stage every description field with symbolic ordinals. This checks both
# pattern arity and whether Finch implements every arithmetic operation used by
# the pattern family. A failure declines the whole description.
function _stages_cleanly(nodes::Vector{R.Node}, depth::Int)::Bool
    probes = ntuple(i -> Symbol(:__probe_, i), depth)
    for node in nodes
        try
            node.count isa Int || R.resolve(node.count, probes, STAGE)
            if node isa R.Segment
                R.resolve(node.coord, (probes..., 0), STAGE)
                node.body === nothing || _stages_cleanly(node.body, depth + 1) ||
                    return false
            end
        catch failure
            failure isa Union{MethodError,ArgumentError,OverflowError,BoundsError} &&
                return false
            rethrow()
        end
    end
    true
end

# Finch phases cover contiguous coordinate ranges, so a realized segment must
# advance coordinates by exactly one. The innermost scalar must be a period-one
# formula whose drift folds to literal `1` for symbolic enclosing ordinals.
function _staged_innermost_drift(pattern::R.Pattern, probes::Tuple)::Any
    if pattern isa R.Nest
        contents = Staged[R.evaluate(param, probes[1], STAGE) for param in pattern.params]
        _staged_innermost_drift(
            R.refill(pattern.child, contents, STAGE), Base.tail(probes))
    else
        pattern.drift
    end
end

function _unit_step(coordinate::R.Pattern, depth::Int)::Bool
    innermost = coordinate
    while innermost isa R.Nest
        innermost = innermost.child
    end
    innermost isa R.PeriodicAffine{1} || return false
    probes = ntuple(i -> Symbol(:__probe_, i), depth - 1)
    _staged_innermost_drift(coordinate, probes) === 1
end

function _coordinates_unit_step(nodes::Vector{R.Node}, depth::Int)::Bool
    for node in nodes
        node isa R.Segment || continue
        _unit_step(node.coord, depth + 1) || return false
        node.body === nothing || _coordinates_unit_step(node.body, depth + 1) ||
            return false
    end
    true
end

can_realize_description(description::R.Description)::Bool =
    _stages_cleanly(description.nodes, 0) && _coordinates_unit_step(description.nodes, 0)

# -- Profitability policy ------------------------------------------------------

function worth_realizing_description(
    description::R.Description, structure::R.Structure
)::Bool
    any(level isa R.CompressedLevel for level in structure.levels) || return false
    R.description_units(description) <= SPECIALIZE_MAX_UNITS[] || return false
    tallies = R.coverage(description, structure)
    total = sum(BigInt, tallies.total)
    claimed = sum(BigInt, tallies.claimed)
    total > 0 && claimed / total >= SPECIALIZE_MIN_COVERAGE[]
end

# -- Freshness snapshot --------------------------------------------------------

# Values checked before a reusable kernel runs. Each compressed level records
# its structural token and generation, plus array identity and length. Each
# dense level records its extent. Finch lifecycle updates change the generation;
# identity and length also catch replacement arrays.
struct GuardSnapshot
    token_ids::Vector{UInt}
    generations::Vector{UInt64}
    ptr_ids::Vector{UInt}
    ptr_lengths::Vector{Int}
    idx_ids::Vector{UInt}
    idx_lengths::Vector{Int}
    dense_extents::Vector{Int}           # One per dense level, in chain order.
end

function _take_snapshot(chain::ExtractedChain)::GuardSnapshot
    snapshot = GuardSnapshot(UInt[], UInt64[], UInt[], Int[], UInt[], Int[], Int[])
    for level in chain.structure.levels
        if level isa R.CompressedLevel
            token = structural_token(level.ptr)
            push!(snapshot.token_ids, objectid(token))
            push!(snapshot.generations, token.generation)
            push!(snapshot.ptr_ids, objectid(level.ptr))
            push!(snapshot.ptr_lengths, length(level.ptr))
            push!(snapshot.idx_ids, objectid(level.idx))
            push!(snapshot.idx_lengths, length(level.idx))
        else
            push!(snapshot.dense_extents, (level::R.DenseLevel).extent)
        end
    end
    snapshot
end

function _runtime_shape(virtual_level::VirtualDenseLevel)
    virtual_level.shape.kind === FinchNotation.value ? virtual_level.shape.val : nothing
end

function _guard_preamble(root::AbstractVirtualLevel, snapshot::GuardSnapshot)::Expr
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
                    Finch.structure_current(
                        $ptr,
                        $(snapshot.token_ids[compressed_index]),
                        $(snapshot.generations[compressed_index]),
                    ) || error(
                        "Regularity: stale specialization; the structural token does " *
                        "not match the mined snapshot (structure was rewritten since mining)",
                    )
                end,
            )
            push!(
                checks,
                quote
                    (
                        objectid($ptr) === $(snapshot.ptr_ids[compressed_index]) &&
                        length($ptr) == $(snapshot.ptr_lengths[compressed_index]) &&
                        objectid($idx) === $(snapshot.idx_ids[compressed_index]) &&
                        length($idx) == $(snapshot.idx_lengths[compressed_index])
                    ) || error(
                        "Regularity: stale specialization; structural arrays were " *
                        "replaced (identity/length diagnostic)")
                end,
            )
        else
            dense_index += 1
            shape_variable = _runtime_shape(virtual_level)
            shape_variable === nothing || push!(
                checks,
                quote
                    $shape_variable == $(snapshot.dense_extents[dense_index]) || error(
                        "Regularity: stale specialization; dense extent changed " *
                        "(extent diagnostic)")
                end,
            )
        end
        virtual_level = virtual_level.lvl
    end
    Expr(:block, checks...)
end

# -- Mine before lowering ------------------------------------------------------

# Before lowering, concrete storage is visible but traversal choices are not.
# Extract and mine the structure, apply description-local checks, and keep a
# candidate on the root virtual level. A declined tensor does not modify the
# compiler context.
function Finch.mine_regular_structure!(ctx, fbr::VirtualFiber)
    root = fbr.lvl
    root isa Union{VirtualDenseLevel,VirtualSparseListLevel} || return nothing
    root.regularity === nothing || return nothing            # Mine once per tensor.
    chain = extract_structure(root)
    chain === nothing && return nothing
    description = R.mine(chain.structure;
        families=(R.PeriodicAffineConfig(; pmax=SPECIALIZE_PMAX[]),),
        min_run=SPECIALIZE_MIN_RUN[],
        leaf_min_run=SPECIALIZE_LEAF_MIN_RUN[],
        max_nodes=SPECIALIZE_MAX_NODES[])
    if !(
        can_realize_description(description) &&
        worth_realizing_description(description, chain.structure)
    )
        root.regularity = Declined()
        return nothing
    end
    snapshot = chain.reusable ? _take_snapshot(chain) : nothing
    root.regularity = Candidate(description, snapshot, chain.reusable)
    nothing
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
    # Capability checks at this use site.
    mode.kind === reader || return nothing
    proto isa Union{typeof(defaultread),typeof(walk)} || return nothing
    # `follow` and `gallop` are not sequential.
    isliteral(fbr.pos) && fbr.pos.val == 1 || return nothing
    # Enter at the root fiber's first position.
    _canonical_traversal(ext, root) || return nothing
    # A windowed loop lacks this anchor.
    _outermost_traversal(ctx) || return nothing      # Preserve storage visit order.
    _alias_free(ctx, root) || return nothing         # Exclude mid-call structural mutation.
    # Use-site policy.
    _sole_candidate(ctx, root) || return nothing     # Avoid multiplying phase partitions.
    # All checks passed; emission may now allocate compiler state.
    emission = Emission(ctx, root, mode)
    body = _realize_fiber(ctx, emission, 1, candidate.description.nodes, (), fbr.pos)
    preamble = if candidate.reusable
        Expr(:block, _guard_preamble(root, candidate.snapshot),
            _cursor_seed_preamble(emission))
    else
        _cursor_seed_preamble(emission)
    end
    Thunk(;
        preamble=preamble,
        body=(ctx_2) -> body,
    )
end

# Position cursors are anchored only when a loop enters a dimension at its
# first coordinate. A window such as `17:48` therefore uses the ordinary path.
# Ending early is safe because phases clip to the loop extent and the cursors
# are local to that traversal.
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

# Return object identities for the compressed arrays in one stashed chain.
function _stashed_arrays(root::AbstractVirtualLevel)::Vector{UInt}
    ids = UInt[]
    virtual_level = root
    while virtual_level isa Union{VirtualDenseLevel,VirtualSparseListLevel}
        stash = concrete_stash(virtual_level)
        if stash isa ConcreteStash && virtual_level isa VirtualSparseListLevel
            push!(ids, objectid(stash.lvl.ptr), objectid(stash.lvl.idx))
        end
        virtual_level = virtual_level.lvl
    end
    ids
end

# Decline when another bound tensor shares a structural array. The compiler
# cannot rule out a write through that alias during the kernel.
function _alias_free(ctx, root::AbstractVirtualLevel)::Bool
    ours = _stashed_arrays(root)
    isempty(ours) && return false
    for (_, bound) in ctx.scope.bindings
        bound.kind === FinchNotation.virtual || continue
        bound.val isa VirtualFiber || continue
        other = bound.val.lvl
        other === root && continue
        isempty(intersect(ours, _stashed_arrays(other))) || return false
    end
    true
end

function _sole_candidate(ctx, root::AbstractVirtualLevel)::Bool
    for (_, bound) in ctx.scope.bindings
        bound.kind === FinchNotation.virtual || continue
        bound.val isa VirtualFiber || continue
        other = bound.val.lvl
        other === root && continue
        other isa Union{VirtualDenseLevel,VirtualSparseListLevel} || continue
        other.regularity isa Candidate && return false
    end
    true
end

# -- Realize descriptions as Finch looplets -----------------------------------

"A staged bound as a looplet phase stop: literals stay literals."
_stop_node(bound::Int, Ti) = literal(Ti(bound))
_stop_node(bound::Staged, Ti) = value(bound, Ti)

# State shared while one description is emitted. Sparse levels receive runtime
# position cursors. `cursor_valid` tracks whether earlier realized fibers have
# supplied every width needed to derive the next start. An opaque ancestor
# invalidates deeper cursors; the next formula-backed fiber reseeds its cursor
# from `ptr`.
mutable struct Emission
    levels::Vector{AbstractVirtualLevel}           # Outermost-first level chain.
    cursor_symbols::Vector{Union{Nothing,Symbol}}  # Sparse levels receive cursors.
    cursor_valid::Vector{Bool}
    mode::Any
end

function Emission(ctx, root::AbstractVirtualLevel, mode)
    levels = AbstractVirtualLevel[]
    virtual_level = root
    while virtual_level isa Union{VirtualDenseLevel,VirtualSparseListLevel}
        push!(levels, virtual_level)
        virtual_level = virtual_level.lvl
    end
    cursor_symbols = Union{Nothing,Symbol}[
        level isa VirtualSparseListLevel ? freshen(ctx, level.tag, :_cursor) : nothing
        for level in levels
    ]
    Emission(levels, cursor_symbols, fill(true, length(levels)), mode)
end

# Structure validation guarantees that each level starts at position 1.
function _cursor_seed_preamble(emission::Emission)::Expr
    seeds = Expr[]
    for (depth, cursor) in pairs(emission.cursor_symbols)
        cursor === nothing && continue
        Tp = postype(emission.levels[depth])
        push!(seeds, :($cursor = $(Tp(1))))
    end
    Expr(:block, seeds...)
end

function _invalidate_below!(emission::Emission, depth::Int)
    for deeper in (depth + 1):length(emission.levels)
        emission.cursor_valid[deeper] = false
    end
    nothing
end

"Return a formula-backed child fiber's width as a staged sum of node counts."
function _staged_fiber_width(nodes::Vector{R.Node}, binders::Tuple)::Staged
    width::Staged = 0
    for node in nodes
        width = add(STAGE, width, _staged_count(node.count, binders))
    end
    width
end

# A formula-backed child fiber passed between Finch loop levels. Sequential
# protocols realize its description. Other protocols use Finch's native
# SubFiber and invalidate deeper derived cursors.
struct SpecializedSubFiber
    emission::Emission
    depth::Int
    nodes::Vector{R.Node}
    binders::Tuple
    pos::Any
end

_level(fiber::SpecializedSubFiber) = fiber.emission.levels[fiber.depth]

FinchNotation.finch_leaf(fiber::SpecializedSubFiber) = virtual(fiber)
function Finch.virtual_fill_value(ctx, fiber::SpecializedSubFiber)
    virtual_level_fill_value(_level(fiber))
end
function Finch.virtual_size(ctx, fiber::SpecializedSubFiber)
    virtual_size(ctx, VirtualSubFiber(_level(fiber), fiber.pos))
end
Finch.instantiate(ctx, fiber::SpecializedSubFiber, mode) = fiber

function Finch.unfurl(ctx, fiber::SpecializedSubFiber, ext, mode,
    ::Union{typeof(defaultread),typeof(walk)})
    _realize_fiber(ctx, fiber.emission, fiber.depth, fiber.nodes, fiber.binders,
        fiber.pos)
end
function Finch.unfurl(ctx, fiber::SpecializedSubFiber, ext, mode, proto)
    _invalidate_below!(fiber.emission, fiber.depth)
    unfurl(ctx, VirtualSubFiber(_level(fiber), fiber.pos), ext, mode, proto)
end

# Descend from one formula-backed child. At a leaf, instantiate the element at
# its derived position. Otherwise pass the shared body to the next level with
# the current child ordinal appended.
function _descend(ctx, emission::Emission, depth::Int, node::R.Segment,
    binders::Tuple, child_ordinal::Staged, position_variable::Symbol, Tpos)
    if node.body === nothing
        Simplify(
            instantiate(ctx,
                VirtualSubFiber(emission.levels[depth].lvl,
                    value(position_variable, Tpos)),
                emission.mode),
        )
    else
        instantiate(ctx,
            SpecializedSubFiber(emission, depth + 1, node.body,
                (binders..., child_ordinal), value(position_variable, Tpos)),
            emission.mode)
    end
end

# Build the per-child looplet shared by dense and sparse levels. It binds the
# derived position, descends, and advances the next level's cursor by the
# completed child fiber width.
function _claimed_lookup(emission::Emission, depth::Int, segment::R.Segment,
    binders::Tuple, first_coordinate::Staged, position_of::Function, Tpos)
    tag = emission.levels[depth].tag
    child_cursor =
        depth < length(emission.levels) ?
        emission.cursor_symbols[depth + 1] : nothing
    Lookup(;
        body=(ctx_2, i) -> begin
            position_variable = freshen(ctx_2, tag, :_q)
            child_ordinal = sub(STAGE, ctx_2(i), first_coordinate)
            epilogue = if segment.body !== nothing && child_cursor !== nothing
                fiber_width = _staged_fiber_width(segment.body,
                    (binders..., child_ordinal))
                :($child_cursor += $fiber_width)
            else
                quote end
            end
            Thunk(;
                preamble=:($position_variable = $(position_of(ctx_2, i))),
                body=(ctx_3) -> _descend(ctx_3, emission, depth, segment,
                    binders, child_ordinal, position_variable, Tpos),
                epilogue=epilogue,
            )
        end,
    )
end

function _realize_fiber(ctx, emission::Emission, depth::Int,
    nodes::Vector{R.Node}, binders::Tuple, pos)
    if emission.levels[depth] isa VirtualDenseLevel
        _realize_dense_fiber(ctx, emission, depth, nodes, binders, pos)
    else
        _realize_sparse_fiber(ctx, emission, depth, nodes, binders, pos)
    end
end

# -- Dense fibers --------------------------------------------------------------
#
# Dense child position is `(parent_position - 1) * extent + coordinate`, so no
# structural array is needed. An Opaque stretch uses the native dense body and
# invalidates deeper derived cursors.
function _realize_dense_fiber(ctx, emission::Emission, depth::Int,
    nodes::Vector{R.Node}, binders::Tuple, pos)
    level = emission.levels[depth]::VirtualDenseLevel
    Ti = level.Ti
    tag = level.tag
    position_of =
        (ctx_2, i) ->
            :(($(ctx_2(pos)) - $(Ti(1))) * $(ctx_2(level.shape)) + $(ctx_2(i)))
    phases = Phase[]
    cursor_coordinate::Staged = 1            # First coordinate of the next node.
    for node in nodes
        stretch = _staged_count(node.count, binders)
        last_coordinate = sub(STAGE, add(STAGE, cursor_coordinate, stretch), 1)
        if node isa R.Opaque
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) -> _stop_node(last_coordinate, Ti),
                    body=(ctx_2, ext_2) -> begin
                        _invalidate_below!(emission, depth)
                        Lookup(;
                            body=(ctx_3, i) -> begin
                                position_variable = freshen(ctx_3, tag, :_q)
                                Thunk(;
                                    preamble=:(
                                        $position_variable =
                                            $(position_of(ctx_3, i))
                                    ),
                                    body=(ctx_4) -> instantiate(ctx_4,
                                        VirtualSubFiber(level.lvl,
                                            value(position_variable, Ti)),
                                        emission.mode),
                                )
                            end,
                        )
                    end,
                ),
            )
        else
            segment = node::R.Segment
            first_coordinate = _staged_first_coordinate(segment.coord, binders)
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) -> _stop_node(last_coordinate, Ti),
                    body=(ctx_2, ext_2) -> _claimed_lookup(emission, depth, segment,
                        binders, first_coordinate, position_of, Ti),
                ),
            )
        end
        cursor_coordinate = add(STAGE, cursor_coordinate, stretch)
    end
    push!(
        phases,
        Phase(;
            body=(ctx_2, ext_2) -> Run(FillLeaf(virtual_level_fill_value(level)))),
    )
    Sequence(phases)
end

# -- Sparse fibers -------------------------------------------------------------
#
# A formula-backed sparse position is
# `cursor + earlier_children + (coordinate - first_coordinate)`. The cursor
# gives this fiber's first stored position, and node counts provide
# `earlier_children`. Opaque nodes use Finch's native sparse walk.
function _realize_sparse_fiber(ctx, emission::Emission, depth::Int,
    nodes::Vector{R.Node}, binders::Tuple, pos)
    level = emission.levels[depth]::VirtualSparseListLevel
    Ti = level.Ti
    Tpos = postype(level)
    fill_leaf = FillLeaf(virtual_level_fill_value(level))
    cursor = emission.cursor_symbols[depth]::Symbol

    # If an opaque ancestor hid this fiber's start, read it from `ptr` once.
    reseed = if emission.cursor_valid[depth]
        quote end
    else
        emission.cursor_valid[depth] = true
        :($cursor = $(level.ptr)[$(ctx(pos))])
    end

    isempty(nodes) &&
        return Thunk(; preamble=reseed, body=(ctx_2) -> Run(fill_leaf))

    first_coordinates = Union{Staged,Nothing}[
        node isa R.Segment ? _staged_first_coordinate(node.coord, binders) : nothing
        for node in nodes
    ]

    phases = Phase[]
    earlier::Staged = 0                      # Stored children before this node.
    for (node_index, node) in enumerate(nodes)
        stretch = _staged_count(node.count, binders)
        if node isa R.Opaque
            following = findnext(!isnothing, first_coordinates, node_index + 1)
            generic_stop = if following === nothing
                nothing
            else
                sub(STAGE, first_coordinates[following], 1)
            end
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) ->
                        generic_stop === nothing ? nothing : _stop_node(generic_stop, Ti),
                    body=(ctx_2, ext_2) -> begin
                        _invalidate_below!(emission, depth)
                        unfurl_sparse_list_walk(ctx_2, VirtualSubFiber(level, pos),
                            ext_2, emission.mode, defaultread)
                    end,
                ),
            )
        else
            segment = node::R.Segment
            first_coordinate = first_coordinates[node_index]
            last_coordinate = sub(STAGE, add(STAGE, first_coordinate, stretch), 1)
            gap_stop = sub(STAGE, first_coordinate, 1)
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) -> _stop_node(gap_stop, Ti),
                    body=(ctx_2, ext_2) -> Run(fill_leaf)),
            )
            base = add(STAGE, cursor, earlier)
            position_of =
                (ctx_2, i) ->
                    add(STAGE, base, sub(STAGE, ctx_2(i), first_coordinate))
            push!(
                phases,
                Phase(;
                    stop=(ctx_2, ext_2) -> _stop_node(last_coordinate, Ti),
                    body=(ctx_2, ext_2) -> _claimed_lookup(emission, depth, segment,
                        binders, first_coordinate, position_of, Tpos),
                ),
            )
        end
        earlier = add(STAGE, earlier, stretch)
    end
    push!(phases, Phase(; body=(ctx_2, ext_2) -> Run(fill_leaf)))

    Thunk(; preamble=reseed, body=(ctx_2) -> Sequence(phases))
end

end # module RegularityExt

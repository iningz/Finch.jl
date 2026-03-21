# ═══════════════════════════════════════════════════════════════════════════════
# Regularity: compiler-independent sparsity pattern mining
# ═══════════════════════════════════════════════════════════════════════════════
#
# This module discovers regular structure in sparse fiber data (ptr/idx arrays).
# It has ZERO dependency on Finch's compiler infrastructure.
#
# Layout:
#   Regularity.jl          - framework: types, interfaces, pipeline
#   empty.jl               - plugin: EmptyPattern + EmptyMiner
#   contiguous.jl          - plugin: ContiguousPattern + ContiguousMiner
#   affine.jl              - plugin: AffinePattern + AffineMiner
#   identical_relative.jl  - plugin: IdenticalRelativePattern + IdenticalRelativeMiner
#
# Adding a new pattern:
#   1. Create a new file with a struct <: AbstractPattern and a struct <: AbstractMiningPass
#   2. Implement mine_pass for the new miner
#   3. Include the file below and add the miner to DEFAULT_MINING_PASSES
#   4. (In Finch) Add an emit_looplet method for each level that should handle it
#
# ═══════════════════════════════════════════════════════════════════════════════
module Regularity

export PatternRegion, FiberSpan
export OpaquePattern
export AbstractMiningPass, mine_pass
export EmptyPattern, EmptyMiner
export ContiguousPattern, ContiguousMiner
export AffinePattern, AffineMiner
export IdenticalRelativePattern, IdenticalRelativeMiner
export DEFAULT_MINING_PASSES
export mine_nd

# ═══════════════════════════════════════════════════════════════════════════════
# Framework
# ═══════════════════════════════════════════════════════════════════════════════

# ── PatternRegion ────────────────────────────────────────────────────────────

"""
    PatternRegion

An N-dimensional rectangular region in parent-index space, where each
dimension corresponds to one Dense level above a sparse level.

## Fields
- `ranges::Vector{UnitRange{Int}}` - one range per Dense dimension above
  the sparse level, ordered outermost-first.
"""
struct PatternRegion
    ranges::Vector{UnitRange{Int}}
end

Base.ndims(r::PatternRegion) = length(r.ranges)

# ── FiberSpan ────────────────────────────────────────────────────────────────

"""
    FiberSpan

Read-only view of a single fiber's index data within shared ptr/idx arrays.
Each pass computes what it needs from this; no shared mutable cache.

## Fields
- `ptr::Vector{Int}` - shared pointer array (length ≥ pos+1)
- `idx::Vector{Int}` - shared index array
- `pos::Int`         - which fiber (1-based position in ptr)
"""
struct FiberSpan
    ptr::Vector{Int}
    idx::Vector{Int}
    pos::Int
end

"""Number of stored indices in this fiber."""
@inline fiber_nnz(f::FiberSpan) = f.ptr[f.pos + 1] - f.ptr[f.pos]

"""View of the stored indices for this fiber."""
@inline function fiber_indices(f::FiberSpan)
    q_lo = f.ptr[f.pos]
    q_hi = f.ptr[f.pos + 1] - 1
    q_lo > q_hi ? view(f.idx, 1:0) : view(f.idx, q_lo:q_hi)
end

"""First stored index (0 if empty)."""
@inline fiber_first(f::FiberSpan) = fiber_nnz(f) == 0 ? 0 : f.idx[f.ptr[f.pos]]

"""Last stored index (0 if empty)."""
@inline fiber_last(f::FiberSpan) = fiber_nnz(f) == 0 ? 0 : f.idx[f.ptr[f.pos + 1] - 1]

"""Whether the indices form a contiguous range first:last."""
@inline fiber_is_contiguous(f::FiberSpan) =
    fiber_nnz(f) == 0 ? true : (fiber_last(f) - fiber_first(f) + 1 == fiber_nnz(f))

"""Hash of the full index set for quick equality checks."""
function fiber_idx_hash(f::FiberSpan)
    n = fiber_nnz(f)
    n == 0 && return zero(UInt)
    h = hash(n)
    q_lo = f.ptr[f.pos]
    for q in q_lo:(q_lo + n - 1)
        h = hash(f.idx[q], h)
    end
    return h
end

"""Element-wise comparison of two fibers' index sets."""
function fiber_indices_equal(a::FiberSpan, b::FiberSpan)
    na = fiber_nnz(a)
    na == fiber_nnz(b) || return false
    qa = a.ptr[a.pos]
    qb = b.ptr[b.pos]
    for k in 0:(na - 1)
        a.idx[qa + k] == b.idx[qb + k] || return false
    end
    return true
end

# ── AbstractPattern ──────────────────────────────────────────────────────────

"""
    AbstractPattern

Base type for all mined sparsity patterns.  Every subtype carries a
`region::PatternRegion` field describing the fiber positions it covers.

Subtype this to define a new pattern kind.
"""
abstract type AbstractPattern end

"""
    OpaquePattern <: AbstractPattern

Positions not claimed by any mining pass. The structure is opaque to the
miner. Codegen emits the generic (unspecialized) loop for these fibers.

This makes the miner total: every position is covered by some pattern,
so downstream codegen is a pure fold with no gap-filling or trailing phases.
"""
struct OpaquePattern <: AbstractPattern
    region::PatternRegion
end

# ── AbstractMiningPass ───────────────────────────────────────────────────────

"""
    AbstractMiningPass

Base type for mining passes.  Subtype this and implement `mine_pass` to add
a new pattern detector.
"""
abstract type AbstractMiningPass end

"""
    mine_pass(pass, fibers, unclaimed) -> Vector{<:AbstractPattern}

Scan fibers at unclaimed positions. Return patterns found.
Each pass reads its own `region_threshold` field to decide the minimum run length.
The caller marks claimed positions in `unclaimed` after each pass.
"""
function mine_pass end

# ── Helper ───────────────────────────────────────────────────────────────────

"""Skip forward to the next unclaimed position ≥ k, or return nothing."""
@inline function _next_unclaimed(unclaimed::BitVector, k::Int)
    k > length(unclaimed) && return nothing
    return findnext(unclaimed, k)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Plugins (one file per pattern + miner pair)
# ═══════════════════════════════════════════════════════════════════════════════

include("empty.jl")
include("contiguous.jl")
include("affine.jl")
include("identical_relative.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

"""Default ordered list of mining passes."""
const DEFAULT_MINING_PASSES = AbstractMiningPass[
    EmptyMiner(),
    AffineMiner(),
    ContiguousMiner(),
    IdenticalRelativeMiner(),
]

"""
    _mine_1d(ptr, idx, n_fibers; passes=DEFAULT_MINING_PASSES)
        -> Vector{AbstractPattern}

Internal helper: run the mining pipeline over 1D fiber positions.
Use [`mine_nd`](@ref) as the public entry point.
"""
function _mine_1d(ptr, idx, n_fibers::Int;
    passes::Vector{AbstractMiningPass}=DEFAULT_MINING_PASSES)
    n_fibers == 0 && return AbstractPattern[]

    fibers = [FiberSpan(ptr, idx, p) for p in 1:n_fibers]
    unclaimed = trues(n_fibers)
    result = AbstractPattern[]

    for pass in passes
        patterns = mine_pass(pass, fibers, unclaimed)
        for g in patterns
            for p in g.region.ranges[1]
                unclaimed[p] = false
            end
        end
        append!(result, patterns)
    end

    # Sweep: every unclaimed run becomes OpaquePattern.
    # This makes the miner total.
    k = _next_unclaimed(unclaimed, 1)
    while k !== nothing
        run_end = k
        while run_end < n_fibers && unclaimed[run_end + 1]
            run_end += 1
        end
        push!(result, OpaquePattern(PatternRegion([k:run_end])))
        k = _next_unclaimed(unclaimed, run_end + 1)
    end

    sort!(result; by=g -> first(g.region.ranges[1]))
    return result
end

"""
    mine_nd(ptr, idx, dense_shapes; passes=DEFAULT_MINING_PASSES)
        -> Vector{AbstractPattern}

Entry point for N-dimensional mining.  Currently only the 1-D case is
implemented; higher dimensions return an empty result.
"""
function mine_nd(ptr, idx, dense_shapes::Vector{Int};
    passes::Vector{AbstractMiningPass}=DEFAULT_MINING_PASSES)
    ndense = length(dense_shapes)
    if ndense == 1
        return _mine_1d(ptr, idx, dense_shapes[1]; passes=passes)
    end
    # Multi-dimensional mining is not yet implemented; fall back to no patterns.
    return AbstractPattern[]
end

end # module Regularity

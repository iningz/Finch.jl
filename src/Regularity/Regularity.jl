# ===============================================================================
# Regularity: compiler-independent sparsity pattern mining
# ===============================================================================
#
# Three contracts:
#   AbstractFiberView  - must implement: indices(f)
#   AbstractPattern    - must implement: region(p)
#   AbstractMiner      - must implement: mine_pass(miner, fibers, unclaimed)
#
# ===============================================================================
module Regularity

export CompressedFiberView
export indices, fiber_nnz, fiber_first, fiber_last
export fiber_is_contiguous, fiber_idx_hash, fiber_indices_equal
export OpaquePattern
export region
export AbstractMiner
export EmptyPattern, EmptyMiner
export ContiguousPattern, ContiguousMiner
export AffinePattern, AffineMiner
export IdenticalRelativePattern, IdenticalRelativeMiner
export DEFAULT_MINING_PASSES
export mine_nd

# ===============================================================================
# Contract 1: AbstractFiberView
# ===============================================================================

"""
    AbstractFiberView

Read-only view of a single fiber's stored indices.

Subtype this to support a new storage layout. Subtypes must implement `indices(f)`.
All other helpers (`fiber_nnz`, `fiber_first`, etc.) are derived from it but may
be overridden for performance.
"""
abstract type AbstractFiberView end

"""
    indices(f::AbstractFiberView) -> AbstractVector{Int}

Return the sorted stored indices for fiber `f`. This is the only method
subtypes of `AbstractFiberView` must implement.
"""
function indices end

"""Number of stored indices in this fiber."""
fiber_nnz(f::AbstractFiberView) = length(indices(f))

"""First stored index (0 if empty)."""
fiber_first(f::AbstractFiberView) = fiber_nnz(f) == 0 ? 0 : first(indices(f))

"""Last stored index (0 if empty)."""
fiber_last(f::AbstractFiberView) = fiber_nnz(f) == 0 ? 0 : last(indices(f))

"""Whether the indices form a contiguous range first:last."""
fiber_is_contiguous(f::AbstractFiberView) =
    fiber_nnz(f) == 0 ? true : (fiber_last(f) - fiber_first(f) + 1 == fiber_nnz(f))

"""Hash of the full index set for quick equality checks."""
function fiber_idx_hash(f::AbstractFiberView)
    n = fiber_nnz(f)
    n == 0 && return zero(UInt)
    h = hash(n)
    for idx in indices(f)
        h = hash(idx, h)
    end
    return h
end

"""Element-wise comparison of two fibers' index sets."""
function fiber_indices_equal(a::AbstractFiberView, b::AbstractFiberView)
    fiber_nnz(a) == fiber_nnz(b) || return false
    for (ai, bi) in zip(indices(a), indices(b))
        ai == bi || return false
    end
    return true
end

# ===============================================================================
# Contract 2: AbstractPattern
# ===============================================================================

"""
    AbstractPattern

A detected sparsity pattern covering a rectangular region of fiber positions.

Subtype this to define a new pattern kind; codegen in Finch dispatches on the
subtype via `emit_looplet`. Subtypes must implement `region(p)`.
"""
abstract type AbstractPattern end

"""
    region(p::AbstractPattern) -> Vector{UnitRange{Int}}

The rectangular region of fiber positions covered by `p`. Returns one range per
Dense dimension above the sparse level, ordered outermost-first. This is the
only method subtypes of `AbstractPattern` must implement.
"""
function region end

"""
    OpaquePattern <: AbstractPattern

Catch-all pattern for positions not claimed by any mining pass. Codegen emits
the generic (unspecialized) loop for these fibers. Ensures the miner is total:
every fiber position is covered by exactly one pattern.
"""
struct OpaquePattern <: AbstractPattern
    ranges::Vector{UnitRange{Int}}
end

region(p::OpaquePattern) = p.ranges

# ===============================================================================
# Contract 3: AbstractMiner
# ===============================================================================

"""
    AbstractMiner

A single-pass sparsity pattern detector.

Subtype this to add a new detector. Subtypes must implement `mine_pass`.
Register an instance in `DEFAULT_MINING_PASSES` to include it in the pipeline.
"""
abstract type AbstractMiner end

"""
    mine_pass(miner, fibers, unclaimed) -> Vector{<:AbstractPattern}

Scan `fibers` at positions where `unclaimed` is true. Return all detected
patterns. The caller marks claimed positions in `unclaimed` after each pass.
This is the only method subtypes of `AbstractMiner` must implement.
"""
function mine_pass end

# ===============================================================================
# Helpers
# ===============================================================================

"""Skip forward to the next unclaimed position >= k, or return nothing."""
@inline function _next_unclaimed(unclaimed::BitVector, k::Int)
    k > length(unclaimed) && return nothing
    return findnext(unclaimed, k)
end

# ===============================================================================
# Concrete implementations
# ===============================================================================

include("compressed_fiber.jl")

include("empty.jl")
include("contiguous.jl")
include("affine.jl")
include("identical_relative.jl")

# ===============================================================================
# Pipeline
# ===============================================================================

"""Default ordered list of mining passes."""
const DEFAULT_MINING_PASSES = AbstractMiner[
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
    passes::Vector{AbstractMiner}=DEFAULT_MINING_PASSES)
    n_fibers == 0 && return AbstractPattern[]

    fibers = [CompressedFiberView(ptr, idx, p) for p in 1:n_fibers]
    unclaimed = trues(n_fibers)
    result = AbstractPattern[]

    for pass in passes
        patterns = mine_pass(pass, fibers, unclaimed)
        for g in patterns
            for p in region(g)[1]
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
        push!(result, OpaquePattern([k:run_end]))
        k = _next_unclaimed(unclaimed, run_end + 1)
    end

    sort!(result; by=g -> first(region(g)[1]))
    return result
end

"""
    mine_nd(ptr, idx, dense_shapes; passes=DEFAULT_MINING_PASSES)
        -> Vector{AbstractPattern}

Entry point for N-dimensional mining. Currently only the 1-D case is
implemented; higher dimensions return an empty result.
"""
function mine_nd(ptr, idx, dense_shapes::Vector{Int};
    passes::Vector{AbstractMiner}=DEFAULT_MINING_PASSES)
    ndense = length(dense_shapes)
    if ndense == 1
        return _mine_1d(ptr, idx, dense_shapes[1]; passes=passes)
    end
    # Multi-dimensional mining is not yet implemented; fall back to no patterns.
    return AbstractPattern[]
end

end # module Regularity

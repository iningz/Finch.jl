# ===============================================================================
# CompressedFiberView: ptr/idx compressed fiber view
# ===============================================================================

# Fiber views where fibers are segments of a shared (ptr, idx) pair.
# ===============================================================================

"""
    CompressedFiberView <: AbstractFiberView

Lazy view of a single fiber within shared ptr/idx compressed arrays.

## Fields
- `ptr::Vector{Int}` - shared pointer array (length >= pos+1)
- `idx::Vector{Int}` - shared index array
- `pos::Int`         - which fiber (1-based position in ptr)
"""
struct CompressedFiberView <: AbstractFiberView
    ptr::Vector{Int}
    idx::Vector{Int}
    pos::Int
end

# -- Contract primitive --------------------------------------------------------

"""View of the stored indices for this fiber."""
@inline function indices(f::CompressedFiberView)
    q_lo = f.ptr[f.pos]
    q_hi = f.ptr[f.pos + 1] - 1
    q_lo > q_hi ? view(f.idx, 1:0) : view(f.idx, q_lo:q_hi)
end

# -- Performance overrides (avoid materializing index views) -------------------

@inline fiber_nnz(f::CompressedFiberView) = f.ptr[f.pos + 1] - f.ptr[f.pos]

@inline fiber_first(f::CompressedFiberView) =
    fiber_nnz(f) == 0 ? 0 : f.idx[f.ptr[f.pos]]

@inline fiber_last(f::CompressedFiberView) =
    fiber_nnz(f) == 0 ? 0 : f.idx[f.ptr[f.pos + 1] - 1]

function fiber_idx_hash(f::CompressedFiberView)
    n = fiber_nnz(f)
    n == 0 && return zero(UInt)
    h = hash(n)
    q_lo = f.ptr[f.pos]
    for q in q_lo:(q_lo + n - 1)
        h = hash(f.idx[q], h)
    end
    return h
end

function fiber_indices_equal(a::CompressedFiberView, b::CompressedFiberView)
    na = fiber_nnz(a)
    na == fiber_nnz(b) || return false
    qa = a.ptr[a.pos]
    qb = b.ptr[b.pos]
    for k in 0:(na - 1)
        a.idx[qa + k] == b.idx[qb + k] || return false
    end
    return true
end

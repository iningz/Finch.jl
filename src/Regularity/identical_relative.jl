# ═══════════════════════════════════════════════════════════════════════════════
# IdenticalRelative: pattern + miner
# ═══════════════════════════════════════════════════════════════════════════════
#
# Every fiber stores the same sparse (non-contiguous) index set.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    IdenticalRelativePattern <: AbstractPattern

Every fiber stores the same sparse (non-contiguous) index set.

## Fields
- `region::PatternRegion`
- `indices::Vector{Int}` - the shared index set
"""
struct IdenticalRelativePattern <: AbstractPattern
    region::PatternRegion
    indices::Vector{Int}
end

struct IdenticalRelativeMiner <: AbstractMiningPass
    region_threshold::Int
end
IdenticalRelativeMiner(; region_threshold::Int=2) = IdenticalRelativeMiner(region_threshold)

function mine_pass(pass::IdenticalRelativeMiner, fibers::Vector{FiberSpan},
    unclaimed::BitVector)
    region_threshold = pass.region_threshold
    patterns = AbstractPattern[]
    k = _next_unclaimed(unclaimed, 1)
    while k !== nothing
        f = fibers[k]
        n = fiber_nnz(f)
        if n > 0 && !fiber_is_contiguous(f)
            h = fiber_idx_hash(f)
            run_end = k
            while run_end < length(fibers) && unclaimed[run_end + 1]
                f2 = fibers[run_end + 1]
                (fiber_nnz(f2) == n && fiber_idx_hash(f2) == h) || break
                fiber_indices_equal(f, f2) || break
                run_end += 1
            end
            rlen = run_end - k + 1
            if rlen >= region_threshold
                indices = Int.(collect(fiber_indices(f)))
                push!(
                    patterns, IdenticalRelativePattern(
                        PatternRegion([k:run_end]), indices)
                )
                k = _next_unclaimed(unclaimed, run_end + 1)
                continue
            end
        end
        k = _next_unclaimed(unclaimed, k + 1)
    end
    return patterns
end

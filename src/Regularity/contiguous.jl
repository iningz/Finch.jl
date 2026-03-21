# ═══════════════════════════════════════════════════════════════════════════════
# Contiguous: pattern + miner
# ═══════════════════════════════════════════════════════════════════════════════
#
# Every fiber stores the same contiguous block [first_idx .. first_idx+nnz-1].
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ContiguousPattern <: AbstractPattern

Every fiber stores the same contiguous block `[first_idx .. first_idx+nnz-1]`.

## Fields
- `region::PatternRegion`
- `first_idx::Int` - starting index of the block
- `nnz::Int`       - number of stored indices per fiber
"""
struct ContiguousPattern <: AbstractPattern
    region::PatternRegion
    first_idx::Int
    nnz::Int
end

struct ContiguousMiner <: AbstractMiningPass
    region_threshold::Int
end
ContiguousMiner(; region_threshold::Int=2) = ContiguousMiner(region_threshold)

function mine_pass(pass::ContiguousMiner, fibers::Vector{FiberSpan},
    unclaimed::BitVector)
    region_threshold = pass.region_threshold
    patterns = AbstractPattern[]
    k = _next_unclaimed(unclaimed, 1)
    while k !== nothing
        f = fibers[k]
        n = fiber_nnz(f)
        if n > 0 && fiber_is_contiguous(f)
            fi = fiber_first(f)
            run_end = k
            while run_end < length(fibers) && unclaimed[run_end + 1]
                f2 = fibers[run_end + 1]
                (fiber_nnz(f2) == n &&
                 fiber_is_contiguous(f2) &&
                 fiber_first(f2) == fi) || break
                run_end += 1
            end
            rlen = run_end - k + 1
            if rlen >= region_threshold
                push!(patterns, ContiguousPattern(
                    PatternRegion([k:run_end]), fi, n))
                k = _next_unclaimed(unclaimed, run_end + 1)
                continue
            end
        end
        k = _next_unclaimed(unclaimed, k + 1)
    end
    return patterns
end

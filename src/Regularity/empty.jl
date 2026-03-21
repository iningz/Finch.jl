# ═══════════════════════════════════════════════════════════════════════════════
# Empty: pattern + miner
# ═══════════════════════════════════════════════════════════════════════════════
#
# All fibers in the region have nnz == 0.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    EmptyPattern <: AbstractPattern

All fibers in the region have nnz == 0.
"""
struct EmptyPattern <: AbstractPattern
    region::PatternRegion
end

struct EmptyMiner <: AbstractMiningPass
    region_threshold::Int
end
EmptyMiner(; region_threshold::Int=2) = EmptyMiner(region_threshold)

function mine_pass(pass::EmptyMiner, fibers::Vector{FiberSpan},
    unclaimed::BitVector)
    region_threshold = pass.region_threshold
    patterns = AbstractPattern[]
    k = _next_unclaimed(unclaimed, 1)
    while k !== nothing
        if fiber_nnz(fibers[k]) == 0
            run_end = k
            while run_end < length(fibers) &&
                      unclaimed[run_end + 1] &&
                      fiber_nnz(fibers[run_end + 1]) == 0
                run_end += 1
            end
            if run_end - k + 1 >= region_threshold
                push!(patterns, EmptyPattern(PatternRegion([k:run_end])))
                k = _next_unclaimed(unclaimed, run_end + 1)
                continue
            end
        end
        k = _next_unclaimed(unclaimed, k + 1)
    end
    return patterns
end

# ═════════════════════════════════════════════════════════════════════════════
# Affine: pattern + miner
# ═════════════════════════════════════════════════════════════════════════════
#
# Every fiber stores a contiguous block of width `nnz`, whose starting index
# shifts linearly: start(p) = base_first + (p - outer_start) * delta.
# ═════════════════════════════════════════════════════════════════════════════

"""
    AffinePattern <: AbstractPattern

Every fiber stores a contiguous block of width `nnz`, whose starting index
shifts linearly: `start(p) = base_first + (p - outer_start) * delta`.

## Fields
- `region::PatternRegion`
- `base_first::Int` - starting index at the first fiber in the region
- `delta::Int`      - shift in starting index per fiber step
- `nnz::Int`        - number of stored indices per fiber
"""
struct AffinePattern <: AbstractPattern
    region::PatternRegion
    base_first::Int
    delta::Int
    nnz::Int
end

struct AffineMiner <: AbstractMiningPass
    region_threshold::Int
end
AffineMiner(; region_threshold::Int=2) = AffineMiner(region_threshold)

function mine_pass(pass::AffineMiner, fibers::Vector{FiberSpan},
    unclaimed::BitVector)
    region_threshold = pass.region_threshold
    patterns = AbstractPattern[]
    k = _next_unclaimed(unclaimed, 1)
    while k !== nothing
        f = fibers[k]
        n = fiber_nnz(f)
        if n > 0 && fiber_is_contiguous(f)
            # Find next unclaimed to compute delta
            k2 = _next_unclaimed(unclaimed, k + 1)
            if k2 !== nothing
                f2 = fibers[k2]
                if fiber_nnz(f2) == n && fiber_is_contiguous(f2)
                    delta = fiber_first(f2) - fiber_first(f)
                    if delta != 0
                        # Scan forward: only include contiguous unclaimed
                        # positions that maintain the affine relationship
                        run_end = k
                        p = k
                        while true
                            p_next = _next_unclaimed(unclaimed, p + 1)
                            p_next === nothing && break
                            # Must be the very next position (contiguous in
                            # parent-index space) to form a rectangular region
                            p_next == p + 1 || break
                            fn = fibers[p_next]
                            expected_first = fiber_first(f) + (p_next - k) * delta
                            (
                                fiber_nnz(fn) == n &&
                                fiber_is_contiguous(fn) &&
                                fiber_first(fn) == expected_first
                            ) || break
                            run_end = p_next
                            p = p_next
                        end
                        rlen = run_end - k + 1
                        if rlen >= region_threshold
                            push!(
                                patterns,
                                AffinePattern(
                                    PatternRegion([k:run_end]),
                                    fiber_first(f), delta, n),
                            )
                            k = _next_unclaimed(unclaimed, run_end + 1)
                            continue
                        end
                    end
                    # delta == 0 falls through to ContiguousMiner
                end
            end
        end
        k = _next_unclaimed(unclaimed, k + 1)
    end
    return patterns
end

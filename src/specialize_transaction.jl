# The specialization transaction: one specialized compilation runs as one
# attempt that owns its budget state and report. The attempt is created by
# `specialize_compile`, carried by the compiler context it compiles under
# (`FinchCompiler.attempt`), and discarded with that context on decline.
#
# Cost is measured where it is created, in the compiler's own units: emitted
# Sequence phases (each lowers to its own loop) and emitted Switch cases (each
# lowers to its own branch). Both Cartesian composition sites check budget
# headroom before materializing a product, and each lowering charges what it
# emits. Charging accrues from attempt creation; enforcement of the budget —
# emitted_sequence_phases + emitted_switch_cases <= limit — begins when an
# extension realizes a candidate and activates the budget. Products
# materialized before activation are baseline generic work: counted, not
# bounded.
#
# Enforcement is fail-closed over exact counts: only a genuinely representable
# total can be proven within the budget, and it compares exactly (a total or
# product equal to its bound passes). Per-counter saturation, cross-counter
# addition overflow, and length-product overflow all decline while the budget
# is active — at activation reconciliation, after a charge, and before a
# composition materializes, through the same checked state.
#
# Finch does not depend on Regularity types: everything here is core-owned,
# and an extension participates only by calling
# `activate_specialization_budget!` when it commits a realization.

"""
    SpecializationAttempt()

The transaction state for one specialized compilation.
`emitted_sequence_phases` and `emitted_switch_cases` accrue what the lowerings
emit; `counts_exact` records whether those counters are still exact — a
counter that saturates makes them lower bounds, which no budget check can
accept. The budget is inactive until an extension realizes a candidate;
`limit` bounds the counters' sum and is meaningful only while
`budget_active`. `verdict` moves from `:inflight` to `:completed`,
`:declined`, or `:errored`.
"""
mutable struct SpecializationAttempt
    emitted_sequence_phases::Int
    emitted_switch_cases::Int
    counts_exact::Bool
    limit::Int
    budget_active::Bool
    realized::Int
    verdict::Symbol
    reason::Union{Symbol,Nothing}
end

SpecializationAttempt() =
    SpecializationAttempt(0, 0, true, typemax(Int), false, 0, :inflight, nothing)

"""
    SpecializeReport

An immutable snapshot of a finished [`SpecializationAttempt`](@ref): the two
emission counts are the final charged totals (the declined attempt's totals
when `declined`; the generic retry is not re-measured).
"""
struct SpecializeReport
    realized::Int
    emitted_sequence_phases::Int
    emitted_switch_cases::Int
    declined::Bool
    reason::Union{Symbol,Nothing}
end

SpecializeReport(attempt::SpecializationAttempt) = SpecializeReport(
    attempt.realized,
    attempt.emitted_sequence_phases,
    attempt.emitted_switch_cases,
    attempt.verdict === :declined,
    attempt.reason,
)

"Control exception: decline the whole specialized attempt and rebuild generic."
struct RegularizeDecline <: Exception
    reason::Symbol
end

"""
    specialization_attempt(ctx)

The [`SpecializationAttempt`](@ref) carried by compiler context `ctx`, or
`nothing`. The default method makes every charging hook inert for contexts
that carry no attempt.
"""
specialization_attempt(ctx) = nothing

# One counter accrual: checked addition that reports whether the new value is
# still exact. Saturation keeps the counter a valid lower bound for reports.
function _accrue(count::Int, n::Int)::Tuple{Int,Bool}
    (total, overflowed) = Base.Checked.add_with_overflow(count, n)
    overflowed ? (typemax(Int), false) : (total, true)
end

# The counters' sum with its provenance: exact only if no counter has
# saturated and the cross-counter addition itself is representable.
function _exact_total(attempt::SpecializationAttempt)::Tuple{Int,Bool}
    (total, overflowed) = Base.Checked.add_with_overflow(
        attempt.emitted_sequence_phases, attempt.emitted_switch_cases)
    (total, attempt.counts_exact && !overflowed)
end

# The one fail-closed enforcement, shared by activation reconciliation and
# post-charge checks: an inexact total cannot be proven within any budget; an
# exact total compares exactly (equality passes).
function _enforce_budget!(attempt::SpecializationAttempt)
    attempt.budget_active || return nothing
    (total, exact) = _exact_total(attempt)
    (!exact || total > attempt.limit) &&
        throw(RegularizeDecline(:emitted_phases_and_cases))
    nothing
end

"""
    activate_specialization_budget!(attempt, limit)

Record one committed realization and, on the first call, activate budget
enforcement: emitted_sequence_phases + emitted_switch_cases must stay within
`limit`. Activation immediately reconciles the charge already accrued through
the same fail-closed check every later charge uses: an exact total above the
limit — or a total that is no longer exact — declines on the spot.
"""
function activate_specialization_budget!(attempt::SpecializationAttempt, limit::Integer)
    attempt.realized += 1
    attempt.budget_active && return nothing
    attempt.budget_active = true
    attempt.limit = Int(limit)
    _enforce_budget!(attempt)
end

# The lengths of the composed lists are multiplied with checked machine
# arithmetic; the mathematical product is never constructed. Overflow is
# reported alongside the (saturated) value so callers can fail closed: an
# unrepresentable product can never be proven within any budget.
function _checked_length_product(lists)::Tuple{Int,Bool}
    n = 1
    for list in lists
        (n, overflowed) = Base.Checked.mul_with_overflow(n, length(list))
        overflowed && return (typemax(Int), true)
    end
    (n, false)
end

"""
    regularize_precompose(ctx, lists)

Called before a Cartesian composition (Sequence phases or Switch cases)
materializes the product of `lists`. While the budget is active, the
composition declines before anything is allocated when the length product
overflows `Int`, when the accrued totals are no longer exact, or when a
representable product exceeds the remaining headroom — a product exactly
equal to the headroom passes.
"""
function regularize_precompose(ctx, lists)
    attempt = specialization_attempt(ctx)
    (attempt === nothing || !attempt.budget_active) && return nothing
    (product, overflowed) = _checked_length_product(lists)
    (total, exact) = _exact_total(attempt)
    (overflowed || !exact || product > attempt.limit - total) &&
        throw(RegularizeDecline(:emitted_phases_and_cases))
    nothing
end

"""
    regularize_charge_sequence_phases!(ctx, n)

Charge `n` emitted Sequence phases to the context's attempt. A saturating
accrual marks the counters inexact. Enforcement applies only while the budget
is active; before activation the charge is baseline accounting.
"""
function regularize_charge_sequence_phases!(ctx, n::Int)
    attempt = specialization_attempt(ctx)
    attempt === nothing && return nothing
    (attempt.emitted_sequence_phases, exact) =
        _accrue(attempt.emitted_sequence_phases, n)
    exact || (attempt.counts_exact = false)
    _enforce_budget!(attempt)
end

"""
    regularize_charge_switch_cases!(ctx, n)

Charge `n` emitted Switch cases to the context's attempt. A saturating
accrual marks the counters inexact. Enforcement applies only while the budget
is active; before activation the charge is baseline accounting.
"""
function regularize_charge_switch_cases!(ctx, n::Int)
    attempt = specialization_attempt(ctx)
    attempt === nothing && return nothing
    (attempt.emitted_switch_cases, exact) = _accrue(attempt.emitted_switch_cases, n)
    exact || (attempt.counts_exact = false)
    _enforce_budget!(attempt)
end

"""
    specialize_compile(build; algebra, mode, report=nothing)

Run one specialized compilation transaction. `build(ctx)` receives a fresh
`FinchCompiler` and must construct everything else itself; the attempt-bearing
context is attempt 1, and on [`RegularizeDecline`](@ref) the same builder runs
again on an attempt-free context — the generic compilation, byte-identical by
determinism. `report`, when given, documents the specialized attempt's settled
verdict: it is written exactly once, when attempt 1 completes or declines, and
never when attempt 1 itself raises a foreign error (which rethrows). A foreign
error during the generic rebuild propagates with the declined report already
assigned — the specialization decision had settled, and the rebuild is
deterministic-generic, so its failure is the program's own.
"""
function specialize_compile(build; algebra=DefaultAlgebra(), mode=:safe, report=nothing)
    attempt = SpecializationAttempt()
    try
        code = build(FinchCompiler(; algebra=algebra, mode=mode, attempt=attempt))
        attempt.verdict = :completed
        report === nothing || (report[] = SpecializeReport(attempt))
        return code
    catch exception
        if exception isa RegularizeDecline
            attempt.verdict = :declined
            attempt.reason = exception.reason
            report === nothing || (report[] = SpecializeReport(attempt))
        else
            attempt.verdict = :errored
            rethrow()
        end
    end
    build(FinchCompiler(; algebra=algebra, mode=mode))
end

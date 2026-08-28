# One specialized compilation runs as one attempt: it carries the policy the
# extension mines under and counts what the lowerings emit, in the compiler's
# own units — emitted Sequence phases (each lowers to its own loop) and
# emitted Switch cases (each lowers to its own branch). The attempt is created
# by `specialize_compile` and carried by the compiler context it compiles
# under (`FinchCompiler.attempt`). Counters are pure observability: nothing
# here bounds compilation — time and memory control belong to the caller.
#
# Finch does not depend on Regularity types: everything here is core-owned.
# An extension reads `specialization_attempt(ctx).policy` for its knobs and
# increments `realized` when it commits a realization.

"""
    SpecializePolicy(; pmax=8, min_run=1, leaf_min_run=1)

Performance-motivated specialization knobs: `pmax` bounds the PeriodicAffine
period search, and claims spanning fewer than `min_run` fibers (interior
levels) or `leaf_min_run` fibers (leaf level) stay opaque. All fields must be
at least 1.
"""
struct SpecializePolicy
    pmax::Int
    min_run::Int
    leaf_min_run::Int
    function SpecializePolicy(pmax::Integer, min_run::Integer, leaf_min_run::Integer)
        min(pmax, min_run, leaf_min_run) >= 1 ||
            throw(ArgumentError("SpecializePolicy fields must be >= 1"))
        new(pmax, min_run, leaf_min_run)
    end
end

SpecializePolicy(; pmax=8, min_run=1, leaf_min_run=1) =
    SpecializePolicy(pmax, min_run, leaf_min_run)

"""
    SpecializationAttempt(policy=SpecializePolicy())

The state of one specialized compilation: the [`SpecializePolicy`](@ref) it
runs under, how many candidates the extension `realized`, and the emitted
`sequence_phases` and `switch_cases` the lowerings charged.
"""
mutable struct SpecializationAttempt
    const policy::SpecializePolicy
    realized::Int
    sequence_phases::Int
    switch_cases::Int
end

SpecializationAttempt(policy::SpecializePolicy=SpecializePolicy()) =
    SpecializationAttempt(policy, 0, 0, 0)

"""
    SpecializeReport

An immutable snapshot of a completed [`SpecializationAttempt`](@ref).
"""
struct SpecializeReport
    realized::Int
    emitted_sequence_phases::Int
    emitted_switch_cases::Int
end

SpecializeReport(attempt::SpecializationAttempt) = SpecializeReport(
    attempt.realized, attempt.sequence_phases, attempt.switch_cases)

"""
    specialization_attempt(ctx)

The [`SpecializationAttempt`](@ref) carried by compiler context `ctx`, or
`nothing`. The default method makes every charging hook inert for contexts
that carry no attempt.
"""
specialization_attempt(ctx) = nothing

"""
    regularize_charge_sequence_phases!(ctx, n)

Record `n` emitted Sequence phases on the context's attempt, if any.
"""
function regularize_charge_sequence_phases!(ctx, n::Int)
    attempt = specialization_attempt(ctx)
    attempt === nothing || (attempt.sequence_phases += n)
    nothing
end

"""
    regularize_charge_switch_cases!(ctx, n)

Record `n` emitted Switch cases on the context's attempt, if any.
"""
function regularize_charge_switch_cases!(ctx, n::Int)
    attempt = specialization_attempt(ctx)
    attempt === nothing || (attempt.switch_cases += n)
    nothing
end

"""
    specialize_compile(build; algebra, mode, policy=SpecializePolicy(), report=nothing)

Run one specialized compilation. `build(ctx)` receives a fresh `FinchCompiler`
carrying a [`SpecializationAttempt`](@ref) under `policy` and must construct
everything else itself. `report`, when given, receives the completed attempt's
[`SpecializeReport`](@ref); a foreign error propagates without writing it.
"""
function specialize_compile(build; algebra=DefaultAlgebra(), mode=:safe,
        policy::SpecializePolicy=SpecializePolicy(), report=nothing)
    attempt = SpecializationAttempt(policy)
    code = build(FinchCompiler(; algebra=algebra, mode=mode, attempt=attempt))
    report === nothing || (report[] = SpecializeReport(attempt))
    code
end

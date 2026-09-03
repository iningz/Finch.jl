# One specialized compilation runs as one attempt: it carries the policy the
# extension mines under and counts what the lowerings emit, in the compiler's
# own units — emitted Sequence phases (each lowers to its own loop) and
# emitted Switch cases (each lowers to its own branch). The attempt is created
# by `specialize_compile` and carried by the compiler context it compiles
# under (`FinchCompiler.attempt`). Counters are pure observability: nothing
# here bounds compilation — time and memory control belong to the caller, and
# a compile that times out is a measurement, not a fallback.
#
# Finch does not depend on Regularity types: everything here is core-owned.
# An extension reads `specialization_attempt(ctx).policy` for its knobs and
# increments `realized` when it commits a realization.

"""
    SpecializePolicy(; pmax=8, code_density=0, visit_density=0)

Performance-motivated specialization knobs. `pmax` (>= 1) bounds the
PeriodicAffine period search during mining.

The two pruning knobs share one numerator, the structural reads a claim
avoids: one `idx` read per entry of a leaf Segment at a compressed level and
two `ptr` reads per child whose origin derives from the description. They
differ in what they count reads against.

`code_density` (`d`) prices code: a span survives iff its subtree avoids at
least `d` reads per abstract span, summed over every parent sharing the
subtree. It bounds the description, and so the emitted code, to `reads / d`
spans.

`visit_density` (`c`) prices execution: one parent visit survives iff it
avoids at least `c` reads per extra concrete span it executes,
`reads(visit) >= c * (concrete_spans(visit) - 1)`; a Series counts
`reps * word` concrete spans. A failing visit demotes its weakest claim and
re-checks.

`0` disables a rule; both default to `0`, which leaves the mined description
unpruned. A demoted span returns to the host's ordinary iterator as an opaque
stretch of the same count, so the partition stays exact. A tensor whose
pruned description avoids no structural read declines to generic code.
"""
struct SpecializePolicy
    pmax::Int
    code_density::Int
    visit_density::Int
    function SpecializePolicy(pmax::Integer, code_density::Integer, visit_density::Integer)
        pmax >= 1 || throw(ArgumentError("SpecializePolicy: pmax must be >= 1"))
        code_density >= 0 ||
            throw(ArgumentError("SpecializePolicy: code_density must be >= 0"))
        visit_density >= 0 ||
            throw(ArgumentError("SpecializePolicy: visit_density must be >= 0"))
        new(pmax, code_density, visit_density)
    end
end

SpecializePolicy(; pmax=8, code_density=0, visit_density=0) =
    SpecializePolicy(pmax, code_density, visit_density)

"""
    SpecializationAttempt(policy=SpecializePolicy())

The state of one specialized compilation: the [`SpecializePolicy`](@ref) it
runs under, how many candidates the extension `realized`, the emitted
`sequence_phases` and `switch_cases` the lowerings charged, and the realized
descriptions' `abstract_spans`, `concrete_spans`, and `structural_reads`
avoided.
"""
mutable struct SpecializationAttempt
    const policy::SpecializePolicy
    realized::Int
    sequence_phases::Int
    switch_cases::Int
    abstract_spans::Int
    concrete_spans::Int
    structural_reads::Int
end

function SpecializationAttempt(policy::SpecializePolicy=SpecializePolicy())
    SpecializationAttempt(policy, 0, 0, 0, 0, 0, 0)
end

"""
    SpecializeReport

An immutable snapshot of a completed [`SpecializationAttempt`](@ref).
`realized` counts realizations committed only after their whole looplet was
constructed; the emitted counters describe the composed kernel's code
structure; `abstract_spans`, `concrete_spans`, and `structural_reads` sum the
realized descriptions' spans and the structural reads they avoid (zero when
nothing realized).
"""
struct SpecializeReport
    realized::Int
    emitted_sequence_phases::Int
    emitted_switch_cases::Int
    abstract_spans::Int
    concrete_spans::Int
    structural_reads::Int
end

function SpecializeReport(attempt::SpecializationAttempt)
    SpecializeReport(
        attempt.realized, attempt.sequence_phases, attempt.switch_cases,
        attempt.abstract_spans, attempt.concrete_spans, attempt.structural_reads)
end

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

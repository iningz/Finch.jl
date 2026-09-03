# The periodic-affine family from the evaluation's `periodic` generator:
# constant width-4 fibers whose start follows a period-8 word
# (start = 1 + r^2 + q*(p^2 + 3) for fiber ordinal j-1 = q*p + r), so the
# structure is recognizable exactly when the policy's period search covers the
# period. The last column breaks the word so that no claim covers the whole
# level.
function policy_periodic_data(n=64; period=8, width=4)
    drift = period^2 + 3
    starts = map(1:(n - 1)) do j
        q, r = divrem(j - 1, period)
        1 + r^2 + q * drift
    end
    data = zeros(Float64, last(starts) + 2width, n)
    for (j, start) in pairs(starts), i in start:(start + width - 1)
        data[i, j] = (13i + 29j) / 2048
    end
    data[end, n] = 1.0
    data
end

function policy_spmv_code(A, x, y; specialize, policy=SpecializePolicy(), report=nothing)
    if specialize
        return string(@finch_code specialize = true policy = policy report = report begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end)
    end
    string(@finch_code begin
        y .= 0.0
        for j in _, i in _
            y[i] += A[i, j] * x[j]
        end
    end)
end

struct DelayedPolicyCount <: Regularity.Scalar
    threshold::Int
    low::Int
    high::Int
end

function Regularity.evaluate(
    pattern::DelayedPolicyCount,
    ordinal::Int,
    ::Regularity.Ops,
)
    ordinal <= pattern.threshold ? pattern.low : pattern.high
end

@testset "specialization policy" begin
    @test SpecializePolicy() == SpecializePolicy(8, 0, 0)
    @test SpecializePolicy(; pmax=16, code_density=4, visit_density=2) ==
        SpecializePolicy(16, 4, 2)
    @test fieldnames(SpecializePolicy) == (:pmax, :code_density, :visit_density)
    @test_throws ArgumentError SpecializePolicy(pmax=0)
    @test_throws ArgumentError SpecializePolicy(code_density=-1)
    @test_throws ArgumentError SpecializePolicy(visit_density=-1)

    data = policy_periodic_data()
    A = Tensor(Dense(SparseList(Element(0.0))), data)
    x = Tensor(Dense(Element(0.0)), [1.0 + j / 32 for j in axes(data, 2)])
    y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    generic = policy_spmv_code(A, x, y; specialize=false)

    # A period search that covers the word realizes the structure.
    deep_report = Ref{F.SpecializeReport}()
    deep = policy_spmv_code(A, x, y;
        specialize=true, policy=SpecializePolicy(; pmax=16), report=deep_report)
    @test deep != generic
    @test deep_report[].realized >= 1

    # Both rules default to disabled.
    @test SpecializePolicy().code_density == 0
    @test SpecializePolicy().visit_density == 0

    # A density above every claim prunes everything: the tensor declines and
    # the emitted code is normalized-AST-identical to generic.
    pruned_report = Ref{F.SpecializeReport}()
    pruned = policy_spmv_code(A, x, y;
        specialize=true,
        policy=SpecializePolicy(; pmax=16, code_density=10^9),
        report=pruned_report)
    @test pruned_report[].realized == 0
    @test pruned == generic

    # A moderate density keeps the dominant claim: emission survives with no
    # more phases than the unpruned attempt, and values stay exact.
    kept_report = Ref{F.SpecializeReport}()
    kept = policy_spmv_code(A, x, y;
        specialize=true,
        policy=SpecializePolicy(; pmax=16, code_density=20),
        report=kept_report)
    @test kept_report[].realized == 1
    @test kept != generic
    @test kept_report[].emitted_sequence_phases <=
        deep_report[].emitted_sequence_phases
    kept_result = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    @finch specialize = true policy = SpecializePolicy(; pmax=16, code_density=20) begin
        kept_result .= 0.0
        for j in _, i in _
            kept_result[i] += A[i, j] * x[j]
        end
    end
    @test ulps_apart(Array(kept_result), data * Array(x)) <= 4

    # `@finch` forwards the policy through `execute` to `execute_specialized`.
    result = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    @finch specialize = true policy = SpecializePolicy(; pmax=16) begin
        result .= 0.0
        for j in _, i in _
            result[i] += A[i, j] * x[j]
        end
    end
    @test ulps_apart(Array(result), data * Array(x)) <= 4
end

# The mining policy pruning re-folds under; the tests never depend on it.
prune_description(description, structure, policy) = RX._prune_description(
    description, structure, policy,
    Regularity.MiningPolicy((Regularity.PeriodicAffineConfig(; pmax=16, coordinate_drift=1),)))

# Direct check of the bottom-up code rule on a hand-built description:
# reads aggregate over enclosing parents while abstract spans count once,
# constant-counted demoted spans merge into one Opaque of the level's binder
# arity, and a light child demotes while its heavy sibling and their parent
# survive.
@testset "code-density pruning tree" begin
    coord = Regularity.PeriodicAffine((1,), 1)
    heavy = Regularity.Segment(64, coord, nothing)
    light = Regularity.Segment(Regularity.PeriodicAffine((1,), 0), coord, nothing)
    body = Regularity.Span[heavy, light, Regularity.Opaque(2)]
    top = Regularity.Span[
        Regularity.Segment(4, coord, body),
        Regularity.Segment(1, coord, nothing),
        Regularity.Opaque(3),
    ]
    description = Regularity.Description(top)
    ptr = [1; cumsum(fill(67, 4)) .+ 1; fill(269, 4)]
    idx = repeat(collect(1:67), 4)
    structure = Regularity.Structure(
        Regularity.DenseLevel(8), Regularity.CompressedLevel(ptr, idx))

    # heavy aggregates 4 parents x 64 = 256 >= 10/span; light aggregates 4.
    # The parent adds 2 ptr reads per child (its origin derives): 264 over
    # 3 abstract spans.
    pruned =
        prune_description(description, structure,
            SpecializePolicy(; code_density=10)).spans
    @test length(pruned) == 2
    parent = pruned[1]
    @test parent isa Regularity.Segment && parent.count == 4
    @test length(parent.body) == 2
    @test parent.body[1] isa Regularity.Segment && parent.body[1].count == 64
    @test parent.body[2] isa Regularity.Opaque             # light + Opaque(2)
    @test parent.body[2].count isa Regularity.Pattern
    @test Regularity.bind_count(parent.body[2].count, (0,)) == 3
    @test pruned[2] isa Regularity.Opaque && pruned[2].count == 4
    @test RX.structural_reads_avoided(Regularity.Description(pruned), structure) == 264

    # A density above everything collapses the list to one opaque span.
    all_pruned =
        prune_description(description, structure,
            SpecializePolicy(; code_density=10^9)).spans
    @test length(all_pruned) == 1
    @test all_pruned[1] isa Regularity.Opaque && all_pruned[1].count == 8
end

@testset "visit-density is per parent visit" begin
    coord = Regularity.PeriodicAffine((1,), 1)
    constant(n) = Regularity.PeriodicAffine((n,), 0)
    ptr = collect(1:5:(1 + 64 * 5))
    idx = repeat([7, 25, 26, 27, 55], 64)
    structure = Regularity.Structure(
        Regularity.DenseLevel(64), Regularity.CompressedLevel(ptr, idx))

    # Sharing this child list across 64 parents must not aggregate its three
    # avoided idx reads. Every parent executes three concrete spans, so each
    # visit fails 3 >= 4 * (3 - 1); the claim demotes and the row stands on
    # its derived origin alone: one exact-count Opaque, two ptr reads saved.
    short_body = Regularity.Span[
        Regularity.Opaque(constant(1)),
        Regularity.Segment(constant(3), coord, nothing),
        Regularity.Opaque(constant(1)),
    ]
    short = Regularity.Description((Regularity.Segment(64, coord, short_body),))
    short_pruned = prune_description(
        short, structure, SpecializePolicy(; visit_density=4))
    @test only(short_pruned.spans) isa Regularity.Segment
    @test only(only(short_pruned.spans).body) isa Regularity.Opaque
    @test RX.structural_reads_avoided(short_pruned, structure) == 2 * 64

    # One concrete span is free under the visit rule, so the band survives.
    band_body = Regularity.Span[
        Regularity.Segment(constant(9), coord, nothing),
    ]
    band = Regularity.Description((Regularity.Segment(64, coord, band_body),))
    band_pruned = prune_description(
        band, structure, SpecializePolicy(; visit_density=4))
    @test only(band_pruned.spans) isa Regularity.Segment
    @test only(only(band_pruned.spans).body) isa Regularity.Segment

    # An originally all-Opaque body still realizes its derived child origin;
    # it is distinct from a failed coordinate claim becoming all-Opaque.
    count_body = Regularity.Span[Regularity.Opaque(constant(5))]
    count_only = Regularity.Description((
        Regularity.Segment(64, coord, count_body),
    ))
    count_pruned = prune_description(
        count_only, structure, SpecializePolicy(; visit_density=4))
    @test only(count_pruned.spans) isa Regularity.Segment
    @test only(only(count_pruned.spans).body) isa Regularity.Opaque
    @test RX.structural_reads_avoided(count_pruned, structure) == 128
end

@testset "Series pruning uses encoded parts and exact spans" begin
    coord = Regularity.PeriodicAffine((1,), 1)
    series_coord = Regularity.Lift(
        coord,
        (
            Regularity.PeriodicAffine((1,), 0),
            Regularity.PeriodicAffine((1,), 0),
        ),
    )
    repeated = Regularity.Series(
        8,
        Regularity.Span[
        Regularity.Segment(
            Regularity.PeriodicAffine((2,), 0),
            series_coord,
            nothing,
        ),
    ],
    )
    description = Regularity.Description((repeated,))
    structure = Regularity.Structure((
        Regularity.CompressedLevel([1, 17], collect(1:16)),
    ))

    # Sixteen claimed children pay for the encoded shell and body once.
    kept =
        prune_description(
            description,
            structure,
            SpecializePolicy(; code_density=8),
        ).spans
    @test length(kept) == 1
    @test only(kept) isa Regularity.Series

    # Demotion preserves the expanded child span, not the repetition count.
    demoted =
        prune_description(
            description,
            structure,
            SpecializePolicy(; code_density=9),
        ).spans
    @test demoted == Regularity.Span[Regularity.Opaque(16)]

    reps = Regularity.PeriodicAffine((2,), 1)
    variable = Regularity.Series(
        reps,
        Regularity.Span[
        Regularity.Opaque(
            Regularity.Lift(
                Regularity.PeriodicAffine((2,), 0),
                (
                    Regularity.PeriodicAffine((2,), 0),
                    Regularity.PeriodicAffine((0,), 0),
                ),
            ),
        ),
    ],
    )
    heavy = Regularity.Segment(100, coord, nothing)
    nested = Regularity.Description((
        Regularity.Segment(
            2,
            coord,
            Regularity.Span[heavy, variable],
        ),
    ))
    nested_structure = Regularity.Structure((
        Regularity.DenseLevel(2),
        Regularity.CompressedLevel(
            [1, 105, 211],
            vcat(collect(1:104), collect(1:106)),
        ),
    ))
    nested_pruned =
        prune_description(
            nested,
            nested_structure,
            SpecializePolicy(; code_density=20),
        ).spans
    parent = only(nested_pruned)
    @test parent isa Regularity.Segment
    @test parent.body[1] isa Regularity.Segment
    variable_demoted = parent.body[2]
    @test variable_demoted isa Regularity.Opaque
    @test variable_demoted.count isa Regularity.Scaled
    for outer in 0:1
        @test Regularity.bind_count(variable_demoted.count, (outer,)) ==
            2 * (2 + outer)
    end

    # A word count that depends on an enclosing ordinal cannot become the
    # integer scale of `Scaled`, and a Series whose body keeps no Segment has
    # nothing for a Stepper to realize: neither lowerable nor demotable, it
    # makes its owner demote instead (the owner's count is exact).
    delayed_count = Regularity.Lift(
        Regularity.PeriodicAffine((2,), 0),
        (
            DelayedPolicyCount(16, 2, 3),
            Regularity.PeriodicAffine((0,), 0),
        ),
    )
    unscalable = Regularity.Series(
        reps,
        Regularity.Span[Regularity.Opaque(delayed_count)],
    )
    guarded = Regularity.Description((
        Regularity.Segment(
            18,
            coord,
            Regularity.Span[heavy, unscalable],
        ),
    ))
    widths = [100 + (2 + ordinal) * (ordinal <= 16 ? 2 : 3) for ordinal in 0:17]
    guarded_ptr = Int[1]
    guarded_idx = Int[]
    for width in widths
        append!(guarded_idx, 1:width)
        push!(guarded_ptr, length(guarded_idx) + 1)
    end
    guarded_structure = Regularity.Structure((
        Regularity.DenseLevel(18),
        Regularity.CompressedLevel(guarded_ptr, guarded_idx),
    ))
    guarded_pruned =
        prune_description(
            guarded,
            guarded_structure,
            SpecializePolicy(; code_density=20),
        ).spans
    @test only(guarded_pruned) == Regularity.Opaque(18)
end

# End-to-end demotion of a whole subtree: columns alternate between heavy
# windows (a 16-row dense block, mined as a top segment whose leaf body
# claims every entry) and light windows (one or three scattered rows,
# alternating so no two neighbors share a leaf skeleton and no template
# lifts: the leaf stays opaque, so their top segment claims interior
# positions but zero leaf entries). Bottom-up scoring demotes each light
# segment WITH its body while the heavy segments survive; emission accepts
# the demoted description and values stay exact.
function nested_mixed_data(; blocks=2, bw=24, width=16)
    data = zeros(Float64, 48, 2 * blocks * bw)
    for j in axes(data, 2)
        if div(j - 1, bw) % 2 == 0
            for i in 1:width
                data[i, j] = (13i + 29j) / 2048
            end
        else
            data[40, j] = (7j + 3) / 512
            if iseven(j)
                data[42, j] = (5j + 1) / 512
                data[45, j] = (3j + 7) / 512
            end
        end
    end
    data
end

@testset "claimed-density nested demotion" begin
    data = nested_mixed_data()
    A = Tensor(Dense(SparseList(Element(0.0))), data)
    x = Tensor(Dense(Element(0.0)), [1.0 + j / 32 for j in axes(data, 2)])
    y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    generic = policy_spmv_code(A, x, y; specialize=false)

    unpruned_report = Ref{F.SpecializeReport}()
    unpruned = policy_spmv_code(A, x, y;
        specialize=true, policy=SpecializePolicy(; pmax=16), report=unpruned_report)
    @test unpruned_report[].realized == 1
    @test unpruned != generic

    # Heavy windows claim 24x16 = 384 leaf entries over ~2 parts and clear
    # 100-per-part; light windows claim zero leaf entries and demote whole.
    demoted_report = Ref{F.SpecializeReport}()
    demoted = policy_spmv_code(A, x, y;
        specialize=true,
        policy=SpecializePolicy(; pmax=16, code_density=100),
        report=demoted_report)
    @test demoted_report[].realized == 1
    @test demoted != generic
    @test demoted != unpruned                    # the light claim is gone
    @test demoted_report[].emitted_switch_cases <=
        unpruned_report[].emitted_switch_cases
    result = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    @finch specialize = true policy = SpecializePolicy(; pmax=16, code_density=100) begin
        result .= 0.0
        for j in _, i in _
            result[i] += A[i, j] * x[j]
        end
    end
    @test ulps_apart(Array(result), data * Array(x)) <= 4
end

# A demoted node in front of a Series whose word LEADS with an Opaque member:
# columns 1-2 carry a light claim and the rest repeat one scattered column
# followed by three width-8 blocks. Fold v2 discovers all six repetitions as
# Series([Opaque, Segment]); pruning demotes only the two-column light Segment
# into the preceding Opaque span. That Opaque phase must stop where the
# Series' first CLAIMED coordinate begins (its Segment member at repetition
# zero), so the Stepper takes over there and the leading Opaque member is
# walked natively inside the chunk.
function leading_opaque_series_data(; reps=6, width=8)
    data = zeros(Float64, 48, 2 + 4reps)
    for j in 1:2, i in 30:31
        data[i, j] = (13i + 29j) / 2048
    end
    for r in 0:(reps - 1)
        j = 3 + 4r
        for i in (1, 7, 20, 41 - r)
            data[i, j] = (7i + 3j) / 512
        end
        for jj in (j + 1):(j + 3), i in 1:width
            data[i, jj] = (13i + 29jj) / 2048
        end
    end
    data
end

@testset "leading-Opaque Series after a demoted node" begin
    data = leading_opaque_series_data()
    structure = series_structure(data; dense_root=true)
    description = Regularity.mine(
        structure;
        schemas=(Regularity.PeriodicAffineConfig(; pmax=16, coordinate_drift=1),),
    )
    pruned =
        prune_description(
            description, structure, SpecializePolicy(; code_density=20)
        ).spans
    @test length(pruned) == 2
    @test pruned[1] isa Regularity.Opaque && pruned[1].count == 2
    series = pruned[2]
    @test series isa Regularity.Series && series.reps == 6
    @test series.body[1] isa Regularity.Opaque
    @test series.body[2] isa Regularity.Segment
    # The Series exposes its first claimed coordinate — the Segment member at
    # repetition zero, column 4 — so the preceding Opaque phase gets a stop.
    @test RX._node_first_coordinate(series, ()) == 4
    @test RX._node_first_coordinate(pruned[1], ()) === nothing

    A = Tensor(Dense(SparseList(Element(0.0))), data)
    x = Tensor(Dense(Element(0.0)), [1.0 + j / 32 for j in axes(data, 2)])
    y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    generic = policy_spmv_code(A, x, y; specialize=false)
    report = Ref{F.SpecializeReport}()
    demoted = policy_spmv_code(A, x, y;
        specialize=true,
        policy=SpecializePolicy(; pmax=16, code_density=20),
        report=report)
    @test report[].realized == 1
    @test demoted != generic
    result = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    @finch specialize = true policy = SpecializePolicy(; pmax=16, code_density=20) begin
        result .= 0.0
        for j in _, i in _
            result[i] += A[i, j] * x[j]
        end
    end
    @test ulps_apart(Array(result), data * Array(x)) <= 4
    # At a compressed level, the preceding Opaque and the Series' leading
    # Opaque have no staged coordinate boundary. The host must use one bounded
    # native walk for that fiber rather than skip the Series member.
    sparse_A = Tensor(SparseList(SparseList(Element(0.0))), data)
    sparse_generic = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    sparse_specialized = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    @finch begin
        sparse_generic .= 0.0
        for j in _, i in _
            sparse_generic[i] += sparse_A[i, j] * x[j]
        end
    end
    @finch specialize = true policy = SpecializePolicy(; pmax=16, code_density=20) begin
        sparse_specialized .= 0.0
        for j in _, i in _
            sparse_specialized[i] += sparse_A[i, j] * x[j]
        end
    end
    @test reinterpret(UInt64, Array(sparse_specialized)) ==
        reinterpret(UInt64, Array(sparse_generic))
    sparse_generic_code = string(@finch_code begin
        sparse_specialized .= 0.0
        for j in _, i in _
            sparse_specialized[i] += sparse_A[i, j] * x[j]
        end
    end)
    sparse_report = Ref{F.SpecializeReport}()
    sparse_code = string(
        @finch_code specialize = true policy =
            SpecializePolicy(; pmax=16, code_density=20) report = sparse_report begin
            sparse_specialized .= 0.0
            for j in _, i in _
                sparse_specialized[i] += sparse_A[i, j] * x[j]
            end
        end
    )
    @test sparse_report[].realized == 0
    @test sparse_report[].structural_reads == 0
    @test sparse_code == sparse_generic_code
end


# Hand-built description pieces. `lift_constant` shares a pattern unchanged
# across one more enclosing ordinal; `member_coord(start, stride)` is a Series
# member's unit-drift coordinate stream whose first coordinate is
# `start + stride * j` in the repetition ordinal.
function lift_constant(child)
    Regularity.Lift(
        child,
        Tuple(Regularity.PeriodicAffine((n,), 0) for n in Regularity.numbers(child)),
    )
end
function member_coord(start, stride)
    Regularity.Lift(
        Regularity.PeriodicAffine((start,), 1),
        (Regularity.PeriodicAffine((start,), stride), Regularity.PeriodicAffine((1,), 0)),
    )
end

# A Series whose exact expanded span is represented family-generically by
# `Scaled` can demote without knowing an interval interpretation for `reps`.
@testset "a Series with no interval span demotes exactly with Scaled" begin
    coord = Regularity.PeriodicAffine((1,), 1)
    word_coord = lift_constant(lift_constant(coord))
    heavy_member = Regularity.Segment(
        lift_constant(Regularity.PeriodicAffine((40,), 0)), word_coord, nothing)
    light_member = Regularity.Segment(
        lift_constant(Regularity.PeriodicAffine((1,), 0)), word_coord, nothing)
    unspanned = Regularity.Series(
        DelayedPolicyCount(16, 3, 3), Regularity.Span[heavy_member, light_member])
    heavy = Regularity.Segment(
        Regularity.PeriodicAffine((2000,), 0), lift_constant(coord), nothing)
    description = Regularity.Description((
        Regularity.Segment(18, coord, Regularity.Span[heavy, unspanned]),
    ))
    width = 2000 + 3 * (40 + 1)
    structure = Regularity.Structure((
        Regularity.DenseLevel(18),
        Regularity.CompressedLevel(
            collect(1:width:(1 + 18width)), repeat(collect(1:width), 18)),
    ))
    span = RX._series_count(unspanned)
    @test span isa Regularity.Scaled
    for outer in 0:17
        @test Regularity.bind_count(span, (outer,)) ==
            Regularity.bind_count(unspanned.reps, (outer,)) * 41
    end

    # The light member demotes first. The enclosing rejected Series then becomes
    # an Opaque Scaled span rather than resurrecting either original claim.
    pruned =
        prune_description(description, structure,
            SpecializePolicy(; code_density=1000)).spans
    parent = only(pruned)
    @test parent isa Regularity.Segment
    @test parent.body[1] == heavy
    demoted = parent.body[2]
    @test demoted isa Regularity.Opaque
    @test demoted.count isa Regularity.Scaled
    for outer in 0:17
        @test Regularity.bind_count(demoted.count, (outer,)) ==
            Regularity.bind_count(unspanned.reps, (outer,)) * 41
    end
end

# A count that throws once the emitter binds it: realization is a
# transaction, so a failed emission leaves the report at zero and the tensor
# on the generic path.
mutable struct EmissionTrapCount <: Regularity.Scalar
    armed::Bool
end
struct EmissionTrap <: Exception end
function Regularity.evaluate(count::EmissionTrapCount, ::Any, ::Regularity.Ops)
    count.armed && throw(EmissionTrap())
    4
end

@testset "realization is counted only after emission succeeds" begin
    trap = EmissionTrapCount(false)
    description = Regularity.Description((
        Regularity.Series(
            2,
            Regularity.Span[
                Regularity.Segment(
                    Regularity.PeriodicAffine((3,), 0), member_coord(1, 10), nothing
                ),
                Regularity.Opaque(trap),
            ],
        ),
    ))
    @test RX._series_count(only(description.spans)) === nothing
    data = zeros(20)
    data[[1:7; 11:17]] .= 1.0
    A = Tensor(SparseList(Element(0.0)), data)

    # Drive the unfurl hook directly with the description planted as the
    # root's candidate, exactly as a mined tensor reaches it.
    function planted_unfurl()
        attempt = F.SpecializationAttempt()
        ctx = F.FinchCompiler(; attempt=attempt)
        root = F.virtualize(ctx.code, :A_lvl, typeof(A.lvl), :A)
        F.stash_concrete!(root, A.lvl, false)
        root.regularity = RX.Candidate(description, nothing)
        function unfurl()
            Finch.regularize_unfurl(
                ctx,
                F.VirtualSubFiber(root, F.literal(1)),
                F.VirtualExtent(F.literal(1), F.literal(length(data))),
                F.FinchNotation.reader(),
                F.defaultread,
            )
        end
        (attempt, root, unfurl)
    end

    attempt, _, unfurl = planted_unfurl()
    @test unfurl() isa F.Thunk
    @test attempt.realized == 1

    trap.armed = true
    attempt, root, unfurl = planted_unfurl()
    @test_throws EmissionTrap unfurl()
    @test attempt.realized == 0
    # The tensor is left to Finch's native traversal once it is declined.
    root.regularity = RX.Declined()
    @test unfurl() === nothing
    @test attempt.realized == 0
    trap.armed = false
end

struct TransactionForeignError <: Exception end

struct TransactionLengthOnly
    n::Int
end

Base.length(list::TransactionLengthOnly) = list.n

mutable struct TransactionNoIterList
    n::Int
    iterated::Bool
end

Base.length(list::TransactionNoIterList) = list.n
function Base.iterate(list::TransactionNoIterList, state...)
    list.iterated = true
    error("Cartesian product iterated before the specialization budget declined")
end

struct TransactionSequenceProbe{L}
    list::L
end

struct TransactionSwitchProbe{L}
    list::L
end

F.get_sequence_phases(ctx, probe::TransactionSequenceProbe, ext) = probe.list
F.get_switch_cases(ctx, probe::TransactionSwitchProbe) = probe.list

mutable struct TransactionReportSink
    writes::Vector{Any}
end

Base.setindex!(sink::TransactionReportSink, report) = push!(sink.writes, report)

function transaction_exception(f)
    try
        f()
    catch exception
        return exception
    end
    nothing
end

function with_transaction_policy(f; limit=nothing)
    old_limit = RX.SPECIALIZE_MAX_EMITTED_PHASES_AND_CASES[]
    try
        limit === nothing || (RX.SPECIALIZE_MAX_EMITTED_PHASES_AND_CASES[] = limit)
        f()
    finally
        RX.SPECIALIZE_MAX_EMITTED_PHASES_AND_CASES[] = old_limit
    end
end

function transaction_spmv_problem(n=64)
    data = unclipped_banded_data(n, 2)
    A = Tensor(Dense(SparseList(Element(0.0))), data)
    x = Tensor(Dense(Element(0.0)), [1.0 + j / 32 for j in 1:n])
    y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    A, x, y, data * Array(x)
end

function transaction_spmv_code(A, x, y; specialize, report=nothing)
    if specialize
        return string(@finch_code specialize = true report = report begin
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

function transaction_spmv_kernel_code(A, x, y; specialize, report=nothing)
    if specialize
        return string(
            @finch_kernel specialize = true report = report function transaction_spmv(
                y, A, x
            )
                y .= 0.0
                for j in _, i in _
                    y[i] += A[i, j] * x[j]
                end
            end
        )
    end
    string(@finch_kernel function transaction_spmv(y, A, x)
        y .= 0.0
        for j in _, i in _
            y[i] += A[i, j] * x[j]
        end
    end)
end

function transaction_spmv_program(A, x, y)
    F.@finch_program_instance begin
        y .= 0.0
        for j in _, i in _
            y[i] += A[i, j] * x[j]
        end
        return y
    end
end

function transaction_pairwise_code(A, B, C; specialize, report=nothing)
    if specialize
        return string(@finch_code specialize = true report = report begin
            C .= 0.0
            for j in _, i in _
                C[i, j] = A[i, j] * B[i, j]
            end
        end)
    end
    string(@finch_code begin
        C .= 0.0
        for j in _, i in _
            C[i, j] = A[i, j] * B[i, j]
        end
    end)
end

function transaction_pairwise_program(A, B, C)
    F.@finch_program_instance begin
        C .= 0.0
        for j in _, i in _
            C[i, j] = A[i, j] * B[i, j]
        end
        return C
    end
end

function transaction_square_code(A, C; specialize, report=nothing)
    if specialize
        return string(@finch_code specialize = true report = report begin
            C .= 0.0
            for j in _, i in _
                C[i, j] = A[i, j] * A[i, j]
            end
        end)
    end
    string(@finch_code begin
        C .= 0.0
        for j in _, i in _
            C[i, j] = A[i, j] * A[i, j]
        end
    end)
end

@testset "specialization transaction" begin
    @testset "separate accounting and exact boundary" begin
        @test fieldnames(F.SpecializeReport) == (
            :realized,
            :emitted_sequence_phases,
            :emitted_switch_cases,
            :declined,
            :reason,
        )

        attempt = F.SpecializationAttempt()
        ctx = F.FinchCompiler(; attempt=attempt)
        F.activate_specialization_budget!(attempt, 5)
        @test F.regularize_charge_sequence_phases!(ctx, 2) === nothing
        @test F.regularize_charge_switch_cases!(ctx, 3) === nothing
        @test attempt.emitted_sequence_phases == 2
        @test attempt.emitted_switch_cases == 3

        exception = transaction_exception() do
            F.regularize_charge_switch_cases!(ctx, 1)
        end
        @test exception isa F.RegularizeDecline
        @test exception.reason === :emitted_phases_and_cases
        @test attempt.emitted_sequence_phases == 2
        @test attempt.emitted_switch_cases == 4

        inactive = F.SpecializationAttempt()
        inactive_ctx = F.FinchCompiler(; attempt=inactive)
        F.regularize_charge_sequence_phases!(inactive_ctx, typemax(Int))
        F.regularize_charge_sequence_phases!(inactive_ctx, 1)
        F.regularize_charge_switch_cases!(inactive_ctx, typemax(Int))
        F.regularize_charge_switch_cases!(inactive_ctx, 1)
        @test inactive.emitted_sequence_phases == typemax(Int)
        @test inactive.emitted_switch_cases == typemax(Int)

        generic_ctx = F.FinchCompiler()
        @test F.regularize_charge_sequence_phases!(generic_ctx, 10) === nothing
        @test F.regularize_charge_switch_cases!(generic_ctx, 10) === nothing
        @test F.regularize_precompose(generic_ctx, (1:100, 1:100)) === nothing
    end

    @testset "both lowering sites charge their own quantity" begin
        sequence_attempt = F.SpecializationAttempt()
        sequence_ctx = F.FinchCompiler(; attempt=sequence_attempt)
        F.activate_specialization_budget!(sequence_attempt, 1)
        sequence_probe = TransactionSequenceProbe([
            [] => F.literal(1),
            [] => F.literal(2),
        ])
        sequence_root = F.loop(
            F.index(:i), F.literal(nothing), F.virtual(sequence_probe))
        exception = transaction_exception() do
            F.lower(sequence_ctx, sequence_root, F.SequenceStyle())
        end
        @test exception isa F.RegularizeDecline
        @test exception.reason === :emitted_phases_and_cases
        @test sequence_attempt.emitted_sequence_phases == 2
        @test sequence_attempt.emitted_switch_cases == 0

        switch_attempt = F.SpecializationAttempt()
        switch_ctx = F.FinchCompiler(; attempt=switch_attempt)
        F.activate_specialization_budget!(switch_attempt, 1)
        switch = F.Switch([
            F.literal(true) => F.literal(1),
            F.literal(true) => F.literal(2),
        ])
        exception = transaction_exception() do
            F.lower(switch_ctx, switch, F.SwitchStyle())
        end
        @test exception isa F.RegularizeDecline
        @test exception.reason === :emitted_phases_and_cases
        @test switch_attempt.emitted_sequence_phases == 0
        @test switch_attempt.emitted_switch_cases == 2
    end

    @testset "Cartesian products are guarded before materialization" begin
        attempt = F.SpecializationAttempt()
        ctx = F.FinchCompiler(; attempt=attempt)
        F.activate_specialization_budget!(attempt, 7)
        F.regularize_charge_sequence_phases!(ctx, 2)
        F.regularize_charge_switch_cases!(ctx, 1)
        @test F.regularize_precompose(ctx, (1:2, 1:2)) === nothing
        exception = transaction_exception() do
            F.regularize_precompose(ctx, (1:5,))
        end
        @test exception isa F.RegularizeDecline
        @test exception.reason === :emitted_phases_and_cases

        sequence_attempt = F.SpecializationAttempt()
        sequence_ctx = F.FinchCompiler(; attempt=sequence_attempt)
        F.activate_specialization_budget!(sequence_attempt, 3)
        sequence_left = TransactionNoIterList(2, false)
        sequence_right = TransactionNoIterList(2, false)
        sequence_node = F.call(
            +,
            F.virtual(TransactionSequenceProbe(sequence_left)),
            F.virtual(TransactionSequenceProbe(sequence_right)),
        )
        exception = transaction_exception() do
            F.get_sequence_phases(sequence_ctx, sequence_node, nothing)
        end
        @test exception isa F.RegularizeDecline
        @test exception.reason === :emitted_phases_and_cases
        @test !sequence_left.iterated
        @test !sequence_right.iterated

        switch_attempt = F.SpecializationAttempt()
        switch_ctx = F.FinchCompiler(; attempt=switch_attempt)
        F.activate_specialization_budget!(switch_attempt, 3)
        switch_left = TransactionNoIterList(2, false)
        switch_right = TransactionNoIterList(2, false)
        switch_node = F.call(
            +,
            F.virtual(TransactionSwitchProbe(switch_left)),
            F.virtual(TransactionSwitchProbe(switch_right)),
        )
        exception = transaction_exception() do
            F.get_switch_cases(switch_ctx, switch_node)
        end
        @test exception isa F.RegularizeDecline
        @test exception.reason === :emitted_phases_and_cases
        @test !switch_left.iterated
        @test !switch_right.iterated
    end

    @testset "maximum-limit equality distinguishes exact from saturated" begin
        exact_product_attempt = F.SpecializationAttempt()
        exact_product_ctx = F.FinchCompiler(; attempt=exact_product_attempt)
        F.activate_specialization_budget!(exact_product_attempt, typemax(Int))
        @test F.regularize_precompose(
            exact_product_ctx, (TransactionLengthOnly(typemax(Int)),)) === nothing

        attempt = F.SpecializationAttempt()
        ctx = F.FinchCompiler(; attempt=attempt)
        F.activate_specialization_budget!(attempt, typemax(Int))
        lists = (
            TransactionLengthOnly(typemax(Int)),
            TransactionLengthOnly(2),
        )
        exception = transaction_exception() do
            F.regularize_precompose(ctx, lists)
        end
        @test exception isa F.RegularizeDecline
        @test exception.reason === :emitted_phases_and_cases

        exact_sum = F.SpecializationAttempt()
        exact_sum_ctx = F.FinchCompiler(; attempt=exact_sum)
        F.regularize_charge_sequence_phases!(exact_sum_ctx, typemax(Int) - 1)
        F.regularize_charge_switch_cases!(exact_sum_ctx, 1)
        @test F.activate_specialization_budget!(exact_sum, typemax(Int)) === nothing

        active_exact_sum = F.SpecializationAttempt()
        active_exact_sum_ctx = F.FinchCompiler(; attempt=active_exact_sum)
        F.activate_specialization_budget!(active_exact_sum, typemax(Int))
        F.regularize_charge_sequence_phases!(
            active_exact_sum_ctx, typemax(Int) - 1)
        exception = transaction_exception() do
            F.regularize_charge_switch_cases!(active_exact_sum_ctx, 1)
        end
        @test exception === nothing

        saturated_sum = F.SpecializationAttempt()
        saturated_sum_ctx = F.FinchCompiler(; attempt=saturated_sum)
        F.regularize_charge_sequence_phases!(saturated_sum_ctx, typemax(Int))
        F.regularize_charge_sequence_phases!(saturated_sum_ctx, 1)
        exception = transaction_exception() do
            F.activate_specialization_budget!(saturated_sum, typemax(Int))
        end
        @test exception isa F.RegularizeDecline &&
            exception.reason === :emitted_phases_and_cases

        saturated_cross_sum = F.SpecializationAttempt()
        saturated_cross_sum_ctx = F.FinchCompiler(; attempt=saturated_cross_sum)
        F.regularize_charge_sequence_phases!(saturated_cross_sum_ctx, typemax(Int))
        F.regularize_charge_switch_cases!(saturated_cross_sum_ctx, 1)
        exception = transaction_exception() do
            F.activate_specialization_budget!(saturated_cross_sum, typemax(Int))
        end
        @test exception isa F.RegularizeDecline &&
            exception.reason === :emitted_phases_and_cases
    end

    @testset "reports, retries, and error hygiene" begin
        completed_report = Ref{F.SpecializeReport}()
        completed = F.specialize_compile(; report=completed_report) do ctx
            attempt = F.specialization_attempt(ctx)
            F.activate_specialization_budget!(attempt, 5)
            F.regularize_charge_sequence_phases!(ctx, 2)
            F.regularize_charge_switch_cases!(ctx, 3)
            :specialized
        end
        @test completed === :specialized
        @test completed_report[].realized == 1
        @test completed_report[].emitted_sequence_phases == 2
        @test completed_report[].emitted_switch_cases == 3
        @test !completed_report[].declined
        @test completed_report[].reason === nothing

        calls = 0
        attempts = Any[]
        declined_report = Ref{F.SpecializeReport}()
        rebuilt = F.specialize_compile(; report=declined_report) do ctx
            calls += 1
            push!(attempts, F.specialization_attempt(ctx))
            if calls == 1
                F.regularize_charge_sequence_phases!(ctx, 2)
                F.regularize_charge_switch_cases!(ctx, 2)
                F.activate_specialization_budget!(F.specialization_attempt(ctx), 3)
            end
            :generic
        end
        @test rebuilt === :generic
        @test calls == 2
        @test attempts[1] isa F.SpecializationAttempt
        @test attempts[2] === nothing
        @test declined_report[].realized == 1
        @test declined_report[].emitted_sequence_phases == 2
        @test declined_report[].emitted_switch_cases == 2
        @test declined_report[].declined
        @test declined_report[].reason === :emitted_phases_and_cases

        inactive_budget_calls = 0
        inactive_budget_report = Ref{F.SpecializeReport}()
        result = F.specialize_compile(; report=inactive_budget_report) do ctx
            inactive_budget_calls += 1
            F.regularize_charge_sequence_phases!(ctx, 10_000)
            F.regularize_charge_switch_cases!(ctx, 10_000)
            :ordinary
        end
        @test result === :ordinary
        @test inactive_budget_calls == 1
        @test inactive_budget_report[].realized == 0
        @test inactive_budget_report[].emitted_sequence_phases == 10_000
        @test inactive_budget_report[].emitted_switch_cases == 10_000
        @test !inactive_budget_report[].declined

        foreign_report = Ref{F.SpecializeReport}()
        exception = transaction_exception() do
            F.specialize_compile(; report=foreign_report) do ctx
                throw(TransactionForeignError())
            end
        end
        @test exception isa TransactionForeignError
        @test !isassigned(foreign_report)

        retry_calls = 0
        retry_report = Ref{F.SpecializeReport}()
        exception = transaction_exception() do
            F.specialize_compile(; report=retry_report) do ctx
                retry_calls += 1
                if retry_calls == 1
                    F.regularize_charge_sequence_phases!(ctx, 2)
                    F.regularize_charge_switch_cases!(ctx, 3)
                    F.activate_specialization_budget!(F.specialization_attempt(ctx), 4)
                end
                throw(TransactionForeignError())
            end
        end
        @test exception isa TransactionForeignError
        @test retry_calls == 2
        @test isassigned(retry_report)
        @test retry_report[].realized == 1
        @test retry_report[].emitted_sequence_phases == 2
        @test retry_report[].emitted_switch_cases == 3
        @test retry_report[].declined
        @test retry_report[].reason === :emitted_phases_and_cases

        completed_sink = TransactionReportSink(Any[])
        @test F.specialize_compile(; report=completed_sink) do ctx
            :completed
        end === :completed
        @test length(completed_sink.writes) == 1

        declined_sink = TransactionReportSink(Any[])
        declined_calls = 0
        @test F.specialize_compile(; report=declined_sink) do ctx
            declined_calls += 1
            declined_calls == 1 && throw(F.RegularizeDecline(:test_decline))
            :rebuilt
        end === :rebuilt
        @test length(declined_sink.writes) == 1
        @test only(declined_sink.writes).reason === :test_decline
    end

    @testset "attempt state is isolated and propagated explicitly" begin
        attempt = F.SpecializationAttempt()
        ctx = F.FinchCompiler(; attempt=attempt)
        F.contain(ctx) do contained
            @test F.specialization_attempt(contained) === attempt
            F.open_scope(contained) do scoped
                @test F.specialization_attempt(scoped) === attempt
            end
        end

        outer_report = Ref{F.SpecializeReport}()
        inner_report = Ref{F.SpecializeReport}()
        outer_attempt = Ref{Any}()
        inner_attempt = Ref{Any}()
        result = F.specialize_compile(; report=outer_report) do outer_ctx
            outer_attempt[] = F.specialization_attempt(outer_ctx)
            inner = F.specialize_compile(; report=inner_report) do inner_ctx
                inner_attempt[] = F.specialization_attempt(inner_ctx)
                F.activate_specialization_budget!(inner_attempt[], 3)
                F.regularize_charge_switch_cases!(inner_ctx, 2)
                :inner
            end
            @test inner === :inner
            F.activate_specialization_budget!(outer_attempt[], 5)
            F.regularize_charge_sequence_phases!(outer_ctx, 4)
            :outer
        end
        @test result === :outer
        @test outer_attempt[] !== inner_attempt[]
        @test outer_report[].emitted_sequence_phases == 4
        @test outer_report[].emitted_switch_cases == 0
        @test inner_report[].emitted_sequence_phases == 0
        @test inner_report[].emitted_switch_cases == 2

        ready = Channel{Int}(2)
        release = Channel{Nothing}(2)
        concurrent_attempts = Vector{Any}(undef, 2)
        concurrent_reports = [Ref{F.SpecializeReport}() for _ in 1:2]
        tasks = map(1:2) do id
            @async F.specialize_compile(; report=concurrent_reports[id]) do task_ctx
                task_attempt = F.specialization_attempt(task_ctx)
                concurrent_attempts[id] = task_attempt
                F.activate_specialization_budget!(task_attempt, 10)
                put!(ready, id)
                take!(release)
                F.regularize_charge_sequence_phases!(task_ctx, id)
                F.regularize_charge_switch_cases!(task_ctx, id + 1)
                id
            end
        end
        seen = sort([take!(ready), take!(ready)])
        put!(release, nothing)
        put!(release, nothing)
        @test seen == [1, 2]
        @test fetch.(tasks) == [1, 2]
        @test concurrent_attempts[1] !== concurrent_attempts[2]
        for id in 1:2
            @test concurrent_reports[id][].emitted_sequence_phases == id
            @test concurrent_reports[id][].emitted_switch_cases == id + 1
            @test !concurrent_reports[id][].declined
        end
    end

    @testset "all specialized entry points discard a declined attempt" begin
        A, x, y, expected = transaction_spmv_problem()
        generic_code = transaction_spmv_code(A, x, y; specialize=false)
        code_report = Ref{F.SpecializeReport}()
        declined_code = with_transaction_policy(; limit=0) do
            transaction_spmv_code(A, x, y; specialize=true, report=code_report)
        end
        @test declined_code == generic_code
        @test code_report[].declined
        @test code_report[].realized >= 1
        @test code_report[].reason === :emitted_phases_and_cases

        generic_kernel = transaction_spmv_kernel_code(A, x, y; specialize=false)
        kernel_report = Ref{F.SpecializeReport}()
        declined_kernel = with_transaction_policy(; limit=0) do
            transaction_spmv_kernel_code(
                A, x, y; specialize=true, report=kernel_report)
        end
        @test declined_kernel == generic_kernel
        @test kernel_report[].declined
        @test kernel_report[].realized >= 1
        @test kernel_report[].reason === :emitted_phases_and_cases

        execute_report = Ref{F.SpecializeReport}()
        result = with_transaction_policy(; limit=0) do
            F.execute_specialized(
                transaction_spmv_program(A, x, y); report=execute_report)
        end
        @test ulps_apart(Array(result.y), expected) <= 4
        @test execute_report[].declined
        @test execute_report[].realized >= 1
        @test execute_report[].reason === :emitted_phases_and_cases

        supplied_ctx = F.FinchCompiler()
        @test_throws ArgumentError F.finch_kernel(
            :transaction_ctx_reuse,
            Any[:y => y, :A => A, :x => x],
            typeof(transaction_spmv_program(A, x, y));
            specialize=true,
            ctx=supplied_ctx,
        )
    end

    @testset "real emissions set the budget boundary" begin
        A, x, y, _ = transaction_spmv_problem()
        high_report = Ref{F.SpecializeReport}()
        high_code = with_transaction_policy(; limit=typemax(Int)) do
            transaction_spmv_code(A, x, y; specialize=true, report=high_report)
        end
        @test !high_report[].declined
        @test high_report[].realized >= 1
        @test high_report[].emitted_sequence_phases > 0
        total = high_report[].emitted_sequence_phases +
                high_report[].emitted_switch_cases
        @test total > 0

        exact_report = Ref{F.SpecializeReport}()
        exact_code = with_transaction_policy(; limit=total) do
            transaction_spmv_code(A, x, y; specialize=true, report=exact_report)
        end
        @test !exact_report[].declined
        @test exact_code == high_code
        @test exact_report[].emitted_sequence_phases ==
            high_report[].emitted_sequence_phases
        @test exact_report[].emitted_switch_cases == high_report[].emitted_switch_cases

        tight_report = Ref{F.SpecializeReport}()
        tight_code = with_transaction_policy(; limit=total - 1) do
            transaction_spmv_code(A, x, y; specialize=true, report=tight_report)
        end
        @test tight_report[].declined
        @test tight_report[].reason === :emitted_phases_and_cases
        @test tight_code == transaction_spmv_code(A, x, y; specialize=false)
    end

    @testset "budget remains inactive without a realization" begin
        data = zeros(32, 32)
        for j in 1:32
            data[1, j] = j
            data[2 + mod(7j, 29), j] = -j
        end
        A = Tensor(Dense(SparseList(Element(0.0))), data)
        x = Tensor(Dense(Element(0.0)), ones(32))
        y = Tensor(Dense(Element(0.0)), zeros(32))
        generic = transaction_spmv_code(A, x, y; specialize=false)
        report = Ref{F.SpecializeReport}()
        inert = with_transaction_policy(; limit=0) do
            transaction_spmv_code(A, x, y; specialize=true, report=report)
        end
        @test inert == generic
        @test report[].realized == 0
        @test !report[].declined
        @test report[].reason === nothing

        unused_report = Ref{F.SpecializeReport}()
        @finch_code report = unused_report begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end
        @test !isassigned(unused_report)
    end

    @testset "independent multi-sparse admission is bounded by actual emissions" begin
        data_a = zeros(70, 64)
        raw_a = unclipped_banded_data(64, 2)
        data_a[axes(raw_a, 1), axes(raw_a, 2)] .= raw_a
        data_b = unclipped_banded_data(64, 3)
        A = Tensor(Dense(SparseList(Element(0.0))), data_a)
        B = Tensor(Dense(SparseList(Element(0.0))), data_b)
        C = Tensor(Dense(Dense(Element(0.0))), zeros(70, 64))
        generic = transaction_pairwise_code(A, B, C; specialize=false)

        admitted_report = Ref{F.SpecializeReport}()
        admitted = transaction_pairwise_code(
            A, B, C; specialize=true, report=admitted_report)
        @test admitted != generic
        @test admitted_report[].realized >= 2
        @test !admitted_report[].declined

        generic_output = Tensor(Dense(Dense(Element(0.0))), zeros(70, 64))
        @finch begin
            generic_output .= 0.0
            for j in _, i in _
                generic_output[i, j] = A[i, j] * B[i, j]
            end
        end
        specialized_output = Tensor(Dense(Dense(Element(0.0))), zeros(70, 64))
        execute_report = Ref{F.SpecializeReport}()
        specialized_result = F.execute_specialized(
            transaction_pairwise_program(A, B, specialized_output);
            report=execute_report,
        )
        reference = data_a .* data_b
        @test ulps_apart(Array(specialized_result.C), Array(generic_output)) <= 4
        @test ulps_apart(Array(specialized_result.C), reference) <= 4
        @test execute_report[].realized >= 2
        @test !execute_report[].declined

        admitted_total =
            admitted_report[].emitted_sequence_phases +
            admitted_report[].emitted_switch_cases
        @test 0 < admitted_total <= RX.SPECIALIZE_MAX_EMITTED_PHASES_AND_CASES[]
        tight_report = Ref{F.SpecializeReport}()
        tight = with_transaction_policy(; limit=admitted_total - 1) do
            transaction_pairwise_code(A, B, C; specialize=true, report=tight_report)
        end
        @test tight == generic
        @test tight_report[].declined
        @test tight_report[].realized >= 2
        @test tight_report[].reason === :emitted_phases_and_cases
        @test tight_report[].emitted_sequence_phases +
              tight_report[].emitted_switch_cases > 0

        same_root_report = Ref{F.SpecializeReport}()
        same_root = transaction_square_code(
            A, C; specialize=true, report=same_root_report)
        same_root_generic = transaction_square_code(A, C; specialize=false)
        @test same_root != same_root_generic
        @test same_root_report[].realized == 1
        @test !same_root_report[].declined

        same_root_total =
            same_root_report[].emitted_sequence_phases +
            same_root_report[].emitted_switch_cases
        @test same_root_total > 0
        same_root_tight_report = Ref{F.SpecializeReport}()
        same_root_tight = with_transaction_policy(; limit=same_root_total - 1) do
            transaction_square_code(
                A, C; specialize=true, report=same_root_tight_report)
        end
        @test same_root_tight == same_root_generic
        @test same_root_tight_report[].declined
        @test same_root_tight_report[].reason === :emitted_phases_and_cases
    end
end

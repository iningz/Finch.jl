struct TransactionForeignError <: Exception end

function transaction_spmv_problem(n=64)
    data = unclipped_banded_data(n, 2)
    A = Tensor(Dense(SparseList(Element(0.0))), data)
    x = Tensor(Dense(Element(0.0)), [1.0 + j / 32 for j in 1:n])
    y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    A, x, y, data * Array(x)
end

function transaction_spmv_code(
    A, x, y; specialize, policy=SpecializePolicy(), report=nothing
)
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

function transaction_spmv_kernel_code(
    A, x, y; policy=SpecializePolicy(), report
)
    string(
        @finch_kernel specialize = true policy = policy report = report function transaction_spmv(
            y, A, x
        )
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end
    )
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

function transaction_pairwise_program(A, B, C)
    F.@finch_program_instance begin
        C .= 0.0
        for j in _, i in _
            C[i, j] = A[i, j] * B[i, j]
        end
        return C
    end
end

@testset "specialization transaction" begin
    @testset "charges accrue on the attempt and are inert elsewhere" begin
        @test fieldnames(F.SpecializeReport) ==
            (:realized, :emitted_sequence_phases, :emitted_switch_cases,
             :abstract_spans, :concrete_spans, :structural_reads)

        attempt = F.SpecializationAttempt()
        @test attempt.policy == F.SpecializePolicy()
        ctx = F.FinchCompiler(; attempt=attempt)
        @test F.regularize_charge_sequence_phases!(ctx, 2) === nothing
        @test F.regularize_charge_switch_cases!(ctx, 3) === nothing
        F.regularize_charge_switch_cases!(ctx, 1)
        @test attempt.sequence_phases == 2
        @test attempt.switch_cases == 4
        @test F.SpecializeReport(attempt) == F.SpecializeReport(0, 2, 4, 0, 0, 0)

        generic_ctx = F.FinchCompiler()
        @test F.regularize_charge_sequence_phases!(generic_ctx, 10) === nothing
        @test F.regularize_charge_switch_cases!(generic_ctx, 10) === nothing
    end

    @testset "switch lowering charges its emitted cases" begin
        attempt = F.SpecializationAttempt()
        ctx = F.FinchCompiler(; attempt=attempt)
        switch = F.Switch([
            F.literal(true) => F.literal(1),
            F.literal(true) => F.literal(2),
        ])
        F.lower(ctx, switch, F.SwitchStyle())
        @test attempt.switch_cases == 2
        @test attempt.sequence_phases == 0
    end

    @testset "every specialized entry point writes a report" begin
        A, x, y, expected = transaction_spmv_problem()
        generic_code = transaction_spmv_code(A, x, y; specialize=false)

        code_report = Ref{F.SpecializeReport}()
        specialized_code = transaction_spmv_code(
            A, x, y; specialize=true, report=code_report)
        @test specialized_code != generic_code
        @test code_report[].realized >= 1
        @test code_report[].emitted_sequence_phases > 0

        kernel_report = Ref{F.SpecializeReport}()
        transaction_spmv_kernel_code(A, x, y; report=kernel_report)
        @test kernel_report[].realized >= 1
        @test kernel_report[].emitted_sequence_phases > 0

        execute_report = Ref{F.SpecializeReport}()
        result = F.execute_specialized(
            transaction_spmv_program(A, x, y); report=execute_report)
        @test ulps_apart(Array(result.y), expected) <= 4
        @test execute_report[].realized >= 1

        unwritten = F.SpecializeReport(-1, -1, -1, 0, 0, 0)
        unused_report = Ref(unwritten)
        @finch_code report = unused_report begin
            y .= 0.0
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end
        @test unused_report[] === unwritten

        supplied_ctx = F.FinchCompiler()
        @test_throws ArgumentError F.finch_kernel(
            :transaction_ctx_reuse,
            Any[:y => y, :A => A, :x => x],
            typeof(transaction_spmv_program(A, x, y));
            specialize=true,
            ctx=supplied_ctx,
        )
    end

    @testset "counters describe emitted code structure, not data size" begin
        small_A, small_x, small_y, _ = transaction_spmv_problem(64)
        large_A, large_x, large_y, _ = transaction_spmv_problem(4096)
        small_report = Ref{F.SpecializeReport}()
        large_report = Ref{F.SpecializeReport}()
        transaction_spmv_code(
            small_A, small_x, small_y; specialize=true, report=small_report)
        transaction_spmv_code(
            large_A, large_x, large_y; specialize=true, report=large_report)
        @test small_report[].realized == large_report[].realized >= 1
        @test small_report[].emitted_sequence_phases ==
            large_report[].emitted_sequence_phases > 0
        @test small_report[].emitted_switch_cases ==
            large_report[].emitted_switch_cases
    end

    @testset "every realization is counted" begin
        data_a = zeros(70, 64)
        raw_a = unclipped_banded_data(64, 2)
        data_a[axes(raw_a, 1), axes(raw_a, 2)] .= raw_a
        data_b = unclipped_banded_data(64, 3)
        A = Tensor(Dense(SparseList(Element(0.0))), data_a)
        B = Tensor(Dense(SparseList(Element(0.0))), data_b)
        C = Tensor(Dense(Dense(Element(0.0))), zeros(70, 64))

        execute_report = Ref{F.SpecializeReport}()
        specialized_result = F.execute_specialized(
            transaction_pairwise_program(A, B, C); report=execute_report)
        @test ulps_apart(Array(specialized_result.C), data_a .* data_b) <= 4
        @test execute_report[].realized >= 2
    end

    @testset "a foreign error propagates without a report" begin
        calls = 0
        unwritten = F.SpecializeReport(-1, -1, -1, 0, 0, 0)
        foreign_report = Ref(unwritten)
        exception = try
            F.specialize_compile(; report=foreign_report) do ctx
                calls += 1
                throw(TransactionForeignError())
            end
            nothing
        catch caught
            caught
        end
        @test exception isa TransactionForeignError
        @test calls == 1
        @test foreign_report[] === unwritten
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
        @test F.specialization_attempt(F.FinchCompiler()) === nothing

        outer_report = Ref{F.SpecializeReport}()
        inner_report = Ref{F.SpecializeReport}()
        result = F.specialize_compile(; report=outer_report) do outer_ctx
            inner = F.specialize_compile(; report=inner_report) do inner_ctx
                F.regularize_charge_switch_cases!(inner_ctx, 2)
                :inner
            end
            @test inner === :inner
            F.regularize_charge_sequence_phases!(outer_ctx, 4)
            :outer
        end
        @test result === :outer
        @test outer_report[] == F.SpecializeReport(0, 4, 0, 0, 0, 0)
        @test inner_report[] == F.SpecializeReport(0, 0, 2, 0, 0, 0)
    end
end

@testset "extension absent" begin
    @test Base.get_extension(Finch, :RegularityExt) === nothing
    @test isdefined(Finch, :mine_regular_structure!)
    @test isdefined(Finch, :regularize_unfurl)

    data = [1.0 0.0 2.0; 0.0 3.0 0.0; 4.0 0.0 5.0]
    A = Tensor(Dense(SparseList(Element(0.0))), data)
    x = Tensor(Dense(Element(0.0)), ones(3))
    y = Tensor(Dense(Element(0.0)), zeros(3))

    generic = string(@finch_code begin
        y .= 0.0
        for j in _, i in _
            y[i] += A[i, j] * x[j]
        end
    end)
    report = Ref{Finch.SpecializeReport}()
    inert = string(@finch_code specialize = true report = report begin
        y .= 0.0
        for j in _, i in _
            y[i] += A[i, j] * x[j]
        end
    end)

    @test inert == generic
    @test isassigned(report)
    @test report[].realized == 0
    @test report[].emitted_sequence_phases > 0
end

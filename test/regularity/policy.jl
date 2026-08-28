# The periodic-affine family from the evaluation's `periodic` generator:
# constant width-4 fibers whose start follows a period-8 word
# (start = 1 + r^2 + q*(p^2 + 3) for fiber ordinal j-1 = q*p + r), so the
# structure is recognizable exactly when the policy's period search covers the
# period. The last column breaks the word so that no claim covers the whole
# level — a claim covering the whole input is exempt from `min_run` by design.
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

@testset "specialization policy" begin
    @test SpecializePolicy() == SpecializePolicy(8, 1, 1)
    @test SpecializePolicy(pmax=16, min_run=4, leaf_min_run=2) ==
        SpecializePolicy(16, 4, 2)
    @test_throws ArgumentError SpecializePolicy(pmax=0)
    @test_throws ArgumentError SpecializePolicy(min_run=0)
    @test_throws ArgumentError SpecializePolicy(leaf_min_run=0)

    data = policy_periodic_data()
    A = Tensor(Dense(SparseList(Element(0.0))), data)
    x = Tensor(Dense(Element(0.0)), [1.0 + j / 32 for j in axes(data, 2)])
    y = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    generic = policy_spmv_code(A, x, y; specialize=false)

    # A period search that covers the word realizes the structure.
    deep_report = Ref{F.SpecializeReport}()
    deep = policy_spmv_code(A, x, y;
        specialize=true, policy=SpecializePolicy(pmax=16), report=deep_report)
    @test deep != generic
    @test deep_report[].realized >= 1

    # A period search below the word realizes nothing and emits generic code.
    shallow_report = Ref{F.SpecializeReport}()
    shallow = policy_spmv_code(A, x, y;
        specialize=true, policy=SpecializePolicy(pmax=4), report=shallow_report)
    @test shallow_report[].realized == 0
    @test shallow == generic

    # A run floor above every claim keeps the whole level opaque.
    coarse_report = Ref{F.SpecializeReport}()
    coarse = policy_spmv_code(A, x, y;
        specialize=true, policy=SpecializePolicy(min_run=10^9), report=coarse_report)
    @test coarse_report[].realized == 0
    @test coarse == generic

    # `@finch` forwards the policy through `execute` to `execute_specialized`.
    result = Tensor(Dense(Element(0.0)), zeros(size(data, 1)))
    @finch specialize = true policy = SpecializePolicy(pmax=16) begin
        result .= 0.0
        for j in _, i in _
            result[i] += A[i, j] * x[j]
        end
    end
    @test ulps_apart(Array(result), data * Array(x)) <= 4
end

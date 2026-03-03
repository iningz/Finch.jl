using Finch

println("="^70)
println("  Staged Dense Size Specialization")
println("="^70)

println("\n── 1D Dense vector (size 10) ──\n")

A = Tensor(Dense(Element(0.0)), rand(10))
s = Tensor(Element(0.0))

println("Normal @finch_code:")
println(@finch_code (
    begin
        s .= 0
        for i in _
            s[] += A[i]
        end
        return s
    end
))

sA = specialize(A)
println("\nSpecialized @finch_code:")
println(@finch_code (
    begin
        s .= 0
        for i in _
            s[] += sA[i]
        end
        return s
    end
))

# Correctness test
s1 = Tensor(Element(0.0))
s2 = Tensor(Element(0.0))
@finch (
    begin
        s1 .= 0
        for i in _
            s1[] += A[i]
        end
        return s1
    end
)
@finch (
    begin
        s2 .= 0
        for i in _
            s2[] += sA[i]
        end
        return s2
    end
)
@assert s1[] ≈ s2[] "Results must match: $(s1[]) vs $(s2[])"
println("\n Results match: $(s1[]) = $(s2[])")

# ── 2D Dense matrix ─────────────────────────────────────────────────────────

println("\n\n── 2D Dense matrix (3×4) ──\n")

B = Tensor(Dense(Dense(Element(0.0))), rand(3, 4))
t = Tensor(Element(0.0))

println("Normal @finch_code:")
println(@finch_code (
    begin
        t .= 0
        for j in _, i in _
            t[] += B[i, j]
        end
        return t
    end
))

sB = specialize(B)
println("\nSpecialized @finch_code:")
println(@finch_code (
    begin
        t .= 0
        for j in _, i in _
            t[] += sB[i, j]
        end
        return t
    end
))

# Correctness test
t1 = Tensor(Element(0.0))
t2 = Tensor(Element(0.0))
@finch (
    begin
        t1 .= 0
        for j in _, i in _
            t1[] += B[i, j]
        end
        return t1
    end
)
@finch (
    begin
        t2 .= 0
        for j in _, i in _
            t2[] += sB[i, j]
        end
        return t2
    end
)
@assert t1[] ≈ t2[] "Results must match: $(t1[]) vs $(t2[])"
println("\n Results match: $(t1[]) = $(t2[])")

# ── Dense outer + SparseList inner ──────────────────────────────────────────

println("\n\n── Dense(SparseList(Element)) matrix (5×3) ──\n")

C = Tensor(Dense(SparseList(Element(0.0))), [1.0 0 2; 0 3 0; 4 0 5; 0 6 0; 7 0 8])
u = Tensor(Element(0.0))

println("Normal @finch_code:")
println(@finch_code (
    begin
        u .= 0
        for j in _, i in _
            u[] += C[i, j]
        end
        return u
    end
))

sC = specialize(C)
println("\nSpecialized @finch_code (outer Dense specialized):")
println(@finch_code (
    begin
        u .= 0
        for j in _, i in _
            u[] += sC[i, j]
        end
        return u
    end
))

# Correctness test
u1 = Tensor(Element(0.0))
u2 = Tensor(Element(0.0))
@finch (
    begin
        u1 .= 0
        for j in _, i in _
            u1[] += C[i, j]
        end
        return u1
    end
)
@finch (
    begin
        u2 .= 0
        for j in _, i in _
            u2[] += sC[i, j]
        end
        return u2
    end
)
@assert u1[] ≈ u2[] "Results must match: $(u1[]) vs $(u2[])"
println("\n Results match: $(u1[]) = $(u2[])")

println("\n" * "="^70)
println("  All tests passed!")
println("="^70)

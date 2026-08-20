function issafe(mode)
    if mode === :debug
        return true
    elseif mode === :safe
        return true
    elseif mode === :fast
        return false
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
end

"""
    instantiate!(ctx, prgm)

A transformation to call `instantiate` on tensors before executing an
expression.
"""
function instantiate!(ctx, prgm)
    prgm = InstantiateTensors(; ctx=ctx)(prgm)
    return prgm
end

@kwdef struct InstantiateTensors{Ctx}
    ctx::Ctx
    escape = Set()
end

function (ctx::InstantiateTensors)(node::FinchNode)
    if node.kind === block
        block(map(ctx, node.bodies)...)
    elseif node.kind === define
        push!(ctx.escape, node.lhs)
        define(node.lhs, ctx(node.rhs), ctx(node.body))
    elseif node.kind === declare
        push!(ctx.escape, node.tns)
        node
    elseif node.kind === freeze
        push!(ctx.escape, node.tns)
        node
    elseif node.kind === thaw
        push!(ctx.escape, node.tns)
        node
    elseif (@capture node access(~tns, ~mode, ~idxs...)) && !(getroot(tns) in ctx.escape)
        #@assert get(ctx.ctx.modes, tns, reader()) === node.mode
        tns_2 = instantiate(ctx.ctx, tns, mode)
        access(tns_2, mode, idxs...)
    elseif istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end

getvalue(::Type{Val{v}}) where {v} = v

"""
    lower_global(ctx, prgm)

lower the program `prgm` at global scope in the context `ctx`.
"""
function lower_global(ctx, prgm)
    prgm = exit_on_yieldbind(prgm)
    prgm = enforce_scopes(prgm)
    prgm = evaluate_partial(ctx, prgm)
    # Give an extension a deterministic pre-lowering view of every bound tensor.
    # Concrete storage is available only when a specialized entry point stashed
    # it on the virtual levels.
    for var in sort!(collect(keys(ctx.scope.bindings)); by=string)
        bound = ctx.scope.bindings[var]
        if bound.kind === virtual && bound.val isa VirtualFiber
            mine_regular_structure!(ctx, bound.val)
        end
    end
    code = contain(ctx) do ctx_2
        quote
            $(
                begin
                    prgm = wrapperize(ctx_2, prgm)
                    prgm = enforce_lifecycles(prgm)
                    prgm = dimensionalize!(prgm, ctx_2)
                    prgm = concordize(ctx_2, prgm)
                    prgm = evaluate_partial(ctx_2, prgm)
                    prgm = simplify(ctx_2, prgm) #appears necessary
                    prgm = instantiate!(ctx_2, prgm)
                    contain(ctx_2) do ctx_3
                        ctx_3(prgm)
                    end
                end
            )
            $(get_result(ctx))
        end
    end
end

function execute_code(
    ex,
    T;
    algebra=DefaultAlgebra(),
    mode=:safe,
    ctx=FinchCompiler(; algebra=algebra, mode=mode),
    concrete=nothing,
)
    code = contain(ctx) do ctx_2
        prgm = nothing
        prgm = virtualize(ctx_2.code, ex, T)
        # Concrete structure is observable only inside a specialization
        # attempt; a context without one compiles the ordinary generic path.
        if concrete !== nothing && specialization_attempt(ctx_2) !== nothing
            attach_concrete!(prgm, concrete, false)
        end
        lower_global(ctx_2, prgm)
    end
end

"""
    attach_concrete!(prgm, instance, reusable)

Match tensor bindings in virtual program `prgm` with concrete tensors in
`instance`, then attach each concrete level chain to its virtual counterpart.
`reusable=false` means the kernel is regenerated for this call and needs no
freshness guard. `finch_kernel` passes `true` because its generated function may
be called again.
"""
function attach_concrete!(prgm, instance, reusable::Bool)
    tensors = Dict{Symbol,Tensor}()
    collect_concrete_tensors!(tensors, instance)
    for node in PostOrderDFS(prgm)
        if node isa FinchNode && node.kind === tag &&
            node.var.kind === variable && node.bind.kind === virtual &&
            node.bind.val isa VirtualFiber
            tns = get(tensors, node.var.name, nothing)
            tns === nothing || stash_concrete!(node.bind.val.lvl, tns.lvl, reusable)
        end
    end
    prgm
end

function collect_concrete_tensors!(tensors::Dict{Symbol,Tensor}, instance)
    if instance isa FinchNotation.TagInstance &&
        instance.var isa FinchNotation.VariableInstance &&
        instance.bind isa Tensor
        tensors[instance.var.name] = instance.bind
    end
    if SyntaxInterface.istree(instance)
        foreach(
            child -> collect_concrete_tensors!(tensors, child),
            SyntaxInterface.arguments(instance),
        )
    end
    tensors
end

@staged function execute_impl(ex, algebra, mode)
    code = execute_code(:ex, ex; algebra=getvalue(algebra), mode=getvalue(mode))
    if mode === :debug
        return quote
            try
                begin
                    $(unblock(code))
                end
            catch
                println("Error executing code:")
                println($(QuoteNode(unquote_literals(pretty(code)))))
                rethrow()
            end
        end
    else
        return quote
            @inbounds @fastmath begin
                $(unquote_literals(pretty(code)))
            end
        end
    end
end

function execute(ex; algebra=DefaultAlgebra(), mode=:safe, specialize=false)
    if specialize
        execute_specialized(ex; algebra=algebra, mode=mode)
    else
        execute_impl(ex, Val(algebra), Val(mode))
    end
end

"""
    execute_specialized(ex; algebra=DefaultAlgebra(), mode=:safe, report=nothing)

Like [`execute`](@ref), but make concrete tensor structure available to a
loaded specialization extension. A fresh kernel is generated on every call,
outside the per-type `@staged` cache, so a structure-specific kernel cannot be
reused for another tensor merely because its type matches. This per-call path
does not need a runtime freshness guard. The compilation runs as one
[`specialize_compile`](@ref) transaction; `report` receives its
[`SpecializeReport`](@ref).
"""
function execute_specialized(ex; algebra=DefaultAlgebra(), mode=:safe, report=nothing)
    code = specialize_compile(; algebra=algebra, mode=mode, report=report) do ctx
        sym = freshen(ctx, :ex)
        body = execute_code(
            sym, typeof(ex); algebra=algebra, mode=mode, ctx=ctx, concrete=ex,
        )
        quote
            $sym = $ex
            @inbounds @fastmath $body
        end
    end
    thunk = eval(:(() -> $code))
    Base.invokelatest(thunk)
end

"""
    @finch [options...] prgm

Run a finch program `prgm`. The syntax for a finch program is a set of nested
loops, statements, and branches over pointwise array assignments. For example,
the following program computes the sum of two arrays `A = B + C`:

```julia
@finch begin
    A .= 0
    for i = _
        A[i] = B[i] + C[i]
    end
    return A
end
```

Finch programs are composed using the following syntax:

 - `arr .= 0`: an array declaration initializing arr to zero.
 - `arr[inds...]`: an array access, the array must be a variable and each index may be another finch expression.
 - `x + y`, `f(x, y)`: function calls, where `x` and `y` are finch expressions.
 - `arr[inds...] = ex`: an array assignment expression, setting `arr[inds]` to the value of `ex`.
 - `arr[inds...] += ex`: an incrementing array expression, adding `ex` to `arr[inds]`. `*, &, |`, are supported.
 - `arr[inds...] <<min>>= ex`: a incrementing array expression with a custom operator, e.g. `<<min>>` is the minimum operator.
 - `for i = _ body end`: a loop over the index `i`, where `_` is computed from array access with `i` in `body`.
 - `if cond body end`: a conditional branch that executes only iterations where `cond` is true.
 - `return (tnss...,)`: at global scope, exit the program and return the tensors `tnss` with their new dimensions. By default, any tensor declared in global scope is returned.

Symbols are used to represent variables, and their values are taken from the environment. Loops introduce
index variables into the scope of their bodies.

Finch uses the types of the arrays and symbolic analysis to discover program
optimizations. If `B` and `C` are sparse array types, the program will only run
over the nonzeros of either.

Semantically, Finch programs execute every iteration. However, Finch can use
sparsity information to reliably skip iterations when possible.

`options` are optional keyword arguments:

 - `algebra`: the algebra to use for the program. The default is `DefaultAlgebra()`.
 - `mode`: the optimization mode to use for the program. Possible modes are:
    - `:debug`: run the program in debug mode, with bounds checking and better error handling.
    - `:safe`: run the program in safe mode, with modest checks for performance and correctness.
    - `:fast`: run the program in fast mode, with no checks or warnings, this mode is for power users.
    The default is `:safe`.

See also: [`@finch_code`](@ref)
"""
macro finch(opts_ex...)
    length(opts_ex) >= 1 ||
        throw(ArgumentError("Expected at least one argument to @finch(opts..., ex)"))
    (opts, ex) = (opts_ex[1:(end - 1)], opts_ex[end])
    prgm = FinchNotation.finch_parse_instance(ex)
    prgm = quote
        $(FinchNotation.block_instance)(
            $prgm,
            $(FinchNotation.yieldbind_instance)(
                $(
                    map(
                        FinchNotation.variable_instance,
                        FinchNotation.finch_parse_default_yieldbind(ex),
                    )...
                ),
            ),
        )
    end
    res = esc(:res)
    thunk = quote
        res = $execute($prgm, ; $(map(esc, opts)...))
    end
    for tns in something(
        FinchNotation.finch_parse_yieldbind(ex),
        FinchNotation.finch_parse_default_yieldbind(ex),
    )
        push!(
            thunk.args,
            quote
                $(esc(tns)) = res[$(QuoteNode(tns))]
            end,
        )
    end
    push!(
        thunk.args,
        quote
            res
        end,
    )
    thunk
end

"""
@finch_code [options...] prgm

Return the code that would be executed in order to run a finch program `prgm`.

See also: [`@finch`](@ref)
"""
macro finch_code(opts_ex...)
    length(opts_ex) >= 1 ||
        throw(ArgumentError("Expected at least one argument to @finch(opts..., ex)"))
    (opts, ex) = (opts_ex[1:(end - 1)], opts_ex[end])
    prgm = FinchNotation.finch_parse_instance(ex)
    prgm = quote
        $(FinchNotation.block_instance)(
            $prgm,
            $(FinchNotation.yieldbind_instance)(
                $(
                    map(
                        FinchNotation.variable_instance,
                        FinchNotation.finch_parse_default_yieldbind(ex),
                    )...
                ),
            ),
        )
    end
    return quote
        let prgm = $prgm
            $finch_code(prgm; $(map(esc, opts)...))
        end
    end
end

"""
    finch_code(prgm; algebra, mode, specialize=false, report=nothing)

Return the code that would execute the finch program instance `prgm`. With
`specialize=true` the compilation runs as one [`specialize_compile`](@ref)
transaction and `report` receives its [`SpecializeReport`](@ref).
"""
function finch_code(
    prgm; algebra=DefaultAlgebra(), mode=:safe, specialize=false, report=nothing
)
    code = if specialize
        specialize_compile(; algebra=algebra, mode=mode, report=report) do ctx
            execute_code(
                :ex, typeof(prgm); algebra=algebra, mode=mode, ctx=ctx, concrete=prgm,
            )
        end
    else
        execute_code(:ex, typeof(prgm); algebra=algebra, mode=mode)
    end
    unquote_literals(dataflow(unresolve(pretty(code))))
end

"""
    finch_kernel(fname, args, prgm; options...)

Return a function definition for which can execute a Finch program of
type `prgm`. Here, `fname` is the name of the function and `args` is a
`iterable` of argument name => type pairs.

See also: [`@finch`](@ref)
"""
function finch_kernel(
    fname,
    args,
    prgm;
    algebra=DefaultAlgebra(),
    mode=:safe,
    specialize=false,
    report=nothing,
    ctx=nothing,
)
    if specialize
        # A specialized compilation is one transaction over a fresh context
        # per attempt; a caller-supplied context cannot be rebuilt for a
        # byte-identical generic retry.
        ctx === nothing || throw(ArgumentError(
            "finch_kernel: specialize=true builds a fresh compiler context per " *
            "attempt and cannot reuse a caller-supplied ctx"))
        return specialize_compile(; algebra=algebra, mode=mode, report=report) do ctx_2
            finch_kernel_code(fname, args, prgm, ctx_2; algebra=algebra, mode=mode)
        end
    end
    finch_kernel_code(fname, args, prgm,
        something(ctx, FinchCompiler(; algebra=algebra, mode=mode));
        algebra=algebra, mode=mode)
end

function finch_kernel_code(fname, args, prgm, ctx; algebra, mode)
    maybe_typeof(x) = x isa Type ? x : typeof(x)
    ex = freshen(ctx, :ex)
    code = contain(ctx) do ctx_2
        foreach(args) do (key, val)
            virt = virtualize(ctx_2.code, key, maybe_typeof(val), key)
            if specialization_attempt(ctx_2) !== nothing && val isa Tensor &&
                virt isa VirtualFiber
                # The generated function can be called again, so mark its
                # concrete structure as reusable and require a freshness guard.
                stash_concrete!(virt.lvl, val.lvl, true)
            end
            set_binding!(
                ctx_2,
                variable(key),
                finch_leaf(virt),
            )
        end
        execute_code(ex, prgm; algebra=algebra, mode=mode, ctx=ctx_2)
    end
    code = quote
        # Bind the program once before executing the generated body.
        $ex = $prgm
        $code
    end
    code = unquote_literals(dataflow(unresolve(pretty(code))))
    arg_defs = map(((key, val),) -> :($key::$(maybe_typeof(val))), args)
    striplines(:(function $fname($(arg_defs...))
        @inbounds @fastmath $(striplines(unblock(code)))
    end))
end

"""
    @finch_kernel [options...] fname(args...) = prgm

Return a definition for a function named `fname` which executes `@finch prgm` on
the arguments `args`. `args` should be a list of variables holding
representative argument instances.

See also: [`@finch`](@ref)
"""
macro finch_kernel(opts_def...)
    length(opts_def) >= 1 ||
        throw(ArgumentError("expected at least one argument to @finch(opts..., def)"))
    (opts, def) = (opts_def[1:(end - 1)], opts_def[end])
    (@capture def :function(:call(~name, ~args...), ~ex)) ||
        (@capture def :(=)(:call(~name, ~args...), ~ex)) ||
        throw(ArgumentError("unrecognized function definition in @finch_kernel"))
    named_args = map(arg -> :($(QuoteNode(arg)) => $(esc(arg))), args)
    prgm = FinchNotation.finch_parse_instance(ex)
    #for arg in args
    #    prgm = quote
    #        let $(esc(arg)) = $(FinchNotation.variable_instance(arg))
    #            $prgm
    #        end
    #    end
    #end
    return quote
        $finch_kernel(
            $(QuoteNode(name)), Any[$(named_args...),], typeof($prgm); $(map(esc, opts)...)
        )
    end
end

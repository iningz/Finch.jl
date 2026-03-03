"""
    inspect_looplets

Utilities for inspecting the looplet tree that Finch constructs during
compilation.  These are meant for interactive debugging and development of
partial-evaluation passes – they let you see the full nesting structure
(Thunk → Sequence → Phase → Stepper → Spike → …) that is normally
ephemeral and consumed immediately by the lowering pipeline.

# Quick start

```julia
using Finch
using SparseArrays

A = Tensor(SparseList(Element(0.0)), sparsevec([2, 5], [1.0, 2.0], 5))
s = Scalar(0.0)

# Print the looplet tree instead of lowering it:
inspect_looplets(Finch.@finch_program_instance for i = _; s[] += A[i] end)

# Or capture it as a data structure:
tree = capture_looplets(Finch.@finch_program_instance for i = _; s[] += A[i] end)
```
"""

# ─── Looplet tree node ──────────────────────────────────────────────────────

"""
    LoopletNode

A lightweight snapshot of one node in the looplet tree.  The original looplet
objects are closures and mutable structs that are hard to introspect after
lowering has consumed them, so we copy the *structural* information into a
simple, printable tree.

# Fields
- `name::String` – the looplet type (`"Stepper"`, `"Spike"`, …)
- `fields::Vector{Pair{String,Any}}` – scalar attributes (stop symbol, preamble, …)
- `children::Vector{LoopletNode}` – sub-looplets (e.g. the chunk of a Stepper)
- `style::Union{Nothing,Any}` – the lowering style this node triggered
"""
mutable struct LoopletNode
    name::String
    fields::Vector{Pair{String,Any}}
    children::Vector{LoopletNode}
    style::Any

    LoopletNode(name; fields=Pair{String,Any}[], children=LoopletNode[], style=nothing) =
        new(name, collect(fields), collect(children), style)
end

# AbstractTrees interface so we get free printing via print_tree
AbstractTrees.children(n::LoopletNode) = n.children
function AbstractTrees.printnode(io::IO, n::LoopletNode)
    printstyled(io, n.name; bold=true)
    if n.style !== nothing
        printstyled(io, "  ⟨", n.style, "⟩"; color=:cyan)
    end
    for (k, v) in n.fields
        print(io, "\n  ", k, " = ")
        _print_field(io, v)
    end
end

function _print_field(io::IO, v)
    s = sprint(show, v; context=:compact => true)
    if length(s) > 120
        s = s[1:117] * "..."
    end
    print(io, s)
end

# ─── Static snapshot builders (no compiler context) ─────────────────────────
# These give a quick structural picture but cannot expand closures.

"""
    snapshot_looplet(obj) -> LoopletNode

Convert a live looplet object into a `LoopletNode` tree.  Closures are recorded
as `"<closure>"` but the structural nesting is always captured.
"""
snapshot_looplet(x) = LoopletNode("Unknown($(typeof(x)))")

function snapshot_looplet(t::Thunk)
    node = LoopletNode("Thunk")
    push!(node.fields, "preamble" => _summarize_expr(t.preamble))
    push!(node.fields, "epilogue" => _summarize_expr(t.epilogue))
    push!(node.fields, "body" => "<closure>")
    node
end

function snapshot_looplet(l::Lookup)
    LoopletNode("Lookup"; style=LookupStyle(), fields=["body" => "<closure (ctx, i) -> ...>"])
end

function snapshot_looplet(r::Run)
    node = LoopletNode("Run"; style=RunStyle())
    push!(node.children, _snapshot_or_leaf(r.body))
    node
end

function snapshot_looplet(r::AcceptRun)
    LoopletNode("AcceptRun"; style=AcceptRunStyle(), fields=["body" => "<closure>"])
end

function snapshot_looplet(s::Spike)
    node = LoopletNode("Spike"; style=SpikeStyle())
    push!(node.children, LoopletNode("body"; children=[_snapshot_or_leaf(s.body)]))
    push!(node.children, LoopletNode("tail"; children=[_snapshot_or_leaf(s.tail)]))
    node
end

function snapshot_looplet(s::Stepper)
    node = LoopletNode("Stepper"; style=StepperStyle())
    push!(node.fields, "preamble" => _summarize_expr(s.preamble))
    push!(node.fields, "stop" => "<closure>")
    push!(node.fields, "next" => "<closure>")
    push!(node.fields, "seek" => "<closure>")
    if s.chunk !== nothing
        push!(node.children, LoopletNode("chunk"; children=[snapshot_looplet(s.chunk)]))
    end
    node
end

function snapshot_looplet(j::Jumper)
    node = LoopletNode("Jumper"; style=JumperStyle())
    push!(node.fields, "preamble" => _summarize_expr(j.preamble))
    push!(node.fields, "stop" => "<closure>")
    push!(node.fields, "next" => "<closure>")
    push!(node.fields, "seek" => "<closure>")
    if j.chunk !== nothing
        push!(node.children, LoopletNode("chunk"; children=[snapshot_looplet(j.chunk)]))
    end
    node
end

function snapshot_looplet(seq::Sequence)
    node = LoopletNode("Sequence"; style=SequenceStyle())
    for phase in seq.phases
        push!(node.children, snapshot_looplet(phase))
    end
    node
end

function snapshot_looplet(p::Phase)
    node = LoopletNode("Phase")
    push!(node.fields, "start" => "<closure>")
    push!(node.fields, "stop" => "<closure>")
    push!(node.fields, "body" => "<closure>")
    node
end

function snapshot_looplet(sw::Switch)
    node = LoopletNode("Switch"; style=SwitchStyle())
    for (guard, body) in sw.cases
        case_node = LoopletNode("case")
        push!(case_node.fields, "guard" => guard)
        push!(case_node.children, _snapshot_or_leaf(body))
        push!(node.children, case_node)
    end
    node
end

function snapshot_looplet(f::FillLeaf)
    LoopletNode("FillLeaf"; fields=["value" => f.body])
end

function snapshot_looplet(s::Simplify)
    node = LoopletNode("Simplify")
    push!(node.children, _snapshot_or_leaf(s.body))
    node
end

function snapshot_looplet(u::Unfurled)
    node = LoopletNode("Unfurled")
    push!(node.fields, "arr" => string(u.arr))
    push!(node.fields, "ndims" => u.ndims)
    push!(node.children, _snapshot_or_leaf(u.body))
    node
end

function snapshot_looplet(n::FinchNode)
    if n.kind === virtual
        return snapshot_looplet(n.val)
    else
        return LoopletNode("FinchNode($(n.kind))"; fields=["val" => n])
    end
end

function _snapshot_or_leaf(x)
    if x isa FinchNode
        if x.kind === virtual
            return snapshot_looplet(x.val)
        else
            return LoopletNode("FinchNode($(x.kind))"; fields=["val" => x])
        end
    else
        return snapshot_looplet(x)
    end
end

function _summarize_expr(ex)
    ex === nothing && return "nothing"
    ex === :(quote end) && return "quote end"
    s = string(ex)
    length(s) > 100 ? s[1:97] * "..." : s
end

# ─── Tracing compiler ──────────────────────────────────────────────────────
#
# The key idea: looplet objects like Thunk, Stepper, Phase, Sequence contain
# closures that can only be evaluated by the lowering pipeline with a live
# compiler context.  Instead of trying to peek at closures from outside, we
# wrap a real compiler in a *tracing layer* that intercepts every `lower()`
# call.  Each time the pipeline encounters a looplet style (ThunkStyle,
# StepperStyle, …), we snapshot the node *at that moment* — when the
# compiler has already expanded parent closures and the children are live
# objects.
#
# The trace is collected as a flat list of (depth, LoopletNode) pairs,
# then reconstructed into a tree afterward.

"""
    TracingCompiler

Wraps a real `FinchCompiler` and records every looplet-style `lower()` call
that the pipeline makes, building up a trace of `LoopletNode` snapshots
with all closures expanded.
"""
mutable struct TracingCompiler <: AbstractCompiler
    inner::FinchCompiler
    trace::Vector{Pair{Int,LoopletNode}}   # depth => snapshot
    depth::Int
    max_depth::Int
end

# Forward the full AbstractCompiler interface to the inner compiler
get_result(ctx::TracingCompiler) = get_result(ctx.inner)
get_mode_flag(ctx::TracingCompiler) = get_mode_flag(ctx.inner)
get_binding(ctx::TracingCompiler, var) = get_binding(ctx.inner, var)
has_binding(ctx::TracingCompiler, var) = has_binding(ctx.inner, var)
set_binding!(ctx::TracingCompiler, var, val) = set_binding!(ctx.inner, var, val)
set_declared!(ctx::TracingCompiler, var, val, op) = set_declared!(ctx.inner, var, val, op)
set_frozen!(ctx::TracingCompiler, var, val) = set_frozen!(ctx.inner, var, val)
set_thawed!(ctx::TracingCompiler, var, val, op) = set_thawed!(ctx.inner, var, val, op)
get_tensor_mode(ctx::TracingCompiler, var) = get_tensor_mode(ctx.inner, var)
push_preamble!(ctx::TracingCompiler, thunk) = push_preamble!(ctx.inner, thunk)
push_epilogue!(ctx::TracingCompiler, thunk) = push_epilogue!(ctx.inner, thunk)
get_task(ctx::TracingCompiler) = get_task(ctx.inner)
freshen(ctx::TracingCompiler, tags...) = freshen(ctx.inner, tags...)
get_algebra(ctx::TracingCompiler) = get_algebra(ctx.inner)
get_static_hash(ctx::TracingCompiler) = get_static_hash(ctx.inner)
prove(ctx::TracingCompiler, root) = prove(ctx.inner, root)
simplify(ctx::TracingCompiler, root) = simplify(ctx.inner, root)

function open_scope(f::F, ctx::TracingCompiler) where {F}
    open_scope(ctx.inner) do inner_2
        old = ctx.inner
        ctx.inner = inner_2
        try
            f(ctx)
        finally
            ctx.inner = old
        end
    end
end

function contain(f, ctx::TracingCompiler; kwargs...)
    contain(ctx.inner; kwargs...) do inner_2
        old = ctx.inner
        ctx.inner = inner_2
        try
            f(ctx)
        finally
            ctx.inner = old
        end
    end
end

# ─── Snapshot-at-lower: intercept each looplet style ───────────────────────

# Helper: build a LoopletNode for a looplet object with *live* context info,
# then let the real lower() proceed.

function _trace_snapshot_thunk(ctx::TracingCompiler, root::FinchNode)
    # We don't snapshot Thunks themselves as meaningful tree nodes —
    # they are just wrappers.  The inner body will be traced when it
    # is lowered.  But we record their presence.
    node = LoopletNode("Thunk")
    push!(node.fields, "preamble" => _summarize_expr_from_ir(root))
    node.style = ThunkStyle()
    node
end

function _trace_snapshot_loop(ctx::TracingCompiler, root::FinchNode, style)
    # For loop-level styles (Stepper, Spike, Sequence, etc.) we want to
    # capture what is *inside* the loop body — the looplet objects.
    node = LoopletNode("loop")
    node.style = style
    idx_name = root.kind === loop ? string(root.idx) : "?"
    ext_str = root.kind === loop ? _format_extent(ctx, root.ext) : "?"
    push!(node.fields, "idx" => idx_name)
    push!(node.fields, "extent" => ext_str)

    # Walk the body to find virtual looplet nodes
    if root.kind === loop
        for child in PostOrderDFS(root.body)
            if child isa FinchNode && child.kind === virtual
                snap = _snapshot_virtual_live(ctx, child.val)
                if snap !== nothing
                    push!(node.children, snap)
                end
            end
        end
    end
    node
end

function _format_extent(ctx::TracingCompiler, ext_node)
    if ext_node isa FinchNode && ext_node.kind === virtual
        ext = ext_node.val
        if ext isa VirtualExtent
            start_str = _format_finch_node(ext.start)
            stop_str = _format_finch_node(ext.stop)
            return "$start_str : $stop_str"
        else
            return string(typeof(ext))
        end
    end
    return string(ext_node)
end

function _format_finch_node(n)
    if n isa FinchNode
        if n.kind === literal
            return string(n.val)
        elseif n.kind === value
            return string(n.val)
        else
            return string(n)
        end
    else
        return string(n)
    end
end

"""
    _snapshot_virtual_live(ctx, obj) -> Union{LoopletNode, Nothing}

Attempt to build a deep LoopletNode for a virtual looplet, evaluating closures
with the live compiler context where possible.  Returns `nothing` for objects
that aren't meaningful looplet types.
"""
function _snapshot_virtual_live(ctx::TracingCompiler, obj)
    _snapshot_virtual_live_inner(ctx, obj)
end

_snapshot_virtual_live_inner(ctx, obj) = nothing

function _snapshot_virtual_live_inner(ctx, t::Thunk)
    node = LoopletNode("Thunk")
    node.style = ThunkStyle()
    push!(node.fields, "preamble" => _summarize_expr(t.preamble))
    push!(node.fields, "epilogue" => _summarize_expr(t.epilogue))
    # Try to expand the body closure
    try
        inner = t.body(ctx.inner)
        child = _snapshot_any_live(ctx, inner)
        if child !== nothing
            push!(node.children, LoopletNode("body"; children=[child]))
        else
            push!(node.fields, "body" => string(typeof(inner)))
        end
    catch e
        push!(node.fields, "body" => "<closure — error: $(sprint(showerror, e; context=:limit=>true))>")
    end
    node
end

function _snapshot_virtual_live_inner(ctx, l::Lookup)
    LoopletNode("Lookup"; style=LookupStyle(), fields=["body" => "(ctx, i) -> ..."])
end

function _snapshot_virtual_live_inner(ctx, r::Run)
    node = LoopletNode("Run"; style=RunStyle())
    child = _snapshot_any_live(ctx, r.body)
    if child !== nothing
        push!(node.children, child)
    end
    node
end

function _snapshot_virtual_live_inner(ctx, r::AcceptRun)
    LoopletNode("AcceptRun"; style=AcceptRunStyle(), fields=["body" => "(ctx, ext) -> ..."])
end

function _snapshot_virtual_live_inner(ctx, s::Spike)
    node = LoopletNode("Spike"; style=SpikeStyle())
    body_snap = _snapshot_any_live(ctx, s.body)
    tail_snap = _snapshot_any_live(ctx, s.tail)
    if body_snap !== nothing
        push!(node.children, LoopletNode("body"; children=[body_snap]))
    end
    if tail_snap !== nothing
        push!(node.children, LoopletNode("tail"; children=[tail_snap]))
    end
    node
end

function _snapshot_virtual_live_inner(ctx, s::Stepper)
    node = LoopletNode("Stepper"; style=StepperStyle())
    push!(node.fields, "preamble" => _summarize_expr(s.preamble))
    push!(node.fields, "stop" => "(ctx, ext) -> ...")
    push!(node.fields, "next" => "(ctx, ext) -> ...")
    if s.chunk !== nothing
        chunk_snap = _snapshot_any_live(ctx, s.chunk)
        if chunk_snap !== nothing
            push!(node.children, LoopletNode("chunk"; children=[chunk_snap]))
        end
    end
    node
end

function _snapshot_virtual_live_inner(ctx, j::Jumper)
    node = LoopletNode("Jumper"; style=JumperStyle())
    push!(node.fields, "preamble" => _summarize_expr(j.preamble))
    push!(node.fields, "stop" => "(ctx, ext) -> ...")
    push!(node.fields, "next" => "(ctx, ext) -> ...")
    if j.chunk !== nothing
        chunk_snap = _snapshot_any_live(ctx, j.chunk)
        if chunk_snap !== nothing
            push!(node.children, LoopletNode("chunk"; children=[chunk_snap]))
        end
    end
    node
end

function _snapshot_virtual_live_inner(ctx, seq::Sequence)
    node = LoopletNode("Sequence"; style=SequenceStyle())
    for phase in seq.phases
        child = _snapshot_any_live(ctx, phase)
        if child !== nothing
            push!(node.children, child)
        end
    end
    node
end

function _snapshot_virtual_live_inner(ctx, p::Phase)
    node = LoopletNode("Phase")

    # Build a dummy extent to evaluate the Phase closures.
    # Phase.stop/start/body all take (ctx, ext) where ext is the loop extent.
    # We use symbolic placeholder values so the closures can execute and return
    # their looplet objects, even though we don't know the real bounds.
    dummy_ext = VirtualExtent(
        value(:__lo__, Int),
        value(:__hi__, Int),
    )

    # Evaluate start
    try
        start_val = p.start(ctx.inner, dummy_ext)
        if start_val === nothing
            push!(node.fields, "start" => "nothing (unbounded)")
        else
            push!(node.fields, "start" => _format_finch_node(start_val))
        end
    catch
        push!(node.fields, "start" => "<closure>")
    end

    # Evaluate stop
    try
        stop_val = p.stop(ctx.inner, dummy_ext)
        if stop_val === nothing
            push!(node.fields, "stop" => "nothing (unbounded)")
        else
            push!(node.fields, "stop" => _format_finch_node(stop_val))
        end
    catch
        push!(node.fields, "stop" => "<closure>")
    end

    # Evaluate body — this is the critical one that returns Stepper, Run, etc.
    try
        inner = p.body(ctx.inner, dummy_ext)
        child = _snapshot_any_live(ctx, inner)
        if child !== nothing
            push!(node.children, LoopletNode("body"; children=[child]))
        else
            push!(node.fields, "body" => string(typeof(inner)))
        end
    catch e
        push!(node.fields, "body" => "<closure — error: $(_truncate_str(sprint(showerror, e), 120))>")
    end
    node
end

function _snapshot_virtual_live_inner(ctx, sw::Switch)
    node = LoopletNode("Switch"; style=SwitchStyle())
    for (guard, body) in sw.cases
        case_node = LoopletNode("case")
        push!(case_node.fields, "guard" => _format_finch_node(guard))
        child = _snapshot_any_live(ctx, body)
        if child !== nothing
            push!(case_node.children, child)
        end
        push!(node.children, case_node)
    end
    node
end

function _snapshot_virtual_live_inner(ctx, f::FillLeaf)
    LoopletNode("FillLeaf"; fields=["value" => _format_finch_node(f.body)])
end

function _snapshot_virtual_live_inner(ctx, s::Simplify)
    node = LoopletNode("Simplify")
    child = _snapshot_any_live(ctx, s.body)
    if child !== nothing
        push!(node.children, child)
    end
    node
end

function _snapshot_virtual_live_inner(ctx, u::Unfurled)
    node = LoopletNode("Unfurled")
    push!(node.fields, "arr" => string(u.arr))
    push!(node.fields, "ndims" => u.ndims)
    child = _snapshot_any_live(ctx, u.body)
    if child !== nothing
        push!(node.children, child)
    end
    node
end

function _snapshot_virtual_live_inner(ctx, n::FinchNode)
    if n.kind === virtual
        return _snapshot_virtual_live_inner(ctx, n.val)
    else
        return LoopletNode("FinchNode($(n.kind))"; fields=["val" => _format_finch_node(n)])
    end
end

"""
    _snapshot_any_live(ctx, obj) -> Union{LoopletNode, Nothing}

Dispatch on the type of `obj` — could be a looplet, a FinchNode, or anything
else.
"""
function _snapshot_any_live(ctx, obj)
    result = _snapshot_virtual_live_inner(ctx, obj)
    result !== nothing && return result
    return LoopletNode("Other($(typeof(obj)))")
end

function _snapshot_any_live(ctx, n::FinchNode)
    if n.kind === virtual
        result = _snapshot_virtual_live_inner(ctx, n.val)
        result !== nothing && return result
        return LoopletNode("virtual($(typeof(n.val)))")
    else
        return LoopletNode("FinchNode($(n.kind))"; fields=["val" => _format_finch_node(n)])
    end
end

function _summarize_expr_from_ir(root::FinchNode)
    s = string(root)
    length(s) > 100 ? s[1:97] * "..." : s
end

# ─── Tracing lower() intercepts ───────────────────────────────────────────
#
# We intercept every style-based lower() call.  For each looplet style, we:
#   1. Snapshot the looplet tree at this point (with live closures expanded)
#   2. Record the snapshot in the trace
#   3. Delegate to the real lower() on the inner compiler
#
# This captures the full tree as the pipeline sees it.

for StyleT in [
    :ThunkStyle, :LookupStyle, :RunStyle, :AcceptRunStyle,
    :SpikeStyle, :StepperStyle, :JumperStyle, :SequenceStyle,
    :SwitchStyle, :FillStyle,
]
    @eval function lower(ctx::TracingCompiler, root::FinchNode, style::$StyleT)
        if ctx.depth <= ctx.max_depth
            snap = _trace_snapshot_loop(ctx, root, style)
            push!(ctx.trace, ctx.depth => snap)
            ctx.depth += 1
            try
                return lower(ctx.inner, root, style)
            finally
                ctx.depth -= 1
            end
        else
            return lower(ctx.inner, root, style)
        end
    end
end

# Phase styles are parameterized
function lower(ctx::TracingCompiler, root::FinchNode, style::PhaseStyle)
    if ctx.depth <= ctx.max_depth
        snap = _trace_snapshot_loop(ctx, root, style)
        push!(ctx.trace, ctx.depth => snap)
        ctx.depth += 1
        try
            return lower(ctx.inner, root, style)
        finally
            ctx.depth -= 1
        end
    else
        return lower(ctx.inner, root, style)
    end
end

# Default style passthrough
function lower(ctx::TracingCompiler, root::FinchNode, style::DefaultStyle)
    lower(ctx.inner, root, style)
end

function lower(ctx::TracingCompiler, root, style::DefaultStyle)
    lower(ctx.inner, root, style)
end

# SimplifyStyle passthrough
function lower(ctx::TracingCompiler, root::FinchNode, style::SimplifyStyle)
    lower(ctx.inner, root, style)
end

# ─── Build tree from flat trace ────────────────────────────────────────────

function _build_trace_tree(trace::Vector{Pair{Int,LoopletNode}})
    isempty(trace) && return LoopletNode[]

    # Find root-level entries (minimum depth)
    min_depth = minimum(first, trace)
    roots = LoopletNode[]

    # Stack-based tree construction
    stack = Pair{Int,LoopletNode}[]  # depth => node

    for (depth, node) in trace
        # Pop everything from the stack that is at the same or deeper depth
        while !isempty(stack) && first(last(stack)) >= depth
            pop!(stack)
        end
        if !isempty(stack)
            push!(last(last(stack)).children, node)
        else
            push!(roots, node)
        end
        push!(stack, depth => node)
    end

    roots
end

# ─── Main capture function ─────────────────────────────────────────────────

"""
    capture_looplets(prgm_instance; algebra=DefaultAlgebra(), mode=:safe, max_depth=50)

Run the Finch compilation pipeline with a **tracing compiler** that intercepts
every looplet-style `lower()` call, capturing the looplet structure *with
closures expanded by the live compiler context*.

Returns a `Vector{LoopletNode}` — typically one root per loop in the program,
with children showing the full nesting of looplet types.

## Two capture strategies

The function uses a two-pronged approach:

1. **Pre-lower snapshot**: Before lowering begins, it calls `unfurl()` on each
   tensor access and uses the live compiler context to expand Thunk body
   closures, producing a "structural" view of the looplet tree.

2. **Tracing lower**: During actual lowering, a `TracingCompiler` intercepts
   every style-dispatched `lower()` call, recording which looplet styles fire
   and in what order.

Both are returned: the structural trees first, then the trace trees.

## Example

```julia
A = Tensor(SparseList(Element(0.0)), sparsevec([2, 5], [1.0, 2.0], 5))
s = Scalar(0.0)
trees = capture_looplets(Finch.@finch_program_instance for i = _; s[] += A[i] end)

using AbstractTrees
for t in trees
    print_tree(t; maxdepth=20)
end
```
"""
function capture_looplets(
    prgm_type::Type;
    algebra=DefaultAlgebra(),
    mode=:safe,
    max_depth=50,
)
    structural_trees = LoopletNode[]

    inner_ctx = FinchCompiler(; algebra=algebra, mode=mode)
    tracing_ctx = TracingCompiler(inner_ctx, Pair{Int,LoopletNode}[], 0, max_depth)

    contain(inner_ctx) do ctx_2
        tracing_ctx.inner = ctx_2
        prgm = virtualize(ctx_2.code, :ex, prgm_type)

        # Run the same passes as lower_global up to the point where we can
        # unfurl and then lower.
        prgm = exit_on_yieldbind(prgm)
        prgm = enforce_scopes(prgm)
        prgm = evaluate_partial(ctx_2, prgm)

        contain(ctx_2) do ctx_3
            tracing_ctx.inner = ctx_3

            prgm = wrapperize(ctx_3, prgm)
            prgm = enforce_lifecycles(prgm)
            prgm = dimensionalize!(prgm, ctx_3)
            prgm = concordize(ctx_3, prgm)
            prgm = evaluate_partial(ctx_3, prgm)
            prgm = simplify(ctx_3, prgm)
            prgm = instantiate!(ctx_3, prgm)

            # Strategy 1: pre-lower structural snapshot via unfurl
            _capture_structural!(structural_trees, tracing_ctx, prgm)

            # Strategy 2: actually lower with the tracing compiler to capture
            # the style dispatch order
            try
                contain(tracing_ctx) do tctx
                    tracing_ctx.inner = tctx.inner
                    tctx(prgm)
                end
            catch
                # Lowering may fail in tracing mode for various reasons;
                # the structural trees are still useful.
            end

            quote end  # satisfy contain()'s return expectation
        end
    end

    trace_trees = _build_trace_tree(tracing_ctx.trace)

    # Merge: structural trees first, then trace trees
    all_trees = LoopletNode[]

    if !isempty(structural_trees)
        wrapper = LoopletNode("── Structural (pre-lower unfurl) ──")
        wrapper.children = structural_trees
        push!(all_trees, wrapper)
    end

    if !isempty(trace_trees)
        wrapper = LoopletNode("── Lowering trace (style dispatch) ──")
        wrapper.children = trace_trees
        push!(all_trees, wrapper)
    end

    all_trees
end

function capture_looplets(prgm_instance; kwargs...)
    capture_looplets(typeof(prgm_instance); kwargs...)
end

# ─── Structural capture via unfurl ─────────────────────────────────────────

function _capture_structural!(trees, ctx::TracingCompiler, node::FinchNode)
    if node.kind === loop
        # For each tensor access in the loop body that is indexed by this
        # loop's index, call unfurl to get the looplet and snapshot it.
        for child in PostOrderDFS(node.body)
            if child isa FinchNode && child.kind === access && !isempty(child.idxs) &&
                    child.idxs[end] == node.idx
                tns = child.tns
                mode_node = child.mode
                ext_val = isvirtual(node.ext) ? node.ext.val : node.ext
                try
                    looplet = unfurl(
                        ctx.inner,
                        tns,
                        ext_val,
                        mode_node,
                        (mode_node.kind === reader ? defaultread : defaultupdate),
                    )
                    # Now snapshot with live context (expands Thunk closures)
                    snap = _snapshot_any_live(ctx, looplet)
                    if snap !== nothing
                        tns_str = _truncate_str(string(tns), 80)
                        idx_str = string(node.idx)
                        ext_str = _format_extent(ctx, node.ext)
                        wrapper = LoopletNode("unfurl")
                        push!(wrapper.fields, "tensor" => tns_str)
                        push!(wrapper.fields, "index" => idx_str)
                        push!(wrapper.fields, "extent" => ext_str)
                        push!(wrapper.children, snap)
                        push!(trees, wrapper)
                    end
                catch e
                    push!(trees, LoopletNode("UnfurlError";
                        fields=["tensor" => _truncate_str(string(tns), 80),
                                "error" => sprint(showerror, e)]))
                end
            end
        end
        # Recurse into the loop body for nested loops
        _capture_structural!(trees, ctx, node.body)
    elseif node isa FinchNode && istree(node)
        for child in arguments(node)
            _capture_structural!(trees, ctx, child)
        end
    end
end

_capture_structural!(trees, ctx, node) = nothing

function _truncate_str(s, maxlen)
    length(s) > maxlen ? s[1:(maxlen-3)] * "..." : s
end

# ─── Pretty-print entry point ──────────────────────────────────────────────

"""
    inspect_looplets(prgm_instance; io=stdout, kwargs...)

Capture and pretty-print the looplet tree(s) for a Finch program instance.

## Example

```julia
using Finch, SparseArrays

A = Tensor(SparseList(Element(0.0)), sparsevec([2, 5], [1.0, 2.0], 5))
s = Scalar(0.0)

inspect_looplets(Finch.@finch_program_instance begin
    for i = _
        s[] += A[i]
    end
end)
```

Output shows both the *structural* looplet tree (from unfurl, with Thunk bodies
expanded) and the *lowering trace* (which styles fired during compilation):

```
━━━ Looplet tree 1 ━━━
── Structural (pre-lower unfurl) ──
└─ unfurl
     tensor = "A"
     index = "i"
   └─ Thunk
      └─ body
         └─ Sequence  ⟨SequenceStyle()⟩
            ├─ Phase
            │  └─ body
            │     └─ Stepper  ⟨StepperStyle()⟩
            │        └─ chunk
            │           └─ Spike  ⟨SpikeStyle()⟩
            │              ├─ body
            │              │  └─ FillLeaf
            │              └─ tail
            │                 └─ Simplify
            │                    └─ Thunk
            │                       └─ body
            │                          └─ ...
            └─ Phase
               └─ body
                  └─ Run  ⟨RunStyle()⟩
                     └─ FillLeaf
```
"""
function inspect_looplets(prgm_instance; io::IO=stdout, kwargs...)
    trees = capture_looplets(prgm_instance; kwargs...)
    if isempty(trees)
        println(io, "No looplet trees captured. ",
            "(The program may not contain any loop-level tensor accesses.)")
        return trees
    end
    for (i, tree) in enumerate(trees)
        println(io, "━━━ Looplet tree $i ━━━")
        AbstractTrees.print_tree(io, tree; maxdepth=25)
        println(io)
    end
    return trees
end

# ─── Diff two looplet trees ────────────────────────────────────────────────

"""
    diff_looplets(tree_a::LoopletNode, tree_b::LoopletNode; io=stdout)

Side-by-side structural comparison of two looplet trees.  Useful for comparing
the looplet tree of a standard tensor vs. a `specialize()`-d tensor.

## Example

```julia
A = Tensor(Dense(Element(0.0)), rand(10))
sA = specialize(A)
s1 = Scalar(0.0)
s2 = Scalar(0.0)

t1 = capture_looplets(Finch.@finch_program_instance for i = _; s1[] += A[i] end)
t2 = capture_looplets(Finch.@finch_program_instance for i = _; s2[] += sA[i] end)
diff_looplets(t1[1], t2[1])
```
"""
function diff_looplets(a::LoopletNode, b::LoopletNode; io::IO=stdout, indent=0)
    prefix = "  " ^ indent
    if a.name != b.name
        printstyled(io, prefix, "- ", a.name, "\n"; color=:red)
        printstyled(io, prefix, "+ ", b.name, "\n"; color=:green)
        return
    end
    println(io, prefix, a.name)

    # Compare fields
    a_dict = Dict(a.fields)
    b_dict = Dict(b.fields)
    all_keys = unique(vcat(first.(a.fields), first.(b.fields)))
    for k in all_keys
        av = get(a_dict, k, nothing)
        bv = get(b_dict, k, nothing)
        if string(av) == string(bv)
            println(io, prefix, "  ", k, " = ", av)
        elseif av === nothing
            printstyled(io, prefix, "  + ", k, " = ", bv, "\n"; color=:green)
        elseif bv === nothing
            printstyled(io, prefix, "  - ", k, " = ", av, "\n"; color=:red)
        else
            printstyled(io, prefix, "  - ", k, " = ", av, "\n"; color=:red)
            printstyled(io, prefix, "  + ", k, " = ", bv, "\n"; color=:green)
        end
    end

    # Compare children
    n = max(length(a.children), length(b.children))
    for i in 1:n
        if i > length(a.children)
            printstyled(io, prefix, "  + child $i: ", b.children[i].name, "\n"; color=:green)
        elseif i > length(b.children)
            printstyled(io, prefix, "  - child $i: ", a.children[i].name, "\n"; color=:red)
        else
            diff_looplets(a.children[i], b.children[i]; io, indent=indent + 1)
        end
    end
end

# ─── Count looplet node types ──────────────────────────────────────────────

"""
    looplet_stats(tree::LoopletNode) -> Dict{String,Int}

Count the number of each looplet type in the tree.

```julia
stats = looplet_stats(tree)
# Dict("Thunk" => 2, "Stepper" => 1, "Spike" => 1, "FillLeaf" => 2, …)
```
"""
function looplet_stats(tree::LoopletNode)
    counts = Dict{String,Int}()
    _count_nodes!(counts, tree)
    counts
end

function _count_nodes!(counts, node::LoopletNode)
    counts[node.name] = get(counts, node.name, 0) + 1
    for child in node.children
        _count_nodes!(counts, child)
    end
end

# ─── Flatten to list ───────────────────────────────────────────────────────

"""
    looplet_nodes(tree::LoopletNode) -> Vector{LoopletNode}

Flatten the tree to a pre-order list of all nodes.
"""
function looplet_nodes(tree::LoopletNode)
    result = LoopletNode[]
    _collect_nodes!(result, tree)
    result
end

function _collect_nodes!(result, node::LoopletNode)
    push!(result, node)
    for child in node.children
        _collect_nodes!(result, child)
    end
end

# ─── Find nodes by type ───────────────────────────────────────────────────

"""
    find_looplets(tree::LoopletNode, name::String) -> Vector{LoopletNode}

Find all nodes in the tree with the given name (e.g. `"Stepper"`, `"Spike"`).
"""
function find_looplets(tree::LoopletNode, name::String)
    filter(n -> n.name == name, looplet_nodes(tree))
end

# Finch exposes three small pieces needed by a structural-data extension:
#
#   - a hook before lowering, where concrete storage can be observed;
#   - a hook at unfurl, where an accepted description can replace one native
#     traversal;
#   - a generation token that lets reusable kernels reject changed structure.
#
# Finch does not depend on Regularity types. When no extension handles a hook,
# compilation continues through Finch's ordinary path.

"""
    mine_regular_structure!(ctx, fbr)

Called once per bound tensor after virtualization and before lowering. Concrete
storage is visible here, but traversal protocols and entry positions are not.
An extension may inspect structure, mine a description, and attach a candidate
to the virtual level. The default does nothing.
"""
mine_regular_structure!(ctx, fbr) = nothing

"""
    regularize_unfurl(ctx, fbr, ext, mode, proto)

Called from dense and SparseList read unfurlers with the actual traversal mode,
protocol, extent, and entry position. Returning `nothing` keeps the native
unfurl. Returning a looplet tree replaces it. An extension must finish all
checks before allocating compiler symbols or adding preamble code.
"""
regularize_unfurl(ctx, fbr, ext, mode, proto) = nothing

# -- Concrete-data stash -------------------------------------------------------
# Virtual levels normally contain symbols rather than concrete arrays.
# Specialized entry points attach the matching concrete levels and record
# whether the generated kernel will be reused. The extension uses that flag to
# decide whether a runtime freshness guard is needed.

"""
    stash_concrete!(vlvl, lvl, reusable::Bool)

Attach concrete level `lvl` and its descendants to the matching virtual chain.
Unsupported level kinds stop the walk. A missing stash makes the extension use
the ordinary path.
"""
stash_concrete!(vlvl, lvl, reusable::Bool) = nothing

"""
    concrete_stash(vlvl) -> Union{ConcreteStash, Nothing}

The stash attached by [`stash_concrete!`](@ref), or `nothing`.
"""
concrete_stash(vlvl) = nothing

"A concrete level attached to a virtual level, plus whether its kernel is reused."
struct ConcreteStash
    lvl::Any
    reusable::Bool
end

# Dense and SparseList levels participate. Any other kind reaches the default
# method and stops the walk; extraction later requires a complete chain.
function stash_concrete!(vlvl::VirtualDenseLevel, lvl::DenseLevel, reusable::Bool)
    vlvl.concrete = ConcreteStash(lvl, reusable)
    stash_concrete!(vlvl.lvl, lvl.lvl, reusable)
end
function stash_concrete!(vlvl::VirtualSparseListLevel, lvl::SparseListLevel, reusable::Bool)
    vlvl.concrete = ConcreteStash(lvl, reusable)
    stash_concrete!(vlvl.lvl, lvl.lvl, reusable)
end

concrete_stash(vlvl::VirtualDenseLevel) = vlvl.concrete
concrete_stash(vlvl::VirtualSparseListLevel) = vlvl.concrete

# -- Structural freshness -----------------------------------------------------
# A reusable kernel may run after Finch has rewritten a sparse structure in
# place. Array identity and length do not detect such a rewrite, so each
# structural array object — `ptr` and `idx` alike — has its own generation
# token. Finch increments the tokens of every array a lifecycle transition may
# change. A kernel records token identity and generation for each array it
# depends on when it is built and compares them before every later run.
# Per-array tokens matter: two levels may hold distinct `ptr` arrays yet share
# one mutable `idx`, and a rewrite reached through the other level must still
# be observed here.
#
# Aliases that hold the same array object share its token. Direct user
# mutation of raw arrays bypasses Finch's lifecycle and is not tracked.

"""
    StructuralToken

The generation token associated with one structural array object. `generation`
counts Finch lifecycle transitions that may rewrite structure.
"""
mutable struct StructuralToken
    generation::UInt64
end

# `WeakKeyDict` uses value equality and the key's mutable hash. Structural
# arrays need identity semantics, so keep weak references and compare with
# `===`. The registry is normally tiny: one entry per structural array
# observed by a reusable specialized kernel.
const _structural_tokens = Tuple{WeakRef,StructuralToken}[]
const _structural_tokens_lock = ReentrantLock()

"""
    structural_token(key::AbstractVector) -> StructuralToken

Return the token for the exact array object `key`, creating it on first use.
References to the same array object receive the same token; separate arrays do
not share one merely because their contents are equal.
"""
function structural_token(key::AbstractVector)::StructuralToken
    lock(_structural_tokens_lock) do
        index = 1
        while index <= length(_structural_tokens)
            weak_key, token = _structural_tokens[index]
            stored_key = weak_key.value
            if stored_key === nothing
                deleteat!(_structural_tokens, index)
            elseif stored_key === key
                return token
            else
                index += 1
            end
        end
        token = StructuralToken(UInt64(0))
        push!(_structural_tokens, (WeakRef(key), token))
        token
    end
end

"""
    touch_structure!(token::StructuralToken)
    touch_structure!(key::AbstractVector)

Record that the guarded structure may have changed. Finch calls this from
emitted `declare!`, `freeze!`, and `thaw!` lifecycle code.
"""
function touch_structure!(token::StructuralToken)
    token.generation += UInt64(1)
    nothing
end

touch_structure!(key::AbstractVector) = touch_structure!(structural_token(key))

"""
    structure_current(key::AbstractVector, token_id::UInt, generation::UInt64) -> Bool

Return whether `key` still has the token identity and generation recorded
when a reusable kernel was built.
"""
function structure_current(key::AbstractVector, token_id::UInt, generation::UInt64)::Bool
    token = structural_token(key)
    objectid(token) === token_id && token.generation === generation
end

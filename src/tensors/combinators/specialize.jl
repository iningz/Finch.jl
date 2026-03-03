"""
    SpecializedTensor{Params, T}

The output of [`specialize`](@ref), a staging wrapper that lifts level params
into the type domain so the Finch compiler can treat them as compile-time
constants rather than symbolic runtime variables.

Because `Params` is a type parameter, two tensors with different params produce
distinct program instance types, causing `@staged` to generate separate
specialized code for each.

# Type Parameters
- `Params`: A tuple of compile-time concrete values collected from the level
  hierarchy via [`level_params`](@ref).
- `T`: The wrapped tensor type.

See also: [`specialize`](@ref), [`level_params`](@ref)
"""
struct SpecializedTensor{Params,Body} <: AbstractCombinator
    body::Body
end

SpecializedTensor{Params}(body::Body) where {Params,Body} = SpecializedTensor{Params,Body}(body)

Base.ndims(arr::SpecializedTensor) = ndims(arr.body)
Base.size(arr::SpecializedTensor) = size(arr.body)
Base.eltype(arr::SpecializedTensor) = eltype(arr.body)
fill_value(arr::SpecializedTensor) = fill_value(arr.body)
countstored(arr::SpecializedTensor) = countstored(arr.body)

function Base.show(io::IO, ex::SpecializedTensor{Params}) where {Params}
    print(io, "SpecializedTensor{", Params, "}(")
    show(io, ex.body)
    print(io, ")")
end

function labelled_show(io::IO, ::SpecializedTensor{Params}) where {Params}
    print(io, "SpecializedTensor (params = ", Params, ")")
end

labelled_children(ex::SpecializedTensor) = [LabelledTree(ex.body)]

"""
    specialize(tensor::Tensor)

Wrap a tensor so that level metadata become compile-time
constants. Returns a `SpecializedTensor` that works transparently with `@finch`.

# Example
```julia
A = Tensor(Dense(Element(0.0)), rand(10))
sA = specialize(A)
@finch for i = _; s[] += sA[i]; end
```
"""
function specialize(tensor::Tensor)
    params = level_params(tensor.lvl)
    SpecializedTensor{params}(tensor)
end

"""
    level_params(lvl)

Collect specialization params from a level hierarchy. Returns a flat tuple
embedded in `SpecializedTensor`'s type parameter. Only Dense is supported now.
"""
level_params(lvl) = ()
function level_params(lvl::DenseLevel)
    (lvl.shape, level_params(lvl.lvl)...)
end

function virtualize(ctx, ex, ::Type{SpecializedTensor{Params,Tensor{Lvl}}}) where {Params,Lvl}
    tag = freshen(ctx, :tns)
    lvl = _virtualize_specialized(ctx, :($ex.body.lvl), Lvl, Params, Symbol(tag, :_lvl))
    VirtualFiber(lvl)
end

#     virtualize_specialized(ctx, ex, LvlType, params, tag)

# Like `virtualize`, but consumes specialization params collected by
# [`level_params`](@ref) to inject compile-time constants. Derived from
# the standard `DenseLevel` virtualizer in `src/tensors/levels/dense_levels.jl`.

# Each level type that contributes to `level_params` should also
# define a `virtualize_specialized` method that peels off its entries from `params`
# and passes the remainder to the child level. Levels that don't contribute
# params just forward unchanged.

# Falls back to standard `virtualize` when params are exhausted.

function _virtualize_specialized(
    ctx, ex, ::Type{DenseLevel{Ti,Lvl}}, params::NTuple{N,Any}, tag
) where {Ti,Lvl,N}
    @assert N > 0 "Specialization params exhausted before all Dense levels were processed"
    
    shape_val = params[1]
    tail = Base.tail(params)
    
    tag = freshen(ctx, tag)
    push_preamble!(ctx, :($tag = $ex))
    shape = literal(Ti(shape_val))
    
    lvl_2 = _virtualize_specialized(ctx, :($tag.lvl), Lvl, tail, tag)
    VirtualDenseLevel(tag, lvl_2, Ti, shape)
end

# Base case: params exhausted → fall back to standard virtualize for the
# remaining part of the level hierarchy.
function _virtualize_specialized(ctx, ex, ::Type{Lvl}, ::Tuple{}, tag) where {Lvl}
    virtualize(ctx, ex, Lvl, tag)
end

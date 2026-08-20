module FinchRegularityTests

using Test
using Finch

const F = Finch

# Load Finch before the weak dependency so the first test covers the inert hook
# path and the remaining tests cover the dynamically loaded extension.
include("without_extension.jl")

@eval using Regularity
const RX = Base.get_extension(Finch, :RegularityExt)

include("with_extension.jl")
include("freshness_alias.jl")
include("transaction.jl")
include("admission.jl")

end # module FinchRegularityTests

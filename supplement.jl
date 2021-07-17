# abstract type NFSPAgents <: MultiAgentManager end

# include("average_learner.jl")
# include("nfsp.jl")

using Random
using StatsBase: sample, Weights
using Flux: softmax

function Base.run(
    p::NFSPAgents,
    env::AbstractEnv,
    stop_condition = StopAfterEpisode(1),
    hook = EmptyHook(),
)
    @assert NumAgentStyle(env) isa MultiAgent
    @assert DynamicStyle(env) === SEQUENTIAL
    @assert RewardStyle(env) === TERMINAL_REWARD
    @assert ChanceStyle(env) === EXPLICIT_STOCHASTIC
    @assert DefaultStateStyle(env) isa InformationSet

    is_stop = false

    while !is_stop
        RLBase.reset!(env)
        hook(PRE_EPISODE_STAGE, p, env)

        while !is_terminated(env) # one episode
            RLBase.update!(p, env)
            hook(POST_ACT_STAGE, p, env)

            if stop_condition(p, env)
                is_stop = true
                break
            end
        end # end of an episode

        if is_terminated(env)
            hook(POST_EPISODE_STAGE, p, env)
        end
    end
    hook(POST_EXPERIMENT_STAGE, p, env)
    hook
end

export WeightedSoftmaxExplorer


"""
    WeightedSoftmaxExplorer(;rng=Random.GLOBAL_RNG)

See also: [`WeightedExplorer`](@ref)
"""
struct WeightedSoftmaxExplorer{R<:AbstractRNG} <: AbstractExplorer
    rng::R
end

function WeightedSoftmaxExplorer(; rng = Random.GLOBAL_RNG)
    WeightedSoftmaxExplorer(rng)
end

(s::WeightedSoftmaxExplorer)(values::AbstractVector{T}) where {T} =
    sample(s.rng, Weights(softmax(values), one(T)))

function (s::WeightedSoftmaxExplorer)(values::AbstractVector{T}, mask) where {T}
    values[.!mask] .= typemin(T)
    s(values)
end

RLBase.prob(explorer::WeightedSoftmaxExplorer, values) = softmax(values)
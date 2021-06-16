using ReinforcementLearning
using Setfield: @set
using Random
using Flux


mutable struct AverageLearner{
    Tq<:AbstractApproximator,
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Tq
    # loss_func::Tf
    min_replay_history::Int
    update_freq::Int
    update_step::Int
    sampler::NStepBatchSampler
    rng::R
    # # for logging
    # loss::Float32
end

function AverageLearner(;
    approximator::Tq,
    # loss_func::Tf,
    stack_size::Union{Int,Nothing} = nothing,
    batch_size::Int = 32,
    update_horizon::Int = 1,
    min_replay_history::Int = 32,
    update_freq::Int = 1,
    traces = SARTS,
    update_step = 0,
    rng = Random.GLOBAL_RNG,
) where {Tq}
    sampler = NStepBatchSampler{traces}(;
        γ = 0.99f0,
        n = update_horizon,
        stack_size = stack_size,
        batch_size = batch_size,
    )
    AverageLearner(
        approximator,
        # loss_func,
        min_replay_history,
        update_freq,
        update_step,
        sampler,
        rng,
    )
end


Flux.functor(x::AverageLearner) = (Q = x.approximator, ), y -> begin
    x = @set x.approximator = y.Q
    x
end

function (learner::AverageLearner)(env)
    env |>
    state |>
    x -> Flux.unsqueeze(x, ndims(x) + 1) |>
    x -> send_to_device(device(learner), x) |>
    learner.approximator |>
    send_to_host |> vec
end


function RLBase.update!(learner::AverageLearner, t::AbstractTrajectory)
    length(t[:terminal]) - learner.sampler.n <= learner.min_replay_history && return

    learner.update_step += 1
    learner.update_step % learner.update_freq == 0 || return

    _, batch = sample(learner.rng, t, learner.sampler)
    update!(learner, batch)

end

function RLBase.update!(learner::AverageLearner, batch::NamedTuple)
    Q = learner.approximator
    _device(x) = send_to_device(device(Q), x)

    local s, a
    @sync begin
        @async s = _device(batch[:state])
        @async a = _device(batch[:action])
    end

    gs = gradient(params(Q)) do
        ŷ = Q(s)
        y = Flux.onehotbatch(a, axes(ŷ, 1)) |> _device
        logitcrossentropy(ŷ, y)
    end

    update!(Q, gs)
end


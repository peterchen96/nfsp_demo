"""
NFSP agents trained on Kuhn Poker game.
"""

using ReinforcementLearning
using Flux
using StableRNGs
using ProgressMeter: @showprogress

include("nfsp.jl")

# Encode the KuhnPokerEnv's states for training.
env = KuhnPokerEnv()
states = [
    (), (:J,), (:Q,), (:K,),
    (:J, :bet), (:J, :pass), (:Q, :bet), (:Q, :pass), (:K, :bet), (:K, :pass),
    (:J, :pass, :bet), (:J, :bet, :bet), (:J, :bet, :pass), (:J, :pass, :pass),
    (:Q, :pass, :bet), (:Q, :bet, :bet), (:Q, :bet, :pass), (:Q, :pass, :pass),
    (:K, :pass, :bet), (:K, :bet, :bet), (:K, :bet, :pass), (:K, :pass, :pass),
    (:J, :pass, :bet, :pass), (:J, :pass, :bet, :bet), (:Q, :pass, :bet, :pass),
    (:Q, :pass, :bet, :bet), (:K, :pass, :bet, :pass), (:K, :pass, :bet, :bet),
] # all states for players 1 & 2

states_indexes_Dict = Dict((i, j) for (j, i) in enumerate(states))

RLBase.state(env::StateTransformedEnv, args...; kwargs...) =
    env.state_mapping(state(env.env, args...; kwargs...), args...)

RLBase.state_space(env::StateTransformedEnv, args...; kwargs...) = 
    env.state_space_mapping(state_space(env.env, args...; kwargs...), args...)

wrapped_env = StateTransformedEnv(
        env;
        state_mapping = (s, player=current_player(env)) -> 
            player == chance_player(env) ? s : [states_indexes_Dict[s]],
        state_space_mapping = (ss, player=current_player(env)) -> 
            player == chance_player(env) ? ss : [[i] for i in 1:length(states)]
    )

# set parameters
seed = 123
anticipatory_param = 0.1f0
used_device = Flux.cpu # Flux.gpu
rng = StableRNG(seed)
# hidden_layers = (64, 64)
eval_every = 10_000
# ϵ_decay = 2_000_000
train_episodes = 10_000_000

# initial NFSPAgents
nfsp = NFSPAgents(wrapped_env;
        η = anticipatory_param,
        _device = used_device, 
        # ϵ_decay = ϵ_decay, 
        # hidden_layers = hidden_layers,
        rng = rng
        )

episodes = []
results = [] # where can use `hook` to record the results

@showprogress for episode in range(1, length=train_episodes)
    reset!(wrapped_env)
    while !is_terminated(wrapped_env)
        RLBase.update!(nfsp, wrapped_env)
    end

    if episode % eval_every == 0
        push!(episodes, episode)
        push!(results, nash_conv(nfsp, wrapped_env) / 2)

    end

end

# save results
ENV["GKSwstype"]="nul" 
using Plots

savefig(plot(episodes, results, xaxis=:log), "result")
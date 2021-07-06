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
    [(), (:J,), (:J, :pass, :bet), (:Q,), (:Q, :pass, :bet), (:K,), (:K, :pass, :bet),
    (:J, :bet, :bet), (:J, :bet, :pass), (:J, :pass, :bet), (:J, :pass, :pass), (:Q, :bet, :bet), 
    (:Q, :bet, :pass), (:Q, :pass, :bet), (:Q, :pass, :pass), (:K, :bet, :bet), (:K, :bet, :pass), 
    (:K, :pass, :bet), (:K, :pass, :pass)], # all states for player1

    [(), (:J, :bet), (:J, :pass), (:Q, :bet), (:Q, :pass), (:K, :bet), (:K, :pass),
    (:J, :bet), (:J, :pass), (:J, :pass, :bet, :bet), (:J, :pass, :bet, :pass), (:Q, :bet), (:Q, :pass), 
    (:Q, :pass, :bet, :bet), (:Q, :pass, :bet, :pass), (:K, :bet), (:K, :pass), (:K, :pass, :bet, :bet), 
    (:K, :pass, :bet, :pass)] # all states for player2
]

states_indexes_Dict = [Dict([(i, j) for (j, i) in enumerate(s)]) for s in states]

wrapped_env = StateTransformedEnv(
        env;
        state_mapping = s -> 
        current_player(env) == chance_player(env) ? s : [states_indexes_Dict[current_player(env)][s]],
        state_space_mapping = ss -> 
        current_player(env) == chance_player(env) ? ss : 1:length(states[current_player(env)])
    )

# global parameters
seed = 123
anticipatory_param = 0.1
used_device = Flux.cpu # Flux.gpu
rng = StableRNG(seed)

hidden_layers = (64, 64)
eval_every = 10_000
ϵ_decay = 2_000_000
train_episodes = 10_000_000
nfsp = NFSPAgents(wrapped_env;
        η = anticipatory_param,
        _device = used_device, 
        ϵ_decay = ϵ_decay, 
        hidden_layers = hidden_layers,
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
        push!(results, reward(wrapped_env))

    end

end

# save results
ENV["GKSwstype"]="nul" 
using Plots

savefig(scatter(episodes, results, xaxis=:log), "scatter_result")
savefig(plot(episodes, results, xaxis=:log), "result")
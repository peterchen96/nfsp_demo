using ReinforcementLearning
using Flux
using StableRNGs
using ProgressMeter: @showprogress

include("NFSPAgent.jl")

env = KuhnPokerEnv()
states = [
    [(), (:J,), (:J, :pass, :bet), (:Q,), (:Q, :pass, :bet), (:K,), (:K, :pass, :bet),
    (:J, :bet, :bet), (:J, :bet, :pass), (:J, :pass, :bet), (:J, :pass, :pass), (:Q, :bet, :bet), 
    (:Q, :bet, :pass), (:Q, :pass, :bet), (:Q, :pass, :pass), (:K, :bet, :bet), (:K, :bet, :pass), 
    (:K, :pass, :bet), (:K, :pass, :pass)], # player 1 all states when playing game

    [(), (:J, :bet), (:J, :pass), (:Q, :bet), (:Q, :pass), (:K, :bet), (:K, :pass),
    (:J, :bet), (:J, :pass), (:J, :pass, :bet, :bet), (:J, :pass, :bet, :pass), (:Q, :bet), (:Q, :pass), 
    (:Q, :pass, :bet, :bet), (:Q, :pass, :bet, :pass), (:K, :bet), (:K, :pass), (:K, :pass, :bet, :bet), 
    (:K, :pass, :bet, :pass)] # player 2 all states when playing game
]

states_indexes_Dict = [Dict([(i, j) for (j, i) in enumerate(s)]) for s in states]

wrapped_env = ActionTransformedEnv(
        StateTransformedEnv(
            env;
            state_mapping = s -> current_player(env) == chance_player(env) ? s : [states_indexes_Dict[current_player(env)][s]],
            state_space_mapping = ss -> current_player(env) == chance_player(env) ? ss : 1:length(states[current_player(env)])
        );
        action_mapping = i -> current_player(env) == chance_player(env) ? i : [action_space(env, current_player(env))[i]],
        action_space_mapping = as -> current_player(env) == chance_player(env) ? as : 1:length(action_space(env, current_player(env)))
    )

# global parameters
η = 0.1
seed = 123
used_device = Flux.cpu # Flux.gpu
rng = StableRNG(seed)

hidden_layers = (64, 64)
eval_every = 10_000
ϵ_decay = 2_000_000
train_episodes = 10_000_000
nfsp = [NFSPAgent(env, states_indexes_Dict, player_id; 
            _device = used_device, 
            ϵ_decay = ϵ_decay, 
            hidden_layers_sizes = hidden_layers,
            ) 
    for player_id in players(env) if player_id != chance_player(env)]

episodes = []
rewards = []

@showprogress for episode in range(1, length=train_episodes)

    reset!(env)
    while current_player(env) == chance_player(env)
        env |> legal_action_space |> rand |> env
    end

    while !is_terminated(env)
        player_id = current_player(env)
        sl_policy = nfsp[player_id].sl_agent
        rl_policy = nfsp[player_id].rl_agent
        
        if rand(rng) < η
            action = rl_policy(wrapped_env)
            sl_policy(PRE_ACT_STAGE, wrapped_env, action)
            rl_policy(PRE_ACT_STAGE, wrapped_env, action)
            env(action)
            sl_policy(POST_ACT_STAGE, wrapped_env)
            rl_policy(POST_ACT_STAGE, wrapped_env)

        else
            action = sl_policy(wrapped_env)
            rl_policy(PRE_ACT_STAGE, wrapped_env, action)
            env(action)
            rl_policy(POST_ACT_STAGE, wrapped_env)

        end
        
    end

    if episode % eval_every == 0
        push!(episodes, episode)
        push!(rewards, reward(env))

    end

end

# show results
ENV["GKSwstype"]="nul" 
using Plots

savefig(plot(episodes, rewards, xaxis=:log), "result")

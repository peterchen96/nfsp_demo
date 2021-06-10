using ReinforcementLearning
using StableRNGs

include("nfsp.jl")

env = KuhnPokerEnv()
states = [
    [(), (:J,), (:J, :pass, :bet), (:Q,), (:Q, :pass, :bet), (:K,), (:K, :pass, :bet),
    (:J, :bet, :bet), (:J, :bet, :pass), (:J, :pass, :bet), (:J, :pass, :pass), (:Q, :bet, :bet), 
    (:Q, :bet, :pass), (:Q, :pass, :bet), (:Q, :pass, :pass), (:K, :bet, :bet), (:K, :bet, :pass), 
    (:K, :pass, :bet), (:K, :pass, :pass)], # player 1 states when playing game

    [(), (:J, :bet), (:J, :pass), (:Q, :bet), (:Q, :pass), (:K, :bet), (:K, :pass),
    (:J, :bet), (:J, :pass), (:J, :pass, :bet, :bet), (:J, :pass, :bet, :pass), (:Q, :bet), (:Q, :pass), 
    (:Q, :pass, :bet, :bet), (:Q, :pass, :bet, :pass), (:K, :bet), (:K, :pass), (:K, :pass, :bet, :bet), 
    (:K, :pass, :bet, :pass)] # player 2 states when playing game
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
rng = StableRNG(seed)

eval_every = 10_000
train_episodes = 10_000_000_000
nfsp = [initial_NFSPAgent(env, states_indexes_Dict, player_id) 
    for player_id in players(env) if player_id != chance_player(env)]

episodes = []
rewards = []

for episode in 1:train_episodes

    reset!(env)
    while current_player(env) == chance_player(env)
        env |> legal_action_space |> rand |> env
    end

    while !is_terminated(env)
        player_id = current_player(env)
        sl_policy = nfsp[player_id]["sl_agent"]
        rl_policy = nfsp[player_id]["rl_agent"]
        reservoir = nfsp[player_id]["reservoir"]
        
        if rand(rng) < η
            action = rl_policy(wrapped_env)
            reservoir(PRE_ACT_STAGE, rl_policy, wrapped_env, action)

            if length(reservoir.records) > nfsp[player_id]["SL_buffer_capacity"]
                popfirst!(reservoir.records[:state])
                popfirst!(reservoir.records[:action])
            end
        
        else
            action = sl_policy(wrapped_env)
        end
       
        rl_policy(PRE_ACT_STAGE, wrapped_env, action)
        env(action)
        rl_policy(POST_ACT_STAGE, wrapped_env)

        if length(reservoir.records) >= nfsp[player_id]["min_buffer_size_to_learn"]
            nfsp[player_id]["SL_iters"] += 1

            if nfsp[player_id]["SL_iters"] % nfsp[player_id]["SL_update_freq"] == 0
                s = BatchSampler{(:state, :action)}(nfsp[player_id]["batch_size"];)
                _, batch = s(reservoir.records)
                RLBase.update!(sl_policy, batch)
            end
        end
    end

    if episode % eval_every == 0
        push!(episodes, episode)
        push!(rewards, reward(env))

    end

end

using Plots

savefig(plot(episodes, rewards, xaxis=:log), "result")

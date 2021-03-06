"""
Neural Fictitious Self-Play (NFSP) agent implemented in Julia.

See the paper https://arxiv.org/abs/1603.01121 for more details.
"""

using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

include("average_learner.jl")
include("supplement.jl")

mutable struct NFSPAgents <: AbstractPolicy
    agents::Dict{Any, Any}
end

mutable struct NFSPAgent <: AbstractPolicy
    η
    rng::StableRNG
    rl_agent::Agent
    sl_agent::Agent
end

function NFSPAgents(env::AbstractEnv; kwargs...)
    NFSPAgents(
        Dict((player, NFSPAgent(env, player; kwargs...)) 
        for player in players(env) if player != chance_player(env)
        )
    )
end

# Neural Fictitious Self-play(NFSP) agent
function NFSPAgent(
    env::AbstractEnv,
    player;

    # parameters setting
    # public parameters
    η = 0.01f0,
    _device = Flux.gpu,
    Optimizer = Flux.Descent,
    rng = StableRNG(123),
    batch_size::Int = 128,
    learn_freq::Int = 128,
    min_buffer_size_to_learn::Int = 1000,
    hidden_layers = (128, 128, 128, 128),

    # Reinforcement Learning(RL) agent parameters
    ϵ_start = 0.06,
    ϵ_end = 0.001,
    ϵ_decay = 20_000_000,
    rl_learning_rate = 0.01,
    RL_buffer_capacity::Int = 200_000,
    discount_factor::Float32 = 1.0f0,
    update_target_network_freq::Int = 19200,

    # Supervisor Learning(SL) agent parameters
    sl_learning_rate = 0.01,
    SL_buffer_capacity::Int = 2_000_000
    )

    # ignore chance_player.
    while current_player(env) == chance_player(env)
        env |> action_space |> rand |> env
    end

    # base Neural network for training DQNLearner
    ns = length(state(env, player))
    na = length(action_space(env, player))
    base_model = Chain(
        Dense(ns, hidden_layers[1], relu; init = glorot_uniform(rng)),
        [Dense(hidden_layers[i], hidden_layers[i+1], relu; init = glorot_uniform(rng)) 
            for i in 1:length(hidden_layers)-1]...,
        Dense(hidden_layers[end], na; init = glorot_uniform(rng))
    )

    # RL agent
    rl_agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = build_dueling_network(base_model) |> _device,
                    optimizer = Optimizer(rl_learning_rate),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = build_dueling_network(base_model) |> _device,
                ),
                γ = discount_factor,
                loss_func = huber_loss,
                batch_size = batch_size,
                update_freq = learn_freq,
                min_replay_history = min_buffer_size_to_learn,
                target_update_freq = update_target_network_freq,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :linear,
                ϵ_init = ϵ_start,
                ϵ_stable = ϵ_end,
                decay_steps = ϵ_decay,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = RL_buffer_capacity,
            state = Vector{Int64} => (ns, )
        ),
    )

    # SL agent
    sl_agent = Agent(
        policy = QBasedPolicy(
            learner = AverageLearner(
                approximator = NeuralNetworkApproximator(
                    model = build_dueling_network(base_model) |> _device,
                    optimizer = Optimizer(sl_learning_rate),
                ),
                batch_size = batch_size,
                update_freq = learn_freq,
                min_replay_history = min_buffer_size_to_learn,
                rng = rng,
            ),
            explorer = WeightedSoftmaxExplorer(),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = SL_buffer_capacity,
            state = Vector{Int64} => (ns, )
        ),
    )

    NFSPAgent(
        η,
        rng,
        rl_agent,
        sl_agent,
    )
    
end

# DQN network relative functions 
function build_dueling_network(network::Chain)
    lm = length(network)
    if !(network[lm] isa Dense) || !(network[lm-1] isa Dense)
        error("The Qnetwork provided is incompatible with dueling.")
    end
    base = Chain([deepcopy(network[i]) for i=1:lm-2]...)
    last_layer_dims = size(network[lm].W, 2)
    val = Chain(deepcopy(network[lm-1]), Dense(last_layer_dims, 1))
    adv = Chain([deepcopy(network[i]) for i=lm-1:lm]...)
    return DuelingNetwork(base, val, adv)
end

function RLBase.update!(π::NFSPAgent, env::AbstractEnv)
    sl = π.sl_agent
    rl = π.rl_agent
    
    if rand(π.rng) < π.η
        action = rl(env)
        sl(PRE_ACT_STAGE, env, action)
        rl(PRE_ACT_STAGE, env, action)
        env(action)
        sl(POST_ACT_STAGE, env)
        rl(POST_ACT_STAGE, env)
    else
        action = sl(env)
        rl(PRE_ACT_STAGE, env, action)
        env(action)
        rl(POST_ACT_STAGE, env)
    end
end

function RLBase.update!(agents::NFSPAgents, env::AbstractEnv)
    player = current_player(env)
    if player == chance_player(env)
        env |> legal_action_space |> rand |> env
    else
        RLBase.update!(agents.agents[player], env)
    end
end

function (nfsp::NFSPAgent)(env::AbstractEnv)
    player = current_player(env)
    if player == chance_player(env)
        env |> legal_action_space |> rand |> env
    else
        nfsp.sl_agent(env)
        # rand(nfsp.rng) < nfsp.η ? nfsp.rl_agent(env) : nfsp.sl_agent(env)
    end
end

function RLBase.prob(nfsp::NFSPAgents, env::AbstractEnv, args...)
    player = current_player(env)
    agent = nfsp.agents[player]
    a = agent.sl_agent
    return prob(a.policy, env, args...)
end

function RLBase.prob(p::QBasedPolicy, env::AbstractEnv) 
    # if typeof(p.explorer) == WeightedSoftmaxExplorer{Random._GLOBAL_RNG}
    prob(p, env, ActionStyle(env))
    # else
    #     prob(p, env, ActionStyle(env)).p
    # end
end
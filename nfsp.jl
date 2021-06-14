using ReinforcementLearning
using CircularArrayBuffers: CircularArrayBuffer
using StableRNGs
using Flux
using Flux.Losses


# SL hook setting
Base.@kwdef struct RecordStateAction <: AbstractHook
    records::Any = VectorSATrajectory(; state = Vector{Int64})
end

function (h::RecordStateAction)(::PreActStage, policy, env, action)
    push!(h.records; state = copy(state(env)), action = action)
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

# construct NFSP agent
function initial_NFSPAgent(
    env::KuhnPokerEnv, 
    states_indexes_Dict,
    player_id;

    # parameters setting
    # public parameters
    device = gpu,
    optimizer_str = "sgd",
    batch_size::Int = 128,
    learn_every::Int = 128,
    min_buffer_size_to_learn::Int = 1000,
    hidden_layers_sizes = (128, 128, 128, 128),

    # QBased Policy parameters (RL agent)
    ϵ_start = 0.06,
    ϵ_end = 0.001,
    ϵ_decay = 20_000_000,
    discount_factor::Float32 = Float32(1.0),
    rl_learning_rate = 0.01,
    RL_buffer_capacity::Int = 200_000,
    update_target_network_freq::Int = 19200,

    # Average Policy parameters (SL agent)
    sl_learning_rate = 0.01,
    SL_buffer_capacity::Int = 2_000_000
    )

    # Neural network construction
    ns = length(states_indexes_Dict[player_id][state(env, player_id)])
    na = length(action_space(env, player_id))
    base_model = Chain(
        Dense(ns, hidden_layers_sizes[1], relu; init = glorot_uniform(rng)),
        [Dense(hidden_layers_sizes[i], hidden_layers_sizes[i+1], relu; init = glorot_uniform(rng)) 
            for i in 1:length(hidden_layers_sizes)-1]...,
        Dense(hidden_layers_sizes[end], na; init = glorot_uniform(rng))
    )

    # RL agent setting (QBased Policy)
    rl_agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = build_dueling_network(base_model) |> device,
                    optimizer = optimizer_str == "sgd" ? Descent(rl_learning_rate) : ADAM(rl_learning_rate),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = build_dueling_network(base_model) |> device,
                ),
                γ = discount_factor,
                loss_func = huber_loss,
                stack_size = nothing,
                batch_size = batch_size,
                min_replay_history = min_buffer_size_to_learn,
                update_freq = learn_every,
                target_update_freq = update_target_network_freq,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
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

    # SL agent setting (Average Policy)
    sl_agent = BehaviorCloningPolicy(
        approximator = NeuralNetworkApproximator(
            model = base_model |> cpu, # device
            optimizer = optimizer_str == "sgd" ? Descent(sl_learning_rate) : ADAM(sl_learning_rate),
            ),
        )
    reservoir = RecordStateAction()
    SL_update_freq = learn_every
    initial_iters = 1

    return Dict(
        "batch_size" => batch_size,
        "min_buffer_size_to_learn" => min_buffer_size_to_learn,
        "rl_agent" => rl_agent,
        "reservoir" => reservoir,
        "sl_agent" => sl_agent,
        "SL_iters" => initial_iters,
        "SL_update_freq" => SL_update_freq,
        "SL_buffer_capacity" => SL_buffer_capacity,
    )
    
end

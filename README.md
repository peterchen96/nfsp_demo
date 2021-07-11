## Neural Fictitious Self-play(NFSP)

Use `RL.jl` to implement `Neural Fictitious Self-play(NFSP)` algorithm and test it on kuhn poker game(use Env `KuhnPokerEnv`).

### NFSPAgent's structure:

* anticipatory parameter($\eta$):

    ```julia
    if rand() < \eta
        use rl_agent's policy
        and sl_agent collect the best response from rl_agent's output
    
    else
        use sl_agent's policy
    
    end
    ```
    
* rl_agent: 

    ```julia
    Agent(
        policy = QBasedPolicy(
            learner = DQNLearner,
            explorer = EpsilonGreedyExplorer,
        ),
        trajectory = CircularArraySARTTrajectory
    )
    ```

    where rl_agent(DQNLearner) is to search for the best response from the self-play process.

* sl_agent:
    
    ```julia
    Agent(
        policy = QBasedPolicy(
            learner = AverageLearner,
            explorer = GreedyExplorer,
        ),
        trajectory = CircularArraySARTTrajectory,
    )
    ```
    
    where sl_agent is an `Average Learner` that learns the best response from rl_agent's data.

### Kuhn Poker Experiment

#### Preperation

Before training the agent, I should encode all states of the players except the chance player.
```julia
wrapped_env = StateTransformedEnv(
        env;
        state_mapping = (s, player=current_player(env)) -> 
            player == chance_player(env) ? s : [states_indexes_Dict[s]],
        state_space_mapping = (ss, player=current_player(env)) -> 
            player == chance_player(env) ? ss : [[i] for i in 1:length(states)]
    )
```

#### parameters setting:

<details>
    <summary> parameters </summary>
    
        * anticipatory_param = 0.1,
        * eval_every = 10_000,
        * learn_freq = 128,
        * batch_size = 128,
        * min_buffer_size_to_learn = 1_000,
        * optimizer = Flux.Descent,

        * SL_buffer_capacity = 2_000_000,
        * SL_learning_rate = 0.01,

        * RL_buffer_capacity = 200_000,
        * update_target_network_every = 19200,
        * discount_factor = 1.0,
        * RL_learning_rate = 0.01,
        * $\epsilon$ _ init = 0.06,
        * $\epsilon$ _ end = 0.001,
        * $\epsilon$ _ decay kind = linear.

        * $\epsilon$ _ decay = 2_000_000,
        * train_episodes = 10_000_000, 
        * hidden_layers_sizes = (64, 64),
        * used_device = Flux.cpu,

</details>

#### result

* used time : about 10 min.
* evaluation_metric: nash_conv

The result looks bad. :disappointed_relieved:

<div align="center">
<img src="./result.png" height="300px" alt="result" >
</div>

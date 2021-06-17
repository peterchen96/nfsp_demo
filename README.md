### NFSP Kuhn

Use `Julia` to implement `Neural Fictitious Self-play(NFSP)` algorithm and test it on `kuhn poker` game(use Env `KuhnPokerEnv`).

* NFSPAgent:
    
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
    
    where `AverageLearner` is an `AbstractLearner` which I imitated the structure from `DQNLearner` and rewrite its network updating process.

#### recent progress

parameters setting:

most parameters are the same as the [paper](https://arxiv.org/abs/2103.00187)'s `NFSP_Kuhn experiment part`.

<details>
    <summary> same parameters </summary>
    
      
      * anticipatory_param = 0.1,
      * eval_every = 10_000,
      * learn_freq = 128,
      * batch_size = 128,
      * min_buffer_size_to_learn = 1_000,
      * optimizer = Descent,

      * SL_buffer_capacity = 2_000_000,
      * SL_learning_rate = 0.01,

      * RL_buffer_capacity = 200_000,
      * update_target_network_every = 19200,
      * discount_factor = 1.0,
      * RL_learning_rate = 0.01,
      * $\epsilon$ _ init = 0.06,
      * $\epsilon$ _ end = 0.001,
      * $\epsilon$ _ decay kind = linear.

</details>

* difference:
  * $\epsilon$ _ decay = 2_000_000,
  * train_episodes = 10_000_000, 
  * hidden_layers_sizes = (64, 64),
  * used_device = cpu,
  * evaluation_metric: player 1's reward


* used time: about 10 min

* result:

<div align="center">
<img src="./scatter_result.png" height="300px" alt="scatter_result" >
<img src="./result.png" height="300px" alt="result" >
</div>

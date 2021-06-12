#### NFSP Kuhn

Use `Julia` to implement `Neural Fictitious Self-play(NFSP)` algorithm and test it on `Kuhn_Poker` game(use `KuhnPokerEnv`).

Based on `DQNLearner` in `QBasedPolicy` (as RL) and `BehaviorCloningPolicy` in `offline_rl` (as SL) to learn the best policy.

##### recent progress

set:

$`\epsilon`$ _ decay = 2000000,
train_episodes = 10000000,
eval_every = 10000.

get player 1 reward based on the trained policy

![result](./result.png)

However, `nash_conv` is a more reliable metric for evaluating and I don't know how to  approximate it.

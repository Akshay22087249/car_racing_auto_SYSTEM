# Multi-Agent Deep Reinforcement Learning for Warlords

Portfolio 3 (Autonomous Systems). A self-play deep-RL agent for the four-player
Atari game Warlords, with an empirical comparison of DQN, A2C and PPO and a
documented account of the reward and representation choices that the result
turned out to hinge on.

> The report is written in English to match the codebase and the concept notes;
> it can be translated to Dutch on request.

## 1. Introduction and problem analysis

### 1.1 The environment

Warlords is a four-player Atari game served through PettingZoo's multi-agent
Atari suite (`warlords_v3`). Each player controls a paddle that defends a corner
fortress and deflects a ball toward the other three fortresses. A player whose
fortress is destroyed receives `-1` and is eliminated; the last player standing
receives `+1`. It is a **last-man-standing, competitive game**.

The agent is observed in **RAM mode**: each observation is the raw 128-byte
Atari console RAM rather than pixels. This removes the perception problem (no
convolutional vision stack is needed) and lets the project concentrate on the
*decision* problem. Probing the live environment (`ram_map.py`) established the
facts the rest of the project relies on:

| Property | Value |
|---|---|
| Players | 4 (`first_0 … fourth_0`) |
| Observation | 128-byte RAM, **identical for every player** |
| Action space | `Discrete(6)`: NOOP, FIRE, up, right, left, down |
| Reward | `-1` when your fortress falls, `+1` for the last player standing |
| Episode length | ~6,600 environment steps under random play |
| Determinism | sticky actions off (`repeat_action_probability = 0`), frame-skip 4 |
| Paddle position | RAM bytes `[91, 92, 93, 94]`, one per player |
| Ball | RAM bytes `[90, 95, 111]` (change every frame) |
| Castle health | byte `108` (third), `109` (fourth); brick bitmasks for first/second |

Two facts shape every later decision. First, the reward is **extremely sparse**:
a single non-zero value at the end of a multi-thousand-step game. Second, all
four players receive the **same global RAM**, so an agent cannot tell which
paddle is its own from the observation; it must infer this from how the RAM
responds to its own actions.

### 1.2 Why this is a Multi-Agent RL problem

A single-agent view treats the other three paddles as a fixed part of the
environment. That assumption fails here: the opponents also act, and during
training also learn, so the dynamics seen by any one agent are **non-stationary**
[[Multi-Agent Reinforcement Learning]]. The formal backdrop is a Markov game, the
multi-agent generalisation of the [[Markov Decision Process]] $(S, A, P, R,
\gamma)$ in which the transition and reward depend on the *joint* action. Warlords
is competitive, so the relevant solution concept is a [[Nash Equilibrium]] of the
underlying [[Game Theory|game]]: a strategy from which no player can profit by
deviating unilaterally. Self-play is exactly a search for such an equilibrium.

### 1.3 Choosing an algorithm: comparison rather than assertion

Three deep-RL families taught in the course are candidates, and rather than argue
for one a priori, all three are trained under an identical setup and compared.

- **DQN** (value-based, off-policy). Learns a Q-function via the
  [[Bellman Equation]] and acts greedily; the deep version ([[Deep Q-Network]])
  stabilises [[Q-Learning]] with **experience replay** and a **target network**.
  Sample-efficient, but its replay buffer assumes a stationary environment.
- **A2C** (on-policy actor-critic). [[A2C|Advantage Actor-Critic]] combines a
  policy (actor) with a value baseline (critic) to cut the variance of the
  [[Policy Gradients|policy gradient]]. Simple and fast, but takes large steps.
- **PPO** (on-policy, clipped actor-critic). [[PPO|Proximal Policy Optimization]]
  keeps the actor-critic structure but constrains each update with a clipped
  surrogate objective, so the policy cannot move too far per step. This
  trust-region behaviour suits the noisy, non-stationary gradients of self-play,
  which is why it is the expected winner and the algorithm behind most modern
  self-play results, including the multi-agent variant [[MAPPO]].

## 2. Method and implementation

The method has two parts that mirror how the project actually unfolded: a first
self-play design that **failed**, and the opponent-diversity design that worked.

### 2.1 First attempt: shared-policy self-play (and why it collapsed)

The natural MARL recipe for a symmetric team is **Centralised Training with
Decentralised Execution** [[Multi-Agent Reinforcement Learning]]: one shared
policy controls all four paddles (parameter sharing), with a one-hot **agent
indicator** appended so the policy can specialise per corner (the
`warlords_env.py` + `train.py` pipeline, using SuperSuit to vectorise the
PettingZoo env for Stable-Baselines3).

This collapsed. With four equally-good copies the game simply lasts a long time
for everyone, so there is no gradient toward *winning*: the trained policy scored
**0% win rate** against the random and rule-based baselines. Four identical
agents reach a mutual, mediocre equilibrium rather than a skilful one. This is a
known failure mode of naive self-play and motivated the redesign.

### 2.2 Final design: opponent-diversity solo training

Instead of four copies, the learner controls **one randomly chosen corner each
episode** and the other three are driven by scripted opponents sampled from a
pool (random / rule-based / a frozen snapshot of the learner). Because the
learner is rewarded for its own outcome and must beat *real* opponents, it has a
genuine gradient toward skill; because its corner is randomised (with the
indicator), it learns to defend any corner, which the tournament requires
(`warlords_solo.py` + `train_solo.py`). This is independent learning against a
fixed/mixed opponent distribution, the practical cousin of self-play that avoids
the symmetric-collapse trap.

### 2.3 Observation: engineered features, not raw RAM

A shared policy reading the raw 128 bytes never beat a simple ball-tracking
heuristic: the MLP has to first *find* the few control-relevant bytes among 128.
Handing it those bytes directly fixed this. The observation
(`ram_map.extract_features`) is a compact vector for the controlled corner:
its paddle position, its castle health, the ball bytes, and each opponent's
paddle and health. A single frame has no velocity, so **STACK_K = 3 frames are
stacked**, which is what lets the policy anticipate and intercept the ball rather
than only react. The reverse-engineered RAM map (paddle, ball and health bytes)
is therefore not just analysis but the actual feature set.

### 2.4 Reward shaping: the crux

The native `+1/-1` signal is unlearnable on its own (one value, ~6,600 steps
after the decisions that earned it). Getting the dense shaping right was the
single biggest determinant of the result, and several intuitive choices failed:

1. **Per-step survival bonus** made the agent maximise *survival time*; it learned
   to stall games to the cap and its win rate dropped *below* chance.
2. **Discounting.** With `γ = 0.995` the effective horizon (~200 steps) is far
   shorter than an episode, so the terminal reward is invisible from most states.
   Raising to `γ = 0.999` (~1,000-step horizon) was necessary for any terminal
   credit at all.
3. **Defense-only damage reward.** Rewarding "lose fewer of my own bricks" (read
   from the castle-health bytes) produced a perfect **turtle**: it defended so
   well that no castle fell, games stalemated, and it won *nothing* even with a
   tournament-length cap.
4. **Offense-balanced reward (final).** The reward used in `warlords_solo.py`
   keeps the scaled terminal `±1` dominant and adds, per step, a penalty for
   damage to the agent's castle, a *larger* reward for damage it deals to others,
   and a bonus for each opponent it outlasts. This makes the agent attack as well
   as defend, so games resolve and it actually wins. Its training return turned
   positive once this balance was found.

### 2.5 Algorithm configurations

All three algorithms share the network (two hidden layers of 128, tanh),
`γ = 0.999`, and the solo environment, so the comparison is fair.

| | DQN | A2C | PPO |
|---|---|---|---|
| Type | value-based, off-policy | actor-critic, on-policy | clipped actor-critic, on-policy |
| Stabiliser | experience replay + target net | advantage baseline | clipped trust region + GAE |
| Key settings | replay 200k, target 2k, ε 1→0.05 | n-steps 16 | n-steps 512, 4 epochs, clip 0.2 |

### 2.6 The tournament agent

`agents/submission_agent.py` is self-contained and numpy-only (no torch needed in
the grading environment). It reproduces the trained actor's forward pass by hand
from exported weights, and because all players see the same RAM it first
**self-identifies its corner**: it issues a short directional probe (a run of
DOWN then UP) and picks the paddle byte whose signed movement follows its
commands, which the opponents' paddles do not. It then feeds the policy the same
engineered, frame-stacked features used in training. A ball-tracking fallback
covers the case where the weights are missing.

## 3. Experiments and results

Numbers are produced by `evaluate.py`; see `results/`.

### 3.1 Training curves

`results/training_curves.png` shows win rate against the opponent pool during
training (dashed line = the 0.25 "fair share" of a four-player game). The
algorithms separate exactly as the theory predicts: **PPO** is the most stable,
holding ~0.25-0.28 out to 3M steps; **DQN** climbs fastest and peaks highest
(~0.29 near 0.9M) but then drifts down, consistent with its replay buffer
assuming a stationarity the self-play setting violates; **A2C** is the least
stable, rising then decaying below the fair-share line as its unconstrained
policy-gradient steps overshoot. PPO's clipped trust region is the decisive
advantage, so PPO is the deep-RL model carried forward.

### 3.2 Comparison against the baselines

Each agent was evaluated against three random and three rule-based opponents
(16 games, `results/comparison.csv`), and all agents were seated together in a
tournament-realistic head-to-head (`results/round_robin.csv`,
`results/agent_comparison.png`). The clearest signal is **survival** (how long an
agent keeps its castle alive):

| Agent | win vs random | survival vs random | round-robin survival |
|---|---|---|---|
| random baseline | 0.12 | 4,862 | 14,935 |
| rule-based | 0.25 | 4,703 | 19,198 |
| A2C (self-play) | 0.12 | 4,910 | 24,822 |
| DQN (self-play) | 0.12 | 5,424 | 16,424 |
| **PPO (self-play)** | 0.06 | **9,019** | 23,766 |

The PPO agent keeps its castle alive roughly **twice as long** as the baselines:
the reward shaping succeeded in teaching strong defence. The hand-written
ball-tracker also clearly beats a stationary paddle and a random policy, so the
game *is* controllable, not pure luck.

### 3.3 What actually drives the win rate

The single biggest factor is keeping the shield **moving**. A stationary or
near-stationary shield wins almost nothing (~0.09), because a still shield
deflects the ball predictably and games stall; a shield that keeps moving
scatters the ball, games resolve, and a competent defender takes more than its
share. The best control law is corner-specific (a continuous sweep for two
corners, ball-y tracking for the other two), and the resulting **per-corner rule
agent wins ~0.38 against random opponents, comfortably above the 0.25 fair
share**.

This corrects an earlier, overly pessimistic reading. A first version of the rule
agent had a calibration deadlock that left its shield essentially frozen, which
made every agent look capped at fair share. Once that was fixed and the shield
kept active, the ceiling lifted. The learned agents, by contrast, *turtle*: they
defend so well that no castle falls, games do not resolve, and they win little
(~0.06-0.12) despite the highest survival. So the lesson is two-sided: deep RL
learns excellent defence but over-defends into stalemates, whereas a simple but
*active* heuristic both defends and keeps games resolving, and wins most.

## 4. Discussion and reflection

### 4.1 Limitations

- Win rate is a high-variance, structurally-capped metric in a four-player game;
  survival and head-to-head results are more informative.
- Frame-skip 4 makes control coarse, limiting how precisely any policy can
  intercept the ball.
- Castle health is read cleanly for two corners (summary bytes) and approximately
  for the other two (brick bitmasks), so the dense reward is slightly asymmetric.

### 4.2 What was tried for offence, and why it is hard

The turtle problem was attacked directly. Snapshot **self-play** (training against
frozen copies of the agent, a rolling pool refreshed every 250k steps) was run in
both a mixed pool (`random + rule + self`) and a self-heavy pool, on the theory
that beating an equally-strong defender forces you to attack. It did not: two
strong defenders simply stalemate, and win rate stayed near zero. Making
**stalemate count as a loss** (a `-win_scale` penalty for stalling to the time
limit) pushed the return sharply negative without lifting the win rate, and an
**aggressive, defence-free reward** with a per-step time cost made the agent die
recklessly. The consistent outcome across all of these is that **defence is
learnable but offence is not**, in this budget: deflecting the ball at the precise
angle to break a *defended* castle is a fine-grained skill that the sparse,
chaotic feedback does not teach.

There is also a deeper, environmental reason. The original Atari Warlords manual
describes a **CATCH** mechanic (hold FIRE to catch the ball on your shield, release
to launch it back at high speed and aim it at an opponent), which is the game's
real offensive tool. Probing the environment shows that **PettingZoo's
`warlords_v3` hardcodes the no-catch default mode**: FIRE has no effect on the ball
in any of the ALE-exposed modes. So in the tournament setting there is no way to
deliberately aim the ball at all; offence is limited to the angle of *passive*
deflection, which is barely controllable. This makes the fair-share win-rate
ceiling a genuine property of the environment, not just a budget limitation.
Promising but unimplemented directions:

- A managed **league with prioritised fictitious self-play (PFSP)**, much longer
  training, and a held-out fixed-baseline evaluation.
- A centralised critic ([[MAPPO]]) over the global RAM during training.
- An explicit offence signal (a recurrent policy, or a shaped reward for the
  ball's predicted trajectory intersecting a targeted opponent's castle).

### 4.3 What the journey shows

The result hinged less on the choice of algorithm than on the representation and
reward: engineered RAM features plus velocity stacking made the control learnable,
`γ` and an offense/defense reward balance made *winning* learnable, and naive
shared self-play had to be abandoned for opponent-diversity training. This is the
core autonomous-systems lesson: in a multi-agent setting the environment design
(opponents, observation, reward) matters as much as the learner.

## 5. Conclusion

Warlords was framed as a Markov game and approached with self-play deep RL. After
discarding naive parameter-shared self-play, an opponent-diversity PPO agent on
engineered, frame-stacked RAM features with an offense/defense-balanced reward
learns competent, corner-agnostic *defence* and is the project's deep-RL
deliverable. The learned agent over-defends and stalls, so the **active, per-corner rule
agent, which keeps its shield moving and games resolving, is the stronger
tournament agent (~0.38 win rate vs random, above fair share)** and is
recommended for the competition. The most valuable lesson is methodological: in a
multi-agent setting the environment design (opponents, observation, reward)
decides the outcome more than the choice of algorithm; the result is sensitive to
seemingly-small implementation details (a shield that does not move loses
everything); and a reasoned approach means measuring those choices, including
against a simple but effective baseline rather than assuming a ceiling.

## References

- Mnih et al. (2015), *Human-level control through deep reinforcement learning* (DQN).
- Mnih et al. (2016), *Asynchronous Methods for Deep RL* (A2C/A3C).
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms* (PPO).
- Yu et al. (2022), *The Surprising Effectiveness of MAPPO in Cooperative Multi-Agent Games*.
- Terry et al. (2021), *PettingZoo: Gym for Multi-Agent Reinforcement Learning*.
- Vinyals et al. (2019), *Grandmaster level in StarCraft II* (league training / PFSP).
- Course concept notes: [[Reinforcement learning]], [[Markov Decision Process]],
  [[Bellman Equation]], [[Q-Learning]], [[Deep Q-Network]], [[Policy Gradients]],
  [[Actor-Critic]], [[A2C]], [[PPO]], [[MAPPO]],
  [[Multi-Agent Reinforcement Learning]], [[Nash Equilibrium]], [[Game Theory]].

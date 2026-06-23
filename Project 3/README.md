# Warlords: Multi-Agent Deep Reinforcement Learning

Portfolio 3 (Autonomous Systems). A self-play deep-RL agent for the four-player
Atari game **Warlords** (PettingZoo, RAM mode). DQN, A2C and PPO are trained
under an identical opponent-diversity setup and compared against a random and a
rule-based baseline; PPO is the most stable and becomes the deep-RL deliverable.

## The problem

Warlords is competitive multi-agent: four paddles each defend a corner fortress
and deflect a ball into the others (last fortress standing wins). From any one
paddle's view the others are a moving, learning environment, so this is a
**Multi-Agent RL (MARL)** problem with non-stationary dynamics.

Facts established by probing the environment (`ram_map.py`):

| Property | Value |
|---|---|
| Observation | 128-byte console RAM, **identical for all four players** |
| Action | `Discrete(6)`: NOOP, FIRE, up, right, left, down |
| Reward | sparse: `-1` when your fortress falls, `+1` for the last standing |
| Episode length | ~6,600 steps under random play; sticky actions off, frame-skip 4 |
| Paddle bytes | RAM `[91, 92, 93, 94]` (one per player) |
| Ball / health | ball `[90, 95, 111]`; castle health `108` (third), `109` (fourth) |

Because all players see the same RAM, an agent must **self-identify its own
paddle** at runtime (a short directional probe; see `agents/submission_agent.py`).

## Approach

- **Opponent-diversity self-play.** A naive shared policy controlling all four
  paddles collapses (0% win rate: four equal copies reach a mediocre mutual
  equilibrium). Instead one learner controls a random corner each episode and the
  other three are scripted opponents from a pool (random / rule / frozen self).
- **Engineered features + frame stacking.** The MLP cannot easily find the
  control bytes in raw RAM, so the observation is a compact vector (paddle, ball,
  castle healths) from the reverse-engineered RAM, with 3 frames stacked for
  velocity.
- **Reward shaping.** The native reward is far too sparse; a dense reward
  balances penalising damage to your own castle, rewarding damage you deal, and
  outlasting opponents, with a dominant scaled terminal. (`γ = 0.999` so the
  terminal is visible over the long episodes.)
- **Try every algorithm.** DQN, A2C and PPO trained identically and compared.

## Files

```
ram_map.py          reverse-engineered RAM: features, castle health, self-paddle detection
warlords_solo.py    opponent-diversity training env (engineered features, reward shaping)
train_solo.py       trainer for the solo env: --algo {dqn,a2c,ppo}
warlords_env.py     first attempt: shared-policy self-play via SuperSuit (collapsed)
train.py            trainer + shared PPO config for warlords_env
baselines.py        RandomAgent and RuleBasedAgent (ball-tracking heuristic)
evaluate.py         vs-baseline + round-robin tournament, plots
export_policy.py    export a trained policy to plain-numpy weights
agents/
  rule_agent.py         self-contained ball-tracking agent (best winner; recommended)
  submission_agent.py   self-contained numpy PPO agent (self-detect + features)
  policy_weights.npz    exported PPO policy
results/            training curves, comparison tables, plots
report/report.md    the written report
```

## Setup

```bash
brew install cmake                      # needed to compile the Atari backend
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/AutoROM --accept-license      # download Atari ROMs
```

## Reproduce

```bash
# train the three algorithms (opponent-diversity self-play)
.venv/bin/python train_solo.py --algo ppo --steps 3000000 --tag solo
.venv/bin/python train_solo.py --algo a2c --steps 1500000 --tag solo
.venv/bin/python train_solo.py --algo dqn --steps 1500000 --tag solo

.venv/bin/python evaluate.py --games 20      # compare + plots
.venv/bin/python export_policy.py --model ppo_solo   # bundle the tournament agent
```

## Results

| Agent | win rate vs random | note |
|---|---|---|
| random baseline | ~0.25 | fair share for a 4-player game |
| DQN / A2C / PPO (self-play) | ~0.06-0.12 | strong *defenders* but they *turtle* (stall games, rarely close them out) |
| **rule-based, per-corner (`rule_agent.py`)** | **~0.38** | actively-moving shield; the best agent |

The deep-RL agents learn strong defence (PPO survives ~2x longer than the
baselines, and is the most stable learner, see `results/training_curves.png`) but
over-defend into stalemates, so they win little. The **per-corner rule agent**
keeps its shield actively moving and wins **~0.38 against random opponents, well
above the 0.25 fair share**. An earlier version of the rule agent had a
calibration bug that left the shield nearly stationary (~0.09 win rate); keeping
the shield moving is the single biggest factor. See `report/report.md`.

## Tournament agent

Two numpy-only agents are provided. Every player sees the same RAM, so each takes
the fixed tournament seat (`Agent1`->`first_0`, ...) via `player_index` for a
guaranteed-correct corner, with a directional-probe self-detector as fallback.

- **`agents/rule_agent.py` (recommended for the tournament).** A ball-tracking
  heuristic. In evaluation it is the best *winner* because its active play keeps
  games resolving. Copy `rule_agent.py` next to the notebook:
  ```python
  from rule_agent import RuleAgent
  agent1 = RuleAgent(player_index=0)
  ```
- **`agents/submission_agent.py` (the deep-RL agent).** Runs the exported PPO
  policy on the engineered, frame-stacked features. It is the strongest *defender*
  (survives ~2x longer) but tends to over-defend and stall, so it wins less in
  practice. Copy `submission_agent.py` + `policy_weights.npz`:
  ```python
  from submission_agent import WarlordsAgent
  agent1 = WarlordsAgent(player_index=0)
  ```

## Related notes (Obsidian)

- Concepts: [[Multi-Agent Reinforcement Learning]], [[MAPPO]], [[PPO]],
  [[Policy Gradients]], [[Actor-Critic]], [[A2C]], [[Deep Q-Network]],
  [[Q-Learning]], [[Markov Decision Process]], [[Bellman Equation]],
  [[Nash Equilibrium]], [[Game Theory]]
- Course: [[ADS&AI/Year 3 - 2025-2026/Autonomous systems/Autonomous systems|Autonomous systems]]
- Sibling project: [[Autonomous Car Racing]] (Portfolio 2, single-agent DQN)

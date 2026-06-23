"""Warlords environment factory for self-play training with Stable-Baselines3.

The four paddles are homogeneous, so we train a single shared policy that
controls all of them (parameter sharing). Because every player receives the
same global RAM, an agent indicator (one-hot of which corner it defends) is
appended so the shared policy can specialise per corner. This is the standard
Centralised-Training / Decentralised-Execution recipe for a symmetric team.
"""
import numpy as np
import supersuit as ss
from pettingzoo.atari import warlords_v3
from pettingzoo.utils import BaseParallelWrapper

RAM_SIZE = 128
NUM_AGENTS = 4
OBS_SIZE = RAM_SIZE + NUM_AGENTS  # RAM + agent indicator one-hot


class SurvivalShaping(BaseParallelWrapper):
    """Dense reward shaping for the sparse last-man-standing objective.

    Warlords' native reward (-1 when your fortress falls, +1 for the last
    player standing) is far too sparse to learn from over ~6000-step games.
    Two dense terms are added and the native terminal reward is kept:
      * alive_bonus  -- small per-step reward for not being eliminated; gives a
        gradient on every frame and directly encodes "defend your castle".
      * outlast_bonus -- reward each surviving agent whenever an opponent is
        eliminated. This rewards *relative* skill, so the signal keeps pushing
        even when self-play opponents are equally good (a pure survival bonus
        would otherwise saturate at the episode cap).
    """

    def __init__(self, env, alive_bonus=0.005, outlast_bonus=0.5):
        super().__init__(env)
        self.alive_bonus = alive_bonus
        self.outlast_bonus = outlast_bonus

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        newly_dead = sum(1 for a in rewards if terminations.get(a, False))
        for a in rewards:
            if not terminations.get(a, False) and not truncations.get(a, False):
                rewards[a] += self.alive_bonus + self.outlast_bonus * newly_dead
        return obs, rewards, terminations, truncations, infos


def raw_parallel_env(max_cycles=4000, shaped=True, alive_bonus=0.005, outlast_bonus=0.5, render_mode=None):
    env = warlords_v3.parallel_env(obs_type="ram", max_cycles=max_cycles, render_mode=render_mode)
    if shaped:
        env = SurvivalShaping(env, alive_bonus=alive_bonus, outlast_bonus=outlast_bonus)
    return env


def make_vec_env(n_envs=8, num_cpus=8, max_cycles=4000, shaped=True, alive_bonus=0.005, outlast_bonus=0.5):
    """Vectorised, parameter-sharing env ready for SB3.

    Pipeline: black_death keeps eliminated paddles present (zero obs) so the
    agent set stays constant; RAM is cast to float and normalised to [0,1];
    the agent indicator is appended; then the 4 agents are unrolled into a
    vector env and replicated n_envs times.
    """
    env = raw_parallel_env(max_cycles=max_cycles, shaped=shaped,
                           alive_bonus=alive_bonus, outlast_bonus=outlast_bonus)
    env = ss.black_death_v3(env)
    env = ss.dtype_v0(env, np.float32)
    env = ss.normalize_obs_v0(env, env_min=0.0, env_max=1.0)
    env = ss.agent_indicator_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, n_envs, num_cpus=num_cpus, base_class="stable_baselines3")
    return env

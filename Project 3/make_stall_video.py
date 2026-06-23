"""Record a 'game never ends' (timeout / no winner) scenario.

Two strong defenders (the exported PPO turtle) sit at seats 0 and 2; the other
two seats are random and die quickly. The two turtles then keep deflecting the
ball and neither castle ever falls, so the game runs to the time limit with no
last-man-standing -> no winner. We record the stalled phase to an mp4.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import imageio
from pettingzoo.atari import warlords_v3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
from submission_agent import WarlordsAgent
from baselines import RandomAgent

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results", "videos", "stalemate_no_winner.mp4")
RECORD_CYCLES = 1500          # ~50s at 30 fps once the stall is reached
RECORD_START = 1200           # let the opening play out, then film the stall
MAX_CYCLES = 60000


def caption(frame, text):
    try:
        from PIL import Image, ImageDraw
        img = Image.fromarray(frame).resize((480, 630), Image.NEAREST)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 480, 16], fill=(0, 0, 0))
        d.text((4, 3), text, fill=(255, 255, 0))
        return np.asarray(img)
    except Exception:
        return frame


def main():
    env = warlords_v3.env(obs_type="ram", render_mode="rgb_array", max_cycles=MAX_CYCLES)
    env.reset(seed=4)
    names = list(env.agents)
    agents = {
        names[0]: WarlordsAgent(player_index=0),
        names[1]: RandomAgent(seed=11),
        names[2]: WarlordsAgent(player_index=2),
        names[3]: RandomAgent(seed=22),
    }
    frames = []
    step = 0
    cyc = 0
    recording = False
    rec_count = 0
    reward = {n: 0.0 for n in names}
    for ag in env.agent_iter():
        obs, r, term, trunc, info = env.last()
        reward[ag] += r
        action = None if (term or trunc) else agents[ag].act(obs)
        env.step(action)
        # one render per cycle (4 agent-steps)
        if step % 4 == 0:
            cyc += 1
            n_alive = len(env.agents)
            if not recording and cyc >= RECORD_START:
                recording = True
            if recording:
                fr = env.render()
                if fr is not None:
                    txt = f"cycle {cyc}  castles alive={n_alive}  -> nobody dies, no winner"
                    frames.append(caption(fr, txt))
                    rec_count += 1
                if rec_count >= RECORD_CYCLES:
                    break
        step += 1
    env.close()
    someone_won = max(reward.values()) > 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imageio.mimsave(OUT, frames, fps=30)
    print(f"alive at stop: still going, recorded {len(frames)} cycles")
    print(f"someone_won so far: {someone_won}  (False = stalemate/no winner)")
    print(f"saved {OUT}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()

"""Export a trained SB3 policy to a plain-numpy weight file.

The tournament agent must run with only numpy (torch may not be present in the
grading environment), so we dump the actor MLP's weights and reproduce the
forward pass by hand in agents/submission_agent.py.

  python export_policy.py --algo ppo
"""
import argparse
import os

import numpy as np

MODELS = os.path.join(os.path.dirname(__file__), "models")
OUT = os.path.join(os.path.dirname(__file__), "agents", "policy_weights.npz")


def export_actor(model, out_path):
    """Dump the SB3 MlpPolicy actor (two tanh layers + action head) to numpy."""
    sd = model.policy.state_dict()
    weights = dict(
        w0=sd["mlp_extractor.policy_net.0.weight"].cpu().numpy(),
        b0=sd["mlp_extractor.policy_net.0.bias"].cpu().numpy(),
        w1=sd["mlp_extractor.policy_net.2.weight"].cpu().numpy(),
        b1=sd["mlp_extractor.policy_net.2.bias"].cpu().numpy(),
        w2=sd["action_net.weight"].cpu().numpy(),
        b2=sd["action_net.bias"].cpu().numpy(),
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **{k: v.astype(np.float32) for k, v in weights.items()})
    return weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ppo_solo", help="model name in models/ (without .zip)")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    from stable_baselines3 import A2C, PPO
    cls = {"ppo": PPO, "a2c": A2C}[args.model.split("_")[0]]
    model = cls.load(os.path.join(MODELS, f"{args.model}.zip"), device="cpu")
    weights = export_actor(model, args.out)
    print(f"exported {args.model} actor -> {args.out}")
    for k, v in weights.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()

"""Export a trained DynamicsMLP checkpoint to a single .npz for the notebook."""
import argparse
import numpy as np
import torch
from pathlib import Path

from dynamics_model import DynamicsMLP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = DynamicsMLP(hidden=ckpt["hidden"], layers=ckpt["layers"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    backbone = model.backbone
    layers = []
    reward_layers = []
    # backbone: [Linear, GELU, ...] — reward head is a separate 1-layer head in
    # this implementation (head_reward).
    linears = [m for m in backbone if isinstance(m, torch.nn.Linear)]
    for i, lin in enumerate(linears):
        layers.append((f"w{i}", lin.weight.numpy()), (f"b{i}", lin.bias.numpy()))
    # reward head: single Linear(hidden, 1)
    reward_layers.append(("rw0", model.head_reward.weight.numpy().squeeze(0)))
    reward_layers.append(("rb0", model.head_reward.bias.numpy().squeeze(0)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        n_layers=len(linears),
        n_reward_layers=1,
        rew_scale=ckpt["rew_scale"],
        obs_dim=ckpt["obs_dim"],
        act_dim=ckpt["act_dim"],
        hw0=model.head_obs.weight.numpy(),
        hb0=model.head_obs.bias.numpy(),
        **{k: v for k, v in layers},
        **{k: v for k, v in reward_layers},
    )
    print("exported", out, "backbone layers:", len(linears))


if __name__ == "__main__":
    main()

"""Learned one-step dynamics model: (obs_t, action_t) -> (obs_{t+1}, reward).

Trains on transition files produced by tools/collect_data.py. The model is a
small MLP that pads variable-length building observations/actions to fixed
widths (512 obs, 160 action). Saved as a .pt for use by MPPI/distillation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OBS_DIM = 512
ACT_DIM = 160  # continuous (up to ~120) + discrete (up to ~11)


class DynamicsMLP(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=512, layers=4):
        super().__init__()
        layers_mlp = [nn.Linear(obs_dim + act_dim, hidden), nn.GELU()]
        for _ in range(layers - 1):
            layers_mlp += [nn.Linear(hidden, hidden), nn.GELU()]
        self.backbone = nn.Sequential(*layers_mlp)
        self.head_obs = nn.Linear(hidden, obs_dim)
        self.head_reward = nn.Linear(hidden, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        h = self.backbone(x)
        return self.head_obs(h), self.head_reward(h).squeeze(-1)


def load_dataset(pattern: str, max_samples: int = 0):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    obs_all, act_all, reward_all = [], [], []
    n = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        m = len(d["reward"]) - 1  # drop the last step (no next_obs)
        obs_all.append(d["obs"][:m])
        act_all.append(np.concatenate([d["cont"][:m], d["disc"][:m].astype(np.float32)], axis=1))
        reward_all.append(d["reward"][:m])
        n += m
        if max_samples and n >= max_samples:
            break
    obs = np.concatenate(obs_all, 0)
    act = np.concatenate(act_all, 0)
    reward = np.concatenate(reward_all, 0)
    # pad
    n_obs = obs.shape[1]
    if n_obs < OBS_DIM:
        obs = np.pad(obs, ((0, 0), (0, OBS_DIM - n_obs)))
    n_act = act.shape[1]
    if n_act < ACT_DIM:
        act = np.pad(act, ((0, 0), (0, ACT_DIM - n_act)))
    idx = np.random.default_rng(0).permutation(len(obs))
    obs, act, reward = obs[idx], act[idx], reward[idx]
    return torch.from_numpy(obs).float(), torch.from_numpy(act).float(), torch.from_numpy(reward).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="glob for transition npz files")
    p.add_argument("--out", default="rl/models/dynamics.pt")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; aborting as required", file=sys.stderr)
        sys.exit(3)
    device = torch.device(args.device)

    t0 = time.time()
    obs_t, act_t, rew_t = load_dataset(args.data, args.max_samples)
    print(f"loaded {len(obs_t)} transitions in {time.time()-t0:.0f}s", flush=True)

    model = DynamicsMLP(hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(obs_t) // args.batch))

    # reward scale
    rew_scale = float(torch.std(rew_t)) + 1e-8

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        nb = 0
        for i in range(0, len(obs_t), args.batch):
            ob = obs_t[i:i+args.batch].to(device, non_blocking=True)
            ac = act_t[i:i+args.batch].to(device, non_blocking=True)
            rw = rew_t[i:i+args.batch].to(device, non_blocking=True)
            pred_obs, pred_r = model(ob, ac)
            loss_obs = ((pred_obs - ob) ** 2).mean()
            loss_r = ((pred_r - rw) / rew_scale) ** 2
            # mask padding dims: only penalize the original width? keep simple: weight pad low
            loss = loss_obs + 0.5 * loss_r
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item()
            nb += 1
        print(f"epoch {epoch+1}/{args.epochs} loss={total/nb:.5f} "
              f"(obs={loss_obs.item():.5f} r={loss_r.item():.5f}) t={time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
                "hidden": args.hidden, "layers": args.layers, "rew_scale": rew_scale},
               args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()

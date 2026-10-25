#!/usr/bin/env python3
"""Collect (obs, action, reward, next_obs) transitions with a behavior policy + noise.

Usage:
  python tools/collect_data.py --policy-module policies/v2_policy.py --policy-class V2Policy \
      --out data/collect-v2 --workers 8 --noise 0.15 --flip 0.05 [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = RUN_ROOT / "starter-kit"
sys.path.insert(0, str(STARTER_KIT))

from smart_buildings.evaluate import (  # noqa: E402
    create_building_meta_data,
    load_hybrid_environment,
    resolve_config_dirs,
)


def load_policy_class(module_path, class_name):
    spec = importlib.util.spec_from_file_location("policy_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def collect_one(args):
    config_dir, policy_module, policy_class, out_dir, noise, flip, steps = args
    config_dir = Path(config_dir)
    out_dir = Path(out_dir)
    cls = load_policy_class(policy_module, policy_class)
    env = load_hybrid_environment(config_dir / "config.gin")
    # Match the organizer harness: num_occupants obs = (occ-125)/126.
    env._occupancy_normalization_constant = 125.0
    meta = create_building_meta_data(env, config_dir)
    spec = env.action_spec()
    policy = cls(meta, spec)
    rng = np.random.default_rng(hash(config_dir.name) & 0xFFFFFFFF)

    n_cont = spec["continuous_action"].shape[0]
    n_disc = spec["discrete_action"].shape[0]

    obs, _ = env.reset()
    n_obs = obs.shape[0]
    conts = []
    discs = []
    obs_list = []
    next_list = []
    rewards = []

    for t in range(steps):
        action = policy.get_action(obs)
        c = np.asarray(action["continuous_action"], dtype=np.float32)
        d = np.asarray(action["discrete_action"], dtype=np.int32)
        if noise > 0:
            c = np.clip(c + rng.normal(0.0, noise, n_cont).astype(np.float32), -1.0, 1.0)
        if flip > 0:
            d = np.where(rng.random(n_disc) < flip, 1 - d, d).astype(np.int32)
        next_obs, reward, terminated, truncated, _ = env.step(
            {"continuous_action": c, "discrete_action": d}
        )
        obs_list.append(obs)
        next_list.append(next_obs)
        conts.append(c)
        discs.append(d)
        rewards.append(float(reward))
        obs = next_obs
        if terminated or truncated:
            break

    name = f"{config_dir.parent.name}__{config_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{name}.npz",
        obs=np.array(obs_list, dtype=np.float32),
        next_obs=np.array(next_list, dtype=np.float32),
        cont=np.array(conts, dtype=np.float32),
        disc=np.array(discs, dtype=np.int32),
        reward=np.array(rewards, dtype=np.float32),
        n_obs=n_obs,
        n_cont=n_cont,
        n_disc=n_disc,
    )
    return {"scenario": name, "steps": len(rewards),
            "total_reward": float(np.sum(rewards)),
            "seconds": 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-module", required=True)
    p.add_argument("--policy-class", required=True)
    p.add_argument("--scenarios-dir", default=str(STARTER_KIT / "smart_buildings" / "training_scenarios"))
    p.add_argument("--buildings", nargs="*", default=None)
    p.add_argument("--weathers", nargs="*", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--flip", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=8640)
    p.add_argument("--limit", type=int, default=0, help="max scenarios (debug)")
    args = p.parse_args()
    args.policy_module = str(Path(args.policy_module).resolve())
    args.scenarios_dir = str(Path(args.scenarios_dir).resolve())
    args.out = str(Path(args.out).resolve())
    os.chdir(STARTER_KIT)

    dirs = resolve_config_dirs(Path(args.scenarios_dir), args.buildings, args.weathers)
    if args.limit:
        dirs = dirs[: args.limit]
    tasks = [(str(d), args.policy_module, args.policy_class, args.out,
              args.noise, args.flip, args.steps) for d in dirs]
    t0 = time.time()
    if args.workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            results = pool.map(collect_one, tasks)
    else:
        results = [collect_one(t) for t in tasks]
    wall = time.time() - t0
    print(json.dumps({"scenarios": len(results), "wall_seconds": wall}, indent=1))
    for r in results:
        print(f"  {r['scenario']:44s} steps={r['steps']} reward={r['total_reward']:.1f}")


if __name__ == "__main__":
    main()

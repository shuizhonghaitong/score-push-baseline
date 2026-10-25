#!/usr/bin/env python3
"""Run a policy across training scenarios and compute the leaderboard metric.

Usage:
  python tools/eval_local.py --policy-module policies/pid_water --policy-class PIDWaterResetPolicy \
      [--buildings building_train_1] [--weathers fairbanks_1] [--steps 8640] [--workers 4]

The policy module must expose a class with __init__(building_meta_data, action_spec)
and get_action(obs) -> dict.
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

import gin  # noqa: E402

from smart_buildings.evaluate import (  # noqa: E402
    create_building_meta_data,
    load_hybrid_environment,
    resolve_config_dirs,
)


def load_policy_class(module_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("policy_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def run_scenario(args) -> dict:
    config_dir, policy_module, policy_class, num_steps, diagnose = args
    config_dir = Path(config_dir)
    cls = load_policy_class(policy_module, policy_class)
    env = load_hybrid_environment(config_dir / "config.gin")
    # Match the organizer harness: num_occupants obs = (occ-125)/126.
    env._occupancy_normalization_constant = 125.0
    meta = create_building_meta_data(env, config_dir)
    spec = env.action_spec()
    policy = cls(meta, spec)
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    t0 = time.time()
    tot_npr = tot_ne = tot_nc = tot_occ = 0.0
    for _ in range(num_steps):
        action = policy.get_action(obs)
        action = {
            "continuous_action": np.asarray(action["continuous_action"], dtype=np.float32),
            "discrete_action": np.asarray(action["discrete_action"], dtype=np.int32),
        }
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if diagnose:
            _, rr = env.get_reward_info_and_response()
            tot_npr += float(rr.normalized_productivity_regret)
            tot_ne += float(rr.normalized_energy_cost)
            tot_nc += float(rr.normalized_carbon_emission)
            tot_occ += float(rr.total_occupancy)
        if terminated or truncated:
            break
    wall = time.time() - t0
    out = {
        "scenario": f"{config_dir.parent.name}/{config_dir.name}",
        "total_reward": total_reward,
        "steps": steps,
        "seconds": wall,
    }
    if diagnose:
        out.update({
            "comfort_loss": tot_npr / steps,
            "energy_loss": tot_ne / steps,
            "carbon_loss": tot_nc / steps,
            "avg_occupancy": tot_occ / steps,
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-module", required=True)
    parser.add_argument("--policy-class", required=True)
    parser.add_argument("--scenarios-dir", default=str(STARTER_KIT / "smart_buildings" / "training_scenarios"))
    parser.add_argument("--buildings", nargs="*", default=None)
    parser.add_argument("--weathers", nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=8640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    args.policy_module = str(Path(args.policy_module).resolve())
    args.scenarios_dir = str(Path(args.scenarios_dir).resolve())
    if args.out:
        args.out = str(Path(args.out).resolve())
    os.chdir(STARTER_KIT)  # gin configs hold paths relative to the starter-kit root

    config_dirs = resolve_config_dirs(
        Path(args.scenarios_dir), args.buildings, args.weathers
    )
    tasks = [(str(d), args.policy_module, args.policy_class, args.steps, args.diagnose) for d in config_dirs]
    t0 = time.time()
    if args.workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            results = pool.map(run_scenario, tasks)
    else:
        results = [run_scenario(t) for t in tasks]
    wall = time.time() - t0

    rows = []
    for r in results:
        rows.append({**r, "mean_reward": r["total_reward"] / r["steps"]})
    rows.sort(key=lambda r: r["scenario"])
    score = float(np.mean([abs(r["mean_reward"]) for r in rows]))
    summary = {
        "policy": f"{args.policy_module}:{args.policy_class}",
        "steps": args.steps,
        "num_scenarios": len(rows),
        "score_mae": score,
        "wall_seconds": wall,
        "scenarios": rows,
    }
    print(json.dumps({k: summary[k] for k in ("policy", "steps", "num_scenarios", "score_mae", "wall_seconds")}, indent=1))
    for r in rows:
        extra = ""
        if "comfort_loss" in r:
            extra = (f"  c={r['comfort_loss']:.4f} e={r['energy_loss']:.4f}"
                     f" k={r['carbon_loss']:.4f} occ={r['avg_occupancy']:.0f}")
        print(f"  {r['scenario']:36s} {r['total_reward']:12.2f}  {r['seconds']:8.1f}s{extra}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

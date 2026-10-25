#!/usr/bin/env python3
"""Run one policy on one scenario and dump per-step reward components for analysis."""
import argparse
import importlib.util
import json
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
)
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker  # noqa: E402


def load_policy_class(module_path, class_name):
    spec = importlib.util.spec_from_file_location("policy_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)  # building/weather
    p.add_argument("--policy-module", required=True)
    p.add_argument("--policy-class", required=True)
    p.add_argument("--steps", type=int, default=8640)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    os.chdir(STARTER_KIT)
    cfg_dir = Path("smart_buildings/training_scenarios") / args.scenario
    env = load_hybrid_environment(cfg_dir / "config.gin")
    meta = create_building_meta_data(env, cfg_dir)
    spec = env.action_spec()
    cls = load_policy_class(args.policy_module, args.policy_class)
    policy = cls(meta, spec)
    tracker = SimulationTracker(env, capture_snapshots=False)
    obs = tracker.last_obs
    t0 = time.time()
    n = 0
    for _ in range(args.steps):
        action = policy.get_action(obs)
        action = {
            "continuous_action": np.asarray(action["continuous_action"], dtype=np.float32),
            "discrete_action": np.asarray(action["discrete_action"], dtype=np.int32),
        }
        obs, _, terminated, truncated, _ = tracker.step(action)
        n += 1
        if terminated or truncated:
            break
    rc = tracker.reward_components
    if rc:
        keys = list(rc[0].keys())
        arr = np.array([[c[k] for k in keys] for c in rc])
        print("keys:", keys)
        print("sum per component (total over episode):")
        for i, k in enumerate(keys):
            print(f"  {k:28s} {arr[:, i].sum():12.2f}")
    print("total reward:", sum(tracker.rewards), "steps:", n, f"wall {time.time()-t0:.0f}s")
    if args.out:
        np.savez_compressed(args.out,
                            rewards=np.array(tracker.rewards),
                            occupancy=np.array(tracker.occupancy, dtype=float),
                            **({k: np.array([c[k] for c in rc]) for k in keys} if rc else {}),
                            zone_temps=np.array(tracker.zone_temps) if tracker.zone_temps else None)
        print("saved", args.out)


if __name__ == "__main__":
    main()

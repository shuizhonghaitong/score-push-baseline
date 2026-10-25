"""Probe per-step regret components for constant action configurations.

Loads one scenario, runs N steps with a fixed action vector, and reports mean
normalized regret terms (NPR/NEC/NCE) plus total reward. Run with 1 worker;
cheap (a few minutes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = RUN_ROOT / "starter-kit"
sys.path.insert(0, str(STARTER_KIT))
os_chdir = __import__("os").chdir
os_chdir(STARTER_KIT)

from smart_buildings.evaluate import (  # noqa: E402
    create_building_meta_data,
    load_hybrid_environment,
)

SCEN = sys.argv[1] if len(sys.argv) > 1 else "smart_buildings/training_scenarios/building_train_8/atlanta_1"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1400

env = load_hybrid_environment(Path(SCEN) / "config.gin")
meta = create_building_meta_data(env, Path(SCEN))
spec = env.action_spec()
nd, nc = meta["n_discrete_action"], meta["n_continuous_action"]
print(f"n_discrete={nd} n_continuous={nc}")
print("temp_indices", meta["temp_indices"])
print("pressure_indices", meta["pressure_indices"])
print("water_indices", meta["water_indices"])
print("recirculation_indices", meta["recirculation_indices"])
print("zone_action_indices", meta["zone_action_indices"])
n_ahu = len(meta["temp_indices"])

def make_action(ahu_run=0, pres=-1.0, sat=0.0, recirc=1.0,
                damper=0.0, reheat=0.0, water_run=0, water_temp=-1.0, water_press=-1.0):
    cont = np.zeros(nc, dtype=np.float32)
    disc = np.zeros(nd, dtype=np.int32)
    for i in range(n_ahu):
        cont[meta["temp_indices"][i]] = sat
        if i < len(meta["pressure_indices"]):
            cont[meta["pressure_indices"][i]] = pres
        if i < len(meta["recirculation_indices"]):
            cont[meta["recirculation_indices"][i]] = recirc
        disc[i] = ahu_run
    for zid, acts in meta["zone_action_indices"].items():
        if "supply_air_damper_percentage_command" in acts:
            cont[acts["supply_air_damper_percentage_command"]] = 2*damper - 1
        if "reheat_valve_percentage_command" in acts:
            cont[acts["reheat_valve_percentage_command"]] = 2*reheat - 1
    # water: last discrete, water temp/pressure from water_indices / extra pressure
    if nd > n_ahu:
        disc[n_ahu] = water_run
    wi = meta["water_indices"]
    if wi:
        cont[wi[0]] = water_temp
    extra_p = meta["pressure_indices"][n_ahu:]
    if extra_p:
        cont[extra_p[0]] = water_press
    return {"continuous_action": cont, "discrete_action": disc}

CONFIGS = {
    "off":            dict(),
    "standby":        dict(ahu_run=1),
    "standby_cold":   dict(ahu_run=1, sat=-1.0),
    "standby_d05":    dict(ahu_run=1, damper=0.5),
    "fan_d05":        dict(ahu_run=1, pres=1.0, damper=0.5),
    "fan_oa_cold":    dict(ahu_run=1, pres=1.0, sat=-1.0, recirc=0.0, damper=0.5),
    "standby_water":  dict(ahu_run=1, water_run=1, water_temp=1.0, water_press=1.0),
    "fan_water_reh":  dict(ahu_run=1, pres=1.0, damper=0.5, reheat=1.0,
                           water_run=1, water_temp=1.0, water_press=1.0),
    "standby_d01":    dict(ahu_run=1, damper=0.1),
}

out = {}
for name, kw in CONFIGS.items():
    env.reset()
    action = make_action(**kw)
    tot_r = 0.0
    npr_s = nec_s = nce_s = 0.0
    n = 0
    for _ in range(STEPS):
        obs, reward, term, trunc, info = env.step(action)
        tot_r += reward
        _, resp = env.get_reward_info_and_response()
        npr_s += resp.normalized_productivity_regret
        nec_s += resp.normalized_energy_cost
        nce_s += resp.normalized_carbon_emission
        n += 1
        if term or trunc:
            break
    out[name] = dict(
        mean_reward=tot_r/n, npr=npr_s/n, nec=nec_s/n, nce=nce_s/n,
        r_ec=-0.2*(nec_s+nce_s)/n,
    )
    o = out[name]
    print(f"{name:16s} r={o['mean_reward']:+.6f}  NPR={o['npr']:+.6f}  "
          f"NEC={o['nec']:.6f}  NCE={o['nce']:.6f}  E+C term={o['r_ec']:+.6f}", flush=True)

json.dump(out, open("results/probe_costs.json", "w"), indent=1)
print("wrote results/probe_costs.json")

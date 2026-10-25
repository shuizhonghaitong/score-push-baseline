"""Run a policy over a full scenario; aggregate NEC/NCE + device power by
hour-of-day and weekday/weekend. One scenario, single process."""
from __future__ import annotations
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "starter-kit"))
os.chdir(RUN_ROOT / "starter-kit")
from smart_buildings.evaluate import (  # noqa: E402
    create_building_meta_data,
    load_hybrid_environment,
    resolve_config_dirs,
)
from smart_buildings.smart_control.utils import conversion_utils  # noqa: E402

def load_policy_class(module_path, class_name):
    spec = importlib.util.spec_from_file_location("policy_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def main():
    mod, cls, scen, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    config_dir = Path(scen)
    env = load_hybrid_environment(config_dir / "config.gin")
    env._occupancy_normalization_constant = 125.0
    meta = create_building_meta_data(env, config_dir)
    spec = env.action_spec()
    policy = load_policy_class(mod, cls)(meta, spec)
    obs, _ = env.reset()
    total_reward = 0.0
    # hour-of-day aggregates: nec, nce, blower_W, ac_W, pump_W, gas_W, occ, nsteps
    hod = defaultdict(lambda: defaultdict(float))
    dow = defaultdict(lambda: defaultdict(float))
    for k in range(steps):
        action = policy.get_action(obs)
        action = {
            "continuous_action": np.asarray(action["continuous_action"], dtype=np.float32),
            "discrete_action": np.asarray(action["discrete_action"], dtype=np.int32),
        }
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += float(reward)
        ri, resp = env.get_reward_info_and_response()
        ts = conversion_utils.proto_to_pandas_timestamp(ri.start_timestamp)
        h = ts.hour
        key_d = "weekday" if ts.dayofweek < 5 else "weekend"
        nec, nce = float(resp.normalized_energy_cost), float(resp.normalized_carbon_emission)
        blower = ac = pump = gas = 0.0
        for aid, ai in ri.air_handler_reward_infos.items():
            blower += float(ai.blower_electrical_energy_rate)
            ac += float(ai.air_conditioning_electrical_energy_rate)
        for bid, b in ri.boiler_reward_infos.items():
            gas += float(b.natural_gas_heating_energy_rate)
            pump += float(b.pump_electrical_energy_rate)
        occ = float(resp.total_occupancy)
        for agg, key in ((hod, h), (dow, key_d)):
            a = agg[key]
            a["nec"] += nec; a["nce"] += nce; a["n"] += 1
            a["blower"] += blower; a["ac"] += ac; a["pump"] += pump
            a["gas"] += gas; a["occ"] += occ
        if term or trunc:
            break
    print(f"scenario={config_dir.parent.name}/{config_dir.name} steps={k+1} total_reward={total_reward:.2f}")
    print("== by hour of day (300s steps; NEC/NCE are step-averaged regrets) ==")
    print(f"{'hod':>3s} {'n':>5s} {'NEC':>9s} {'NCE':>9s} {'blower_W':>9s} {'ac_W':>8s} {'pump_W':>8s} {'gas_W':>9s} {'occ':>6s}")
    for h in range(24):
        a = hod.get(h)
        if not a or a["n"] == 0:
            continue
        n = a["n"]
        print(f"{h:3d} {n:5.0f} {a['nec']/n:9.6f} {a['nce']/n:9.6f} {a['blower']/n:9.0f} {a['ac']/n:8.0f} {a['pump']/n:8.0f} {a['gas']/n:9.0f} {a['occ']/n:6.1f}")
    print("== weekday vs weekend (step-averaged) ==")
    print(f"{'key':>8s} {'n':>6s} {'NEC':>9s} {'NCE':>9s} {'blower_W':>9s} {'ac_W':>8s} {'pump_W':>8s} {'gas_W':>9s} {'occ':>6s}")
    for key in ("weekday", "weekend"):
        a = dow.get(key)
        if not a or a["n"] == 0:
            continue
        n = a["n"]
        print(f"{key:>8s} {n:6.0f} {a['nec']/n:9.6f} {a['nce']/n:9.6f} {a['blower']/n:9.0f} {a['ac']/n:8.0f} {a['pump']/n:8.0f} {a['gas']/n:9.0f} {a['occ']/n:6.1f}")
    tot = defaultdict(float)
    for a in hod.values():
        for kk in a:
            tot[kk] += a[kk]
    n = tot["n"]
    print(f"== totals: NEC_share_elec vs gas not separable; device W means: "
          f"blower={tot['blower']/n:.0f}W ac={tot['ac']/n:.0f}W pump={tot['pump']/n:.0f}W gas={tot['gas']/n:.0f}W "
          f"meanNEC={tot['nec']/n:.6f} meanNCE={tot['nce']/n:.6f} meanOcc={tot['occ']/n:.1f} steps={n:.0f}")

if __name__ == "__main__":
    main()

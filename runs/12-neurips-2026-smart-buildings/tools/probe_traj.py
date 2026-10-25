"""Log zone-temp trajectory + plant power for the first N steps of a scenario."""
from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "starter-kit"))
os.chdir(RUN_ROOT / "starter-kit")
from smart_buildings.evaluate import (  # noqa: E402
    create_building_meta_data,
    load_hybrid_environment,
)

def load_policy_class(module_path, class_name):
    spec = importlib.util.spec_from_file_location("pm", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def main():
    mod, cls, scen, steps, every = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    config_dir = Path(scen)
    env = load_hybrid_environment(config_dir / "config.gin")
    env._occupancy_normalization_constant = 125.0
    meta = create_building_meta_data(env, config_dir)
    policy = load_policy_class(mod, cls)(meta, env.action_spec())
    obs, _ = env.reset()
    n_z = 0
    for k in range(steps):
        action = policy.get_action(obs)
        action = {
            "continuous_action": np.asarray(action["continuous_action"], dtype=np.float32),
            "discrete_action": np.asarray(action["discrete_action"], dtype=np.int32),
        }
        obs, reward, term, trunc, _ = env.step(action)
        if k % every == 0:
            zt = [policy._denorm(obs[z["temp_idx"]], policy.zt_mean, policy.zt_std)
                  for z in policy.zones
                  if z["temp_idx"] is not None and np.isfinite(obs[z["temp_idx"]])]
            gas = 0.0
            try:
                ri, _ = env.get_reward_info_and_response()
                for b in ri.boiler_reward_infos.values():
                    gas += float(b.natural_gas_heating_energy_rate)
                ac = sum(float(a.air_conditioning_electrical_energy_rate)
                         for a in ri.air_handler_reward_infos.values())
            except Exception:
                ac = 0.0
            t = 6.0 + k * 300.0 / 3600.0
            day = int(t // 24)
            hod = int(t % 24)
            comfort = obs[policy.comfort_now_index]
            print(f"day={day} hod={hod:02d} zt={float(np.mean(zt)):.2f}K "
                  f"zmin={float(np.min(zt)):.2f} comfort={comfort:.0f} gas={gas:.0f}W ac={ac:.0f}W", flush=True)
        if term or trunc:
            break

if __name__ == "__main__":
    main()

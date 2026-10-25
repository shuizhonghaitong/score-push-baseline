import sys, os
import os as _os
_p = _os.path.abspath("starter-kit")
sys.path.insert(0, _p)
_os.chdir(_p)
from pathlib import Path
import numpy as np
from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data

env = load_hybrid_environment(Path("smart_buildings/training_scenarios/building_train_8/atlanta_1/config.gin"))
meta = create_building_meta_data(env, Path("smart_buildings/training_scenarios/building_train_8/atlanta_1"))
nc, nd = meta["n_continuous_action"], meta["n_discrete_action"]
n_ahu = len(meta["temp_indices"])
cont = np.zeros(nc, dtype=np.float32); disc = np.zeros(nd, dtype=np.int32)
disc[:n_ahu] = 1
for i in range(n_ahu):
    cont[meta["temp_indices"][i]] = 0.0
    cont[meta["pressure_indices"][i]] = float(sys.argv[1]) if len(sys.argv) > 1 else -1.0
    cont[meta["recirculation_indices"][i]] = 1.0
action = {"continuous_action": cont, "discrete_action": disc}
env.reset()
sim = env.building.simulator
for k in range(2880):
    obs, reward, term, trunc, info = env.step(action)
    if k % 288 == 0:
        ah = sim._hvac.air_handler
        s = [a for a in dir(ah) if a.startswith("ahu_1_")]
        pres = None
        for cand in ("ahu_1_supply_air_static_pressure_sensor",):
            pres = getattr(ah, cand, None)
        sp = getattr(ah, "ahu_1_supply_air_static_pressure_setpoint", None)
        fl = getattr(ah, "ahu_1_supply_air_flowrate_sensor", None)
        ri, _ = env.get_reward_info_and_response()
        b = list(ri.air_handler_reward_infos.values())[0].blower_electrical_energy_rate
        t = sim._current_timestamp
        print(f"t={t} cmd={sys.argv[1] if len(sys.argv)>1 else -1} setpoint={sp} sensor={pres} flow={fl} blower={b:.0f}W", flush=True)

"""Print per-device energy rates + averaged NEC for a constant action config."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "starter-kit"))
__import__("os").chdir(RUN_ROOT / "starter-kit")
from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data

SCEN = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1400
SAT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

env = load_hybrid_environment(Path(SCEN) / "config.gin")
meta = create_building_meta_data(env, Path(SCEN))
nc, nd = meta["n_continuous_action"], meta["n_discrete_action"]
n_ahu = len(meta["temp_indices"])

cont = np.zeros(nc, dtype=np.float32)
disc = np.zeros(nd, dtype=np.int32)
disc[:n_ahu] = 1
for i in range(n_ahu):
    cont[meta["temp_indices"][i]] = SAT
    if i < len(meta["pressure_indices"]):
        cont[meta["pressure_indices"][i]] = -1.0
    if i < len(meta["recirculation_indices"]):
        cont[meta["recirculation_indices"][i]] = 1.0
action = {"continuous_action": cont, "discrete_action": disc}

env.reset()
nec_sum = blower_sum = 0.0
n = 0
for k in range(STEPS):
    obs, reward, term, trunc, info = env.step(action)
    _, resp = env.get_reward_info_and_response()
    nec_sum += resp.normalized_energy_cost
    n += 1
    if term or trunc: break

hvac = env.building.simulator._hvac
ah = hvac.air_handler
# AirHandlerSystem?
attrs = [a for a in dir(ah) if "flow" in a or "pressure" in a or "energy" in a]
print("AHU type:", type(ah).__name__)
handlers = getattr(ah, "air_handlers", None) or getattr(ah, "_air_handlers", None)
if handlers:
    for i, h in enumerate(handlers):
        print(f"AHU{i}: flow={h.air_flow_rate:.3f} m3/s pres_sensor={h.supply_air_static_pressure_sensor:.0f} Pa sat={h.supply_air_temperature_sensor:.2f}")
else:
    print("flow:", ah.air_flow_rate, "pres:", ah.supply_air_static_pressure_sensor)
ri, _ = env.get_reward_info_and_response()
for aid, ahinfo in ri.air_handler_reward_infos.items():
    print(f"AHU {aid}: blower={ahinfo.blower_electrical_energy_rate:.1f} W  AC={ahinfo.air_conditioning_electrical_energy_rate:.1f} W")
for bid, b in ri.boiler_reward_infos.items():
    print(f"Boiler {bid}: gas={b.natural_gas_heating_energy_rate:.1f} W  pump={b.pump_electrical_energy_rate:.1f} W")
hws = hvac.hot_water_system
print("water setpoint:", hws.supply_water_temperature_setpoint, "sensor:", hws.supply_water_temperature_sensor, "flow:", hws.total_flow_rate)
print(f"avg NEC over {n} steps: {nec_sum/n:.6f}")

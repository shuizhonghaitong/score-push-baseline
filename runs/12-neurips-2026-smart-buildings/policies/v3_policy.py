"""V3 rule-based HVAC policy (zone-PI architecture).

Architecture
------------
* PI hours: weekdays from one hour before the morning comfort start until the
  evening start, plus a pre-conditioning lead. Each zone runs a PI loop on
  (temperature - target) with a single signal u in [-1, 1] (u > 0 = cooling
  demand). The damper opens with |u|, the reheat valve with -u, so one signal
  coordinates both actuators per zone.
* While any zone of an AHU has demand, the AHU is forced on. Its supply-air
  temperature follows the net demand of its zones: maximum (no chiller) for
  net heating, ramping cold for net cooling. Fan pressure follows effort.
* A cooling-run hold keeps a unit that started cooling running for at least
  HOLD_STEPS steps at its last commands: the shipped baseline's 50-100
  on/off cycles per day were a major loss in hot climates.
* Outside PI hours the zones are (almost) unoccupied, so comfort costs
  nothing and the plant shuts down: the building free-floats. Nighttime
  comfort is therefore never bought, and pre-conditioning before the morning
  start does all the work of reaching the day band on time.
* Economizer: while cooling, outside air is blended proportionally onto the
  supply-air target when the ambient advantage beats the fan breakeven.
* Hot water: demand reset on the worst reheat demand; full temperature while
  any zone heats during PI hours; pump runs with hysteresis.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

SUPPLY_AIR_MIN_K = 285.0
SUPPLY_AIR_MAX_K = 305.0
STATIC_PRESSURE_MAX_PA = 2000.0
MIN_STATIC_PRESSURE_PA = 200.0
ECON_K_PER_PA = 0.00322


class V3Policy:
    KP = 0.2
    KI = 0.02
    INTEGRAL_LIMIT_FRAC = 1.0
    INTEGRAL_LEAK = 0.0
    MIN_DAMPER = 0.1
    HEAT_DAMPER_GAIN = 1.0
    HOLD_STEPS = 288
    PRESTART_H = 3
    EVENING_STOP_H = 0
    COLD_SAT = -0.7
    SAT_RAMP = 0.10        # max cooling demand that reaches COLD_SAT
    WATER_SCALE = 1.0
    TARGET_SHIFT_K = 0.30
    NIGHT_COOL_CEILING = True
    OCC_GATE = True
    OCC_MIN = 0.5

    def __init__(self, building_meta_data, action_spec):
        self.meta = building_meta_data
        self.action_spec = action_spec
        self.n_discrete = building_meta_data["n_discrete_action"]
        self.n_continuous = building_meta_data["n_continuous_action"]
        self.temp_indices = list(building_meta_data["temp_indices"])
        self.pressure_indices = list(building_meta_data["pressure_indices"])
        self.water_indices = list(building_meta_data["water_indices"])
        self.recirculation_indices = list(building_meta_data["recirculation_indices"])
        self.zone_action_indices = building_meta_data["zone_action_indices"]
        self.field_names = list(building_meta_data["observation_field_names"])
        self.ahu_zone_assignments = building_meta_data["ahu_zone_assignments"]
        self.comfort_temp_window = building_meta_data["comfort_temp_window"]
        self.eco_temp_window = building_meta_data["eco_temp_window"]
        self.morning_start_hour = building_meta_data["morning_start_hour"]
        self.evening_start_hour = building_meta_data["evening_start_hour"]

        norm_meta = building_meta_data["observation_normalization_metadata"]

        def norm(field, default_mean, default_std):
            entry = norm_meta.get(field)
            if entry is None:
                return default_mean, default_std
            return entry["sample_mean"], entry["sample_std"]

        self.zt_mean, self.zt_std = norm("zone_air_temperature_sensor", 295.0, 5.0)
        self.oat_mean, self.oat_std = norm("outside_air_temperature_sensor", 295.0, 10.0)

        names = self.field_names
        self.hod_sin_index = names.index("hod_sin_000")
        self.hod_cos_index = names.index("hod_cos_000")
        self.dow_sin_index = names.index("dow_sin_000")
        self.dow_cos_index = names.index("dow_cos_000")
        self.comfort_now_index = names.index("comfort_mode_now")
        self.comfort_soon_index = names.index("comfort_mode_soon")
        self.occ_index = names.index("num_occupants") if "num_occupants" in names else None

        self._build_ahu_mapping()
        self._build_zone_mapping()
        self._reset_state()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _numeric_suffix(name):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 0

    def _build_ahu_mapping(self):
        ahu_ids = sorted(self.ahu_zone_assignments.keys(), key=self._numeric_suffix)
        self.n_ahu = len(ahu_ids)
        names = self.field_names

        def find(field):
            try:
                return names.index(field)
            except ValueError:
                return None

        self.ahus = []
        for i, ahu_id in enumerate(ahu_ids):
            prefix = f"ahs_{ahu_id}_"
            self.ahus.append({
                "temp_idx": self.temp_indices[i],
                "pressure_idx": self.pressure_indices[i] if i < len(self.pressure_indices) else None,
                "recirc_idx": self.recirculation_indices[i] if i < len(self.recirculation_indices) else None,
                "run_discrete_idx": i,
                "oat_idx": find(prefix + "outside_air_temperature_sensor"),
            })
        self.water_temp_idx = self.water_indices[0] if self.water_indices else None
        self.water_pressure_idx = (
            self.pressure_indices[-1] if len(self.pressure_indices) > self.n_ahu else None)
        self.water_discrete_idx = self.n_discrete - 1

    def _build_zone_mapping(self):
        names = self.field_names

        def find(field):
            try:
                return names.index(field)
            except ValueError:
                return None

        self.zones = []
        for vav_id, action_indices in self.zone_action_indices.items():
            prefix = f"{vav_id}_"
            self.zones.append({
                "vav_id": vav_id,
                "temp_idx": find(prefix + "zone_air_temperature_sensor"),
                "heat_sp_idx": find(prefix + "zone_air_heating_temperature_setpoint_sensor"),
                "cool_sp_idx": find(prefix + "zone_air_cooling_temperature_setpoint_sensor"),
                "damper_idx": action_indices.get("supply_air_damper_percentage_command"),
                "reheat_idx": action_indices.get("reheat_valve_percentage_command"),
                "ahu_idx": None,
            })
        for ahu_idx, (_, rooms) in enumerate(sorted(
                self.ahu_zone_assignments.items(), key=lambda kv: self._numeric_suffix(kv[0]))):
            if ahu_idx >= self.n_ahu:
                break
            for room in rooms:
                vav_id = room.replace("room_", "VAV_")
                for z in self.zones:
                    if z["vav_id"] == vav_id:
                        z["ahu_idx"] = ahu_idx
                        break

    def _reset_state(self):
        self._zone_integral = {z["vav_id"]: 0.0 for z in self.zones}
        self._on_steps = [0] * self.n_ahu
        self._last_on = [None] * self.n_ahu
        self._pump_on = False

    # ---------------------------------------------------------------- helpers
    def _cyclic_index(self, sin_v, cos_v, period):
        angle = float(np.arctan2(sin_v, cos_v))
        if angle < 0:
            angle += 2 * np.pi
        return int(np.floor(period * angle / (2 * np.pi)))

    def _denorm(self, value, mean, std):
        return float(value) * std + mean

    def _pi_active(self, obs):
        hour = self._cyclic_index(obs[self.hod_sin_index], obs[self.hod_cos_index], 24)
        dow = self._cyclic_index(obs[self.dow_sin_index], obs[self.dow_cos_index], 7)
        comfort_now = bool(obs[self.comfort_now_index] > 0.5)
        comfort_soon = bool(obs[self.comfort_soon_index] > 0.5)
        if dow >= 5:
            return False
        active = (
            self.morning_start_hour - self.PRESTART_H <= hour
            < self.evening_start_hour - self.EVENING_STOP_H
        ) or comfort_now or comfort_soon
        if self.OCC_GATE and self.occ_index is not None and active:
            # After the arrival hour, require reported occupancy. This skips
            # holidays and empty days, where conditioning would be pure cost.
            if hour >= self.morning_start_hour + 2 and not comfort_now:
                if float(obs[self.occ_index]) < self.OCC_MIN:
                    return False
        return active

    def _zone_band(self, obs, zone, low, high):
        lo_idx, hi_idx = zone["heat_sp_idx"], zone["cool_sp_idx"]
        if lo_idx is not None and hi_idx is not None:
            lo = float(obs[lo_idx])
            hi = float(obs[hi_idx])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return lo, hi
        if self.NIGHT_COOL_CEILING:
            # Outside work hours: no overnight drift above the day cooling
            # setpoint; full heating setback kept.
            return (min(low, self.comfort_temp_window[0]), self.comfort_temp_window[1])
        return low, high

    # ------------------------------------------------------------------ action
    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        hour = self._cyclic_index(obs[self.hod_sin_index], obs[self.hod_cos_index], 24)
        pi_active = self._pi_active(obs)

        comfort_lo, comfort_hi = self.comfort_temp_window
        eco_lo, eco_hi = self.eco_temp_window
        if pi_active:
            band_lo, band_hi = comfort_lo, comfort_hi
        else:
            band_lo, band_hi = eco_lo, eco_hi

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)

        # Default: everything off.
        for i in range(self.n_ahu):
            discrete[self.ahus[i]["run_discrete_idx"]] = 0
        if 0 <= self.water_discrete_idx < self.n_discrete:
            discrete[self.water_discrete_idx] = 0

        max_heat = 0.0
        zone_u = {}

        if pi_active:
            # Outside-air side of the band for the target shift.
            oat_any = next((a["oat_idx"] for a in self.ahus if a["oat_idx"] is not None), None)
            outside = (
                self._denorm(obs[oat_any], self.oat_mean, self.oat_std) if oat_any is not None else None
            )

            for z in self.zones:
                vav = z["vav_id"]
                if z["temp_idx"] is None:
                    zone_u[vav] = 0.0
                    continue
                lo, hi = self._zone_band(obs, z, band_lo, band_hi)
                # Before occupancy the sensor band is the eco band; aim at comfort.
                if pi_active and hi - lo > (comfort_hi - comfort_lo) + 1e-6:
                    lo, hi = comfort_lo, comfort_hi
                centre = 0.5 * (lo + hi)
                if self.TARGET_SHIFT_K > 0 and outside is not None:
                    shift = min(self.TARGET_SHIFT_K, 0.4 * (hi - lo))
                    target = centre - shift if outside < centre else centre + shift
                else:
                    target = centre

                t = self._denorm(obs[z["temp_idx"]], self.zt_mean, self.zt_std)
                err = t - target  # > 0: hot -> cooling demand
                integral = self._zone_integral[vav]
                u_raw = self.KP * err + self.KI * integral
                integral *= 1.0 - self.INTEGRAL_LEAK
                if abs(u_raw) < 1.0 or np.sign(err) != np.sign(u_raw):
                    limit = self.INTEGRAL_LIMIT_FRAC / self.KI
                    integral = float(np.clip(integral + err, -limit, limit))
                self._zone_integral[vav] = integral
                u = float(np.clip(self.KP * err + self.KI * integral, -1.0, 1.0))
                zone_u[vav] = u
                max_heat = max(max_heat, -u)

                if z["damper_idx"] is not None:
                    gain = self.HEAT_DAMPER_GAIN if u < 0 else 1.0
                    damper = self.MIN_DAMPER + (1.0 - self.MIN_DAMPER) * abs(u) * gain
                    continuous[z["damper_idx"]] = 2.0 * damper - 1.0
                if z["reheat_idx"] is not None:
                    reheat = max(0.0, -u)
                    continuous[z["reheat_idx"]] = 2.0 * reheat - 1.0
        else:
            for z in self.zones:
                zone_u[z["vav_id"]] = 0.0
                self._zone_integral[z["vav_id"]] *= 0.9
            for i in range(self.n_ahu):
                self._on_steps[i] = 0
                self._last_on[i] = None

        # ---- AHU run / supply air temperature / pressure / recirculation ----
        ahu_demand = []
        for i in range(self.n_ahu):
            ahu = self.ahus[i]
            us = [zone_u[z["vav_id"]] for z in self.zones if z["ahu_idx"] == i]
            demand = float(np.mean(us)) if us else 0.0
            cool_demand = float(max((u for u in us), default=0.0))
            heat_demand = float(-min((u for u in us), default=0.0))
            ahu_demand.append(demand)
            run = bool(discrete[ahu["run_discrete_idx"]])
            if pi_active and (demand > 0.02 or demand < -0.02) and not run:
                discrete[ahu["run_discrete_idx"]] = 1
                run = True
            if run:
                self._on_steps[i] += 1
                # SAT from the extreme zone: any zone that needs cooling pulls
                # the supply cold (dampers split the air); any zone that needs
                # heat pins the supply warm (no chiller work).
                if heat_demand > 0.02:
                    sat = 1.0
                else:
                    frac = float(np.clip(cool_demand / self.SAT_RAMP, 0.0, 1.0))
                    sat = 1.0 + frac * (self.COLD_SAT - 1.0)
                continuous[ahu["temp_idx"]] = sat
                effort = float(np.clip(max(cool_demand, heat_demand) / self.SAT_RAMP, 0.0, 1.0))
                if ahu["pressure_idx"] is not None:
                    continuous[ahu["pressure_idx"]] = float(
                        np.clip(-1.0 + 2.0 * (0.3 + 0.7 * effort), -1.0, 1.0))
                if ahu["recirc_idx"] is not None:
                    continuous[ahu["recirc_idx"]] = self._recirculation(obs, ahu, demand)
                self._last_on[i] = (
                    float(sat),
                    float(continuous[ahu["pressure_idx"]]) if ahu["pressure_idx"] is not None else 0.0,
                    float(continuous[ahu["recirc_idx"]]) if ahu["recirc_idx"] is not None else 1.0,
                )
            else:
                # Cooling-run hold: keep a recently cooling unit on.
                if (self._last_on[i] is not None and 0 < self._on_steps[i] < self.HOLD_STEPS
                        and self._last_on[i][0] < 0.5):
                    discrete[ahu["run_discrete_idx"]] = 1
                    self._on_steps[i] += 1
                    continuous[ahu["temp_idx"]] = self._last_on[i][0]
                    if ahu["pressure_idx"] is not None:
                        continuous[ahu["pressure_idx"]] = self._last_on[i][1]
                    if ahu["recirc_idx"] is not None:
                        continuous[ahu["recirc_idx"]] = self._last_on[i][2]

        # ---- hot water ----
        any_heating = max_heat > 0.02
        if any_heating:
            self._pump_on = True
        elif max_heat < -0.05 or not pi_active:
            self._pump_on = False
        if 0 <= self.water_discrete_idx < self.n_discrete:
            discrete[self.water_discrete_idx] = 1 if self._pump_on else 0
        if self.water_temp_idx is not None:
            if self._pump_on:
                frac = min(1.0, 0.4 + 0.6 * max_heat) * self.WATER_SCALE
                continuous[self.water_temp_idx] = 2.0 * frac - 1.0
            else:
                continuous[self.water_temp_idx] = -1.0
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if self._pump_on else -1.0

        return {"continuous_action": continuous, "discrete_action": discrete}

    # --------------------------------------------------------------- economizer
    def _recirculation(self, obs, ahu, demand: float) -> float:
        if demand >= 0.0:  # heating or neutral: keep the heat in
            return 1.0
        oat_idx = ahu["oat_idx"]
        if oat_idx is None:
            return 1.0
        outside = self._denorm(obs[oat_idx], self.oat_mean, self.oat_std)
        inside_vals = [
            self._denorm(obs[z["temp_idx"]], self.zt_mean, self.zt_std)
            for z in self.zones if z["ahu_idx"] is not None
            and self.ahus[z["ahu_idx"]]["temp_idx"] == ahu["temp_idx"]
            and z["temp_idx"] is not None
        ]
        if not inside_vals:
            return 1.0
        inside = float(np.mean(inside_vals))
        if not np.isfinite(outside) or outside >= inside:
            return 1.0
        effort = float(np.clip(-demand / self.SAT_RAMP, 0.0, 1.0))
        pressure_pa = max(MIN_STATIC_PRESSURE_PA,
                          (-1.0 + 2.0 * (0.3 + 0.7 * effort)) * 0.5
                          * STATIC_PRESSURE_MAX_PA)
        breakeven_k = pressure_pa * ECON_K_PER_PA
        if (inside - outside) <= breakeven_k:
            return 1.0
        frac = float(np.clip(-demand / self.SAT_RAMP, 0.0, 1.0))
        sat_norm = 1.0 + frac * (self.COLD_SAT - 1.0)
        target = SUPPLY_AIR_MIN_K + (sat_norm + 1.0) * 0.5 * (
            SUPPLY_AIR_MAX_K - SUPPLY_AIR_MIN_K)
        target = max(target, self.eco_temp_window[0])
        if inside <= target:
            return 1.0
        f = (inside - target) / (inside - outside)
        recirculation = float(np.clip(1.0 - f, 0.0, 1.0))
        return 2.0 * recirculation - 1.0

"""V2.2 rule-based HVAC policy.

Design notes
------------
Comfort regret is weighted by per-zone occupancy, and occupancy follows a
work schedule (comfort mode only, roughly weekdays 06:00-19:00 local). The
consequences this policy exploits:

* Comfort mode: hold every zone in its band. The AHU loop tracks the WORST
  zone (the penalty is unbounded and per-zone), per-zone PI dampers split the
  air, proportional reheat heats, an economizer blends outside air when it is
  cheap, and the hot-water loop resets against the worst reheat demand.
  While a zone is already in band the AHU idles in *standby*: minimum
  pressure, saturated recirculation, neutral supply temperature, so drift is
  arrested at the first sign instead of being corrected after a full
  excursion.
* Outside comfort mode: zones are (almost) unoccupied, so comfort costs
  nothing. The plant is shut down and the building free-floats, except in the
  hours leading into the next comfort mode, where it pre-conditions to the
  day band (which sits inside the eco band, so the pre-conditioning itself
  is comfort-free).
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

# Plant constants (from the shipped action normalizers).
SUPPLY_AIR_MIN_K = 285.0
SUPPLY_AIR_MAX_K = 305.0
STATIC_PRESSURE_MAX_PA = 2000.0
MIN_STATIC_PRESSURE_PA = 200.0
# Breakeven K of outside-air advantage per Pa of fan pressure (see pid_plus).
ECON_K_PER_PA = 0.00322


class V22Policy:
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

        def norm(field):
            entry = norm_meta.get(field)
            if entry is None:
                return 0.0, 1.0
            return entry["sample_mean"], entry["sample_std"]

        self.zt_mean, self.zt_std = norm("zone_air_temperature_sensor")
        self.sat_mean, self.sat_std = norm("supply_air_temperature_sensor")
        self.oat_mean, self.oat_std = norm("outside_air_temperature_sensor")

        names = self.field_names
        self.hod_sin_index = names.index("hod_sin_000")
        self.hod_cos_index = names.index("hod_cos_000")
        self.dow_sin_index = names.index("dow_sin_000")
        self.dow_cos_index = names.index("dow_cos_000")
        self.comfort_now_index = names.index("comfort_mode_now")
        self.comfort_soon_index = names.index("comfort_mode_soon")

        # Controller gains.
        self.kp = 0.5          # AHU loop: full effort at 2 K worst-zone error
        self.ki = 0.02
        self.kd = 0.05
        self.integral_limit = 10.0
        self.damper_full_k = 2.0    # zone damper full open at 2 K excursion
        self.reheat_full_k = 2.0    # reheat valve full open at 2 K shortfall
        self.damper_ki = 0.005      # per-zone damper integrator (anti-offset)
        self.damper_integral_limit = 2.0
        self.run_off_k = 0.25       # standby off deadband (K)
        self.pre_lead_hours = 3.0   # start pre-conditioning this many h early
        self.water_reset_full_k = 4.0

        self._build_ahu_mapping()
        self._build_zone_mapping()
        self._state = None

    # ------------------------------------------------------------------ setup
    def _numeric_suffix(self, name: str) -> int:
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else 0

    def _build_ahu_mapping(self) -> None:
        ahu_ids = sorted(self.ahu_zone_assignments.keys(), key=self._numeric_suffix)
        self.n_ahu = len(ahu_ids)
        ahu_ids = ahu_ids[: self.n_ahu]

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
                "pressure_idx": self.pressure_indices[i]
                if i < len(self.pressure_indices) else None,
                "recirc_idx": self.recirculation_indices[i]
                if i < len(self.recirculation_indices) else None,
                "run_discrete_idx": i,
                "sat_sensor_idx": find(prefix + "supply_air_temperature_sensor"),
                "oat_sensor_idx": find(prefix + "outside_air_temperature_sensor"),
                "sat_setpoint_idx": find(prefix + "supply_air_temperature_setpoint"),
            })

        self.water_temp_idx = self.water_indices[0] if self.water_indices else None
        self.water_pressure_idx = (
            self.pressure_indices[-1] if len(self.pressure_indices) > self.n_ahu else None
        )
        self.water_discrete_idx = self.n_discrete - 1

    def _build_zone_mapping(self) -> None:
        names = self.field_names

        def find(field):
            try:
                return names.index(field)
            except ValueError:
                return None

        self.zones = []
        for vav_id, action_indices in self.zone_action_indices.items():
            n = int(re.search(r"(\d+)", vav_id).group(1))
            prefix = f"{vav_id}_"
            self.zones.append({
                "vav_id": vav_id,
                "num": n,
                "temp_idx": find(prefix + "zone_air_temperature_sensor"),
                "heat_sp_idx": find(prefix + "zone_air_heating_temperature_setpoint_sensor"),
                "cool_sp_idx": find(prefix + "zone_air_cooling_temperature_setpoint_sensor"),
                "damper_idx": action_indices.get("supply_air_damper_percentage_command"),
                "reheat_idx": action_indices.get("reheat_valve_percentage_command"),
                "ahu_idx": None,
            })
        # zone -> ahu
        for ahu_idx, (ahu_id, rooms) in enumerate(sorted(
                self.ahu_zone_assignments.items(), key=lambda kv: self._numeric_suffix(kv[0]))):
            if ahu_idx >= self.n_ahu:
                break
            for room in rooms:
                vav_id = room.replace("room_", "VAV_")
                for z in self.zones:
                    if z["vav_id"] == vav_id:
                        z["ahu_idx"] = ahu_idx
                        break

    def _reset_state(self) -> None:
        self._state = {
            "ahu_integral": [0.0] * self.n_ahu,
            "ahu_prev_error": [0.0] * self.n_ahu,
            "damper_integral": {z["vav_id"]: 0.0 for z in self.zones},
            "standby": [True] * self.n_ahu,
            "pump_on": False,
            "prev_hod": None,
        }

    # ----------------------------------------------------------------- helpers
    def _cyclic_index(self, sin_v: float, cos_v: float, period: int) -> int:
        angle = float(np.arctan2(sin_v, cos_v))
        if angle < 0:
            angle += 2 * np.pi
        return int(np.floor(period * angle / (2 * np.pi)))

    def _denorm(self, value: float, mean: float, std: float) -> float:
        return float(value) * std + mean

    @staticmethod
    def _deadband_error(temp: float, low: float, high: float) -> float:
        if temp < low:
            return low - temp
        if temp > high:
            return temp - high
        return 0.0

    def _zone_band(self, obs, zone, fallback):
        lo_idx, hi_idx = zone["heat_sp_idx"], zone["cool_sp_idx"]
        if lo_idx is not None and hi_idx is not None:
            lo = float(obs[lo_idx])
            hi = float(obs[hi_idx])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return lo, hi
        return fallback

    def _occupied(self, obs) -> bool:
        names = self.field_names
        if not hasattr(self, "_occ_idx"):
            self._occ_idx = names.index("num_occupants") if "num_occupants" in names else None
        if self._occ_idx is None:
            return True
        # obs = (occ - 125)/126; occ==0 -> -0.992, occ==1 -> -0.984
        return float(obs[self._occ_idx]) > -0.988

    # ------------------------------------------------------------- main action
    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        if self._state is None:
            self._reset_state()
        st = self._state

        hour = self._cyclic_index(obs[self.hod_sin_index], obs[self.hod_cos_index], 24)
        dow = self._cyclic_index(obs[self.dow_sin_index], obs[self.dow_cos_index], 7)
        is_weekday = dow < 5
        comfort_now = bool(obs[self.comfort_now_index] > 0.5)
        comfort_soon = bool(obs[self.comfort_soon_index] > 0.5)

        # Pre-condition on weekdays in the pre_lead_hours window before the
        # morning comfort start. The day band sits inside the eco band, so the
        # pre-run itself costs no comfort regret while the building is empty.
        gap = (self.morning_start_hour - hour) % 24
        pre_condition = (
            is_weekday and not comfort_now
            and gap <= self.pre_lead_hours
            and hour < self.morning_start_hour
        )
        # Occupancy gate: after the arrival window a building that reports no
        # occupants is a holiday / empty day - conditioning it is pure cost.
        occupied = self._occupied(obs)
        if (comfort_now and is_weekday
                and hour >= self.morning_start_hour + 3 and not occupied):
            comfort_now = False
            comfort_soon = False
            pre_condition = False

        use_day_band = comfort_now or comfort_soon or pre_condition
        if use_day_band:
            day_low, day_high = self.comfort_temp_window
        else:
            day_low, day_high = self.eco_temp_window

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)

        # ---------------- zones: band, errors, zone-level commands ----------
        zone_err = {}      # vav_id -> signed deadband error (+ = cold)
        worst_heat = 0.0   # max heating shortfall across building
        worst_cool = 0.0
        for z in self.zones:
            if z["temp_idx"] is None:
                zone_err[z["vav_id"]] = 0.0
                continue
            lo, hi = self._zone_band(obs, z, (day_low, day_high))
            t = self._denorm(obs[z["temp_idx"]], self.zt_mean, self.zt_std)
            err = self._deadband_error(t, lo, hi)
            # signed: positive cold, negative hot
            signed = err if t < lo else (-err if t > hi else 0.0)
            zone_err[z["vav_id"]] = signed
            worst_heat = max(worst_heat, signed)
            worst_cool = max(worst_cool, -signed)

        plant_off = not (comfort_now or comfort_soon or pre_condition)

        # ---------------- AHU loops -----------------------------------------
        any_heating = False
        any_cooling = False
        for i, ahu in enumerate(self.ahus):
            zone_ids = [z["vav_id"] for z in self.zones if z["ahu_idx"] == i]
            errs = [zone_err[v] for v in zone_ids if v in zone_err]
            if not errs:
                if plant_off:
                    discrete[ahu["run_discrete_idx"]] = 0
                continue

            # Worst magnitude wins; sign of the worst decides the direction.
            worst = max(errs, key=abs)
            error = worst

            if plant_off:
                discrete[ahu["run_discrete_idx"]] = 0
                st["ahu_integral"][i] *= 0.5
                st["ahu_prev_error"][i] = 0.0
                continue

            state_i = i
            if abs(error) < self.run_off_k:
                # In band: decay integrator and hold a standby setpoint that
                # counters the expected drift. Hot mean zone -> mildly cold
                # supply (small chiller work at minimum pressure); cold mean
                # zone -> warm supply (the chiller is off, so this is free).
                st["ahu_integral"][state_i] *= 0.9
                st["ahu_prev_error"][state_i] = 0.0
                zone_temps = [
                    self._denorm(obs[z["temp_idx"]], self.zt_mean, self.zt_std)
                    for z in self.zones
                    if z["ahu_idx"] == state_i and z["temp_idx"] is not None
                ]
                tmean = float(np.mean(zone_temps)) if zone_temps else 295.5
                u = float(np.clip((296.0 - tmean) / 1.5, -0.7, 1.0))
            else:
                st["ahu_integral"][state_i] = float(np.clip(
                    st["ahu_integral"][state_i] + error,
                    -self.integral_limit, self.integral_limit))
                derivative = error - st["ahu_prev_error"][state_i]
                st["ahu_prev_error"][state_i] = error
                u = float(np.clip(
                    self.kp * error
                    + self.ki * st["ahu_integral"][state_i]
                    + self.kd * derivative, -1.0, 1.0))

            discrete[ahu["run_discrete_idx"]] = 1
            continuous[ahu["temp_idx"]] = u
            if ahu["pressure_idx"] is not None:
                if abs(error) < self.run_off_k:
                    continuous[ahu["pressure_idx"]] = -1.0  # standby: minimum
                else:
                    continuous[ahu["pressure_idx"]] = float(
                        np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))
            if ahu["recirc_idx"] is not None:
                continuous[ahu["recirc_idx"]] = self._recirculation(obs, ahu, u, error)

            if u > 0.0 or error > 0.0:
                any_heating = True
            if u < 0.0 or error < 0.0:
                any_cooling = True

        # ---------------- zone commands --------------------------------------
        for z in self.zones:
            v = zone_err.get(z["vav_id"], 0.0)
            ahu = self.ahus[z["ahu_idx"]] if z["ahu_idx"] is not None else None
            ahu_running = (
                ahu is not None and not plant_off
                and bool(discrete[ahu["run_discrete_idx"]])
            )
            if z["damper_idx"] is not None:
                if v < 0.0:  # hot: open damper proportionally + integral
                    integ = st["damper_integral"][z["vav_id"]]
                    integ = float(np.clip(
                        integ + v * 0.1, -self.damper_integral_limit,
                        self.damper_integral_limit))
                    st["damper_integral"][z["vav_id"]] = integ
                    damper = min(1.0, max(0.0,
                                           abs(v) / self.damper_full_k
                                           + self.damper_ki * integ * 10.0))
                else:
                    st["damper_integral"][z["vav_id"]] *= 0.9
                    damper = 0.0
                if not ahu_running:
                    damper = 0.0
                continuous[z["damper_idx"]] = 2.0 * damper - 1.0
            if z["reheat_idx"] is not None:
                if v > 0.0:
                    opening = min(1.0, v / self.reheat_full_k)
                else:
                    opening = 0.0
                if not ahu_running:
                    opening = 0.0
                continuous[z["reheat_idx"]] = 2.0 * opening - 1.0

        # ---------------- water loop -----------------------------------------
        water_demand = float(np.clip(worst_heat / self.water_reset_full_k, 0.0, 1.0))
        pump_target = any_heating and worst_heat > 0.05
        if pump_target:
            st["pump_on"] = True
        elif worst_heat < 0.0:
            st["pump_on"] = False
        if self.water_discrete_idx is not None:
            discrete[self.water_discrete_idx] = 1 if st["pump_on"] else 0
        if self.water_temp_idx is not None:
            continuous[self.water_temp_idx] = (
                2.0 * float(np.clip(0.15 + 0.85 * water_demand, 0.0, 1.0)) - 1.0
                if st["pump_on"] else -1.0)
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = (
                1.0 if st["pump_on"] else -1.0)

        return {"continuous_action": continuous, "discrete_action": discrete}

    def _standby_econ(self, obs, ahu) -> float:
        ahu_idx = self.ahus.index(ahu)
        inside_vals = [
            self._denorm(obs[z["temp_idx"]], self.zt_mean, self.zt_std)
            for z in self.zones
            if z["ahu_idx"] == ahu_idx and z["temp_idx"] is not None
        ]
        oat_idx = ahu.get("oat_sensor_idx")
        if not inside_vals or oat_idx is None:
            return 1.0
        inside = float(np.mean(inside_vals))
        outside = self._denorm(obs[oat_idx], self.oat_mean, self.oat_std)
        if not np.isfinite(outside) or outside >= inside:
            return 1.0
        adv = inside - outside
        breakeven_k = MIN_STATIC_PRESSURE_PA * ECON_K_PER_PA  # min pressure
        if adv <= breakeven_k:
            return 1.0
        # Blend on proportionally to the advantage; the ventilation floor is
        # 30% outside air (recirculation capped at 0.70).
        frac = float(np.clip((adv - breakeven_k) / 4.0, 0.0, 1.0))
        recirc = 1.0 - 0.3 * frac
        return 2.0 * recirc - 1.0

    # ---------------------------------------------------------------- economizer
    def _recirculation(self, obs, ahu, u: float, error: float) -> float:
        """Proportional economizer: blend outside air onto the SAT target."""
        if u >= 0.0 and error > 0.0:  # heating: keep the heat in
            return 1.0
        if u >= 0.0:
            # Standby in band (or mild cooling): outside air that beats the
            # fan breakeven is free cooling, up to the 30% ventilation cap.
            return self._standby_econ(obs, ahu)
        oat_idx = ahu.get("oat_sensor_idx")
        if oat_idx is None:
            return 1.0
        outside = self._denorm(obs[oat_idx], self.oat_mean, self.oat_std)
        sat_idx = ahu.get("sat_sensor_idx")
        if sat_idx is None:
            return 1.0
        # Use the current zone temperatures of this AHU as the inside air.
        inside_vals = [
            self._denorm(obs[z["temp_idx"]], self.zt_mean, self.zt_std)
            for z in self.zones
            if z["ahu_idx"] == self.ahus.index(ahu) and z["temp_idx"] is not None
        ]
        if not inside_vals:
            return 1.0
        inside = float(np.mean(inside_vals))
        if not np.isfinite(outside) or outside >= inside:
            return 1.0

        pressure_pa = max(MIN_STATIC_PRESSURE_PA,
                          (float(np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0)) + 1.0)
                          * 0.5 * STATIC_PRESSURE_MAX_PA)
        breakeven_k = pressure_pa * ECON_K_PER_PA
        if (inside - outside) <= breakeven_k:
            return 1.0

        target = SUPPLY_AIR_MIN_K + (u + 1.0) * 0.5 * (SUPPLY_AIR_MAX_K - SUPPLY_AIR_MIN_K)
        target = max(target, 289.0)  # never mix below the eco band's lower edge
        if inside <= target:
            return 1.0
        f = (inside - target) / (inside - outside)
        recirculation = float(np.clip(1.0 - f, 0.0, 1.0))
        return 2.0 * recirculation - 1.0

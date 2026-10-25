"""Smart Buildings submission.

Policy: V6PolicyStandalone
"""
import sys
sys.path.append('/kaggle/input/competitions/neurips-2026-smart-buildings/kaggle_evaluation')

import numpy as np
from kaggle_evaluation import smart_buildings_inference_server

COMPETITION_DATA_FOLDER = '/kaggle/input/competitions/neurips-2026-smart-buildings/'

import re
from typing import Any

import numpy as np

SUPPLY_AIR_MIN_K = 285.0
SUPPLY_AIR_MAX_K = 305.0
STATIC_PRESSURE_MAX_PA = 2000.0
MIN_STATIC_PRESSURE_PA = 200.0
ECON_K_PER_PA = 0.00322
OCC_EMPTY = -0.988  # obs = (occ-125)/126; occ==0 -> -0.992


class V4Policy:
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
        self.oat_mean, self.oat_std = norm("outside_air_temperature_sensor")

        names = self.field_names
        self.hod_sin_index = names.index("hod_sin_000")
        self.hod_cos_index = names.index("hod_cos_000")
        self.dow_sin_index = names.index("dow_sin_000")
        self.dow_cos_index = names.index("dow_cos_000")
        self.comfort_now_index = names.index("comfort_mode_now")
        self.comfort_soon_index = names.index("comfort_mode_soon")
        if "num_occupants" in names:
            self.occ_index = names.index("num_occupants")
        else:
            self.occ_index = None

        # Controller gains (per-zone PI, from the 0.01582 notebook).
        self.kp = 0.10
        self.ki = 0.005
        self.integral_limit = 10.0 / self.ki  # integral share of full output
        self.min_damper = 0.10

        # Plant / pre-conditioning.
        self.pre_lead_hours = 3.0
        self.pre_start_k = 0.35
        self.pre_max_horizon_hours = 48.0
        self.cool_sat_u = -0.7          # SAT at full cooling demand (~288 K)
        self.sat_ramp = 0.3             # mean zone u at which full cold SAT
        self.pressure_cool_k = 0.4      # sustained cooling demand threshold

        self._build_ahu_mapping()
        self._build_zone_mapping()
        self._state = None

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _numeric_suffix(name):
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else 0

    def _build_ahu_mapping(self):
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
                "oat_sensor_idx": find(prefix + "outside_air_temperature_sensor"),
            })
        self.water_temp_idx = self.water_indices[0] if self.water_indices else None
        self.water_pressure_idx = (
            self.pressure_indices[-1] if len(self.pressure_indices) > self.n_ahu else None
        )
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

    def _reset_state(self):
        self._state = {
            "zone_integral": {z["vav_id"]: 0.0 for z in self.zones},
            "pre_latch": False,
            "pump_on": False,
            "pump_off_steps": 0,
            "prev_minute_of_week": None,
        }

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _cyclic_index(sin_v, cos_v, period):
        angle = float(np.arctan2(sin_v, cos_v))
        if angle < 0:
            angle += 2 * np.pi
        return int(np.floor(period * angle / (2 * np.pi)))

    def _denorm(self, value, mean, std):
        return float(value) * std + mean

    def _zone_band(self, obs, zone, fallback):
        lo_idx, hi_idx = zone["heat_sp_idx"], zone["cool_sp_idx"]
        if lo_idx is not None and hi_idx is not None:
            lo = float(obs[lo_idx])
            hi = float(obs[hi_idx])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                # Sensor setpoints arrive in raw kelvin; the day band does too.
                return lo, hi
        return fallback

    def _zone_target(self, zone, lo, hi):
        """Target temperature for a zone PI loop; band centre by default."""
        return 0.5 * (lo + hi)

    def _band_shift(self, obs, t_next):
        """Whole-band shift (K) for pre-latch operation; 0 = unchanged."""
        return 0.0

    def _plant_override(self, st, plant_on, comfort_now, t_next, hour):
        """Final say on whether the plant runs; default: unchanged."""
        return plant_on

    def _update_zone_ema(self, zone, u):
        pass

    def _worst_deficit(self, obs):
        lo_d, hi_d = self.comfort_temp_window
        worst = 0.0
        for z in self.zones:
            if z["temp_idx"] is None:
                continue
            lo, hi = self._zone_band(obs, z, (lo_d, hi_d))
            t = self._denorm(obs[z["temp_idx"]], self.zt_mean, self.zt_std)
            if not np.isfinite(t):
                continue
            worst = max(worst, 0.0, lo - t, t - hi)
        return worst

    # ------------------------------------------------------------- main action
    def get_action(self, obs):
        if self._state is None:
            self._reset_state()
        st = self._state

        hod_minutes = self._cyclic_index(obs[self.hod_sin_index], obs[self.hod_cos_index], 1440)
        hour = hod_minutes // 60
        dow = self._cyclic_index(obs[self.dow_sin_index], obs[self.dow_cos_index], 7)

        minute_of_week = dow * 1440 + hod_minutes
        if st["prev_minute_of_week"] is not None:
            delta = (minute_of_week - st["prev_minute_of_week"]) % 10080
            if delta > 60:
                self._reset_state()
                st = self._state
        st["prev_minute_of_week"] = minute_of_week

        is_weekday = dow < 5
        comfort_now = bool(obs[self.comfort_now_index] > 0.5)

        # Empty comfort day (special closure): nobody reported after the
        # arrival window starts -> free float.
        if (comfort_now and is_weekday and self.occ_index is not None
                and hour >= self.morning_start_hour + 3
                and float(obs[self.occ_index]) <= OCC_EMPTY):
            comfort_now = False

        # Hours until the next comfort-mode start.
        if comfort_now:
            t_next = 0.0
        elif is_weekday:
            if hour < self.morning_start_hour:
                t_next = self.morning_start_hour - hour
            elif dow == 4:
                t_next = (24.0 - hour) + 24.0 + self.morning_start_hour
            else:
                t_next = (24.0 - hour) + self.morning_start_hour
        elif dow == 5:
            t_next = (24.0 - hour) + 24.0 + self.morning_start_hour
        else:
            t_next = (24.0 - hour) + self.morning_start_hour

        deficit = self._worst_deficit(obs)
        pre_trigger = not comfort_now and (
            (t_next <= self.pre_lead_hours)
            or (deficit > self.pre_start_k and t_next <= self.pre_max_horizon_hours)
        )
        if pre_trigger:
            st["pre_latch"] = True
        if comfort_now:
            st["pre_latch"] = False
        plant_on = comfort_now or st["pre_latch"]
        plant_on = self._plant_override(st, plant_on, comfort_now, t_next, hour)

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)

        lo_d, hi_d = self.comfort_temp_window
        ahu_u = {i: [] for i in range(self.n_ahu)}
        max_heat = 0.0

        # ---------------- per-zone PI ----------------------------------------
        band_shift = self._band_shift(obs, t_next)
        for z in self.zones:
            if z["temp_idx"] is None:
                continue
            lo, hi = self._zone_band(obs, z, (lo_d, hi_d))
            lo, hi = lo - band_shift, hi - band_shift
            t = self._denorm(obs[z["temp_idx"]], self.zt_mean, self.zt_std)
            centre = self._zone_target(z, lo, hi)
            err = t - centre  # >0 hot -> cool; <0 cold -> heat

            integ = st["zone_integral"][z["vav_id"]]
            u_raw = self.kp * err + self.ki * integ
            if abs(u_raw) < 1.0 or np.sign(err) != np.sign(u_raw):
                integ = float(np.clip(integ + err, -self.integral_limit, self.integral_limit))
            st["zone_integral"][z["vav_id"]] = integ
            u = float(np.clip(self.kp * err + self.ki * integ, -1.0, 1.0))

            if z["ahu_idx"] is not None:
                ahu_u[z["ahu_idx"]].append(u)
            if u < 0:
                max_heat = max(max_heat, -u)
            self._update_zone_ema(z, u)

            damper = self.min_damper + (1.0 - self.min_damper) * abs(u) if plant_on else 0.0
            reheat = max(0.0, -u) if plant_on else 0.0
            if z["damper_idx"] is not None:
                continuous[z["damper_idx"]] = 2.0 * damper - 1.0
            if z["reheat_idx"] is not None:
                continuous[z["reheat_idx"]] = 2.0 * reheat - 1.0

        # ---------------- AHU loops -------------------------------------------
        any_cooling = False
        for i, ahu in enumerate(self.ahus):
            if not plant_on:
                discrete[ahu["run_discrete_idx"]] = 0
                for z in self.zones:
                    if z["ahu_idx"] == i:
                        st["zone_integral"][z["vav_id"]] *= 0.9
                continue
            discrete[ahu["run_discrete_idx"]] = 1
            us = ahu_u.get(i, [])
            mean_u = float(np.mean(us)) if us else 0.0
            # Smooth supply-temperature ramp: chiller off at zero cooling
            # demand, full cold SAT at mean_u = sat_ramp.
            if mean_u > 0.0:
                any_cooling = True
            frac = float(np.clip(mean_u / self.sat_ramp, 0.0, 1.0))
            sat = 1.0 + frac * (self.cool_sat_u - 1.0)
            continuous[ahu["temp_idx"]] = sat
            if ahu["pressure_idx"] is not None:
                if mean_u > self.pressure_cool_k:
                    continuous[ahu["pressure_idx"]] = float(
                        np.clip(2.0 * mean_u - 1.0, -1.0, 1.0))
                else:
                    continuous[ahu["pressure_idx"]] = -1.0
            if ahu["recirc_idx"] is not None:
                continuous[ahu["recirc_idx"]] = self._recirculation(obs, ahu, mean_u)

        # ---------------- water loop ------------------------------------------
        pump_target = plant_on and max_heat > 0.15
        if pump_target:
            st["pump_on"] = True
            st["pump_off_steps"] = 0
        elif not plant_on or max_heat < 0.05:
            st["pump_off_steps"] += 1
            if st["pump_off_steps"] > 12:
                st["pump_on"] = False
        if self.water_discrete_idx is not None:
            discrete[self.water_discrete_idx] = 1 if st["pump_on"] else 0
        if self.water_temp_idx is not None:
            continuous[self.water_temp_idx] = (
                2.0 * float(np.clip(0.15 + 0.85 * max_heat, 0.0, 1.0)) - 1.0
                if st["pump_on"] else -1.0)
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if st["pump_on"] else -1.0

        return {"continuous_action": continuous, "discrete_action": discrete}

    # ---------------------------------------------------------------- economizer
    def _recirculation(self, obs, ahu, mean_u):
        """Blend outside air when it beats the fan breakeven (v2 formula)."""
        if mean_u <= 0.0:  # heating or neutral: keep the heat in
            return 1.0
        oat_idx = ahu.get("oat_sensor_idx")
        if oat_idx is None:
            return 1.0
        outside = self._denorm(obs[oat_idx], self.oat_mean, self.oat_std)
        if not np.isfinite(outside):
            return 1.0
        ahu_i = self.ahus.index(ahu)
        zone_temps = [
            self._denorm(obs[z["temp_idx"]], self.zt_mean, self.zt_std)
            for z in self.zones
            if z["ahu_idx"] == ahu_i and z["temp_idx"] is not None
        ]
        if not zone_temps:
            return 1.0
        inside = float(np.mean(zone_temps))
        if outside >= inside:
            return 1.0
        p_norm = -1.0 if mean_u <= self.pressure_cool_k else float(
            np.clip(2.0 * mean_u - 1.0, -1.0, 1.0))
        frac = float(np.clip(mean_u / self.sat_ramp, 0.0, 1.0))
        pressure_pa = max(MIN_STATIC_PRESSURE_PA, (p_norm + 1.0) * 0.5 * STATIC_PRESSURE_MAX_PA)
        breakeven_k = pressure_pa * ECON_K_PER_PA
        if (inside - outside) <= breakeven_k:
            return 1.0
        target = SUPPLY_AIR_MIN_K + (
            1.0 + frac * (self.cool_sat_u - 1.0) + 1.0) * 0.5 * (
            SUPPLY_AIR_MAX_K - SUPPLY_AIR_MIN_K)
        if inside <= target:
            return 1.0
        f = (inside - target) / (inside - outside)
        recirc = float(np.clip(1.0 - f, 0.0, 1.0))
        return 2.0 * recirc - 1.0


class V5Policy(V4Policy):
    """v4 + per-zone economic target shift toward the dominant demand side."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.shift_k = 0.6
        self.shift_full_u = 0.2
        self.shift_ema_steps = 144
        self._u_ema = {}

    def _reset_state(self):
        super()._reset_state()
        self._u_ema = {}

    def _zone_target(self, zone, lo, hi):
        vav = zone["vav_id"]
        if vav not in self._u_ema:
            return 0.5 * (lo + hi)
        ema = self._u_ema[vav]
        frac = float(np.clip(ema / self.shift_full_u, -1.0, 1.0))
        shift = min(self.shift_k, 0.4 * (hi - lo)) * frac
        return float(np.clip(0.5 * (lo + hi) + shift, lo + 0.3, hi - 0.3))

    def _update_zone_ema(self, zone, u):
        vav = zone["vav_id"]
        prev = self._u_ema.get(vav, float(u))
        self._u_ema[vav] = prev + (float(u) - prev) / self.shift_ema_steps

V5PolicyStandalone = V5Policy

class V6Policy(V5Policy):
    """v5 economic target shift + eco-edge hold for long pre-latches.

    While pre-latched with the comfort start more than fade_h away, hold the
    band at the eco edge on the demand side (full at >= full_h), fading back
    over the last fade_h. V5's PI then regulates around the shifted band;
    its EMA-based extra target shift stays ~0 while the plant holds, so the
    two mechanisms are orthogonal (hold depth vs. in-band economic target).
    """

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.fade_h = 3.0
        self.full_h = 12.0

    def _band_shift(self, obs, t_next):
        if not self._state or not self._state.get("pre_latch"):
            return 0.0
        if t_next <= self.fade_h:
            return 0.0
        fade = min(1.0, (t_next - self.fade_h) / max(1e-6, self.full_h - self.fade_h))
        lo_d, hi_d = self.comfort_temp_window
        eco_lo, eco_hi = self.eco_temp_window
        centre = 0.5 * (lo_d + hi_d)
        temps = [self._denorm(obs[z["temp_idx"]], self.zt_mean, self.zt_std)
                 for z in self.zones if z["temp_idx"] is not None
                 and np.isfinite(obs[z["temp_idx"]])]
        if not temps:
            return 0.0
        mean_t = float(np.mean(temps))
        if mean_t < centre:
            # heating side: hold at eco low (band shifted down)
            target = max(eco_lo, lo_d - 4.0)
            return (centre - target) * fade   # positive shift = band down
        target = min(eco_hi, hi_d + 4.0)
        return (centre - target) * fade


V6PolicyStandalone = V6Policy

smart_buildings_inference_server.SmartBuildingsInferenceServer(V6PolicyStandalone).run(
    competition_data_folder=COMPETITION_DATA_FOLDER,
)

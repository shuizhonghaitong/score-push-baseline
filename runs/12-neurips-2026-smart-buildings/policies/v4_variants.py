"""Tuning variants of V4Policy (subclass, no logic changes).

v4a: earlier pre-latch (deficit 0.25 K, 4 h lead)
v4b: faster PI (kp 0.15, ki 0.008)
v4c: more aggressive cooling SAT ramp (0.2) + earlier pressure rise
v4d: zone economic target shift (zt-best idea): 0.6 K, 12 h EMA of u
"""
import numpy as np

import importlib.util as _ilu
from pathlib import Path as _Path

_spec = _ilu.spec_from_file_location(
    "v4_policy_base", _Path(__file__).with_name("v4_policy.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
V4Policy = _mod.V4Policy


class V4aPolicy(V4Policy):
    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.pre_lead_hours = 4.0
        self.pre_start_k = 0.25


class V4bPolicy(V4Policy):
    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.kp = 0.15
        self.ki = 0.008
        self.integral_limit = 10.0 / self.ki


class V4cPolicy(V4Policy):
    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.sat_ramp = 0.2
        self.pressure_cool_k = 0.3


class V4dPolicy(V4Policy):
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


class V4fPolicy(V4Policy):
    """v4 + pre-latch setback: while pre-latched with comfort far away, hold the
    band shifted 3 K toward the outside-weather side, fading to full band as
    the comfort start approaches."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.pre_setback_k = 3.0
        self.pre_setback_full_at_h = 6.0
        self.pre_setback_fade_h = 3.0

    def _band_shift(self, obs, t_next):
        if not self._state or not self._state.get("pre_latch"):
            return 0.0
        if t_next <= self.pre_setback_fade_h:
            return 0.0
        fade = min(1.0, (t_next - self.pre_setback_fade_h) /
                   max(1e-6, self.pre_setback_full_at_h - self.pre_setback_fade_h))
        lo_d, hi_d = self.comfort_temp_window
        centre = 0.5 * (lo_d + hi_d)
        temps = [self._denorm(obs[z["temp_idx"]], self.zt_mean, self.zt_std)
                 for z in self.zones if z["temp_idx"] is not None
                 and np.isfinite(obs[z["temp_idx"]])]
        if not temps:
            return 0.0
        mean_t = float(np.mean(temps))
        sign = -1.0 if mean_t < centre else 1.0
        return float(self.pre_setback_k * fade * sign)


class V4gPolicy(V4Policy):
    """v4 + adaptive eco-edge hold for long pre-latches (weekend setback).

    While pre-latched with the comfort start more than fade_h away, hold the
    band so its centre sits at the eco edge on the demand side (heating: eco
    low; cooling: eco high). Energy saved by running the building at the
    lowest useful temperature; the fade to the full band gives the plant
    fade_h of pre-conditioning. v4 already recovers from free-float (colder)
    in all local winter scenarios, so the comfort risk is small."""

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

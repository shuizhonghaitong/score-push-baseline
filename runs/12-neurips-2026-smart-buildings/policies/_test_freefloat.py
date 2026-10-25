"""Experiment: v6 with night plant-off (free float), pre-condition only in the
last 3h before comfort. Measures thermal-drift feasibility of deep setback."""
import importlib.util
import os
from pathlib import Path

FF_LEAD_H = float(os.environ.get("FF_LEAD", "3.0"))

_spec = importlib.util.spec_from_file_location("v6pol", str(Path(__file__).parent / "v6_policy.py"))
_v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v6)


class FreeFloat3h(_v6.V6Policy):
    def _plant_override(self, st, plant_on, comfort_now, t_next, hour, obs):
        if comfort_now:
            return True
        if t_next <= FF_LEAD_H:
            return True
        d = self._worst_deficit(obs)
        return d > 1.5 and t_next <= 12.0


FreeFloat3hStandalone = FreeFloat3h

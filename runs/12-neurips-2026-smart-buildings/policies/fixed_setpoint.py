"""Wrapper around the FixedSetpointPolicy baseline (local eval)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter-kit"))

from smart_buildings.baselines.fixed_setpoint import FixedSetpointPolicy  # noqa: E402,F401

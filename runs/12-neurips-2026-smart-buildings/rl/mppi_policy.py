"""MPPI policy: model-predictive control on a learned one-step dynamics model.

Self-contained numpy implementation for the competition notebook. The dynamics
model weights (backbone + reward head) are exported from the PyTorch training
run by rl/export_model.py into a single .npz.

The controller samples S action sequences of horizon H, rolls them through the
model, keeps the elite fraction by predicted cumulative reward, and applies a
smoothed average of the elites' first-step actions.

Action packing (fixed width, same layout as collect_data.py):
  [continuous (C), discrete run commands (D)]
"""
from __future__ import annotations

from typing import Any

import numpy as np

OBS_DIM = 512
ACT_DIM = 160
MODEL_PATH = "/kaggle/input/datasets/lilac1996/sb-dynamics-v1/dynamics-numpy.npz"


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class MPPIPolicy:
    HORIZON = 10
    SAMPLES = 128
    ELITE_FRAC = 0.2
    NOISE = 0.25
    RUN_P = 0.5
    SMOOTH = 0.5

    def __init__(self, building_meta_data, action_spec):
        self.meta = building_meta_data
        self.action_spec = action_spec
        self.n_cont = building_meta_data["n_continuous_action"]
        self.n_disc = building_meta_data["n_discrete_action"]
        self.obs_len = len(building_meta_data["observation_field_names"])

        d = np.load(MODEL_PATH)
        n = int(d["n_layers"])
        self.w = [d[f"w{i}"] for i in range(n)]
        self.b = [d[f"b{i}"] for i in range(n)]
        self.rw = [d[f"rw{i}"] for i in range(int(d["n_reward_layers"]))]
        self.rb = [d[f"rb{i}"] for i in range(int(d["n_reward_layers"]))]
        self.rew_scale = float(d["rew_scale"])
        self.hw = d["hw0"]
        self.hb = d["hb0"]
        self.act_dim = int(d["act_dim"])

        self.prev = np.zeros(self.n_cont + self.n_disc, dtype=np.float32)
        self._t = 0

    # ------------------------------------------------------------- model step
    def _forward(self, x: np.ndarray, a: np.ndarray):
        """x (N, OBS_DIM), a (N, ACT_DIM) -> (next_obs, reward)"""
        if a.shape[1] < self.act_dim:
            pad = np.zeros((a.shape[0], self.act_dim - a.shape[1]), dtype=np.float32)
            a = np.concatenate([a, pad], axis=1)
        h = np.concatenate([x, a], axis=1)
        for i in range(len(self.w)):
            h = h @ self.w[i] + self.b[i]
            if i < len(self.w) - 1:
                h = _gelu(h)
        next_obs = h @ self.hw + self.hb
        r = h
        for j in range(len(self.rw)):
            r = r @ self.rw[j] + self.rb[j]
            if j < len(self.rw) - 1:
                r = _gelu(r)
        return next_obs, r * self.rew_scale

    # ------------------------------------------------------------------ step
    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        x = np.zeros(OBS_DIM, dtype=np.float32)
        x[: self.obs_len] = obs
        rng = np.random.default_rng(1000 + self._t % 100000)
        self._t += 1

        n_c, n_d = self.n_cont, self.n_disc
        cont_prev = self.prev[:n_c]
        disc_prev = self.prev[n_c:]

        seqs = np.empty((self.HORIZON, n_c + n_d), dtype=np.float32)
        for h in range(self.HORIZON):
            c = np.clip(cont_prev + rng.normal(0.0, self.NOISE, n_c), -1.0, 1.0)
            d = (rng.random(n_d) < self.RUN_P).astype(np.float32)
            # anchor the run commands: keep the previous run state as the base
            d = np.where(d > 0.5, 1.0 - disc_prev, disc_prev)
            seqs[h] = np.concatenate([c, d])

        X = np.tile(x, (self.SAMPLES, 1))
        rewards = np.zeros(self.SAMPLES, dtype=np.float32)
        first = None
        for h in range(self.HORIZON):
            A = np.tile(seqs[h], (self.SAMPLES, 1))
            X, r = self._forward(X, A)
            rewards += r
            if h == 0:
                first = A.copy()

        k = max(4, int(self.SAMPLES * self.ELITE_FRAC))
        elite = np.argsort(-rewards)[:k]
        mean_first = first[elite].mean(axis=0)

        action = np.empty(n_c + n_d, dtype=np.float32)
        action[:n_c] = np.clip(
            (1.0 - self.SMOOTH) * mean_first[:n_c] + self.SMOOTH * cont_prev,
            -1.0, 1.0)
        action[n_c:] = (mean_first[n_c:] > 0.5).astype(np.float32)
        self.prev = action

        return {
            "continuous_action": action[:n_c].astype(np.float32),
            "discrete_action": action[n_c:].astype(np.int32),
        }

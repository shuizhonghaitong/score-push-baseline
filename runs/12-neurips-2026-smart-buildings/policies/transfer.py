"""Zero-action transfer policy (local eval)."""
import numpy as np


class TransferPolicy:
    def __init__(self, building_meta_data, action_spec):
        self.action = {
            "continuous_action": np.zeros(building_meta_data["n_continuous_action"], dtype=np.float32),
            "discrete_action": np.ones(building_meta_data["n_discrete_action"], dtype=np.int32),
        }

    def get_action(self, obs):
        return self.action

"""StochasticGoose-style exploration agent (adapted, standalone).

CNN predicts which actions change the frame; action choice is biased toward
predicted-change actions. Trains online from its own experience; resets the
model at each new level.
"""
import random
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arcengine import GameAction, GameState


class ActionModel(nn.Module):
    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.action_pool = nn.MaxPool2d(4, 4)
        action_flattened_size = 256 * 16 * 16
        self.action_fc = nn.Linear(action_flattened_size, 512)
        self.action_head = nn.Linear(512, self.num_action_types)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = F.relu(self.conv4(x))
        action_features = self.action_pool(conv_features)
        action_features = action_features.view(action_features.size(0), -1)
        action_features = F.relu(self.action_fc(action_features))
        action_features = self.dropout(action_features)
        action_logits = self.action_head(action_features)
        coord_features = F.relu(self.coord_conv1(conv_features))
        coord_features = F.relu(self.coord_conv2(coord_features))
        coord_features = F.relu(self.coord_conv3(coord_features))
        coord_logits = self.coord_conv4(coord_features)
        coord_logits = coord_logits.view(coord_logits.size(0), -1)
        return torch.cat([action_logits, coord_logits], dim=1)


class Agent:
    MAX_ACTIONS = float("inf")
    _MAX_FRAMES = 10

    def __init__(self, game_id: str = "", *args: Any, **kwargs: Any) -> None:
        self.game_id = game_id
        seed = int(time.time() * 1000000) + abs(hash(game_id)) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_score = -1
        self.grid_size = 64
        self.num_coordinates = self.grid_size * self.grid_size
        self.num_colours = 16
        self.action_model: ActionModel | None = None
        self.optimizer = None
        self.experience_buffer = deque(maxlen=200000)
        self.experience_hashes = set()
        self.batch_size = 64
        self.train_frequency = 5
        self.prev_frame = None
        self.prev_action_idx = None
        self.action_list = [
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION5,
        ]
        self.action_counter = 0

    def _make_model(self):
        self.action_model = ActionModel(input_channels=self.num_colours, grid_size=self.grid_size).to(self.device)
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=1e-4)

    def _get_level(self, frame):
        return getattr(frame, "levels_completed", 0) or 0

    def _compute_experience_hash(self, frame_np, action_idx):
        key = (frame_np.tobytes()[:4096], int(action_idx))
        return hash(key)

    def _frame_to_tensor(self, frame_data):
        try:
            frame = np.array(frame_data.frame, dtype=np.int64)
            frame = frame[-1]
            if frame.shape != (self.grid_size, self.grid_size):
                return None
            one_hot = torch.zeros(self.num_colours, self.grid_size, self.grid_size)
            one_hot[torch.from_numpy(frame)] = 1.0
            return one_hot.to(self.device)
        except Exception:
            return None

    def _train_action_model(self):
        if len(self.experience_buffer) < self.batch_size:
            return
        states, actions, rewards = [], [], []
        for _ in range(self.batch_size):
            exp = random.choice(list(self.experience_buffer))
            states.append(exp["state"])
            actions.append(exp["action_idx"])
            rewards.append(exp["reward"])
        states_tensor = torch.from_numpy(np.stack(states)).float().to(self.device)
        with torch.no_grad():
            logits = self.action_model(states_tensor)
        if any(a >= 5 for a in actions):
            action_targets = torch.full((len(states), 5), 0.0)
            coord_targets = torch.zeros(len(states), self.num_coordinates)
            for i, a in enumerate(actions):
                if a < 5:
                    action_targets[i, a] = rewards[i]
                else:
                    coord_targets[i, a - 5] = rewards[i]
            action_loss = F.binary_cross_entropy_with_logits(logits[:, :5], action_targets)
            coord_loss = F.binary_cross_entropy_with_logits(logits[:, 5:], coord_targets)
            loss = action_loss + coord_loss
        else:
            action_targets = torch.full((len(states), 5), 0.0)
            for i, a in enumerate(actions):
                action_targets[i, a] = rewards[i]
            loss = F.binary_cross_entropy_with_logits(logits[:, :5], action_targets)
            loss = loss + 0.01 * torch.mean(torch.sigmoid(logits[:, :5]) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state in (GameState.WIN, GameState.GAME_OVER)

    def choose_action(self, frames, latest_frame):
        self.action_counter += 1
        try:
            current_level = self._get_level(latest_frame)
            if current_level != self.current_score:
                self.experience_buffer.clear()
                self.experience_hashes.clear()
                if self.action_model is not None:
                    self._make_model()
                self.prev_frame = None
                self.prev_action_idx = None
                self.current_score = current_level

            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                return GameAction.RESET

            current_frame = self._frame_to_tensor(latest_frame)
            if current_frame is None or self.action_model is None:
                if self.action_model is None:
                    self._make_model()
                action = random.choice(self.action_list)
                self.prev_frame = None
                self.prev_action_idx = None
                return action

            if self.prev_frame is not None:
                h = self._compute_experience_hash(self.prev_frame, self.prev_action_idx)
                if h not in self.experience_hashes:
                    frame_changed = not np.array_equal(self.prev_frame, current_frame.cpu().numpy().astype(bool))
                    self.experience_buffer.append({
                        "state": self.prev_frame,
                        "action_idx": self.prev_action_idx,
                        "reward": 1.0 if frame_changed else 0.0,
                    })
                    self.experience_hashes.add(h)

            available_actions = getattr(latest_frame, "available_actions", None)
            with torch.no_grad():
                combined_logits = self.action_model(current_frame.unsqueeze(0)).squeeze(0)
                action_logits = combined_logits[:5]
                coord_logits = combined_logits[5:]
                if available_actions is not None and len(available_actions) > 0:
                    mask = torch.full_like(action_logits, float("-inf"))
                    action6_available = False
                    for a in available_actions:
                        aid = a.value if hasattr(a, "value") else int(a)
                        if 1 <= aid <= 5:
                            mask[aid - 1] = 0.0
                        elif aid == 6:
                            action6_available = True
                    action_logits = action_logits + mask
                    if not action6_available:
                        coord_logits = coord_logits + torch.full_like(coord_logits, float("-inf"))
                action_probs = torch.sigmoid(action_logits)
                coord_probs = torch.sigmoid(coord_logits) / self.num_coordinates
                all_probs = torch.cat([action_probs, coord_probs])
                all_probs = all_probs / all_probs.sum()
                selected_idx = int(np.random.choice(len(all_probs), p=all_probs.cpu().numpy()))
            if selected_idx < 5:
                action = self.action_list[selected_idx]
                chosen_idx = selected_idx
            else:
                coord_idx = selected_idx - 5
                y = coord_idx // self.grid_size
                x = coord_idx % self.grid_size
                action = GameAction.ACTION6
                action.set_data({"x": int(x), "y": int(y)})
                chosen_idx = 5 + coord_idx
            frame_np = current_frame.cpu().numpy().astype(bool)
            self.prev_frame = frame_np
            self.prev_action_idx = chosen_idx
            if self.action_counter % self.train_frequency == 0:
                self._train_action_model()
            return action
        except Exception:
            return random.choice(self.action_list)

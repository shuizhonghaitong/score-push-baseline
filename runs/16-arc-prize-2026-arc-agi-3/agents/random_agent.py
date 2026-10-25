import random
import time

from arcengine import GameAction, GameState


class Agent:
    def __init__(self, game_id: str = "") -> None:
        self.game_id = game_id
        random.seed(int(time.time() * 1000000) + hash(game_id) % 1000000)

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state in (GameState.WIN, GameState.GAME_OVER)

    def choose_action(self, frames, latest_frame):
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        options = [a for a in GameAction if a not in (GameAction.RESET,)]
        action = random.choice(options)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        return action

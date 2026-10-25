"""Local offline evaluation driver for ARC-AGI-3 agents on the public games.

Usage:
  .venv312/bin/python eval/eval_local.py --agent goose --budget 120 [--games ls20,cd82]

Score = official formula: per-level (baseline/actions)^2*100 cap 115,
per-game weighted avg by 1-indexed level, total = mean over played games.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
os.environ.setdefault("ENVIRONMENTS_DIR", str(RUN / "environment_files"))
os.environ.setdefault("MPLCONFIGDIR", str(RUN / ".mpl"))
os.environ.setdefault("RECORDINGS_DIR", str(RUN / "recordings"))
sys.path.insert(0, str(RUN / "agents"))

import arc_agi  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402


def compute_game_score(baselines, level_actions, total_actions, completed_levels, win: bool):
    """Official per-game score: per-level min((b/a)^2*100, 115); weighted avg by 1-indexed
    level over ALL baseline levels; capped by max possible from completed levels."""
    scores, indices = [], []
    prev = 0
    for level_idx in range(len(baselines)):
        if level_idx < len(level_actions):
            la = level_actions[level_idx]
            prev = prev + la
            comp = level_idx < completed_levels or win
        else:
            la = max(total_actions - prev, 0)
            prev = total_actions
            comp = win
        idx = level_idx + 1
        if comp and la > 0:
            sc = min((baselines[level_idx] / la) ** 2 * 100.0, 115.0)
        else:
            sc = 0.0
        scores.append(sc)
        indices.append(idx)
    if not scores:
        return 0.0, []
    total_w = sum(indices)
    total_s = sum(sc * w for sc, w in zip(scores, indices))
    max_w = sum(w for sc, w in zip(scores, indices) if sc > 0)
    score = total_s / total_w
    score = min(score, max_w / total_w * 100.0)
    return score, scores


def load_agent(name: str):
    path = RUN / "agents" / f"{name}_agent.py"
    spec = importlib.util.spec_from_file_location(f"{name}_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent


def load_baselines(game_id: str):
    meta_file = RUN / "environment_files" / game_id.split("-")[0]
    for m in meta_file.rglob("metadata.json"):
        try:
            data = json.loads(m.read_text())
            return data.get("baseline_actions") or []
        except Exception:
            continue
    return []


def play_game(arc: arc_agi.Arcade, agent_cls, game_id: str, card_id: str,
              budget_s: float, max_actions: int = 100000):
    agent = agent_cls(game_id=game_id)
    env = arc.make(game_id, scorecard_id=card_id)
    if env is None:
        return {"game": game_id, "error": "make failed", "actions": 0}
    deadline = time.time() + budget_s
    frames = []
    obs = env.reset()
    frames.append(obs)
    n_actions = 0
    t_start = time.time()
    prev_levels = obs.levels_completed
    level_actions = []
    prev_total = 0
    while time.time() < deadline and n_actions < max_actions:
        if agent.is_done(frames, obs):
            break
        if obs.state in (GameState.WIN, GameState.GAME_OVER):
            break
        action = agent.choose_action(frames, obs)
        if action is None:
            break
        n_actions += 1
        obs = env.step(action)
        if obs is None:
            break
        frames.append(obs)
        if obs.levels_completed > prev_levels:
            for _ in range(obs.levels_completed - prev_levels):
                level_actions.append(n_actions - prev_total)
            prev_total = n_actions
            prev_levels = obs.levels_completed
    baselines = load_baselines(game_id)
    win = obs.state == GameState.WIN
    score, level_scores = compute_game_score(
        baselines, level_actions, n_actions, obs.levels_completed, win
    )
    return {
        "game": game_id,
        "state": str(obs.state),
        "levels": obs.levels_completed,
        "win_levels": obs.win_levels,
        "actions": n_actions,
        "score": round(score, 3),
        "level_scores": [round(x, 2) for x in level_scores],
        "level_actions": level_actions,
        "elapsed": round(time.time() - t_start, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="random")
    p.add_argument("--budget", type=float, default=120.0, help="seconds per game")
    p.add_argument("--games", default="", help="comma list, default all")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    agent_cls = load_agent(args.agent)
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    game_ids = [e.game_id for e in arc.available_environments]
    if args.games:
        wanted = {g.split("-")[0] for g in args.games.split(",")}
        game_ids = [g for g in game_ids if g.split("-")[0] in wanted]
    card_id = arc.open_scorecard()
    results = []
    t0 = time.time()
    for gid in game_ids:
        r = play_game(arc, agent_cls, gid, card_id, args.budget)
        results.append(r)
        print(f"[{time.time()-t0:6.0f}s] {gid}: {r}", flush=True)
    card = arc.close_scorecard(card_id)
    per_env = {}
    if card is not None:
        for env_list in card.environments:
            for run in env_list.runs:
                per_env[run.id] = {
                    "score": run.score,
                    "level_scores": run.level_scores,
                    "level_actions": run.level_actions,
                    "level_baseline_actions": run.level_baseline_actions,
                    "actions": run.actions,
                    "levels_completed": run.levels_completed,
                }
    played = [e for e in per_env.values()]
    total = sum(e["score"] for e in played) / len(played) if played else 0.0
    card_score = card.score if card is not None else None
    out = {
        "agent": args.agent,
        "budget": args.budget,
        "games": results,
        "scorecard": {e: v for e, v in per_env.items()},
        "total_score": total,
        "card_score": card_score,
        "n_games": len(per_env),
        "wall_s": round(time.time() - t0, 1),
    }
    if card is not None:
        out["scorecard_meta"] = {k: v for k, v in card.model_dump().items() if k != "environments"}
    tag = f"{args.agent}_{int(args.budget)}{('_' + args.tag) if args.tag else ''}"
    outp = RUN / "results" / f"eval_{tag}.json"
    outp.write_text(json.dumps(out, indent=1))
    print(f"\nTOTAL SCORE: {total:.2f}  (n={len(per_env)}, wall={out['wall_s']}s)")
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Operator-only service. Do not run inside the agent's filesystem namespace."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.run_guard import Guard, Server
from shared.guard_transport import KaggleTransport


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", required=True)
    p.add_argument("--socket", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--competition", required=True)
    p.add_argument("--max-submissions", type=int, default=1)
    p.add_argument("--notebook", action="append", default=[],
                   help="Allow only this run's dedicated notebook handle; repeatable")
    args = p.parse_args()
    if args.max_submissions < 1:
        p.error("max-submissions must be positive")
    project = Path(args.project).resolve()
    socket_path = Path(args.socket).resolve()
    if socket_path.exists():
        p.error("Socket already exists; do not replace a live service")
    if socket_path.parent == project or project in socket_path.parents:
        p.error("Socket directory must be operator-owned and outside project")
    socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    guard = Guard(args.state, args.project, args.competition, KaggleTransport(),
                  args.max_submissions, args.notebook)
    try:
        with Server(socket_path, guard) as server:
            server.serve_forever()
    finally:
        socket_path.unlink(missing_ok=True)
        guard.store.lockfile.close()


if __name__ == "__main__":
    main()

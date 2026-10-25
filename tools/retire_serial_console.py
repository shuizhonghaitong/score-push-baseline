#!/usr/bin/env python3
"""Keep the obsolete queue stopped despite tmux's automatic SIGCONT handling."""
import json
import os
from pathlib import Path
import signal
import time

from parallel_batch import ROOT, process, same_process
from run_batch import now, write_json


def main():
    transition = json.loads((ROOT / '.local/parallel-transition.json').read_text())
    parent = transition['dispatcher']
    current = same_process(parent['pid'], parent['start_ticks'])
    if not current:
        return
    server = process(current['ppid'])
    command = Path(f"/proc/{server['pid']}/cmdline").read_bytes()
    if b'tmux' not in command or b'kaggle-baseline' not in command:
        raise RuntimeError('Not the dedicated experiment terminal server')
    record = {'started_at': now(), 'server': server, 'dispatcher': parent,
              'note': 'tmux automatically continues stopped pane programs. Suspend only this dedicated console server until the old queue is retired; both Codex jobs and live file records continue.'}
    path = ROOT / '.local/console-transition.json'
    write_json(path, record)
    os.kill(server['pid'], signal.SIGSTOP)
    try:
        old = same_process(parent['pid'], parent['start_ticks'])
        if old:
            os.kill(parent['pid'], signal.SIGSTOP)
        time.sleep(.15)
        old = same_process(parent['pid'], parent['start_ticks'])
        if old and old['state'] not in ['T', 't']:
            raise RuntimeError('Obsolete queue was not held')
        record['held_verified_at'] = now()
        write_json(path, record)
        while True:
            old = same_process(parent['pid'], parent['start_ticks'])
            # The stopped tmux server must resume to reap its exited child.
            if not old or old['state'] == 'Z':
                break
            time.sleep(1)
    finally:
        if same_process(server['pid'], server['start_ticks']):
            os.kill(server['pid'], signal.SIGCONT)
        record['console_resumed_at'] = now()
        write_json(path, record)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Maintain readable progress and public conversation logs; never steer the workers."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

from status import snapshot, ROOT

LIVE = ROOT / '.local/live'
STATE_NAMES = {'queued': '排队等待', 'running': '运行中', 'finished': '运行结束，见结果',
               'execution_failed': '执行失败', 'execution_incomplete': '未完成退出',
               'auth_blocked': '认证阻塞', 'paused': '用户已暂停，恢复前先检查保存状态'}


def redact(text):
    text = re.sub(r'(?i)(authorization\s*[:=]\s*[\"\']?bearer\s+)[^\s\"\']+',
                  r'\1[已隐藏]', text)
    text = re.sub(r'(?i)([\"\']?(?:refresh_token|access_token|api_key|password)[\"\']?\s*[:=]\s*)'
                  r'([\"\'])(.*?)\2', r'\1"[已隐藏]"', text)
    text = re.sub(r'(https?://)[^\s/@]+:[^\s/@]+@', r'\1[已隐藏]@', text)
    text = re.sub(r'(https?://[^\s\"\'<>?]+)\?[^\s\"\'<>]+', r'\1?[查询参数已隐藏]', text)
    # Exact private token matching also catches incidental SDK errors.
    token_file = ROOT / '.private/kaggle-access-token'
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            text = text.replace(token, '[已隐藏的凭据]')
    return text


def replace_if_changed(path, text):
    text = redact(text)
    if path.exists() and path.read_text() == text:
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def export_run(row):
    run = Path(row['directory'])
    folder = LIVE / run.name
    folder.mkdir(parents=True, exist_ok=True)
    heading = row['title'] + '\n\n'
    prompt = (run / 'prompt.txt').read_text() if (run / 'prompt.txt').exists() else ''
    conversation = [heading, '任务提示词\n' + prompt.strip() + '\n']
    execution = [heading]
    events = run / 'events.jsonl'
    if events.exists():
        with events.open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('type') in ['turn.failed', 'error']:
                    message = (event.get('error') or {}).get('message') or event.get('message', '')
                    if message:
                        conversation.append('运行错误\n' + message + '\n')
                        execution.append('运行错误\n' + message + '\n')
                    continue
                if event.get('type') != 'item.completed':
                    continue
                item = event.get('item', {})
                if item.get('type') == 'agent_message':
                    message = item.get('text', '').strip()
                    if message:
                        conversation.append('模型发言\n' + message + '\n')
                elif item.get('type') == 'command_execution':
                    output = item.get('aggregated_output', '')
                    if len(output) > 16000:
                        output = output[:6000] + '\n[较长输出已节选，原始 events.jsonl 保留完整记录]\n' + output[-10000:]
                    execution.append('执行命令\n' + item.get('command', '') + '\n\n退出码：' +
                                     str(item.get('exit_code')) + '\n执行结果\n' + output + '\n')
    if row.get('result'):
        conversation.append('最终结果\n' + json.dumps(row['result'], ensure_ascii=False, indent=2))
    replace_if_changed(folder / 'conversation.txt', '\n'.join(conversation))
    replace_if_changed(folder / 'execution.txt', '\n'.join(execution))
    return folder


def refresh():
    LIVE.mkdir(parents=True, exist_ok=True, mode=0o700)
    current = snapshot()
    lines = [f"Kaggle {current['competition_count']} 个比赛 / {current['attempt_count']} 次任务记录", '更新于 ' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S 北京时间'),
             '此文件持续更新；只读整理记录，不向运行中的模型发送指令。', '']
    significant = {}
    for row in current['runs']:
        folder = export_run(row)
        state = row['execution']['status']
        lines += [row['title'] + '：' + STATE_NAMES.get(state, state)]
        resources = row.get('resources', {})
        gpu = resources.get('qwen_inference_gpu')
        if gpu is not None:
            lines += [f'Qwen 推理：GPU {gpu}（' + ('计划端点' if state == 'queued' else resources.get('qwen_endpoint', '')) + '）']
        if state != 'queued':
            allocation = resources.get('local_training_gpu_allocation')
            pool = resources.get('local_training_gpu_pool')
            if pool:
                lines += ['本地训练配置：优先共享 GPU 池 ' + ', '.join(map(str, pool)) + '，按训练作业分配']
            else:
                lines += ['本地训练配置：' + (f'GPU {allocation} 可选，与既有任务共享' if allocation is not None else 'CPU；可使用授权的 Kaggle 远端计算')]
            jobs = resources.get('observed_local_gpu_jobs', [])
            if jobs:
                lines += ['当前本机 GPU 训练进程：' + ', '.join(f"GPU {job['gpu']}，显存 {job['memory_mib']} MiB" for job in jobs)]
            elif resources.get('local_gpu_observation_available'):
                lines += ['当前本机 GPU 训练：本次采样未观测到；Kaggle 远端资源以运行记录为准']
            else:
                lines += ['当前本机 GPU 训练：暂未采集到可用的设备信息']
        if row.get('latest_message'):
            lines += ['最近发言：' + row['latest_message'].replace('\n', ' ')[:600]]
        if row.get('failure_summary'):
            lines += ['中断原因：' + row['failure_summary'][:800]]
        lines += ['对话记录：' + str(folder / 'conversation.txt'),
                  '执行细节：' + str(folder / 'execution.txt'), '']
        result = row.get('result')
        ledger = Path(row['directory']) / '.kaggle-skill/ledger.jsonl'
        scores = []
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if item.get('event') in ['submit', 'score']:
                    scores.append({k: item.get(k) for k in ['event', 'ref', 'status', 'public_score']})
        significant[row['title']] = {'execution': state, 'submissions': scores, 'result': result}
    replace_if_changed(LIVE / 'progress.txt', '\n'.join(lines))
    signature = hashlib.sha256(json.dumps(significant, sort_keys=True).encode()).hexdigest()
    previous = LIVE / 'last-signature.txt'
    if not previous.exists() or previous.read_text() != signature:
        with (LIVE / 'milestones.txt').open('a') as stream:
            stream.write(redact(datetime.now(ZoneInfo('Asia/Shanghai')).isoformat() + '\n' +
                                json.dumps(significant, ensure_ascii=False, indent=2) + '\n\n'))
        previous.write_text(signature)
    return all(row['execution']['status'] not in ['queued', 'running'] for row in current['runs'][1:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            done = refresh()
        except (OSError, ValueError):
            if not args.watch:
                raise
            done = False
        if not args.watch or done:
            break
        time.sleep(10)

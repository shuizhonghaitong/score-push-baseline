#!/usr/bin/env python
"""usage: submit_nb.py SLUG VERSION DESCRIPTION"""
import sys
sys.path.insert(0, '/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle')
from shared import mcp_client
slug, ver, desc = sys.argv[1], int(sys.argv[2]), sys.argv[3]
r = mcp_client.request('create_code_competition_submission', {
    'competitionName': '2026-ieee-big-data-traffic-flow-bench',
    'kernelOwner': 'lilac1996',
    'kernelSlug': slug,
    'kernelVersion': ver, 'hasKernelVersion': True,
    'fileName': 'submission.csv', 'hasFileName': True,
    'submissionDescription': desc[:220],
    'hasSubmissionDescription': True,
})
t = r.response['result']['content'][0]['text']
print(t)

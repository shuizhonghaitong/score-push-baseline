"""Manual Kaggle competition file submission (skill `submit` fails on this comp).
Usage: python src/kaggle_submit.py <file> <description>"""
import sys, os, time, json, hashlib
import requests
from kaggle.api.kaggle_api_extended import (
    KaggleApi, ApiStartSubmissionUploadRequest, ApiCreateSubmissionRequest)

SLUG = 'low-cost-motor-imagery-decoding-for-rehab-cross-subject'
LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      '.kaggle-skill', 'ledger.jsonl')

def submit(path, message):
    api = KaggleApi()
    api.with_retry = lambda f, *a, **k: f
    api.authenticate()
    data = open(path, 'rb').read()
    sha = hashlib.sha256(data).hexdigest()
    with api.build_kaggle_client() as k:
        req = ApiStartSubmissionUploadRequest()
        req.competition_name = SLUG
        req.file_name = os.path.basename(path)
        req.content_length = len(data)
        req.last_modified_epoch_seconds = int(os.path.getmtime(path))
        resp = k.competitions.competition_api_client.start_submission_upload(req)
        r = requests.put(
            resp.create_url.replace('https://www.googleapis.com',
                                    'https://storage.googleapis.com', 1),
            data=data, timeout=(15, 180))
        print('upload status:', r.status_code)
        assert r.status_code == 200, r.text[:500]
        sr = ApiCreateSubmissionRequest()
        sr.competition_name = SLUG
        sr.blob_file_tokens = resp.token
        sr.submission_description = message
        out = k.competitions.competition_api_client.create_submission(sr)
        ref = out.ref
    rec = {'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': 'submit',
           'competition': SLUG, 'message': message, 'file': path, 'bytes': len(data),
           'sha256': sha, 'ref': ref}
    with open(LEDGER, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print('SUBMITTED ref', ref)
    return ref

if __name__ == '__main__':
    submit(sys.argv[1], sys.argv[2])

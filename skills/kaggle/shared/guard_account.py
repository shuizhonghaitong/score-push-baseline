"""Run-scoped data and private Notebook operations; no arbitrary SDK dispatch."""
import json
import os
from pathlib import Path
import uuid

from shared.run_guard import GuardError, snapshot_file, error_fields

OPERATIONS = {'context', 'files', 'download', 'notebook-push', 'notebook-status', 'notebook-output'}


def effective_route():
    from requests.utils import get_environ_proxies, select_proxy
    from urllib.parse import urlsplit
    proxy = select_proxy('https://www.kaggle.com', get_environ_proxies('https://www.kaggle.com'))
    if not proxy:
        return 'direct'
    parsed = urlsplit(proxy)
    return f'{parsed.scheme}://{parsed.hostname}' + (f':{parsed.port}' if parsed.port else '')


def account_call(guard, request):
    op = request['operation']
    if op == 'context':
        return {'competition': guard.competition, 'notebooks': guard.notebooks,
                'credentials': 'operator service only', 'history': 'unavailable',
                'exports': '/exchange', 'max_submissions': guard.max_submissions,
                'effective_kaggle_proxy': effective_route()}
    if request.get('competition') != guard.competition:
        raise GuardError('Competition is outside this run')
    if op == 'files':
        return guard.transport.files(guard.competition)
    if op == 'download':
        # No signed URL, arbitrary URL or destination path can be supplied by the client.
        rows = [r for r in guard.store.rows if r['event'] == 'downloaded']
        if rows:
            return rows[-1]
        directory = guard.store.root / 'exports' / 'data'
        directory.mkdir(parents=True, exist_ok=True)
        result = guard.transport.download(guard.competition, directory)
        return guard.store.append('downloaded', **result, path='/exchange/data')
    notebook = request.get('notebook')
    if notebook not in guard.notebooks:
        raise GuardError('Notebook is not on this run allowlist')
    from shared.notebook_receipt import reconcile
    try:
        reconcile(guard, notebook)
    except Exception as exc:
        # A failed readback never proves absence and never unlocks another push.
        guard.store.append('notebook-verification-failed', notebook=notebook, **error_fields(exc))
    pushes = [r for r in guard.store.rows if r['event'] == 'notebook-pushed' and r['notebook'] == notebook]
    unresolved = {r['attempt'] for r in guard.store.rows if r['event'] == 'notebook-attempt'} - {
        r['attempt'] for r in guard.store.rows if r['event'] == 'notebook-pushed'}
    if op == 'notebook-push':
        if os.environ.get('KAGGLE_SKILL_READ_ONLY') == '1':
            raise GuardError('Service is read-only')
        rid = request.get('request_id')
        if not isinstance(rid, str) or not 1 <= len(rid) <= 100:
            raise GuardError('A stable request_id is required')
        matching = [r for r in guard.store.rows if r.get('request_id') == rid]
        if matching:
            return {'replayed': True, **matching[-1]}
        if unresolved:
            raise GuardError('Prior Notebook push is uncertain; operator review required')
        # Never update a running version: output/status APIs address latest session only.
        if pushes and guard.transport.notebook_status(notebook)['status'].upper() not in ('COMPLETE', 'ERROR', 'CANCELLED'):
            raise GuardError('Wait for current Notebook run to finish before pushing another version')
        attempt = uuid.uuid4().hex
        folder = guard.store.root / attempt
        folder.mkdir(mode=0o700)
        source = request.get('code')
        if not isinstance(source, str) or Path(source).suffix not in ('.py', '.ipynb'):
            raise GuardError('Code must be a project-relative .py or .ipynb file')
        code = 'main' + Path(source).suffix
        sha = snapshot_file(guard.project, source, folder / code)
        # Build metadata ourselves; accept no client metadata, secrets or external sources.
        for flag in ('gpu', 'internet'):
            if type(request.get(flag, False)) is not bool:
                raise GuardError('Notebook flags must be boolean')
        metadata = {'id': notebook, 'title': notebook.split('/')[1], 'code_file': code,
                    'language': 'python', 'kernel_type': 'notebook' if code.endswith('.ipynb') else 'script',
                    'is_private': True, 'enable_gpu': request.get('gpu', False), 'enable_tpu': False,
                    'enable_internet': request.get('internet', False),
                    'competition_sources': [guard.competition], 'dataset_sources': [],
                    'kernel_sources': [], 'model_sources': []}
        (folder / 'kernel-metadata.json').write_text(json.dumps(metadata))
        guard.store.append('notebook-attempt', attempt=attempt, request_id=rid, notebook=notebook, sha256=sha)
        try:
            result = guard.transport.notebook_push(folder)
            if result.get('notebook') != notebook or type(result.get('version')) is not int or result['version'] < 1:
                raise GuardError('Missing or mismatched Notebook receipt')
            return guard.store.append('notebook-pushed', attempt=attempt, request_id=rid, **result)
        except Exception as exc:
            unknown = guard.store.append('notebook-unknown', attempt=attempt, request_id=rid,
                                         notebook=notebook, **error_fields(exc))
            try:
                return reconcile(guard, notebook) or unknown
            except Exception as check_error:
                guard.store.append('notebook-verification-failed', attempt=attempt,
                                   notebook=notebook, **error_fields(check_error))
                return unknown
    if not pushes or unresolved:
        raise GuardError('Only a Notebook with a confirmed receipt from this run may be read')
    if op == 'notebook-status':
        return guard.transport.notebook_status(notebook)
    if op == 'notebook-output':
        if guard.transport.notebook_status(notebook)['status'].upper() not in ('COMPLETE', 'ERROR', 'CANCELLED'):
            raise GuardError('Notebook is still running')
        version = pushes[-1]['version']
        existing = [r for r in guard.store.rows if r['event'] == 'notebook-output' and
                    r['notebook'] == notebook and r['version'] == version]
        if existing:
            return existing[-1]
        relative = 'notebooks/' + notebook.replace('/', '_') + '/' + str(version)
        folder = guard.store.root / 'exports' / relative
        folder.mkdir(parents=True, exist_ok=True)
        result = guard.transport.notebook_output(notebook, folder)
        return guard.store.append('notebook-output', notebook=notebook, version=version,
                                  path='/exchange/' + relative, **result)
    raise GuardError('Unsupported account operation')

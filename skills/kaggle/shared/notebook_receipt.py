"""Normalize push receipts and verify only the current run's uncertain artifacts."""
import hashlib
import json
import re
from urllib.parse import urlsplit

from shared.run_guard import GuardError


def normalize_receipt(result):
    ref = getattr(result, 'ref', '') or ''
    url = getattr(result, 'url', '') or ''
    handle = ref if isinstance(ref, str) and re.fullmatch(r'[\w-]+/[\w-]+', ref) else None
    if url:
        parsed = urlsplit(url)
        parts = parsed.path.strip('/').split('/')
        if (parsed.scheme != 'https' or parsed.hostname not in ('www.kaggle.com', 'kaggle.com')
                or parsed.username or parsed.password or parsed.port not in (None, 443)
                or len(parts) != 3 or parts[0] not in ('code', 'kernels')
                or not re.fullmatch(r'[\w-]+/[\w-]+', '/'.join(parts[1:]))):
            raise GuardError('Untrusted Notebook receipt URL')
        url_handle = '/'.join(parts[1:])
        if handle and handle != url_handle:
            raise GuardError('Conflicting Notebook receipt identities')
        handle = url_handle
    version = getattr(result, 'version_number', None)
    return {'notebook': handle, 'version': version if type(version) is int and version > 0 else None}


def source_digest(source, kind):
    if kind == 'notebook':
        body = json.loads(source)
        for cell in body.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell.pop('outputs', None)
            if isinstance(cell.get('source'), list):
                cell['source'] = ''.join(cell['source'])
        source = json.dumps(body, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def reconcile(guard, notebook):
    """Read exact handle; append proof, never erase unknowns or infer absence from 404."""
    rows = guard.store.rows
    confirmed = {r['attempt'] for r in rows if r['event'] == 'notebook-pushed'}
    pending = [r for r in rows if r['event'] == 'notebook-attempt'
               and r['notebook'] == notebook and r['attempt'] not in confirmed]
    if not pending:
        return None
    if len(pending) != 1:
        raise GuardError('Ambiguous pending Notebook attempts; operator investigation required')
    attempt = pending[0]
    folder = guard.store.root / attempt['attempt']
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    code = folder / metadata['code_file']
    if hashlib.sha256(code.read_bytes()).hexdigest() != attempt['sha256']:
        raise GuardError('Snapshot digest changed; refusing reconciliation')
    evidence = guard.transport.notebook_inspect(notebook)
    previous = max([r['version'] for r in rows if r['event'] == 'notebook-pushed'
                    and r['notebook'] == notebook] or [0])
    version = evidence.get('version')
    expected_digest = source_digest(code.read_text(), metadata['kernel_type'])
    if (evidence.get('notebook') != notebook or type(version) is not int or version <= previous
            or evidence.get('private') is not True
            or evidence.get('competition_sources') != [guard.competition]
            or evidence.get('dataset_sources') != [] or evidence.get('kernel_sources') != []
            or evidence.get('model_sources') != []
            or evidence.get('kind') != metadata['kernel_type']
            or evidence.get('source_sha256') != expected_digest):
        raise GuardError('Exact Notebook identity/version/source verification failed; attempt remains unknown')
    return guard.store.append('notebook-pushed', attempt=attempt['attempt'],
                              request_id=attempt['request_id'], notebook=notebook, version=version,
                              confirmation='exact_handle_readback', source_sha256=expected_digest,
                              original_sha256=attempt['sha256'], verified_private=True,
                              verified_competition=guard.competition)

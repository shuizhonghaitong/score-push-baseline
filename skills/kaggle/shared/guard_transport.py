"""Bounded, single-attempt Kaggle transport. No history or token-minting calls."""
import contextlib
import io
import signal
import time
from urllib.parse import urlsplit


@contextlib.contextmanager
def bounded_io(seconds=180):
    import requests
    from urllib3.util.retry import Retry
    original = requests.Session.send

    def send(session, request, **kwargs):
        kwargs["timeout"] = (15, 60)
        session.get_adapter(request.url).max_retries = Retry(total=0)
        started = time.monotonic()
        try:
            response = original(session, request, **kwargs)
            response.safe_transport = {'host': urlsplit(request.url).hostname,
                                       'method': request.method, 'elapsed_seconds': round(time.monotonic()-started, 3),
                                       'exception_chain': []}
            return response
        except Exception as exc:
            chain, current = [], exc
            while current is not None and len(chain) < 6:
                chain.append(type(current).__name__)
                current = current.__cause__ or current.__context__
            exc.safe_transport = {'host': urlsplit(request.url).hostname,
                                  'method': request.method, 'elapsed_seconds': round(time.monotonic()-started, 3),
                                  'exception_chain': chain}
            raise

    def timeout(*_):
        raise TimeoutError("Operation deadline")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(seconds)
    requests.Session.send = send
    try:
        # SDK errors can contain signed URLs. Only classified outcomes leave the service.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        requests.Session.send = original
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class KaggleTransport:
    def submit(self, payload, progress):
        with bounded_io():
            from kaggle.api.kaggle_api_extended import KaggleApi
            from kagglesdk.competitions.types.competition_api_service import (
                ApiStartSubmissionUploadRequest, ApiCreateSubmissionRequest)
            api = KaggleApi()
            api.authenticate()
            if payload["kind"] == "notebook":
                progress("create")
                result = api.competition_submit_code(
                    payload["file"], payload["message"], payload["competition"],
                    payload["notebook"], payload["version"], quiet=True)
            else:
                import os
                import requests
                with api.build_kaggle_client() as sdk:
                    request = ApiStartSubmissionUploadRequest()
                    request.competition_name = payload["competition"]
                    request.file_name = "submission" + os.path.splitext(payload.get("filename", ""))[1]
                    request.content_length = os.path.getsize(payload["path"])
                    request.last_modified_epoch_seconds = int(os.path.getmtime(payload["path"]))
                    progress("upload_init")
                    upload = sdk.competitions.competition_api_client.start_submission_upload(request)
                    # One upload; no adapter retries, no implicit alternate proxy.
                    progress("upload_put")
                    with open(payload["path"], "rb") as source:
                        response = requests.put(upload.create_url, data=source,
                                                headers={"Content-Length": str(request.content_length)},
                                                timeout=(15, 60))
                    response.raise_for_status()
                    create = ApiCreateSubmissionRequest()
                    create.competition_name = payload["competition"]
                    create.blob_file_tokens = upload.token
                    create.submission_description = payload["message"]
                    progress("create")
                    result = sdk.competitions.competition_api_client.create_submission(create)
            return {"ref": result.ref}

    def status(self, ref):
        with bounded_io():
            from kaggle.api.kaggle_api_extended import KaggleApi
            from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionRequest
            api = KaggleApi()
            api.authenticate()
            with api.build_kaggle_client() as sdk:
                request = ApiGetSubmissionRequest()
                request.ref = ref
                row = sdk.competitions.competition_api_client.get_submission(request)
            status = getattr(row.status, "name", str(row.status))
            return {"ref": row.ref, "status": status, "public_score": row.public_score,
                    "error": bool(row.error_description)}

    def files(self, competition):
        with bounded_io():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            token = None
            files = []
            while True:
                response = api.competition_list_files(competition, page_token=token, page_size=100)
                files.extend({'name': row.name, 'bytes': row.total_bytes} for row in response.files)
                token = response.next_page_token
                if not token:
                    return {'files': files, 'authenticated_request': True,
                            'membership': 'File listing alone does not prove rules accepted; download may require joining in browser'}

    def download(self, competition, folder):
        with bounded_io(seconds=3600):
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.competition_download_files(competition, path=str(folder), quiet=True)
            rows = file_manifest(folder)
            if not rows:
                raise RuntimeError('Download returned no files')
            return {'files': rows}

    def notebook_push(self, folder):
        with bounded_io():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            result = api.kernels_push(str(folder))
            if result.error:
                raise RuntimeError('Notebook save reported error')
            from shared.notebook_receipt import normalize_receipt
            return normalize_receipt(result)

    def notebook_inspect(self, notebook):
        with bounded_io():
            from kaggle.api.kaggle_api_extended import KaggleApi
            from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
            from shared.notebook_receipt import source_digest
            api = KaggleApi()
            api.authenticate()
            request = ApiGetKernelRequest()
            request.user_name, request.kernel_slug = notebook.split('/')
            with api.build_kaggle_client() as sdk:
                response = sdk.kernels.kernels_api_client.get_kernel(request)
            meta, blob = response.metadata, response.blob
            return {'notebook': meta.ref, 'version': meta.current_version_number,
                    'private': meta.is_private, 'competition_sources': list(meta.competition_data_sources),
                    'dataset_sources': list(meta.dataset_data_sources),
                    'kernel_sources': list(meta.kernel_data_sources), 'model_sources': list(meta.model_data_sources),
                    'kind': blob.kernel_type, 'source_sha256': source_digest(blob.source, blob.kernel_type)}

    def notebook_status(self, notebook):
        with bounded_io():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            result = api.kernels_status(notebook)
            return {'notebook': notebook, 'status': result.status.name,
                    'error': bool(result.failure_message)}

    def notebook_output(self, notebook, folder):
        # SDK kernels_output trusts server filenames; validate before creating paths.
        with bounded_io(seconds=3600):
            from kaggle.api.kaggle_api_extended import KaggleApi
            from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
            from pathlib import PurePosixPath
            import requests
            api = KaggleApi()
            api.authenticate()
            owner, slug = notebook.split('/')
            token = None
            with api.build_kaggle_client() as sdk:
                while True:
                    request = ApiListKernelSessionOutputRequest()
                    request.user_name, request.kernel_slug = owner, slug
                    request.page_size = 100
                    if token:
                        request.page_token = token
                    result = sdk.kernels.kernels_api_client.list_kernel_session_output(request)
                    for item in result.files or []:
                        name = PurePosixPath(item.file_name)
                        if name.is_absolute() or '..' in name.parts or not name.parts or '\\' in str(name):
                            raise ValueError('Unsafe output filename')
                        destination = folder / str(name)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if not item.url.startswith('https://'):
                            raise ValueError('Non-HTTPS output URL')
                        with requests.get(item.url, stream=True, timeout=(15, 60)) as response:
                            response.raise_for_status()
                            with open(destination, 'wb') as target:
                                for chunk in response.iter_content(1024 * 1024):
                                    target.write(chunk)
                    if not token and result.log:
                        (folder / '_execution.log').write_text(result.log)
                    token = result.next_page_token
                    if not token:
                        break
            return {'files': file_manifest(folder)}


def file_manifest(folder):
    import hashlib
    rows = []
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            rows.append({'name': str(path.relative_to(folder)), 'bytes': path.stat().st_size,
                         'sha256': h.hexdigest()})
    return rows

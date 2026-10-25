#!/usr/bin/env python3
"""Export a verified, renewable OAuth access token to a private local file.

Uses an existing authorized OAuth login, leaves its source untouched and never
prints token contents. Set KAGGLE_API_TOKEN to the resulting file path for CLI,
SDK and skill clients. Repeat after expiry to renew the session.
"""
import argparse
import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path.home() / '.kaggle/credentials.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--yes', action='store_true')
    a = p.parse_args()
    if not a.yes:
        print(json.dumps({'dry_run': True, 'source': str(a.source), 'output': str(a.output),
                          'action': 'Verify existing OAuth login and write private access-token file'}))
        return 0
    if a.output.resolve() == a.source.resolve():
        print(json.dumps({'error': 'Output must not overwrite the OAuth source'}))
        return 5
    if os.environ.get('KAGGLE_SKILL_READ_ONLY') == '1':
        print(json.dumps({'error': 'Read-only mode refuses token storage'}))
        return 5
    try:
        import requests
        from urllib3.util.retry import Retry
        send = requests.Session.send

        def bounded(session, request, **kwargs):
            kwargs['timeout'] = (12, 45)
            session.get_adapter(request.url).max_retries = Retry(total=0)
            return send(session, request, **kwargs)

        requests.Session.send = bounded
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from kagglesdk.kaggle_client import KaggleClient
            from kagglesdk.kaggle_creds import KaggleCredentials
            from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
            client = KaggleClient()
            credentials = KaggleCredentials.load(client, str(a.source))
            if credentials is None:
                raise ValueError('No existing OAuth login')
            data = json.loads(a.source.read_text())
            expiry = datetime.fromisoformat(data['access_token_expiration']) if data.get('access_token_expiration') else None
            token = data.get('access_token')
            if not token or not expiry or expiry < datetime.now(timezone.utc) + timedelta(minutes=10):
                reply = credentials.generate_access_token()
                token = reply.token
                expiry = datetime.now(timezone.utc) + timedelta(seconds=reply.expires_in)
            request = IntrospectTokenRequest()
            request.token = token
            reply = client.security.oauth_client.introspect_token(request)
            if not reply.active or not reply.username:
                raise ValueError('OAuth token rejected')
        a.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(prefix='.token-', dir=a.output.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(token)
            os.chmod(temporary, 0o600)
            os.replace(temporary, a.output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(json.dumps({'verified': True, 'username': reply.username,
                          'expires_at': expiry.isoformat(), 'token_file': str(a.output)}))
        return 0
    except Exception as exc:
        print(json.dumps({'verified': False, 'error_type': type(exc).__name__}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

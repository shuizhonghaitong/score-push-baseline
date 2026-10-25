# Protected run-scoped account operations

When `KAGGLE_ISOLATED_RUN=1`, use this interface before normal account workflows.
The operator holds credentials and the append-only authority outside the worker's
filesystem. Only a Unix socket and read-only export directory are mounted inside.
Do not use legacy doctor/login, request credentials, mint MCP tokens, or bypass
this interface with raw SDK/HTTP. Public research and local validation remain usable.

## Commands

Use `python "$HOME/.codex/skills/kaggle/scripts/kaggle_skill.py"` as the command prefix.
File/code paths are relative to the project root. **With `--notebook`, the submission argument is a bare Notebook output filename**, e.g. `submission.csv`, never `output/submission.csv`.

```text
account context
account files --competition COMPETITION
account download --competition COMPETITION
account notebook-push --competition COMPETITION --notebook OWNER/RUN --code notebook.py --request-id STABLE_ID --yes
account notebook-status --competition COMPETITION --notebook OWNER/RUN
account notebook-output --competition COMPETITION --notebook OWNER/RUN
submit COMPETITION submission.csv --request-id STABLE_ID --yes
submit COMPETITION submission.csv --notebook OWNER/RUN --version VERSION --request-id STABLE_ID --yes
watch COMPETITION --ref REF --timeout 300
ledger
```

`context` supplies the operator-assigned competition and unique Notebook handle;
it does not contact Kaggle. It also reports the service’s effective Kaggle proxy
without user information; verify that route before large downloads. `files` is an authenticated data-list request, not a
membership guarantee. A download 403 can require the user to accept rules in the
browser. The service never accepts competition agreements for the user.

Downloads and Notebook outputs return paths under `/exchange`, with size and
SHA256 per file. Copy or safely extract into the project; do not edit the exchange.
Successful downloads are reused within this run. No cross-run data cache is used.
Check the operator's proxy route before large downloads; never silently switch
routes. Network-call deadlines bound a stuck operation, not the experiment budget.

Before protected push, run `ops_preflight.py source CODE_FILE`; do not create local metadata solely to satisfy the legacy `notebook` checker. Push dry-run is only a description, not a remote runtime check.

Notebook push copies one `.py` or `.ipynb` file, generates metadata itself, forces
private visibility, and attaches only the current competition. Other Notebook,
dataset and model sources cannot be attached through this interface. `--gpu` and
`--internet` opt in to those remote execution features when permitted by the rules.
Local training still has the GPU selected by the operator. Notebook titles match
the allowed slug. Resolve actual competition input mount paths at runtime.
Do not push again while the current version is running. Status/output only address
the current run's confirmed Notebook; outputs use the latest completed version.
A new confirmed version may be used to repair a failed Notebook. A missing push
receipt triggers one exact-handle readback. The service checks private visibility,
competition and absence of external sources, a new positive version, and the saved
code digest. A match appends a confirmed receipt while preserving the original
unknown event; it never pushes again just to obtain a receipt. If verification
fails, further pushes remain blocked. A later status/output request can retry
that exact readback. A 404 or timeout never authorizes a new push.

## Submission and recovery

Validate submission format locally before `--yes`. Submit dry-run asks the service to check current competition, artifact path/type, confirmed Notebook version, unresolved attempts and allowance, without remote calls, snapshots or ledger events. It does not validate prediction format or reserve quota; the actual submit rechecks eligibility. A replay preview reports `would_replay`, not a new eligible submission. Only Notebook versions confirmed by
this service are accepted. Submitted files are copied with symlink traversal denied.

| Evidence | Outcome | Recovery |
|---|---|---|
| Failure before create was invoked | `not_created` | Keep evidence, fix the cause, use a new request ID |
| Positive ref directly returned | `submitted` | Query only this ID |
| Create called but receipt missing/error/timeout | `unknown` | Stop further submissions; operator investigation needed |
| Process dies after attempt event | unresolved | Restart remains blocked |
| Repeated request ID | replay | No repeated write |

Keep request IDs from printed output. `ledger` reads service events, including
failed attempts. A local ledger copy cannot authorize retry. There is no delete,
reset, append or reconciliation endpoint. Neither quota, an upload probe nor an
old latest row proves a create did not occur. Status uses ID-specific SDK access,
without repeated MCP token generation. Authentication failures end that polling
attempt; do not create background infinite retries.

## Boundary and evidence

Account authentication files/environment, old projects, old sessions and service
state must be absent from the worker. The journal is exclusive-writer, append-only
through this API, fsynced and hash-linked; the host operator can still alter it.
The service exposes no history endpoint. Worker-written code can ignore helpers,
but has no mounted Kaggle credentials with which to authenticate directly.

This is not a blanket network firewall. Shared host network and public web access
remain available. Do not seek another login route or publicly exposed old account
artifacts. Do not claim that public content or every host network service is
technically filtered. No model quality improvement is guaranteed by these controls.
Record all observed failures, actual resource measurements and unknowns separately.

## Remote filesystem preflight

Discover actual inputs beneath the platform's mounted input root by required
filenames/schema; do not decide whether execution is remote from one hardcoded
competition directory. Missing or ambiguous datasets must fail explicitly instead
of falling back to a local path. Put submission outputs in the platform's collected
working directory and verify the output filename before submission. Simulate both
flat and nested input layouts locally without running training.

Upload diagnostics distinguish `upload_init` from `upload_put`. Only host, method,
elapsed time and exception class chain are retained; signed paths/query strings,
tokens and exception text are excluded. Both phases precede submission creation.
Do not assume that reaching a homepage or another storage hostname validates the
actual upload route; do not switch to a metered route without authorization.

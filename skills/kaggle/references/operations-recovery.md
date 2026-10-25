# Operations and recovery

## Reuse the working environment

Find an interpreter with compatible installed packages before creating another
environment. Check the installed version requirements and available environments;
do not retry an incompatible pip install. Use one explicit Python
path and its CLI for all operations. Discover the actual installed skill root
from its SKILL.md; do not hardcode a launcher-specific path or assume the script
lives at the plugin root.

`ops_preflight.py runtime` reports only installed distribution versions, paths and
redacted proxy endpoints. It imports no Kaggle module, reads no credentials and
makes no network calls. It is an inventory, not a connectivity or authentication
test. A successful public page request does not validate credentials.

## Route and credential diagnosis

Honor the host's AGENTS.md network instructions and preferred proxy route.
Do not silently switch routes for a large download. Verify both uppercase and
lowercase HTTP(S) variables and retain localhost/private-network NO_PROXY values.
Probe the intended route before attributing a failure to an API/provider. For
local proxies, first check the proxy service and its listener. Test www.kaggle.com,
api.kaggle.com and the actual storage hostname separately when applicable.

Separate these failure classes:

| Evidence | Interpretation / action |
|---|---|
| DNS failure, timeout, connection refused | Check intended route/listener and inherited environment |
| Sandbox `PermissionError` | Check execution permission; this alone is not proof the destination is unreachable |
| Automatic approval service rejects because its model is missing | Host/reviewer configuration issue; report the exact blocker, do not bypass approval |
| HTTP 401 / Unauthenticated | Route reached a server; verify credential source/method on this interface |
| HTTP 403 | Read response: participation, scope, private resource, consent or other access denial |
| HTTP 404 / Not found | Check exact slug/handle and access; do not assume a network outage |
| Read-only credentials.json during refresh | OAuth needs a writable credential location supported by the installed client |
| Unknown flag/field/command | Consult installed command `--help` or SDK signature once, then use the supported interface |
| 429 / transient 5xx | Bounded backoff within budget; inspect state before retrying a write |

Token files can exist but be revoked. Different tools may resolve different
credential sources or have different method/scope support. Treat
CLI, skill MCP reads and kagglehub as separate authentication checks when switching
between them. Do not transplant OAuth tokens into another tool and assume scopes
are compatible. Inspect method compatibility; never print the token.

Do not patch installed SDK code as the default fix for credential paths. Prefer
a client-supported writable credential location. If the client only supports a
fixed read-only location, prepare the required permission or launch-configuration
repair instead of silently changing global environment or installed SDK files.
Copy only required credential files when storage is authorized, with
directory mode 700 and files 600. Do not re-copy a known revoked token to the new
location. If credential storage authorization is absent, use a permitted existing
location or prepare the fix for approval. Preserve the original credentials.

## Stop repeating unchanged deterministic failures

After one failure, collect one targeted discriminator. Retry when it identifies
a meaningful change: correct slug/flag, refreshed auth, accepted rules, fixed
schema or a different documented interface. If the same failure remains with
unchanged evidence, record a blocker and proceed with independent work.

No generic retry count can repair missing consent or an incompatible Python.
If a user reports that rules were accepted but downloads remain denied, verify
the account/membership on that route. A private notebook with legitimately
attached competition data can be a fallback only when access is confirmed and
that run/download is authorized; it does not bypass participation requirements.

## Keep process failure visible

Avoid `cmd | tail`, `cmd | head`, `cmd; ls` as a success signal. Use the original
process return code, plus semantic response and expected outputs. If a pipeline
is necessary, use Bash `set -o pipefail`, retain a log and inspect the producer's
status. `head` can itself induce SIGPIPE, so prefer capture-then-display for
machine decisions. Even the CLI can emit a write error with status zero; check
messages such as `Kernel push error` and fetch the created resource/status.

Retain the execution session ID for long jobs. Poll within the host's time limits
and send meaningful progress updates. A detached `&` process may die with its
execution session; use a supported persistent session or a shell that waits for
its children. Do not restart an expensive run merely because a wait timed out.

## Notebook and output contract

Use `kaggle kernels init` when available to create compatible metadata, then
check before pushing:

- `id` is `owner/slug`, title-derived slug matches, `code_file` exists inside the
  notebook directory, language is Python, kernel type matches .py/.ipynb;
- private visibility unless requested otherwise; competition/dataset/model
  sources, internet and accelerator match the actual competition rules;
- clean, self-contained code and the intended checkpoint/data versions; no
  dependence on a local training directory, local secrets or example test IDs;
- inference budget scales to hidden input size and checkpoints/fallbacks keep
  output valid on difficult inputs.

The helper's slug check covers ASCII titles only. For other titles, inspect the
returned canonical notebook ref after push. Always track the actual returned
owner/slug and exact version, not only the requested metadata.

Download to a fresh version-specific output directory and check the manifest.
Do not infer freshness from file existence: the CLI can skip old files. Use the
installed CLI's documented overwrite/output flags only when needed.

Check the installed validator's supported formats; a CSV-only validator cannot
validate Parquet or a nested JSON contract. Choose a matching validator and add
competition-specific label/range/shape checks. For code submissions, explicitly
pass the required output filename when the wrapper's default differs.
Inspect `submit --help`; do not assume a notebook
run's success means a submission was created.

Before any write retry, query the existing kernel/submission state. A timeout
after a write is an unknown result, not evidence that the write did not happen.

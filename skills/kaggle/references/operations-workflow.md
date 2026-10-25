# Competition workflow


Use this skill’s command wrapper for platform operations. Discover actual executable/skill paths;
if an operation or flag is uncertain, consult the installed command's `--help`.

## Resume first

Reuse current checkpoints, authorized historical solutions and official data caches.
Only an explicitly independent rerun excludes prior experiments or account results.
Do not apply the old comparison experiment restrictions to ordinary competitions.
Reuse working environments and credentials; avoid rebuilding an established
setup. Preserve pending job/submission IDs. Read only the reference needed for
the current operation instead of repeatedly loading full manuals or trace files.

Keep non-secret checkpoints in a project-local state file: competition slug,
Python/CLI path and versions, verified credential method/account/time, membership
evidence, data manifest hashes, submission mode/output filename, notebook ref and
version, pending job/submission IDs, remaining authorized budget and next action.
Reuse checks while their dependencies are unchanged; revalidate auth/membership
after relevant state changes or an access denial. A checkpoint is evidence with
provenance and time, not a permanent assertion.

## Resolve dependencies

The usual order is environment/route → authenticated account operation → exact
competition identity/contract → participation → data → solution/validation →
delivery. Public competition research can happen before authentication, and
authorized offline work can proceed while participation needs a website action.

1. Run `ops_preflight.py runtime` with the selected interpreter: it reads package metadata without importing Kaggle or triggering login. Reuse that interpreter for installs and execution (`PYTHON -m pip`); use an explicit cwd. Follow host package-index/proxy preferences, probe the intended destination with a bounded timeout before waiting on an install, and run a tiny API/serialization check before training. Missing torch alone does not establish GPU availability.
2. Verify the chosen credential with an account operation. Public pages or an
   on-disk token file do not prove authentication. Different clients can resolve
   different sources and support different credential methods/scopes.
3. Resolve the exact URL/slug and read evaluation, data and code requirements.
   Record metric/direction, file versus hidden notebook rerun, output filename,
   runtime/memory/internet rules and submission allowance. Do not guess a slug.
4. Check participation separately. Read denial bodies before interpreting 403.
   If the site requires manual consent, provide its URL and resume blocked calls
   after state changes; do not repeat them with unchanged evidence.
5. Check data version/provenance and sample schema before expensive computation.
   Mirrors and placeholder test files need independent verification.

Use [operations-recovery.md](operations-recovery.md) when access, runtime, output freshness
or an operation fails. It covers cause-specific recovery and code submissions.

## Evidence before advancement

For each dependent step, require the operation's real outcome plus expected
artifacts. Shell status zero alone is insufficient; some API wrappers print an
error and return zero. Preserve producer status when formatting logs with pipes.
After one failure collect one targeted discriminator; retry only when there is
new evidence, a concrete fix, or a bounded transient-retry policy.

Run a representative inference/schema check before a full run. For code
competitions, use dynamic input IDs/sizes and the scoring environment's mounts.
Download outputs to a new version-specific directory and verify the exact
version, manifest and required filename. A notebook run is not a competition
submission. On uncertain writes, query existing status before retrying.

The bundled offline helper checks environment inventory or Python notebook
metadata, with no network, credential reads, uploads or code execution:

```bash
python <skill-dir>/scripts/ops_preflight.py runtime
python <skill-dir>/scripts/ops_preflight.py source ./notebook.py  # protected account mode
python <skill-dir>/scripts/ops_preflight.py inputs ./input-root train.csv test.csv
# Legacy direct push only: metadata is user-supplied
python <skill-dir>/scripts/ops_preflight.py notebook ./notebook --competition SLUG --offline
```

Choose a submission-format validator appropriate to the competition. This
helper does not validate predictions, metrics, allowed accelerators, source
licenses, hidden runtime feasibility or account participation.

## Preserve scope and verify completion

Use the user's current compute/submission authorization and limits; do not ask
again for already authorized work. A workflow/audit request alone is not consent
to publish, train indefinitely or spend submission/GPU allowances. Respect the
installed tool's concrete preview/confirmation controls within that authorization.

Record the final file hash or notebook version, submission ID, status and score
when available. If blocked, report the last good checkpoint, evidence and exact
next action. A timeout after a write means the result is unknown; do not create
a duplicate submission to resolve uncertainty.


## Restricted history access

When prior account history is excluded, do not use status/watch/ledger or another
listing command without checking whether it exposes historical submissions. Use a
current-run ID-specific query. If only broad listing is supported, report that
limitation instead of reading history or resubmitting.

Use a single source for local and remote model logic; record actual remote dependency versions. Verify missing and ambiguous inputs explicitly, not only flat/nested success. Preserve the true child exit code (for shell pipelines, `set -o pipefail` and capture status before echo/date); keep failed logs. For requested resource tracking, use a command recorder and operator-side session accounting rather than reconstructing measurements from memory. Similar validation/public scores do not prove absence of leakage or runtime equivalence.

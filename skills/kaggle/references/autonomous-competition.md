# Autonomous competition runs

When the user requests an unattended run, complete the competition workflow using
the available tools and resources. Do not wait for modelling advice or manual
repairs. An unavoidable human login, agreement or permission blocks the affected
task: record the evidence and finish it as blocked rather than claiming success.

Read the actual rules, metric, data and submission contract. Reuse authorized
past solutions, compatible environments, official data caches and public research.
Choose modelling methods freely within the competition rules. Use a realistic
validation split, prevent leakage and ID misalignment, and check inference and
output constraints before expensive runs or submissions.

First obtain a valid baseline, then try useful improvements, parameters,
pretrained models or ensembles. Keep the best working artifacts. Local validation
is the main signal; public feedback may help assess generalization. Multiple
submissions are allowed when authorized and supported by evidence, within the
live platform quota. There is no default three-round limit, one-submission cap,
or mandatory numerical improvement threshold. Stop when useful avenues are
exhausted or progress stalls, and finish with the best available result.

For preauthorized submissions and required private Notebook runs, inspect the
dry run and proceed within that authorization without asking again. Do not
publish unrelated datasets, models or public notebooks. Preserve pending IDs;
an uncertain creation must be reconciled before another write. A Notebook run
alone does not establish that a competition submission succeeded.

Record evaluated attempts compactly in experiments.csv (method/configuration,
validation protocol and score, time, artifact/code version, outcome). Save
RESULT.json containing competition, completion status, successful scored
submission IDs, best score obtained by this run, local best score, best artifacts,
leaderboard evidence/time, account rank, score-based rank interval if available,
elapsed time, failures and any human intervention. Account rank may reflect an
older better submission: distinguish it from this run's score position. Unknown
fields stay null; never infer success or a rank from missing evidence. Preserve
raw sessions locally for later analysis, outside Git.

Environment and proxy settings belong to the host/run configuration. Preserve
them in subprocesses, verify the actual route before large transfers, and never
expose credentials. If a stale API token shadows a working OAuth login, use the
setup module's auth-session helper to obtain a private token file from that login;
do not remove or overwrite unrelated credential files.

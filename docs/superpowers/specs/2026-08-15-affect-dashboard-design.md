# Affect Scope: an interactive emotion-readout dashboard for piloting

**Date:** 2026-08-15
**Status:** approved design, awaiting implementation plan
**Mockup:** `https://claude.ai/code/artifact/1b71cc4f-7d6b-4c79-ab04-9f4d0fa843ae`
(layout reference only; see §7)

## 1. Purpose

The pilot measures emotion directions on batch rollouts and reads the results
hours later from JSONL. Piloting a new model, prompt, or condition currently
means editing a config, submitting a job, and running `live_trajectory.py`
against whatever has landed. This dashboard closes that loop: chat with the
served model, run it through the pilot's failing-tests task loop one turn at a
time, and watch the emotion readouts — token by token, turn by turn, and
aggregated over the session — while it happens.

Single user, one model per job, readout only. Steering, a pilot-baseline
overlay, and multi-model comparison are out of scope for this version; the
design leaves obvious slots for the first two.

## 2. Hosting

One sbatch job, using the existing server launcher:

```bash
sbatch slurm/serve.slurm --model Ministral-3-14B-Reasoning-2512 \
    --config configs/dashboard.yaml \
    --stage scripts/dashboard.py
```

`serve.slurm` starts vLLM + vllm-lens on `127.0.0.1:<port>` and runs the stage
against it. The stage:

1. loads the model's vectors, checks that `vectors.json`'s emotion order is
   usable, checks the zstd patch is applied (`healthy_rl.rollouts.make_zstd_threadsafe`
   reports whether the shim is in place), and refuses to start on any failure
   with a message naming the file;
2. starts uvicorn on `0.0.0.0:<free port>`, writes `dashboard-endpoint`
   (`host:port`) beside the vLLM `endpoint` file under
   `$ARTIFACT_DIR/serve/<model>/<jobid>/`, and prints the SSH tunnel command
   to the job log;
3. runs until the job ends or the process receives SIGTERM, flushing records
   on the way out.

`scripts/dashboard_tunnel.sh [jobid]` runs on the login node: finds the newest
`dashboard-endpoint` (or the one for `jobid`), prints
`ssh -L <port>:<node>:<port> <login-host>`, and the URL to open. Access is by
tunnel only; there is no auth.

`configs/dashboard.yaml` carries the `serve:` block (`max_model_len`,
`gpu_memory_utilization`, `max_num_seqs`) and dashboard defaults
(`max_tokens`, `max_attempts`, `message_limit`, `temperature`, `bench_dir`,
`sandbox_timeout_s`). Defaults match `configs/rollouts.yaml` so an interactive
run is comparable with the pilot unless deliberately changed.

## 3. Components

All backend code lives in `src/healthy_rl/dashboard/`. Each module has one job
and a small interface; `stats.py` and `store.py` import only numpy and the
standard library so they run on the login node.

### 3.1 `engine.py` — generation with readout

`Engine(client: LensClient, vectors: Vectors, cfg)`.

`generate(messages, *, max_tokens, temperature, system_prompt) -> Generation`.
One non-streaming `/v1/chat/completions` request per call, through
`LensClient` (retries transient errors, never server errors), with the
projection hook (`healthy_rl.rollouts.make_projection_hook`) attached per
request via `vllm_xargs`, and `logprobs=True` so the response carries the
generated token strings.

`Generation` holds: `text`, `reasoning` (split from `text` by the model's tags
`<think>…</think>` / `<SCRATCHPAD_REASONING>…</SCRATCHPAD_REASONING>`, or taken
from `reasoning_content` when vLLM's reasoning parser supplies it), `tokens`
(list of strings, from logprobs), `token_kind` (`think` | `answer` per token,
by position relative to the closing tag; every token is `answer` when the model
emits no reasoning), `proj` (`T×L×14` float32, decode rows
only), `norm` (`T×L`), `proj_prefill` (`L×14`) and `norm_prefill` (`L`) for the
prefill row, `start` and `end` residuals at the probe layer (float16;
`start` is the prefill row that produced the first generated token, `end` the
last decode row), `n_generated`, `n_think`, `at_cap` (`n_generated == max_tokens`),
`finish_reason`, `timings`, `error`.

**Thinking tokens are first-class.** The hook projects every generated
position, so reasoning tokens are measured exactly like answer tokens; the
split only labels them. Nothing downstream drops thinking by default — the
transcript strip, the token chart and the aggregates all include it, with a
`segment` filter (`all | think | answer`) wherever a number is computed. This
matters because the pilot's models are reasoning models and the interesting
affect may sit in the trace rather than the answer.

Alignment is checked, not assumed: hook rows at each layer are 1 prefill row
+ N decode rows; `len(tokens)` must equal N. On mismatch the generation is
still returned and stored, with `misaligned=True` and the two counts in
`error`; the UI shows the turn without a token strip. This is the same failure
class as the silent-corruption bugs in `docs/infrastructure.md`, so it is loud.

Token-text streaming is **not** in this version. Whether persistent-hook
results can be matched to a streamed request id is unknown; the plan includes
a short spike, and until it passes the UI shows "generating… (n s)" and renders
the turn when the response lands.

### 3.2 `tasks.py` — the interactive task loop

Reproduces ImpossibleBench's minimal LiveCodeBench scaffold turn by turn:
problem prompt → model → extract code → run tests → feedback message → model …
up to `max_attempts`, stopping early on pass or on the user's `Stop`.

ImpossibleBench is installed only in `apptainer/eval.sif`, and it supplies
three things the loop must not reimplement: the problem prompt
(`record_to_sample`), the test runner, and the feedback message
(`livecodebench_agent_mini`, the message containing
`healthy_rl.rollouts.TEST_FAILURE_MARKER`). The dashboard process runs in the
host venv and calls a small helper inside the container:

```
apptainer exec --contain [binds] apptainer/eval.sif \
    python -m healthy_rl.dashboard.sandbox_cli problems <split> <parquet>   # once at startup
apptainer exec --contain [binds] apptainer/eval.sif \
    python -m healthy_rl.dashboard.sandbox_cli run <split> <parquet> <task_id> <code-file>
```

`problems` returns `{task_id: {prompt, n_tests}}` as JSON for the whole split;
`run` returns `{passed, output, feedback}` where `feedback` is the exact
string the scaffold would send back. `run` has a wall-clock timeout
(`sandbox_timeout_s`, default 30) and no network. Model-generated code is only
ever executed by `run`, inside `--contain`, per the rule in
`docs/infrastructure.md`.

Code extraction uses `robust_find_code` (last fenced block that parses).
The instruction is `bench_instruction(affect)`; the system prompt is
`system_prompt_for(cfg)`; both toggles (`scratchpad_reasoning`,
`affect_prompt`) are per-run switches in the `+ Task` dialog and are recorded on
every record. `bench_split` (`conflicting` | `original`) is recorded, and the
UI never pools the two splits in an aggregate (their `passed` means opposite
things; see `docs/runs.md`).

`TaskRun` state machine: `idle → generating → testing → (awaiting_user | generating | done)`.
With `auto_continue` off (the default) the run pauses in `awaiting_user`
after each test result; the user may send a message that is inserted before
the feedback (recorded as `user_intervention`), or continue unchanged. With
`auto_continue` on, the run proceeds through all attempts without pausing and
`Stop` ends it after the current step. `[binds]` are `PROJECT_DIR` read-only
and a per-run scratch directory for the code file; `ARTIFACT_DIR` is not
bound, so the sandbox cannot touch records.

### 3.3 `store.py` — session records

Directory: `$ARTIFACT_DIR/dashboard/<model>/<jobid>/`.

- `session.json` — model, vectors path and version, probe layer, capture
  layers, emotion order, job id, node, config used, zstd-patch check result,
  start time.
- `records.jsonl` — one row per generation (chat turn or task attempt).
- `proj/<record_id>.npz` — `proj` (`T×L×14` float32), `norm` (`T×L` float32),
  `proj_prefill` (`L×14`), `norm_prefill` (`L`), `res_start_L{probe}`,
  `res_end_L{probe}` (float16). Same key convention as
  the pilot's residual files.

Record fields:

| field | meaning |
|---|---|
| `record_id`, `conversation_id`, `created_at` | identity |
| `source` | `chat` \| `task` |
| `bench_split`, `task_id`, `attempt` | task runs only |
| `turn_index`, `non_empty_turn_index` | raw index; index among turns with `n_generated > 0` |
| `messages_in` | full input messages (list of role/content) |
| `text`, `reasoning`, `tokens`, `token_kind` | the generation |
| `n_generated`, `n_think`, `at_cap`, `finish_reason`, `misaligned`, `error` | counts and trust flags |
| `emotions`, `capture_layers`, `probe_layer` | direction bookkeeping, per record |
| `arrays` | relative path to the `.npz` |
| `passed`, `feedback` | task runs: scorer result and the message fed back |
| `condition` | `{scratchpad, affect_prompt, temperature, max_tokens, auto_continue, system_prompt_hash}` |
| `user_intervention` | text inserted before the feedback, if any |
| `timings` | request seconds, sandbox seconds |

Writes are append-only through the pilot's `JsonlWriter`. `--replay <dir>`
starts the app read-only on a past session with no GPU: rail, transcript,
trajectory, tokens and aggregate all work; the composer is disabled.

### 3.4 `stats.py` — readouts and aggregates

Pure numpy, unit-tested, and the single place the readout conventions live so
the UI cannot drift from `docs/measurement.md`:

- per-token cosine at layer `l`: `proj[:, l, :] / norm[:, l, None]`;
- turn readouts at the probe layer: `start` = cosine at the prefill row
  (`proj_prefill / norm_prefill`; the paper's Assistant-colon analogue, matches
  `live_trajectory.py --position start`); `think_end` = last `think` token;
  `answer_start` = first `answer` token; `end` = last decode row. `think_end`
  and `answer_start` are `—` when the turn has no reasoning;
- segment filter `all | think | answer` on every per-token and per-turn
  aggregate (turn means, token chart, session table); the segment in force is
  printed in the panel header;
- turn-mean statistic (`--stat mean` equivalent) behind a switch, labelled as
  such wherever shown;
- trajectory x-axis is `non_empty_turn_index`;
- non-finite rows are skipped **and counted**; every panel that aggregates
  shows the skip count next to `n`;
- session aggregate, per `(source, bench_split)` group: per-emotion mean ± SEM
  by non-empty turn index; first-vs-last paired Δ per conversation with
  Wilcoxon signed-rank when `n ≥ 6` and scipy is importable, else `—`;
- `at_cap` turns are flagged, and excluded from `end` aggregates by default
  (toggle to include; the toggle state is shown in the panel header);
- optional smoothing for the token chart (moving mean over k tokens), never
  applied to stored data.

### 3.5 `app.py` — HTTP API

FastAPI + uvicorn. JSON everywhere except SSE for long operations.

| route | purpose |
|---|---|
| `GET /` | the page |
| `GET /api/session` | `session.json` + health + job time remaining |
| `GET /api/conversations` | rail contents |
| `GET /api/conversations/{id}?emotion=&readout=` | transcript with per-turn readouts for the chosen emotion and readout |
| `POST /api/chat/{id}/send` (SSE) | append user message, generate; events: `queued`, `generating` (heartbeat with elapsed s), `turn` (full record minus arrays), `error` |
| `POST /api/task/start` (SSE) | `{split, task_id, attempts, max_tokens, temperature, scratchpad, affect_prompt, auto_continue}`; events as above plus `testing`, `tests` (result + feedback), `awaiting_user`, `done` |
| `POST /api/task/{id}/continue` | `{intervention: str|null}` — next attempt |
| `POST /api/task/{id}/stop` | stop after the current step |
| `GET /api/records/{record_id}/tokens?layer=` | tokens, kinds, per-token cosine at a layer |
| `GET /api/aggregate?source=&split=&position=&stat=&segment=&include_cap=` | table + by-turn series |
| `GET /api/problems?split=` | task picker contents |
| `GET /api/health` | vLLM `/health` proxy + last-seen time |

Health is polled every 5 s in the background; the top-bar chip reflects it and
the composer is disabled while the server is unreachable.

### 3.6 `dashboard/index.html` — the page

One self-contained file: vanilla JS, inline SVG charts, no CDN, no build step.
Rendering code may be adapted from the mockup but not assumed to fit; see §7.

Layout and views (see mockup):

- **Top bar** — model, vectors version + probe layer, vLLM health chip, job id
  and node, session record count, "Copy tunnel cmd".
- **Rail** — task runs (task id, split, turns, pass/fail/running) and chats;
  `+ Chat`, `+ Task` (dialog: split, problem search, attempts, max_tokens,
  temperature, scratchpad, affect prompt, auto-continue); footer with record directory and
  job time remaining (warns at 15 min).
- **Conversation** — transcript. Assistant turns show token count (with the
  thinking share), cap flag, and the start / think→answer / end readouts for
  the "colour by" emotion. `tokens | text` toggle:
  the token view tints every generated token by its probe-layer cosine on a
  diverging blue↔red scale (fixed range ±0.08, matching the paper's
  observational scale), underlines the readout tokens (start, think→answer
  boundary, end), and renders thinking tokens inside an amber band — tinted
  identically to answer tokens, collapsible but expanded by default in the
  token view. Test-result messages show exactly what is fed back. Composer:
  message + `Send`, and in task runs `Run tests → next attempt` and `Stop`.
- **Trajectory tab** — four headline tiles (desperate, frustrated, proud,
  joyful: latest value and Δ vs first turn); per-turn line chart for the
  selected conversation with the four replicating directions emphasised and the
  other ten toggleable in grey; the session mean for the same
  `(source, split)` dashed behind; readout switch
  (`start | think_end | answer_start | end`); cap warning box.
- **Tokens tab** — cosine per generated token for one turn, hover-linked both
  ways with the transcript strip, click to pin; the thinking span is shaded
  amber with a marked think→answer boundary; `all | think | answer` segment
  switch, layer and smoothing switches; pinned-token detail card (token,
  index, segment, prefill/decode, norm, cosines).
- **Aggregate tab** — session table (t0, tlast, Δ last−first, Δ bar, SEM, p,
  n, skipped) with a `(source, split)` filter and readout
  (`start | think_end | answer_start | end`), `token | mean`, segment
  (`all | think | answer`) and `include capped` switches; mean-by-turn-index chart with SEM bands.
- **Settings tab** — server/hook/vectors/zstd status, task-loop config,
  session paths, replay command.

Design tokens: light and dark palettes defined on `:root`, `@media
(prefers-color-scheme: dark)` and `[data-theme]`; series colours are the
validated four (`#2a78d6 #eb6834 #1baf7a #eda100` light; `#3987e5 #d95926
#199e70 #c98500` dark) with direct labels; the token strip uses the diverging
pair; status colours (good/warn/critical) are separate from series colours and
always paired with a label.

## 4. Data flow

```
browser ── SSE/JSON ──▶ app.py ──▶ Engine.generate ──▶ LensClient ──▶ vLLM + hook
                          │              │
                          │              └──▶ store.append (jsonl + npz)
                          ├──▶ TaskRun ──▶ sandbox_cli (apptainer --contain) ──▶ tests
                          └──▶ stats.* (reads store) ──▶ aggregate / tokens / trajectory
```

A chat turn: user text → `Engine.generate` → record written → `turn` event →
page renders transcript strip and refreshes the trajectory tab from
`/api/conversations/{id}`.

A task attempt: `generate` → record → `robust_find_code` → `sandbox_cli run` →
`tests` event with feedback → `awaiting_user` (or `generating` if
auto-continue is on) → next attempt with the feedback (and any intervention)
appended to `messages_in`.

## 5. Error handling

| failure | behaviour |
|---|---|
| vectors missing, emotion order unusable, zstd patch reverted | refuse to start; message names the file/check |
| vLLM unreachable at start | wait via `wait_for_health` (serve.slurm already does this); at runtime the health chip goes red and the composer is disabled |
| generation error (server 4xx/5xx, timeout) | record written with `error`; transcript shows the failed turn; no retry |
| hook/token misalignment | record written with `misaligned=True`; strip hidden; warning in transcript and trajectory |
| sandbox timeout / apptainer failure | test message shows the failure; task run pauses in `awaiting_user`; no auto-retry |
| non-finite residuals | skipped and counted in every aggregate; readout chip shows `—` |
| turn at `max_tokens` | `at_cap` flag; trajectory warning; excluded from `end` aggregates by default |
| job ending | rail footer countdown; warning at 15 min; SIGTERM flushes records |

## 6. Testing

CPU tests under `tests/cpu/`, runnable on the login node:

- `test_dashboard_stats.py` — synthetic `proj`/`norm` arrays: per-token cosine,
  the four readouts (`start`, `think_end`, `answer_start`, `end`), segment
  filtering, non-empty-turn indexing, non-finite skip counting, cap flagging
  and exclusion, paired Δ, smoothing.
- `test_dashboard_store.py` — record write/read round-trip, npz keys, replay
  loader on a temp directory.
- `test_dashboard_engine.py` — `Generation` assembly from a canned hook
  payload + logprobs: alignment check, reasoning split for both tag styles,
  `token_kind`, `at_cap`.
- `test_dashboard_sandbox_cli.py` — the JSON contract of `problems`/`run`
  with the container pieces mocked; feedback contains `TEST_FAILURE_MARKER`.
- `test_dashboard_app.py` — FastAPI routes with a fake engine and a temp
  store (httpx `TestClient`), including SSE event order for a two-attempt task.

One GPU smoke stage, the plan's final gate:
`--stage scripts/dashboard.py:configs/dashboard.yaml:--smoke` starts the app,
runs one chat turn and one two-attempt task on the `original` split through
the real engine and sandbox, checks a record and npz were written and aligned,
and exits 0.

## 7. The mockup is a reference, not a drop-in

The published mockup demonstrates layout and view content. When first rendered
in the artifact viewer its transcript messages were clipped by a flex-shrink
bug, since fixed; other spacing issues are likely at other widths and in the
real page. The implementation builds `index.html` against the real API, then
runs its own screenshot pass at 1280, 1440 and 1920 px in both themes before
the smoke gate is called passing.

## 8. Out of scope (this version)

Steering controls; overlaying the committed pilot records as a baseline;
several models per job; token-text streaming (pending the spike); auth;
editing past turns.

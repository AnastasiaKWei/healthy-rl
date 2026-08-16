# Rollout viewer: reading pilot rollouts, per token, in the Affect Scope dashboard

**Date:** 2026-08-16
**Status:** implemented 2026-08-16 (branch feature/rollout-viewer)
**Builds on:** `2026-08-15-affect-dashboard-design.md` (the dashboard this extends)
**Reading a run:** the command and the tunnel line are in `docs/runs.md`
("Reading a run"); the token-strip and alignment rules in `docs/measurement.md`
("Rollout token strips"); the traps met building it in `docs/infrastructure.md`.

## 1. Purpose

The pilot's rollout records (`$ARTIFACT_DIR/rollouts/<model>/<version>/`) are read
today through `scripts/live_trajectory.py` (numbers) and
`scripts/read_transcript.sh` (text), separately. Since the mindset merge
(2026-08-16) each new record also carries every generated position's projection
onto the 14 emotion directions, so a transcript can be read with its emotions
painted on the tokens — but nothing renders that. The Affect Scope dashboard
already renders exactly this for its own records (token strip, Tokens chart,
per-turn trajectory, session aggregate) and has a read-only `--replay` mode.

This design opens rollout cells in that dashboard: literal per-token transcripts
where the arrays exist, per-turn readouts where only boundary residuals exist,
and aggregates across models and cells so a mindset arm can be read against its
base. Login-node, read-only, no GPU.

Two deliverables were agreed; this spec is the first. The second — a
self-contained static HTML export for sharing, `viewer/export_rollouts.py` over
the same store — is out of scope here and is planned to reuse §3.

## 2. Usage

```bash
python -m healthy_rl.dashboard --rollouts <path> [<path> ...] [--host H] [--port P]
```

Each path is a **cell** (`<rollouts>/<model>/<version>`), a **model**
(`<rollouts>/<model>`, all its cells) or a **root** (`<rollouts>`, everything).
A directory is a cell when it holds one or more `rollouts*.jsonl` files; a
model or root is walked one or two levels down for cells. Anything else
(`inspect-logs`, `residuals`, `scratchpad-sanity*` with its `transcripts.jsonl`)
is ignored, and the list of ignored directories is printed once at startup.

Startup prints a table of cells (model, version, split, mindset, rollouts, how
many have per-token arrays) and one line per model saying whether its tokenizer
(`$MODEL_DIR/<model>`) and its vectors artifact (`$ARTIFACT_DIR/vectors/<model>/v1`)
were found. Neither is required to start: a model without a tokenizer shows text
and turn readouts but no token strip; a model without vectors cannot read the
older, residual-only cells (§3.4). Tokenisation is lazy, so a whole root opens
in seconds. Access is by SSH tunnel, as for `--replay`.

## 3. `RolloutStore` (`src/healthy_rl/dashboard/rollout_store.py`)

Presents cell directories through the interface `SessionStore` already exposes —
`session`, `root`, `records()`, `record(rid)`, `arrays(rid)`,
`conversations()` — plus `refresh()`. Read-only; `append`/`close` raise.
`app.py` types the store as a `Protocol` with those members so either class fits.

### 3.1 One record per rollout turn

Built from the JSONL row, its npz, and the sample's `.eval` log under the
cell's `inspect-logs/`.

| field | source |
|---|---|
| `record_id` | `<model>/<version>/<task_id>/s<sample>/t<turn>` — deterministic, so a URL to a turn survives a restart (deviation 1: `s<sample>`, not `ep<epoch>`) |
| `conversation_id` | `<model>/<version>/<task_id>/s<sample>` |
| `sample` | the row's global sample index; with `task_id` it identifies a rollout within a cell |
| `source` | `"rollout"`; never pooled with `chat` or `task` |
| `model`, `version`, `mindset`, `mindset_version`, `scratchpad_reasoning`, `affect_prompt`, `bench_split`, `task_id`, `epoch`, `passed`, `shard`, `run_id`, `condition_name` | copied from the row. Rows predating a key get the documented default (`bench_split="conflicting"`, `mindset=[]`, `mindset_version=0`) |
| `turn_index` | position in the row's per-turn lists |
| `non_empty_turn_index` | index among turns with `n_generated > 0`, `None` for an empty turn (`stats.non_empty_index`) |
| `n_generated` | `turn_n_generated[t]` |
| `at_cap` | `n_generated >= max_tokens`, `max_tokens` read from the cell's `manifest.json` (`config.max_tokens`); `None` when neither records it, and the page then says "cap unknown" instead of "not capped" |
| `after_test_failure` | `turn_after_test_failure[t]` |
| `text`, `reasoning`, `answer` | `turn_completion[t]` split by `generation.split_reasoning` (`<think>`, `[THINK]`, scratchpad tags; `reasoning=None` when there is no span) |
| `messages_in` | reconstructed from the `.eval` sample: every message before this assistant turn — task prompt, prior completions, test-feedback messages — as `{role, content}`. `[]` plus a `warnings` entry when the `.eval` is missing, unreadable, or lacks the sample |
| `feedback` | the user message that followed this turn in the `.eval`, if any |
| `tokens`, `token_kind` | `turn_completion[t]` re-tokenised with the model's tokenizer (§3.3); `token_kind` from `generation.token_kinds` |
| `misaligned`, `error` | true when `len(tokens)` does not match the decode-row count under the EOS rule (§3.3); `error` carries both counts |
| `has_token_arrays` | the npz has `t{t}_proj_L*` |
| `emotions`, `capture_layers`, `probe_layer` | copied from the row |
| `warnings` | list; e.g. no `.eval`, tokenizer missing, vectors missing, cap unknown |
| `arrays` | `"virtual"`; `arrays(rid)` builds them |
| `created_at` | the JSONL file's mtime, ISO; used only for rail ordering |

`turn_completion` is the text the hook rows correspond to and is the source of
truth for the transcript; the `.eval` supplies only the surrounding messages.
`turn_stat` (the superseded mean statistic) is not surfaced.

### 3.2 `arrays(rid)`

Returns the dashboard's array dict:

- `proj` — `T×L×E` float32, from `t{t}_proj_L{n}` for every capture layer,
  decode rows only (`kind == 0`), cast from float16;
- `norm` — `T×L`;
- `proj_prefill`, `norm_prefill` — the `kind == 1` row (`L×E`, `L`);
- `res_start_L{probe}`, `res_end_L{probe}` — copied when present.

Layer order is the record's `capture_layers`. When the npz holds a layer the
row does not list, or vice versa, the record is marked `misaligned` with a
message naming the layer.

Cells with per-token arrays: 5 layers × ~10 k tokens per rollout, ~2 MB per
rollout as float32 in memory; arrays are read on request from the npz and not
cached (npz reads are ~ms; the tokeniser is the slow part and is cached).

### 3.3 Tokenisation and the EOS rule

Tokenizer: `transformers.AutoTokenizer.from_pretrained($MODEL_DIR/<model>)`,
one per model, loaded on first use, `add_special_tokens=False`. Token strings
are decoded per id (`convert_ids_to_tokens` then `convert_tokens_to_string`
per token, so byte-level markers render as text).

Alignment, checked per turn: with `D` decode rows and `N` re-tokenised tokens,

- `N == D` — aligned;
- `N + 1 == D` — aligned; the last decode row is the end-of-sequence token,
  which the completion text does not contain. A token `"<eos>"` with kind
  equal to the previous token's kind is appended so the strip has one cell per
  row and the `end` readout is the row it always was;
- otherwise — `misaligned=True`, `error = "re-tokenised N tokens, D decode rows"`.
  Text and turn readouts (`start`, `end`) still render; `think_end`,
  `answer_start`, turn means and the token strip do not, and the row is
  skipped-and-counted in aggregates that need them.

Reasoning models whose completions omit the reasoning span (a vLLM reasoning
parser that returned it separately) will misalign systematically; that is a
property of the record, and the cell table in Settings shows the misaligned
share per cell so it is visible before anyone reads a strip.

Tokenisation results (`tokens`, `token_kind`, `misaligned`, `error`) are
cached in memory by `record_id`. A disk cache is not built in this version;
add one if opening a root proves slow.

### 3.4 Cells without per-token arrays

Records written before the mindset merge hold only `t{t}_res_{start,end}_L{probe}`.
For these `arrays(rid)` returns `proj` with `T = 0`, `norm` empty, and
`proj_prefill`/`norm_prefill` computed by projecting `res_start` onto the
model's directions at the probe layer (`load_vectors`), and an extra
`proj_end`/`norm_end` from `res_end`. `stats.turn_readout` gains an optional
`proj_end`/`norm_end` pair used for the `end` readout when `proj` is empty. So
`start` and `end` work as `live_trajectory.py --stat token` computes them;
`think_end`, `answer_start`, and turn means are `None`. Without the vectors
artifact every readout is `None` and `warnings` says why. Non-finite residuals
give `None` and are counted as skipped, as everywhere else.

### 3.5 Refresh and growth

`records()` re-reads any `*.jsonl` whose mtime or size changed and discovers new
shard files, so a cell mid-run grows in the rail; the app calls it on every
request that lists records, as it does today with `SessionStore` (which
re-reads the file each call). Cached tokenisation is keyed by `record_id` and
never invalidated: a rollout row, once written, does not change.

### 3.6 `session`

```
{ "mode": "rollouts",
  "roots": [paths given],
  "models": { "<model>": { "emotions": [...], "capture_layers": [...], "probe_layer": n,
                           "tokenizer": "ok" | "missing", "vectors": "ok" | "missing" } },
  "cells":  [ { "model", "version", "bench_split", "mindset", "scratchpad_reasoning",
                "affect_prompt", "n_rollouts", "n_with_token_arrays", "n_misaligned",
                "max_tokens" } ] }
```

`root` is the first path given (the page prints it in the rail footer).

## 4. App changes (`app.py`)

`AppState` gains `mode: "live" | "replay" | "rollouts"` and `store` typed by the
store protocol. `st.vectors` is `None` in rollouts mode.

**Per-record direction metadata.** A `VectorsMeta(emotions, capture_layers,
probe_layer)` derived per record from its own fields. Helpers become

- `_meta(rec) -> VectorsMeta` — in live/replay mode returns the session's
  `Vectors`; in rollouts mode the record's;
- `_layer(layer, rec)` — default `meta.probe_layer`, must be in
  `meta.capture_layers`, 400 otherwise;
- `_emotion_order_mismatch(rec, meta)` — in rollouts mode `meta` is the model's
  `vectors.json` order when the artifact exists, else the first record seen for
  that model; the check and its consequence (readouts `None`,
  `emotion_order_mismatch: true`, skipped-and-counted in aggregates) are
  unchanged;
- `_readouts_for(rec, arrays, meta)` — as now, at `meta.probe_layer`.

Live and replay paths behave exactly as before; the tests that pin them keep
passing unchanged.

**Routes.**

| route | change |
|---|---|
| `GET /api/session` | adds `mode`; in rollouts mode `models` and `cells` from `RolloutStore.session`; `emotions`/`capture_layers`/`probe_layer` are those of the first model (kept for the page's existing code paths) |
| `GET /api/conversations` | new optional `model=`, `version=` filters (repeatable); rollout conversations carry `model`, `version`, `mindset`, `bench_split`, `epoch`, `passed`, `n_turns`, `has_token_arrays` (any turn), `n_misaligned` |
| `GET /api/conversations/{cid}` | unchanged shape; turns carry `feedback`, `messages_in`, `has_token_arrays`, `misaligned`, `warnings`; readouts at the record's probe layer |
| `GET /api/records/{rid}/tokens` | `layer` validated against the record's layers |
| `GET /api/aggregate` | see below |
| chat/task POST routes | 409 in rollouts mode, as in replay |

**Grouped aggregate.** New params `source=rollout`, `model` (repeatable),
`version` (repeatable), `layer=probe|<int>` (default `probe`). Selection:
records whose `(model, version)` is in the cross product of the given lists
(all when a list is empty). The split-pooling refusal stays: mixed splits with
no `split=` is 400. Response:

```
{ "groups": [ { "model", "version", "bench_split", "mindset", "layer",
                "n_conversations", "n_records", "excluded_cap", "skipped",
                "by_turn", "delta" } ],
  "emotions": [...],          # union in first-seen order; a group's series is
                              # aligned to it, None where the model lacks one
  "params": {...} }
```

`layer=probe` reads each group at its model's probe layer; an integer must be a
capture layer of every selected model, else 400 naming the first that lacks it.
Live/replay requests return a single group (the session), so the page's current
consumers read `groups[0]`. `by_turn` and `delta` are computed by
`stats.by_turn_index` / `stats.paired_delta` per group exactly as now.

## 5. Page (`static/index.html`)

One file, rollouts mode switched on `session.mode`.

- **Rail** — `model ▸ cell ▸ rollout`, cells collapsible (default: expanded
  when one cell is loaded, collapsed otherwise). Cell header: version, split
  chip, mindset chips, `n` rollouts, `k` with token arrays, misaligned count if
  any. Rollout row: task id, `ep n` when the cell has epochs > 1, pass/fail
  mark, turn count, a dot when per-token data exists. A text filter on task id
  and cell name. `+ Chat`, `+ Task`, the composer, health chip and job
  countdown are hidden; the footer prints the roots.
- **Conversation** — existing transcript renderer. Turn 0's `messages_in`
  supplies the task prompt (collapsible, as the static viewer does); each
  assistant turn renders as now (token strip in `tokens` view, text in `text`
  view, its readouts for the colour-by emotion); the `feedback` message
  follows it in a collapsible. Turns without token arrays render text only with
  a one-line note; misaligned turns show the two counts. Header shows model,
  cell, split, mindset, epoch, pass/fail.
- **Trajectory** — unchanged; the dashed session-mean overlay is the
  conversation's own `(model, version, split)` group.
- **Tokens** — unchanged; the layer switch offers the record's layers.
- **Aggregate** — a groups picker replaces the source/split filter: model
  checklist → cell checklist (each cell shows its split; the split radio filters
  cells), plus the existing readout / stat / segment / include-cap switches and
  the layer switch (`probe` or a common layer). Chart: one line + SEM band per
  selected group for the chosen emotion, one colour per group with a direct
  label at line end; more than eight groups is allowed but the legend collapses
  to a table. Table: one row per group × emotion (t0, tlast, Δ, SEM, p, n,
  skipped). A **with base** button adds a selected mindset arm's base cell
  (`d6` / `aff6` / `sp6` chosen by the arm's `affect_prompt` and
  `scratchpad_reasoning` flags) when it is loaded. Colours follow the
  dashboard's existing categorical set; status colours are not reused for groups.
- **Settings** — the model table and cell table from `session`.

The trajectory's rule from the dashboard deviations still holds: it ignores the
layer/segment/stat switches and reads single tokens at the probe layer.

## 6. Error handling

| failure | behaviour |
|---|---|
| path is not a cell / holds no rows | listed in the startup ignore line; if nothing loads, exit 2 with the paths |
| tokenizer missing for a model | `models[m].tokenizer="missing"`; records `misaligned=True`, `error="no tokenizer"`; strip hidden, text and readouts shown |
| vectors missing for a model | old cells' readouts `None` with a warning; new cells unaffected |
| `.eval` missing / unreadable / sample absent | `messages_in=[]`, `feedback=None`, warning; the transcript shows the completions only and says the prompt is unavailable |
| npz missing or key missing | record `misaligned=True` with the key named; readouts `None` |
| re-tokenised count off by more than the EOS rule | `misaligned=True`, counts in `error` (§3.3) |
| non-finite residuals / projections | `None`, skipped and counted, as elsewhere |
| emotion order differs from the model's `vectors.json` | readouts `None`, `emotion_order_mismatch`, skipped and counted |
| layer requested that a selected model lacks | 400 naming the model |
| mixed splits in one aggregate without `split=` | 400 (unchanged) |

## 7. Testing

CPU, `tests/cpu/`:

- `test_rollout_store.py` — a synthetic cell written to a temp dir (two JSONL
  shards, one npz per rollout, `manifest.json` with `max_tokens`, and a minimal
  `.eval` written through `inspect_ai.log` — if that proves awkward, the `.eval`
  is stubbed by a monkeypatched loader and one test runs with it absent).
  Covers: record ids and fields; `non_empty_turn_index` over a zero-token
  leading turn; EOS-tolerant alignment (`N == D`, `N + 1 == D`, `N + 2 == D`
  → misaligned); token kinds across a `[THINK]` span; `arrays()` shape and
  layer order; the old-cell path with fake directions (start/end readouts
  finite, others `None`); vectors-missing path; `refresh()` seeing an appended
  row and a new shard; `messages_in`/`feedback` reconstruction; the fake
  tokenizer is a whitespace tokenizer injected through a constructor argument
  so `MODEL_DIR` is never touched.
- `test_dashboard_app.py` additions — rollouts-mode `AppState` over the
  synthetic store: `/api/session` shape; `/api/conversations` filters;
  `/api/conversations/{cid}` readouts at the record's probe layer;
  `/api/records/{rid}/tokens` layer validation; grouped `/api/aggregate`
  (two models with different probe layers, `layer=probe` ok, `layer=<int>`
  400 naming the model, split refusal, `groups[0]` shape equal to the live
  shape); POST routes 409.
- `test_dashboard_stats.py` — `turn_readout` with empty `proj` and the
  `proj_end`/`norm_end` pair.
- Existing live/replay tests must pass unchanged.

Manual gate before calling it done: open `--rollouts $ARTIFACT_DIR/rollouts`
on the login node, read one Ministral `appr6` rollout in tokens view, one
`d6` (old) rollout in text view with start/end readouts, and an aggregate of
`appr6` with base `d6`; screenshot pass at 1280 and 1920 px, both themes.

## 8. Documentation

- `docs/runs.md` "Reading a run": the `--rollouts` command and the tunnel line.
- `docs/measurement.md`: token strips on rollouts are re-tokenised
  `turn_completion` text under the EOS rule; misalignment means the strip is
  withheld, not shifted.
- `docs/infrastructure.md`: any tokenizer or `.eval`-reading trap met while
  building.

## 9. Out of scope

The static export (`viewer/export_rollouts.py`); a disk tokenisation cache;
per-token residuals (not stored); steering; editing; comparing cosines across
models as if commensurable — the page draws them side by side and labels each
group with its model and layer, and leaves the interpretation to the reader.

## Deviations

Where the implementation departs from the design above, and why. Written at the
end of the build (2026-08-16), after `tests/cpu` and the manual gate on the real
`$ARTIFACT_DIR/rollouts` root.

1. **A rollout is identified by its sample, not its epoch** (§3.1). `record_id` is
   `<model>/<version>/<task_id>/s<sample>/t<turn>` and `conversation_id` is
   `<model>/<version>/<task_id>/s<sample>`, where `sample` is the row's global
   sample index. `ep<epoch>` does not identify anything: a cell holds several
   samples of each task and every one of them sits at Inspect epoch 1, and a
   resumed shard restarts the numbering, so ids built from the epoch would collide
   for most of a cell. `sample` is added to §3.1's field list, and `epoch` is kept
   and shown but is not part of any id. A steering sweep is the one case where
   `(task_id, sample)` is not enough: it re-runs each pair once per steering
   condition, so a row whose `condition_name` is neither absent nor `readout`
   appends `/c<condition_name>` to both ids (`m/v1/lcbhard_0/s0/ccalm+0.1`). The
   unsteered `readout` arm every ordinary cell writes stays bare, so no other
   cell's ids move.

2. **`records()` is light; `record(rid)` tokenises** (§3.3, §3.6). `records()`
   returns records without `tokens`/`token_kind`/`misaligned`, so opening the whole
   root is a JSON parse; `record(rid)` tokenises that turn, fills them in and
   caches the result in `_full`. The cell table therefore cannot report a
   misaligned *share*: it reports `n_tokenised` and `n_misaligned` **among the
   tokenised**, both of which grow as the session is read. `_full` entries are
   dropped on refresh when the underlying row changed (`created_at` excluded from
   that comparison: it is the shard file's mtime and moves for every row each time
   the file grows). Two rows with the same identity collapse last-writer-wins and
   are counted per cell as `n_duplicate_rows`. Since the steering condition joined
   the id (deviation 1) no cell on disk has any: `Olmo-3.1-32B-Think/v1`, the one
   sweep, shows all 172 of its rows as 172 rollouts over 36 task-sample pairs and
   counts 0 duplicates.

3. **Token strings come from offsets, not from decoding ids** (§3.3). `tokenise`
   uses the fast tokenizer's `offset_mapping` and slices the completion, so the
   tokens tile the text exactly (`"".join(tokens) == text`) and a strip cell is a
   literal piece of what the model wrote; the true span start (not the tiled start)
   decides the think/answer boundary, so folding a dropped leading space into the
   following token cannot move it. Per-id decode was the design's route and loses
   this property on byte-level and SentencePiece vocabularies. A slow tokenizer
   falls back to per-id decode with cumulative offsets; that path is untested —
   every model in `$MODEL_DIR` loads a fast tokenizer.

4. **`.eval` samples are matched by completion text, and messages are indexed by
   non-empty turn** (§3.1). Several samples share a task id at epoch 1, so
   `sample_messages` matches on the assistant messages equalling `turn_completion`;
   an all-empty rollout (nothing generated anywhere) has no text to match on and
   falls back to the id, but only where exactly one sample carries it. A turn that
   generated nothing wrote no assistant message, so alignment is by
   `non_empty_turn_index`, and an empty turn is shown with the context that
   precedes the next assistant message there is. Per-turn `passed` is `False` when
   that turn drew test feedback, the rollout's own result on the last turn, and
   `None` otherwise; the conversation keeps the rollout-level `passed`.

5. **`arrays_from_npz` takes the emotion count from the record, and refuses a
   vectors artifact that disagrees with it** (§3.2, §3.4). The signature is
   `arrays_from_npz(z, *, turn, capture_layers, probe_layer, n_emotions,
   vectors=None, emotions=None)`. Before it projects an old cell's boundary
   residuals it cross-checks the artifact against the record — probe layer,
   emotion count, emotion order — and on any disagreement drops the artifact and
   returns a named problem instead of projecting; columns are never quietly
   relabelled onto a different order. `residuals: null` is the honest array-less
   case and is left unmarked, while a missing or unreadable npz, a `t{t}_proj_L*`
   without its `t{t}_kind_L*`, and a layer list that differs from the record's all
   set `misaligned=True` with the path, key or layer named.

6. **`/api/aggregate` returns `groups` in every mode** (§4). Live and replay are a
   group of one rather than a separate response shape, so the page has one consumer
   to write. The response is `{groups, emotions, params}`; `emotions` is the union
   over the groups in first-seen order and every group's series is widened onto it,
   with `None` where a model lacks an emotion. `layer` is `probe` (each group at its
   own model's probe layer) or an integer that every selected model must have
   captured. In rollouts mode a `split=` that is not one of the splits on disk is a
   400 naming the ones that are — there is no sandbox to validate it against, and a
   typo would otherwise return an empty aggregate.

7. **`RolloutStore` holds an `RLock`** (§3). The routes are sync, so FastAPI runs
   them in a threadpool and concurrent `/api/aggregate` requests shared one store:
   `refresh()` rebuilding the record index in place while another thread iterated
   it produced `RuntimeError: dictionary changed size during iteration`. Every
   entry point takes the lock; it is an RLock because the methods call each other.
   Recorded as a trap in docs/infrastructure.md.

8. **Page details not in §5.** Rollouts-mode aggregate requests are chained one at
   a time (each re-reads npz files, and the store answers them serially anyway).
   `startup_report` prints at most 8 ignored directories and then `(+N more)` — the
   real root has ~270, because `discover_cells` walks dot-directories and
   `.scratch/<jobid>` is one per job; that is noted, not changed. The first
   conversation opened against a real cell takes ~20 s (the tokenizer load), which
   the page covers with a "loading transcript" ticker. In rollouts mode the
   composer's controls are hidden but `#cstat` stays, so a status line still has
   somewhere to land.

9. **Old cells have no completion text at all** (§3.4). §3.4 assumed the
   pre-mindset-merge cells were missing only their per-token arrays. They are also
   missing `turn_completion`: the two keys arrived in the same merge, and of the
   1645 rows on disk on 2026-08-16 the same 1031 have both, with no row having one
   without the other. So a `d6`/`aff6`/`sp6`/`v1` rollout renders with its problem
   statement, its test feedback, its `start`/`end` readouts and its trajectory, but
   with **empty assistant bubbles**. The text is not lost — turns 1..n−1 appear as
   the next turn's `messages_in`, read out of the `.eval` — but the page does not
   re-attribute it, and its "text and start/end readouts only" note promises a text
   those records do not carry. Worth a wording fix, and worth knowing before
   opening an old cell expecting a transcript.

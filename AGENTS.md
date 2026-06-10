# AGENTS.md — lms-mon development guide

Instructions for AI coding agents working in this repository.

## What this project is

**lms-mon** is a single-file [Textual](https://textual.textualize.io/) TUI that monitors
[LM Studio](https://lmstudio.ai): loaded models, live inference logs, system/GPU metrics, and
rolling sparkline history.

| File | Role |
|------|------|
| `lms_mon.py` | Entire application (~1.2k lines): widgets, polling, log stream, CLI |
| `pyproject.toml` / `setup.cfg` | Package metadata and `lms-mon` console entry-point |
| `requirements.txt` | Runtime deps for `pip` / `uv pip install` |
| `README.md` | User-facing install, usage, shortcuts |
| `images/` | Screenshots for README only |

There is **no test suite yet** and **no CI**. Keep changes focused and verify manually.

---

## Prerequisites

**Required for development**

- Python ≥ 3.10 (3.12 recommended)
- `git`

**Required for full runtime behaviour** (not needed to edit UI code)

- LM Studio ≥ 0.3.26 with local server running
- `lms` CLI on `PATH` (`~/.lmstudio/bin/lms bootstrap`)
- Optional: `nvidia-smi` for GPU metrics

---

## Install

From the repo root:

### Recommended — uv

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -r requirements.txt
# optional: install CLI entry-point
uv pip install -e .
```

### pip + venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Verify install

```bash
python3 -m py_compile lms_mon.py
python lms_mon.py --help
# if editable install:
lms-mon --help
```

---

## Run

```bash
# direct (no install)
python lms_mon.py

# installed entry-point
lms-mon

# remote / custom port
lms-mon --host 192.168.1.100 --port 8080

# log stream options
lms-mon --log-source model --log-filter input,output
lms-mon --no-stats
```

**LM Studio must be reachable** at `--host`:`--port` (default `localhost:1234`) for the models
pane and log stream to work. CPU/RAM metrics work without LM Studio.

Quit with `q` or Textual’s `Ctrl+Q`.

---

## Tests

There are **no automated tests** in this repo today.

### What to run before submitting changes

```bash
python3 -m py_compile lms_mon.py
```

### Manual smoke test

1. Start LM Studio server: `lms server start`
2. Load a model (or use `l` in the TUI)
3. Run `python lms_mon.py` and confirm:
   - Models table populates
   - Log pane shows `lms log stream` events during inference
   - CPU/RAM sparklines advance together every second
   - `Space` filters logs to the cursor model; `s` / `l` / `u` work on models pane
   - Clean exit on `q` (no subprocess / event-loop errors)

### Adding tests (welcome)

Pure helpers at the top of `lms_mon.py` are the easiest to unit-test without a terminal:

- `_parse_log_filter`, `_log_event_data`, `_log_event_matches_selection`
- `_model_matches`, `_stats_float`, `_stats_int`, `_traffic_color`
- `LogStreamOptions.build_cmd()`

Suggested layout when introducing tests:

```
tests/
  test_log_parsing.py
  test_log_stream_options.py
```

Use `pytest`. Add `pytest` as an optional dev dependency only if a `pyproject.toml`
`[project.optional-dependencies]` dev group is added — do not add test deps to
`requirements.txt` unless the maintainer asks.

Textual UI integration tests are possible via `textual.pilot` but are slow and need a
pseudo-terminal; prefer unit tests for logic.

---

## Architecture (where to edit)

All logic lives in **`lms_mon.py`**. Key areas:

```
constants (top)          — HISTORY_*, POLL_INTERVAL, SPARKLINE_STRIDE, etc.
helpers                  — _log_* , _nvidia_smi , _model_matches , …
GradientSparkline*       — utilization charts (green→red, strided bars)
LoadModelScreen          — modal for `lms load` extra flags
ModelTable               — model list; cursor vs log-filter selection
MetricSparkline          — labeled sparkline wrapper
SysPanel / InferenceStatsPanel
LMSMon (App)             — layout CSS, bindings, polling, log subprocess
main()                   — argparse → LMSMon.run()
```

### External integrations

| Integration | Mechanism |
|-------------|-----------|
| Model list | `GET /api/v0/models` via `httpx.AsyncClient` |
| Log stream | Subprocess: `lms log stream --json [--stats] …` |
| Load / unload | Subprocess: `lms load …` / `lms unload …` |
| CPU / RAM | `psutil` in a thread worker |
| GPU | `nvidia-smi` subprocess in `_try_nvidia_smi()` |

### Textual workers (important)

Do not merge poll and log work into one `@work` group — it caused log stream cancellation:

- `@work(..., group="poll")` — model list refresh
- `@work(..., group="log")` — `lms log stream` subprocess loop
- Metrics sparklines use `set_interval(METRICS_INTERVAL, self._tick_sparklines)` on the main
  thread; CPU/RAM/GPU samples are cached and pushed together each tick.

### Log JSON shape

`lms log stream --json` events look like:

```json
{
  "timestamp": "...",
  "data": {
    "type": "llm.prediction.input|output|server.log|runtime.log",
    ...
  }
}
```

Parsers use `_log_event_data()` — always read `data.type`, not top-level `type`.

### Tunable UI constants

```python
METRICS_INTERVAL   = 1.0    # sparkline sample rate (seconds)
HISTORY_WINDOW_SEC = 240    # history duration
SPARKLINE_STRIDE   = 4        # bar char + gap columns
grid-columns: 3fr 1fr        # left column wider than right (in LMSMon.CSS)
```

---

## Development guidelines

1. **Minimize scope** — single-file app; avoid splitting files unless the maintainer requests it.
2. **Match existing style** — plain functions + Textual widgets; minimal comments; no over-abstraction.
3. **Textual 8** — uses `RichLog` (not `Log`), `Sparkline`, command palette (`Ctrl+P`) enabled by default.
4. **Do not commit or push** unless the user explicitly asks. Same for README/images unless requested.
5. **Never update git config** in this environment; use env vars for author identity if committing:
   `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME`, `GIT_COMMITTER_EMAIL`.
6. **Secrets** — never commit `.env`, API keys, or local paths with credentials.
7. **Dependencies** — bump `requirements.txt`, `pyproject.toml`, and `setup.cfg` together if versions change.

---

## Common tasks

| Task | Where |
|------|-------|
| New keyboard shortcut | `LMSMon.BINDINGS` and/or `ModelTable.BINDINGS`; implement `action_*` |
| Change pane layout | `LMSMon.CSS` (`#main` grid) and `compose()` |
| Log source / filters | `LogStreamOptions`, `_dispatch_*_log`, CLI args in `main()` |
| Sparkline appearance | `GradientSparklineRenderable`, `MetricSparkline`, constants |
| Model table columns | `_poll_models()` row building |
| GPU backend | `_try_nvidia_smi()` |

Document new user-visible shortcuts in `README.md` when adding bindings (including hidden ones
discoverable via `Ctrl+P`).

---

## Git / releases

- Default branch: `main`
- Remote: `https://github.com/dcolley/lms-mon`
- License: MIT

No release automation. Version is `0.1.0` in `pyproject.toml` / `setup.cfg`.

---

## Getting help

- User docs: [README.md](README.md)
- Textual command palette guide: https://textual.textualize.io/guide/command_palette/
- LM Studio CLI: `lms --help`, `lms log stream --help`

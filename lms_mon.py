#!/usr/bin/env python3
"""
lms-mon — htop-style TUI monitor for LM Studio
Usage: lms-mon [--host HOST] [--port PORT]
"""

import asyncio
import subprocess
import json
import sys
import argparse
import shlex
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

try:
    import psutil
except ImportError:
    print("psutil not found. Install: pip install psutil")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("httpx not found. Install: pip install httpx")
    sys.exit(1)

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, DataTable, Static, RichLog, Sparkline, Label, Input,
)
from textual.reactive import reactive
from textual import work
from rich.markup import escape as rich_escape
from rich.text import Text
from rich.table import Table


# ─── constants ─────────────────────────────────────────────
HISTORY_LEN   = 60
POLL_INTERVAL = 2.0
GPU_INTERVAL  = 1.5
LOG_SOURCES   = ("model", "server", "runtime")


@dataclass
class LogStreamOptions:
    source: Literal["model", "server", "runtime"] = "model"
    filter_input: bool = False
    filter_output: bool = False
    stats: bool = True

    def build_cmd(self) -> list[str]:
        cmd = ["lms", "log", "stream", "-s", self.source, "--json"]
        if self.source == "model" and self.stats:
            cmd.append("--stats")
        if self.source == "model":
            parts = []
            if self.filter_input:
                parts.append("input")
            if self.filter_output:
                parts.append("output")
            if parts:
                cmd.extend(["--filter", ",".join(parts)])
        return cmd

    def summary(self) -> str:
        parts = [self.source]
        if self.source == "model":
            filt = []
            if self.filter_input:
                filt.append("in")
            if self.filter_output:
                filt.append("out")
            parts.append("+".join(filt) if filt else "all")
        if self.source == "model" and self.stats:
            parts.append("stats")
        return " | ".join(parts)


def _parse_log_filter(value: Optional[str]) -> tuple[bool, bool]:
    if not value:
        return False, False
    parts = {p.strip().lower() for p in value.split(",") if p.strip()}
    return "input" in parts, "output" in parts


def _stats_float(stats: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        val = stats.get(key)
        if val is not None:
            return float(val)
    return default


def _stats_int(stats: dict, *keys: str, default: int = 0) -> int:
    for key in keys:
        val = stats.get(key)
        if val is not None:
            return int(val)
    return default


def _log_timestamp(obj: dict) -> str:
    ts_ms = obj.get("timestamp")
    if isinstance(ts_ms, (int, float)) and ts_ms > 1e12:
        return datetime.fromtimestamp(ts_ms / 1000).strftime("%H:%M:%S")
    if isinstance(ts_ms, (int, float)):
        return datetime.fromtimestamp(ts_ms).strftime("%H:%M:%S")
    return datetime.now().strftime("%H:%M:%S")


# ─── helpers ──────────────────────────────────────────────

def _nvidia_smi_num(val: str) -> float:
    val = val.strip()
    if val in ("N/A", "[N/A]"):
        return 0.0
    return float(val)


def _try_nvidia_smi() -> Optional[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
            text=True,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) < 5:
            return None
        return {
            "util":      _nvidia_smi_num(parts[0]),
            "mem_used":  _nvidia_smi_num(parts[1]),
            "mem_total": _nvidia_smi_num(parts[2]),
            "temp":      _nvidia_smi_num(parts[3]),
            "power":     _nvidia_smi_num(parts[4]),
        }
    except Exception:
        return None


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _model_matches(selected: Optional[str], log_model: str) -> bool:
    """True if a log event belongs to the selected model (paths may differ)."""
    if not selected:
        return True
    if not log_model:
        return False
    a = selected.replace("\\", "/").rstrip("/")
    b = log_model.replace("\\", "/").rstrip("/")
    if a == b:
        return True
    return b.endswith("/" + a) or a.endswith("/" + b)


def _log_event_data(obj: dict) -> dict:
    """Unwrap a DiagnosticsLogEvent from `lms log stream --json`."""
    data = obj.get("data")
    return data if isinstance(data, dict) else obj


def _log_event_models(obj: dict) -> list[str]:
    data = _log_event_data(obj)
    models: list[str] = []
    for key in ("modelIdentifier", "modelPath", "model"):
        val = data.get(key)
        if val:
            models.append(str(val))
    return models


def _log_event_matches_selection(selected: Optional[str], obj: dict) -> bool:
    if not selected:
        return True
    models = _log_event_models(obj)
    if not models:
        return False
    return any(_model_matches(selected, m) for m in models)


def _bar(value: float, max_value: float, width: int = 12, fg: str = "green") -> Text:
    filled = int(round(value / max(max_value, 1) * width))
    bar = "█" * filled + "░" * (width - filled)
    t = Text()
    t.append(bar, style=fg)
    t.append(f" {value:5.1f}%", style="bold white")
    return t


# ─── Widgets ──────────────────────────────────────────────

class LoadModelScreen(Screen[Optional[str]]):
    """Modal to enter extra `lms load` flags for the model under the cursor."""

    DEFAULT_CSS = """
    LoadModelScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }
    #load-dialog {
        width: 72;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #load-dialog Static {
        margin-bottom: 1;
    }
    #load-args {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name

    def compose(self) -> ComposeResult:
        with Container(id="load-dialog"):
            yield Static(f"[bold]Load model[/]\n[dim]{rich_escape(self._model_name)}[/]")
            yield Static(
                "Extra [bold]lms load[/] flags  [dim](Enter to run, Esc to cancel)[/]\n"
                "[dim]e.g. -c 128000  --gpu max  --parallel 2[/]"
            )
            yield Input(placeholder="-c 128000", id="load-args")

    def on_mount(self) -> None:
        self.query_one("#load-args", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "load-args":
            self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ModelTable(DataTable):
    """Model list: ↑↓ move cursor; click or Space toggles model selection."""

    BINDINGS = [
        Binding("space", "toggle_row", "Select", show=False),
        Binding("s", "toggle_sort", "Sort", show=False),
        Binding("l", "load_model", "Load", show=False),
        Binding("u", "unload_model", "Unload", show=False),
    ]

    def _on_mouse_move(self, event) -> None:
        # Suppress the hover cursor so moving the mouse doesn't mimic row selection.
        self._set_hover_cursor(False)

    def _emit_row_toggle(self) -> None:
        if self.cursor_type == "row" and self.show_cursor and self.row_count:
            self._post_selected_message()

    async def _on_click(self, event) -> None:
        await super()._on_click(event)
        self._emit_row_toggle()

    def action_toggle_row(self) -> None:
        self._emit_row_toggle()

    def action_toggle_sort(self) -> None:
        if isinstance(self.app, LMSMon):
            self.app.toggle_model_sort()

    def action_load_model(self) -> None:
        if isinstance(self.app, LMSMon):
            self.app.prompt_load_model()

    def action_unload_model(self) -> None:
        if isinstance(self.app, LMSMon):
            self.app.prompt_unload_model()


class MetricSparkline(Vertical):
    """Labeled sparkline backed by a rolling deque."""

    DEFAULT_CSS = """
    MetricSparkline {
        height: 6;
        border: solid $panel;
        padding: 0 1;
        margin-bottom: 1;
    }
    MetricSparkline Label {
        color: $text-muted;
        text-style: bold;
        height: 1;
    }
    MetricSparkline Sparkline {
        height: 4;
    }
    """

    def __init__(self, label: str, max_val: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self._label   = label
        self._max_val = max_val
        self._data: deque = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)

    def compose(self) -> ComposeResult:
        yield Label(self._label)
        yield Sparkline(list(self._data), summary_function=max)

    def push(self, value: float) -> None:
        self._data.append(value)
        self.query_one(Sparkline).data = list(self._data)


class SysPanel(Static):
    """CPU / RAM / GPU resource meters."""

    DEFAULT_CSS = """
    SysPanel { padding: 1 2; height: auto; }
    """

    cpu_pct:       reactive[float] = reactive(0.0)
    ram_pct:       reactive[float] = reactive(0.0)
    ram_used:      reactive[float] = reactive(0.0)
    ram_total:     reactive[float] = reactive(1.0)
    gpu_util:      reactive[float] = reactive(0.0)
    gpu_mem_pct:   reactive[float] = reactive(0.0)
    gpu_mem_used:  reactive[float] = reactive(0.0)
    gpu_mem_total: reactive[float] = reactive(1.0)
    gpu_temp:      reactive[float] = reactive(0.0)
    gpu_pwr:       reactive[float] = reactive(0.0)
    has_gpu:       reactive[bool]  = reactive(False)

    def render(self) -> Table:
        t = Table.grid(expand=True, padding=(0, 1))
        t.add_column("label", width=14)
        t.add_column("bar",   width=22)
        t.add_column("value", min_width=18)
        t.add_row("[bold cyan]CPU[/]",
                  _bar(self.cpu_pct, 100, fg="cyan"),
                  f"[dim]{psutil.cpu_count()} cores[/]")
        t.add_row("[bold yellow]RAM[/]",
                  _bar(self.ram_pct, 100, fg="yellow"),
                  f"[dim]{_fmt_bytes(self.ram_used)} / {_fmt_bytes(self.ram_total)}[/]")
        if self.has_gpu:
            tc = "red bold" if self.gpu_temp > 80 else "white"
            t.add_row("[bold green]GPU Util[/]",
                      _bar(self.gpu_util, 100, fg="green"),
                      f"[dim]Temp: [{tc}]{self.gpu_temp:.0f}°C[/]  Pwr: {self.gpu_pwr:.0f}W[/]")
            t.add_row("[bold magenta]VRAM[/]",
                      _bar(self.gpu_mem_pct, 100, fg="magenta"),
                      f"[dim]{_fmt_bytes(self.gpu_mem_used*1e6)} / {_fmt_bytes(self.gpu_mem_total*1e6)}[/]")
        else:
            t.add_row("[dim]GPU[/]", Text("[dim]nvidia-smi not detected[/]"), "")
        return t


class InferenceStatsPanel(Static):
    """Rolling inference metrics from lms log stream."""

    DEFAULT_CSS = """
    InferenceStatsPanel {
        padding: 1 2; height: 9;
        border: solid $panel; margin-top: 1;
    }
    """

    tps_last:  reactive[float] = reactive(0.0)
    tps_avg:   reactive[float] = reactive(0.0)
    tps_peak:  reactive[float] = reactive(0.0)
    ttft_ms:   reactive[float] = reactive(0.0)
    req_count: reactive[int]   = reactive(0)
    _history: deque = deque(maxlen=HISTORY_LEN)

    def render(self) -> str:
        bar = "█" * min(int(self.tps_last / 5), 20) + "░" * max(0, 20 - int(self.tps_last / 5))
        return (
            f"[bold]⚡ Inference Metrics[/]\n"
            f"  [dim]Tok/s  last [/][bold cyan]{self.tps_last:7.1f}[/]  {bar}\n"
            f"  [dim]Tok/s  avg  [/][bold]{self.tps_avg:7.1f}[/]\n"
            f"  [dim]Tok/s  peak [/][bold]{self.tps_peak:7.1f}[/]\n"
            f"  [dim]TTFT        [/][bold yellow]{self.ttft_ms:7.0f} ms[/]\n"
            f"  [dim]Total reqs  [/][bold]{self.req_count}[/]"
        )

    def record(self, tps: float, ttft: float) -> None:
        self._history.append(tps)
        self.tps_last  = tps
        self.tps_avg   = sum(self._history) / len(self._history)
        self.tps_peak  = max(self._history)
        self.ttft_ms   = ttft * 1000
        self.req_count += 1


# ─── Main App ──────────────────────────────────────────────

class LMSMon(App):

    TITLE     = "lms-mon"
    SUB_TITLE = "LM Studio Monitor  •  Tab/S-Tab navigate  •  Space select model  •  q quit"

    CSS = """
    Screen { background: $background; }
    #main {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-rows: 2fr 3fr;
        grid-gutter: 1 2;
        height: 1fr;
        padding: 0 1;
    }
    .pane { border: round $panel; padding: 0 1; height: 100%; }
    .pane.active-pane { border: round $accent; }
    .pane-title { text-style: bold; color: $accent; height: 1; padding: 0 1; }
    #pane-charts { overflow-y: auto; }
    DataTable { height: 1fr; }
    RichLog { height: 1fr; }
    """

    BINDINGS = [
        Binding("tab",       "next_pane",        "Next pane"),
        Binding("shift+tab", "prev_pane",        "Prev pane"),
        Binding("r",         "refresh",          "Refresh"),
        Binding("c",         "clear_log",        "Clear log"),
        Binding("1",         "log_source_model", "Log: model",   show=False),
        Binding("2",         "log_source_server","Log: server",  show=False),
        Binding("3",         "log_source_runtime","Log: runtime", show=False),
        Binding("i",         "toggle_log_input", "Log: input",   show=False),
        Binding("o",         "toggle_log_output","Log: output",  show=False),
        Binding("S",         "toggle_log_stats", "Log: stats",   show=False),
        Binding("q",         "quit",             "Quit"),
    ]

    _PANES  = ["pane-models", "pane-sys", "pane-log", "pane-charts"]
    _pane_i = reactive(0)
    _sel_mdl: Optional[str] = None    # log subscription filter (Space / click)
    _cursor_mdl: Optional[str] = None  # table cursor position (↑↓ / click)
    _model_sort: Literal["alpha", "loaded"] = "alpha"
    _load_target: Optional[str] = None

    def __init__(self, host="localhost", port=1234, log_opts: Optional[LogStreamOptions] = None, **kw):
        super().__init__(**kw)
        self._host     = host
        self._port     = port
        self._base_url = f"http://{host}:{port}"
        self._client       = httpx.AsyncClient(timeout=5.0)
        self._lms_proc     = None
        self._log_opts     = log_opts or LogStreamOptions()
        self._log_shutdown = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with Vertical(id="pane-models", classes="pane"):
                yield Static("", id="models-pane-title", classes="pane-title")
                tbl = ModelTable(id="model-table", cursor_type="row", zebra_stripes=True)
                tbl.add_columns(
                    ("Model", "model"), "State", "Ctx", "GPU%", "VRAM", "Arch", "Quant",
                )
                yield tbl
            with Vertical(id="pane-sys", classes="pane"):
                yield Static("◈ System Resources", classes="pane-title")
                yield SysPanel(id="sys-panel")
                yield InferenceStatsPanel(id="infer-panel")
            with Vertical(id="pane-log", classes="pane"):
                yield Static("", id="log-pane-title", classes="pane-title")
                yield RichLog(id="log-view", max_lines=600, markup=True)
            with ScrollableContainer(id="pane-charts", classes="pane"):
                yield Static("◈ Metrics History (last 60s)", classes="pane-title")
                yield MetricSparkline("CPU %",      id="spark-cpu")
                yield MetricSparkline("RAM %",      id="spark-ram")
                yield MetricSparkline("GPU Util %", id="spark-gpu")
                yield MetricSparkline("Tok/s", max_val=300.0, id="spark-tps")
        yield Footer()

    def on_mount(self) -> None:
        self._update_models_pane_title()
        self._update_log_pane_title()
        self._highlight_pane(0)
        self.set_interval(POLL_INTERVAL, self._poll_models)
        self.set_interval(POLL_INTERVAL, self._do_sys_poll)
        self.set_interval(GPU_INTERVAL,  self._do_gpu_poll)
        self._start_log_stream()

    def _highlight_pane(self, idx: int) -> None:
        focus_map = {0: "#model-table", 1: "#sys-panel", 2: "#log-view", 3: "#pane-charts"}
        for i, pid in enumerate(self._PANES):
            try:
                pane = self.query_one(f"#{pid}")
                pane.add_class("active-pane") if i == idx else pane.remove_class("active-pane")
            except Exception:
                pass
        try:
            self.query_one(focus_map.get(idx, "#model-table")).focus()
        except Exception:
            pass

    def action_next_pane(self) -> None:
        self._pane_i = (self._pane_i + 1) % len(self._PANES)
        self._highlight_pane(self._pane_i)

    def action_prev_pane(self) -> None:
        self._pane_i = (self._pane_i - 1) % len(self._PANES)
        self._highlight_pane(self._pane_i)

    def action_clear_log(self) -> None:
        self.query_one("#log-view", RichLog).clear()

    def action_refresh(self) -> None:
        self._poll_models()

    def action_log_source_model(self) -> None:
        self._set_log_source("model")

    def action_log_source_server(self) -> None:
        self._set_log_source("server")

    def action_log_source_runtime(self) -> None:
        self._set_log_source("runtime")

    def action_toggle_log_input(self) -> None:
        if self._log_opts.source != "model":
            return
        self._log_opts.filter_input = not self._log_opts.filter_input
        self._apply_log_opts_change(
            f"input filter: {'on' if self._log_opts.filter_input else 'off'}"
        )

    def action_toggle_log_output(self) -> None:
        if self._log_opts.source != "model":
            return
        self._log_opts.filter_output = not self._log_opts.filter_output
        self._apply_log_opts_change(
            f"output filter: {'on' if self._log_opts.filter_output else 'off'}"
        )

    def action_toggle_log_stats(self) -> None:
        if self._log_opts.source != "model":
            return
        self._log_opts.stats = not self._log_opts.stats
        self._apply_log_opts_change(
            f"stats: {'on' if self._log_opts.stats else 'off'}"
        )

    def _set_log_source(self, source: Literal["model", "server", "runtime"]) -> None:
        if self._log_opts.source == source:
            return
        self._log_opts.source = source
        if source != "model":
            self._log_opts.filter_input = False
            self._log_opts.filter_output = False
        self._apply_log_opts_change(f"log source: {source}")

    def _apply_log_opts_change(self, msg: str) -> None:
        self._update_log_pane_title()
        self.query_one("#log-view", RichLog).write(f"[bold cyan]── {msg} ──[/]")
        self._request_log_restart()

    def _update_models_pane_title(self) -> None:
        try:
            sort_label = "A→Z" if self._model_sort == "alpha" else "loaded first"
            self.query_one("#models-pane-title", Static).update(
                "◈ Models  "
                f"[dim][sort: {sort_label}][/]  "
                "(Space select  s sort  l load  u unload)"
            )
        except Exception:
            pass

    def _update_log_pane_title(self) -> None:
        try:
            title = self.query_one("#log-pane-title", Static)
            title.update(
                "◈ lms log stream  "
                f"[dim][{self._log_opts.summary()}][/]  "
                "(1/2/3 source  i/o filter  S stats  c clear)"
            )
        except Exception:
            pass

    def _cursor_model_name(self) -> Optional[str]:
        if self._cursor_mdl:
            return self._cursor_mdl
        try:
            tbl = self.query_one("#model-table", ModelTable)
            if tbl.row_count:
                return self._model_name_from_row(tbl.get_row_at(tbl.cursor_row))
        except Exception:
            pass
        return None

    def _log_notice(self, msg: str) -> None:
        self.query_one("#log-view", RichLog).write(msg)

    def toggle_model_sort(self) -> None:
        self._model_sort = "loaded" if self._model_sort == "alpha" else "alpha"
        self._update_models_pane_title()
        self._poll_models()

    def prompt_load_model(self) -> None:
        name = self._cursor_model_name()
        if not name:
            self._log_notice("[yellow]⚠ No model under cursor[/]")
            return
        self._load_target = name
        self.push_screen(LoadModelScreen(name), self._on_load_model_dismiss)

    def prompt_unload_model(self) -> None:
        name = self._cursor_model_name()
        if not name:
            self._log_notice("[yellow]⚠ No model under cursor[/]")
            return
        self._exec_lms(["lms", "unload", name], f"unload {name}")

    def _on_load_model_dismiss(self, extra: Optional[str]) -> None:
        name = self._load_target
        self._load_target = None
        if extra is None or not name:
            return
        try:
            args = shlex.split(extra)
        except ValueError as e:
            self._log_notice(f"[red]✗ Invalid args: {e}[/]")
            return
        cmd = ["lms", "load", name, "-y", *args]
        self._exec_lms(cmd, f"load {name}")

    @work(thread=True)
    def _exec_lms(self, cmd: list[str], label: str) -> None:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            out = (proc.stdout or proc.stderr or "").strip()
            lines = out.splitlines()[:25] if out else []
            def show() -> None:
                self._log_notice(f"[bold cyan]── {label} ──[/]")
                for line in lines:
                    self._log_notice(f"[dim]{rich_escape(line)}[/]")
                if proc.returncode == 0:
                    self._log_notice(f"[green]✓ {label}[/]")
                else:
                    self._log_notice(f"[red]✗ {label} (exit {proc.returncode})[/]")
                self._poll_models()
            self.call_from_thread(show)
        except FileNotFoundError:
            self.call_from_thread(
                self._log_notice, "[red]✗ 'lms' binary not found[/]"
            )
        except subprocess.TimeoutExpired:
            self.call_from_thread(
                self._log_notice, f"[red]✗ {label} timed out[/]"
            )
        except Exception as e:
            self.call_from_thread(
                self._log_notice, f"[red]✗ {label}: {rich_escape(str(e))}[/]"
            )

    def _sort_models(self, models: list) -> None:
        if self._model_sort == "alpha":
            models.sort(key=lambda m: str(m.get("id", m.get("path", "?"))).lower())
        else:
            models.sort(
                key=lambda m: (
                    0 if m.get("state") == "loaded" else 1,
                    str(m.get("id", m.get("path", "?"))).lower(),
                )
            )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Track cursor row independently of log subscription."""
        try:
            tbl  = self.query_one("#model-table", ModelTable)
            name = self._model_name_from_row(tbl.get_row(event.row_key))
            if name:
                self._cursor_mdl = name
        except Exception:
            pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._toggle_model(event.row_key)

    def _model_name_from_row(self, row) -> Optional[str]:
        if not row:
            return None
        name = str(row[0]).strip()
        if name.startswith("● "):
            name = name[2:]
        if name.startswith("[") or name == "–":
            return None
        return name or None

    def _toggle_model(self, row_key) -> None:
        try:
            tbl  = self.query_one("#model-table", ModelTable)
            row  = tbl.get_row(row_key)
            name = self._model_name_from_row(row)
            if not name:
                return
            if name == self._sel_mdl:
                self._deselect_model()
            else:
                self._select_model(name)
        except Exception:
            pass

    def _select_model(self, name: str) -> None:
        self._sel_mdl = name
        self.query_one("#log-view", RichLog).write(
            f"[bold cyan]── Filtering log to: {name} ──[/]"
        )
        self._refresh_model_selection_markers()

    def _deselect_model(self) -> None:
        self._sel_mdl = None
        self.query_one("#log-view", RichLog).write(
            "[bold cyan]── Showing all models in log ──[/]"
        )
        self._refresh_model_selection_markers()

    def _request_log_restart(self) -> None:
        """Stop the current stream; the log worker loop reconnects with new options."""
        proc = self._lms_proc
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    def _refresh_model_selection_markers(self) -> None:
        try:
            tbl = self.query_one("#model-table", ModelTable)
            for key in tbl.rows:
                row = tbl.get_row(key)
                if not row:
                    continue
                name = self._model_name_from_row(row)
                if not name:
                    continue
                marked = f"● {name}" if name == self._sel_mdl else name
                if str(row[0]) != marked:
                    tbl.update_cell(key, "model", marked)
        except Exception:
            pass

    def _restore_model_cursor(self, tbl: ModelTable) -> None:
        """Restore table cursor after rebuild; independent of log subscription."""
        if not self._cursor_mdl:
            return
        try:
            row = tbl.get_row_index(self._cursor_mdl)
            if tbl.cursor_row != row:
                tbl.move_cursor(row=row, scroll=True)
        except Exception:
            self._cursor_mdl = None

    @work(exclusive=True, thread=False, group="poll")
    async def _poll_models(self) -> None:
        tbl = self.query_one("#model-table", ModelTable)
        try:
            r = await self._client.get(f"{self._base_url}/api/v0/models")
            r.raise_for_status()
            data   = r.json()
            models = data if isinstance(data, list) else data.get("data", [])
            self._sort_models(models)
            saved_cursor = self._cursor_mdl
            tbl.clear()
            for m in models:
                mid   = m.get("id", m.get("path", "?"))
                state = m.get("state", "loaded")
                ctx   = str(m.get("max_context_length", m.get("context_length", "?")))
                gpu   = m.get("gpu_offload", {})
                gpu_s = f"{gpu.get('ratio',0)*100:.0f}%" if isinstance(gpu, dict) else "?"
                vram  = m.get("loaded_model_info", {})
                vram_s = _fmt_bytes(vram.get("gpu_memory_bytes", 0)) if isinstance(vram, dict) else "?"
                st    = Text(state, style="bold green" if state == "loaded" else "yellow")
                label = f"● {mid}" if mid == self._sel_mdl else mid
                tbl.add_row(
                    label, st, ctx, gpu_s, vram_s,
                    m.get("arch", "?"), m.get("quantization", "?"),
                    key=mid,
                )
            self._cursor_mdl = saved_cursor
            self._restore_model_cursor(tbl)
        except httpx.ConnectError:
            tbl.clear()
            tbl.add_row("[red]LM Studio not reachable[/]", "–", "–", "–", "–", "–", "–")
        except Exception as e:
            tbl.clear()
            tbl.add_row(f"[red]{e}[/]", "–", "–", "–", "–", "–", "–")

    @work(exclusive=False, thread=True)
    def _do_sys_poll(self) -> None:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        self.call_from_thread(self._apply_sys, cpu, mem.percent, float(mem.used), float(mem.total))

    def _apply_sys(self, cpu, ram_pct, used, total) -> None:
        p = self.query_one("#sys-panel", SysPanel)
        p.cpu_pct = cpu; p.ram_pct = ram_pct; p.ram_used = used; p.ram_total = total
        self.query_one("#spark-cpu", MetricSparkline).push(cpu)
        self.query_one("#spark-ram", MetricSparkline).push(ram_pct)

    @work(exclusive=False, thread=True)
    def _do_gpu_poll(self) -> None:
        self.call_from_thread(self._apply_gpu, _try_nvidia_smi())

    def _apply_gpu(self, gpu) -> None:
        p = self.query_one("#sys-panel", SysPanel)
        if gpu:
            p.has_gpu = True
            p.gpu_util      = gpu["util"]
            p.gpu_mem_pct   = gpu["mem_used"] / max(gpu["mem_total"], 1) * 100
            p.gpu_mem_used  = gpu["mem_used"]
            p.gpu_mem_total = gpu["mem_total"]
            p.gpu_temp      = gpu["temp"]
            p.gpu_pwr       = gpu["power"]
            self.query_one("#spark-gpu", MetricSparkline).push(gpu["util"])

    async def _reap_subprocess(self, proc: asyncio.subprocess.Process) -> None:
        """Terminate the lms log child and wait for it to exit."""
        if proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                return
            await proc.wait()

    async def _stop_log_stream(self) -> None:
        proc = self._lms_proc
        self._lms_proc = None
        if proc is not None:
            await self._reap_subprocess(proc)

    @work(exclusive=False, thread=False, group="log")
    async def _start_log_stream(self) -> None:
        log   = self.query_one("#log-view", RichLog)
        infer = self.query_one("#infer-panel", InferenceStatsPanel)
        while not self._log_shutdown:
            cmd   = self._log_opts.build_cmd()
            proc: asyncio.subprocess.Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                self._lms_proc = proc
                log.write(
                    f"[bold green]✓ lms log stream connected[/] "
                    f"[dim]({self._log_opts.summary()})[/]"
                )
                async for raw in proc.stdout:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if line.startswith("Streaming logs"):
                        continue
                    if line.startswith("Error:"):
                        log.write(f"[bold red]{rich_escape(line)}[/]")
                        continue
                    try:
                        self._dispatch_log(json.loads(line), log, infer)
                    except json.JSONDecodeError:
                        log.write(f"[dim]{rich_escape(line[:200])}[/]")
            except asyncio.CancelledError:
                raise
            except FileNotFoundError:
                log.write(
                    "[bold red]✗ 'lms' binary not found.[/]\n"
                    "[dim]  Add ~/.lmstudio/bin to PATH and relaunch lms-mon.[/]"
                )
                return
            except Exception as e:
                log.write(f"[red]✗ log stream error: {e}[/]")
            finally:
                if proc is not None:
                    if self._lms_proc is proc:
                        self._lms_proc = None
                    await self._reap_subprocess(proc)

            if self._log_shutdown:
                break

    def _dispatch_log(self, obj: dict, log: RichLog, infer: InferenceStatsPanel) -> None:
        source = self._log_opts.source
        if source == "model":
            self._dispatch_model_log(obj, log, infer)
        elif source == "server":
            self._dispatch_server_log(obj, log)
        else:
            self._dispatch_runtime_log(obj, log)

    def _dispatch_model_log(
        self, obj: dict, log: RichLog, infer: InferenceStatsPanel,
    ) -> None:
        if not _log_event_matches_selection(self._sel_mdl, obj):
            return
        data  = _log_event_data(obj)
        kind  = data.get("type", "")
        model = (
            data.get("modelIdentifier")
            or data.get("modelPath")
            or data.get("model", "")
        )
        ts    = _log_timestamp(obj)
        stats = data.get("stats") or {}
        mc    = "cyan"
        if kind in ("llm.prediction.input", "input"):
            snippet = rich_escape(
                str(data.get("input", data.get("content", "")))[:180].replace("\n", " ")
            )
            log.write(
                f"[dim]{ts}[/] [bold {mc}]→ {rich_escape(model)}[/] [dim]{snippet}[/]"
            )
        elif kind in ("llm.prediction.output", "output"):
            snippet = rich_escape(
                str(data.get("output", data.get("content", "")))[:180].replace("\n", " ")
            )
            tps = _stats_float(
                stats, "tokensPerSecond", "tokens_per_second", default=0.0,
            )
            ttft = _stats_float(
                stats,
                "timeToFirstTokenSec",
                "time_to_first_token_seconds",
                "time_to_first_token",
                default=0.0,
            )
            ntok = _stats_int(
                stats,
                "predictedTokensCount",
                "total_output_tokens",
                "num_tokens",
                default=0,
            )
            gent = _stats_float(stats, "totalTimeSec", "generation_time", default=0.0)
            log.write(
                f"[dim]{ts}[/] [bold {mc}]← {rich_escape(model)}[/]  "
                f"[green]{tps:.1f}tok/s[/]  [yellow]TTFT:{ttft*1000:.0f}ms[/]  "
                f"[dim]{ntok}tok/{gent:.1f}s[/]  {snippet}"
            )
            if tps > 0:
                infer.record(tps, ttft)
                self.query_one("#spark-tps", MetricSparkline).push(tps)
        else:
            log.write(
                f"[dim]{ts}[/] [dim]{rich_escape(kind)}[/] "
                f"{rich_escape(json.dumps(data)[:220])}"
            )

    def _dispatch_server_log(self, obj: dict, log: RichLog) -> None:
        data    = _log_event_data(obj)
        kind    = data.get("type", "")
        ts      = _log_timestamp(obj)
        level   = str(data.get("level", "info")).lower()
        style   = {
            "debug": "dim",
            "info": "white",
            "warn": "yellow",
            "warning": "yellow",
            "error": "bold red",
        }.get(level, "white")
        if kind == "server.log":
            content = rich_escape(str(data.get("content", "")).replace("\n", " ")[:300])
            log.write(f"[dim]{ts}[/] [{style}]{level}[/] {content}")
        else:
            log.write(
                f"[dim]{ts}[/] [dim]{rich_escape(kind)}[/] "
                f"{rich_escape(json.dumps(data)[:240])}"
            )

    def _dispatch_runtime_log(self, obj: dict, log: RichLog) -> None:
        data = _log_event_data(obj)
        kind = data.get("type", "runtime")
        ts   = _log_timestamp(obj)
        log.write(
            f"[dim]{ts}[/] [magenta]{rich_escape(kind)}[/] "
            f"{rich_escape(json.dumps(data)[:240])}"
        )

    async def on_unmount(self) -> None:
        self._log_shutdown = True
        await self._stop_log_stream()
        await self._client.aclose()


# ─── entry point ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(prog="lms-mon",
                                description="htop-style TUI monitor for LM Studio")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", default=1234, type=int)
    p.add_argument(
        "--log-source", choices=LOG_SOURCES, default="model",
        help="lms log stream --source (default: model)",
    )
    p.add_argument(
        "--log-filter",
        help="lms log stream --filter for model source: comma-separated input, output",
    )
    p.add_argument(
        "--no-stats", action="store_true",
        help="omit --stats from lms log stream",
    )
    args = p.parse_args()
    filt_in, filt_out = _parse_log_filter(args.log_filter)
    log_opts = LogStreamOptions(
        source=args.log_source,
        filter_input=filt_in,
        filter_output=filt_out,
        stats=not args.no_stats,
    )
    LMSMon(host=args.host, port=args.port, log_opts=log_opts).run()


if __name__ == "__main__":
    main()

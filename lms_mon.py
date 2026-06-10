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
from collections import deque
from datetime import datetime
from typing import Optional

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
from textual.widgets import (
    Header, Footer, DataTable, Static, Log, Sparkline, Label,
)
from textual.reactive import reactive
from textual import work
from rich.text import Text
from rich.table import Table


# ─── constants ─────────────────────────────────────────────
HISTORY_LEN   = 60
POLL_INTERVAL = 2.0
GPU_INTERVAL  = 1.5


# ─── helpers ──────────────────────────────────────────────

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
        return {
            "util":      float(parts[0]),
            "mem_used":  float(parts[1]),
            "mem_total": float(parts[2]),
            "temp":      float(parts[3]),
            "power":     float(parts[4]) if parts[4] not in ("N/A", "[N/A]") else 0.0,
        }
    except Exception:
        return None


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _bar(value: float, max_value: float, width: int = 12, fg: str = "green") -> Text:
    filled = int(round(value / max(max_value, 1) * width))
    bar = "█" * filled + "░" * (width - filled)
    t = Text()
    t.append(bar, style=fg)
    t.append(f" {value:5.1f}%", style="bold white")
    return t


# ─── Widgets ──────────────────────────────────────────────

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
    SUB_TITLE = "LM Studio Monitor  •  Tab/S-Tab navigate  •  ↑↓ select model  •  q quit"

    CSS = """
    Screen { background: $background; }
    #main {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-rows: 1fr 1fr;
        grid-gutter: 1 2;
        height: 1fr;
        padding: 0 1;
    }
    .pane { border: round $panel; padding: 0 1; height: 100%; }
    .pane.active-pane { border: round $accent; }
    .pane-title { text-style: bold; color: $accent; height: 1; padding: 0 1; }
    #pane-charts { overflow-y: auto; }
    DataTable { height: 1fr; }
    Log { height: 1fr; }
    """

    BINDINGS = [
        Binding("tab",       "next_pane", "Next pane"),
        Binding("shift+tab", "prev_pane", "Prev pane"),
        Binding("r",         "refresh",   "Refresh"),
        Binding("c",         "clear_log", "Clear log"),
        Binding("q",         "quit",      "Quit"),
    ]

    _PANES  = ["pane-models", "pane-sys", "pane-log", "pane-charts"]
    _pane_i = reactive(0)
    _sel_mdl: Optional[str] = None

    def __init__(self, host="localhost", port=1234, **kw):
        super().__init__(**kw)
        self._host     = host
        self._port     = port
        self._base_url = f"http://{host}:{port}"
        self._client   = httpx.AsyncClient(timeout=5.0)
        self._lms_proc = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with Vertical(id="pane-models", classes="pane"):
                yield Static("◈ Loaded Models  (↑↓ to select)", classes="pane-title")
                tbl = DataTable(id="model-table", cursor_type="row", zebra_stripes=True)
                tbl.add_columns("Model", "State", "Ctx", "GPU%", "VRAM", "Arch", "Quant")
                yield tbl
            with Vertical(id="pane-sys", classes="pane"):
                yield Static("◈ System Resources", classes="pane-title")
                yield SysPanel(id="sys-panel")
                yield InferenceStatsPanel(id="infer-panel")
            with Vertical(id="pane-log", classes="pane"):
                yield Static("◈ lms log stream  (c = clear)", classes="pane-title")
                yield Log(id="log-view", max_lines=600, markup=True)
            with ScrollableContainer(id="pane-charts", classes="pane"):
                yield Static("◈ Metrics History (last 60s)", classes="pane-title")
                yield MetricSparkline("CPU %",      id="spark-cpu")
                yield MetricSparkline("RAM %",      id="spark-ram")
                yield MetricSparkline("GPU Util %", id="spark-gpu")
                yield MetricSparkline("Tok/s", max_val=300.0, id="spark-tps")
        yield Footer()

    def on_mount(self) -> None:
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
        self.query_one("#log-view", Log).clear()

    def action_refresh(self) -> None:
        self.run_worker(self._poll_models())

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        try:
            tbl  = self.query_one("#model-table", DataTable)
            row  = tbl.get_row(event.row_key)
            name = str(row[0]) if row else None
            if name and name != self._sel_mdl:
                self._sel_mdl = name
                self.query_one("#log-view", Log).write_line(
                    f"[bold cyan]── Selected model: {name} ──[/]"
                )
        except Exception:
            pass

    @work(exclusive=True, thread=False)
    async def _poll_models(self) -> None:
        tbl = self.query_one("#model-table", DataTable)
        try:
            r = await self._client.get(f"{self._base_url}/api/v0/models")
            r.raise_for_status()
            data   = r.json()
            models = data if isinstance(data, list) else data.get("data", [])
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
                tbl.add_row(mid, st, ctx, gpu_s, vram_s,
                            m.get("arch", "?"), m.get("quantization", "?"))
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

    @work(exclusive=True, thread=False)
    async def _start_log_stream(self) -> None:
        log   = self.query_one("#log-view", Log)
        infer = self.query_one("#infer-panel", InferenceStatsPanel)
        cmd   = ["lms", "log", "stream", "--source", "model",
                 "--filter", "input,output", "--json", "--stats"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self._lms_proc = proc
            log.write_line("[bold green]✓ lms log stream connected[/]")
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    self._dispatch_log(json.loads(line), log, infer)
                except json.JSONDecodeError:
                    log.write_line(f"[dim]{line[:200]}[/]")
            log.write_line("[yellow]⚠ lms log stream closed[/]")
        except FileNotFoundError:
            log.write_line(
                "[bold red]✗ 'lms' binary not found.[/]\n"
                "[dim]  Add ~/.lmstudio/bin to PATH and relaunch lms-mon.[/]"
            )
        except Exception as e:
            log.write_line(f"[red]✗ log stream error: {e}[/]")

    def _dispatch_log(self, obj: dict, log: Log, infer: InferenceStatsPanel) -> None:
        ts    = datetime.now().strftime("%H:%M:%S")
        kind  = obj.get("type", "")
        model = obj.get("model", "")
        stats = obj.get("stats", {})
        hi    = self._sel_mdl and self._sel_mdl in model
        mc    = "cyan" if hi else "dim white"
        if kind == "input":
            snippet = obj.get("content", "")[:180].replace("\n", " ")
            log.write_line(f"[dim]{ts}[/] [bold {mc}]→ {model}[/] [dim]{snippet}[/]")
        elif kind == "output":
            snippet = obj.get("content", "")[:180].replace("\n", " ")
            tps  = stats.get("tokens_per_second", 0.0)
            ttft = stats.get("time_to_first_token", 0.0)
            ntok = stats.get("num_tokens", 0)
            gent = stats.get("generation_time", 0.0)
            log.write_line(
                f"[dim]{ts}[/] [bold {mc}]← {model}[/]  "
                f"[green]{tps:.1f}tok/s[/]  [yellow]TTFT:{ttft*1000:.0f}ms[/]  "
                f"[dim]{ntok}tok/{gent:.1f}s[/]  {snippet}"
            )
            if tps > 0:
                infer.record(tps, ttft)
                self.query_one("#spark-tps", MetricSparkline).push(tps)
        else:
            log.write_line(f"[dim]{ts} {json.dumps(obj)[:240]}[/]")

    async def on_unmount(self) -> None:
        if self._lms_proc:
            try:
                self._lms_proc.terminate()
            except Exception:
                pass
        await self._client.aclose()


# ─── entry point ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(prog="lms-mon",
                                description="htop-style TUI monitor for LM Studio")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", default=1234, type=int)
    args = p.parse_args()
    LMSMon(host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()

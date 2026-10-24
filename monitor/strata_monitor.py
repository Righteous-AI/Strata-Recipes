#!/usr/bin/env python3
"""Read-only terminal monitor for a running Strata server.

Draws the overview from Niko's `strata monitor` TUI: a LIVE header, then
SERVING | PERFORMANCE, BATCHING / PARALLEL | HARDWARE, REQUESTS | ACTIVITY.
Speeds, hit rate, and the PCIe fraction come from the engine log the service
writes (the `strata serve:` lines), not from a second calculation. Quitting
this process does not stop the server.
"""
from __future__ import annotations

import glob
import json
import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import termios
import time
import tty
from datetime import datetime
import urllib.error
import urllib.request

URL = "http://127.0.0.1:12345"
INTERVAL = 1.0
# The running 0.2.0 FP4-Local server. /metrics and /slots need the LAN token.
LOG_PATH = os.environ.get("STRATA_LOG", "/opt/strata/Strata/strata-fp4-local.log")
JOURNAL_UNIT = "strata-engine.service"
FRAME_JSON = "/tmp/strata-monitor.json"
FRAME_TXT = "/tmp/strata-monitor.frame"

# Plum background and mint rules, matched to the published screenshot.
BG = (30, 12, 24)
FG = {
    "bg": BG,
    "border": (57, 224, 166),
    "title": (72, 232, 176),
    "ready": (110, 245, 196),
    "text": (236, 230, 234),
    "dim": (164, 146, 158),
    "bar": (57, 224, 166),
    "bar_empty": (46, 96, 82),
    "bad": (255, 109, 132),
}
BOLD = {"title", "ready"}
# 256-color fallback. The plum has no cube match, so the background stays black
# and the rules stay mint (43 / 49).
FG256 = {
    "bg": 0,
    "border": 43,
    "title": 49,
    "ready": 49,
    "text": 255,
    "dim": 245,
    "bar": 49,
    "bar_empty": 23,
    "bad": 197,
}
BG256 = 0


def http_get(path: str):
    headers = {}
    key = os.environ.get("STRATA_API_KEY", "local")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(URL + path, headers=headers)
    with urllib.request.urlopen(req, timeout=2) as resp:
        return json.loads(resp.read().decode())


def toks(value, nd=1):
    if value is None:
        return "—"
    return f"{value:,.{nd}f}"


def comma_exact(num_text: str) -> str:
    """Keep the fractional digits the log wrote. Only add thousands separators."""
    if not num_text or num_text == "—":
        return "—"
    if "." in num_text:
        whole, frac = num_text.split(".", 1)
        return f"{int(whole):,}.{frac}"
    return f"{int(num_text):,}"


SERVE_RE = re.compile(
    r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ "
    r"(?P<read_n>\d+)(?: of \d+)? read in (?P<read_ms>[\d.]+) ms "
    r"\((?P<pp>[\d.]+) tok/s\), "
    r"(?P<gen>\d+) generated in (?P<gen_ms>[\d.]+) ms \((?P<tg>[\d.]+) tok/s\)"
    r"(?:, drafts accepted (?P<acc>\d+) of (?P<off>\d+))?"
)
HIT_RE = re.compile(r"decode expert cache hit rate: (?P<hit>[\d.]+)%")
PCIE_RE = re.compile(r"pcie_frac (?P<frac>[\d.]+)")
READ_RE = re.compile(r"reading the prompt: (?P<done>[\d,]+) of (?P<total>[\d,]+) tokens")
# Thinking and tool-call lines use the same rate, not only "answering:".
PROG_RE = re.compile(
    r"\[strata\] (?P<phase>.+?): (?P<done>\d+) of "
    r"(?:budget \d+ \(max (?P<budget_max>\d+)\)|max (?P<mx>\d+)) tokens, "
    r"(?P<rate>[\d.]+) tok/s"
)
STAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\+\d{2}:\d{2}|Z)?)")


def parse_engine_log(text: str):
    """Requests and pcie_frac from the current engine session only."""
    cut = text.rfind("session is up")
    if cut >= 0:
        text = text[cut:]
    reqs = []
    pcie = None
    pending = None
    for line in text.splitlines():
        match = SERVE_RE.search(line)
        if match:
            pending = match.groupdict()
            pending["hit"] = None
            pending["when"] = None
            reqs.append(pending)
            continue
        if pending is not None and pending["hit"] is None:
            hit = HIT_RE.search(line)
            if hit:
                pending["hit"] = hit.group("hit")
        found = PCIE_RE.search(line)
        if found:
            pcie = found.group("frac")
    for index, req in enumerate(reqs):
        req["rid"] = f"r{index:02d}"
    return reqs, pcie


def _stamp(line: str):
    match = STAMP_RE.search(line)
    if not match:
        return None
    try:
        return datetime.fromisoformat(match.group(1)).timestamp()
    except ValueError:
        return None


def parse_journal(text: str, reqs: list):
    """Pair service-journal timestamps with engine lines, and keep a live read/answer."""
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if "almost ready" in line or "filling the GPU" in line:
            start = index
    live = None
    dones = []
    for line in lines[start:]:
        if "[strata] done:" in line:
            dones.append(_stamp(line))
            live = None
            continue
        reading = READ_RE.search(line)
        if reading and "[strata]" in line:
            live = {
                "kind": "reading",
                "done": int(reading.group("done").replace(",", "")),
                "total": int(reading.group("total").replace(",", "")),
            }
            continue
        answering = PROG_RE.search(line)
        if answering:
            live = {
                "kind": "generating",
                "phase": answering.group("phase"),
                "done": int(answering.group("done")),
                "max": int(answering.group("budget_max") or answering.group("mx")),
                "rate": answering.group("rate"),
            }
    if len(dones) >= len(reqs):
        dones = dones[-len(reqs):]
    for req, when in zip(reqs, dones):
        req["when"] = when
    return live


def age_text(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 10:
        return f"-{seconds:.1f}s"
    if seconds < 60:
        return f"-{seconds:.0f}s"
    if seconds < 3600:
        return f"-{seconds / 60:.1f}m"
    return f"-{seconds / 3600:.1f}h"


def span_width(spans) -> int:
    return sum(len(text) for text, _style in spans)


def fit(spans, width: int):
    out = []
    used = 0
    for text, style in spans:
        if used >= width:
            break
        if used + len(text) > width:
            text = text[: width - used]
        if text:
            out.append((text, style))
            used += len(text)
    if used < width:
        out.append((" " * (width - used), "bg"))
    return out


def bar(frac: float, cells: int):
    frac = 0.0 if frac is None else max(0.0, min(1.0, float(frac)))
    cells = max(1, int(cells))
    filled = int(round(frac * cells))
    return [("█" * filled, "bar"), ("░" * (cells - filled), "bar_empty")]


def inner_of(width: int) -> int:
    return max(4, width - 2)


def grow_bar(frac: float, prefix, width: int):
    """A block bar that uses whatever of the row the prefix left free."""
    cells = max(1, inner_of(width) - span_width(prefix))
    return list(prefix) + bar(frac, cells)


def distribute(groups, width: int):
    """Lay groups across the row so the box contents follow its width."""
    groups = [list(group) for group in groups if group]
    if not groups:
        return []
    if len(groups) == 1:
        return groups[0]
    inner = inner_of(width)
    used = sum(span_width(group) for group in groups)
    gaps = len(groups) - 1
    spare = inner - used
    if spare < gaps:
        spare = gaps
    base, extra = divmod(spare, gaps)
    out = []
    for index, group in enumerate(groups):
        out.extend(group)
        if index < gaps:
            out.append((" " * (base + (1 if index < extra else 0)), "text"))
    return out


def label(name: str, width: int = 8):
    text = " " + name
    if len(text) < width:
        text += " " * (width - len(text))
    return [(text[:width], "text")]


def border_top(width: int, title: str):
    inner = width - 2
    shown = f" {title} "
    if len(shown) > inner:
        shown = shown[:inner]
    left = (inner - len(shown)) // 2
    right = inner - len(shown) - left
    return [("┌" + "─" * left, "border"), (shown, "title"), ("─" * right + "┐", "border")]


def border_bot(width: int):
    return [("└" + "─" * (width - 2) + "┘", "border")]


def content(spans, width: int):
    return [("│", "border"), *fit(spans, width - 2), ("│", "border")]


def panel(title: str, rows: list, width: int, n_content: int):
    lines = [border_top(width, title)]
    for index in range(n_content):
        lines.append(content(rows[index] if index < len(rows) else [], width))
    lines.append(border_bot(width))
    return lines


def hjoin(left, right, gap: int = 2):
    gutter = [(" " * gap, "bg")]
    return [a + gutter + b for a, b in zip(left, right)]


def blank(cols: int):
    return [(" " * cols, "bg")]


def display_model(name: str) -> str:
    if not name:
        return "—"
    names = {
        "qwen3.8-flash-next-nvfp4-nvidia": "Qwen3.8-Flash-Next-NVFP4",
        "qwen3.8-flash-next-fp4-local": "Qwen3.8-Flash-Next-FP4-Local",
    }
    return names.get(name, name)


def short_gpu(name: str) -> str:
    if not name:
        return "GPU"
    for token in ("NVIDIA GeForce ", "NVIDIA ", "AMD "):
        name = name.replace(token, "")
    return name


def gib(nbytes) -> str:
    if not nbytes:
        return "—"
    return f"{nbytes / (1024 ** 3):.1f}"


class Watch:
    """Poll snapshots plus a short activity log."""

    def __init__(self):
        self.health = {}
        self.metrics = {}
        self.slots = []
        self.error = ""
        self.fetched_at = time.time()
        self.ok = False
        self.events: list[tuple[float | None, str]] = []
        self.book = {"reqs": [], "pcie": None, "live": None}
        self.journal_text = ""
        self.journal_at = 0.0
        # Highest speeds seen this run. Each poll can raise them; they do not fall.
        self.max_pp_n = -1.0
        self.max_tg_n = -1.0
        self.max_pp_txt = None
        self.max_tg_txt = None

    def poll(self):
        try:
            health = http_get("/health")
            metrics = http_get("/metrics")
            slots = http_get("/slots")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.ok = False
        else:
            self.health = health
            self.metrics = metrics
            self.slots = slots if isinstance(slots, list) else []
            self.ok = bool(health.get("loaded"))
            if self.ok:
                self.error = ""
        self._read_logs()
        self._raise_speeds()
        self.fetched_at = time.time()
        self._note()

    def _consider_speed(self, which: str, value):
        """Keep a higher tok/s. `value` is a log string or a live number."""
        if value is None or value == "":
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if number <= 0:
            return
        text = comma_exact(value if isinstance(value, str) else f"{number:.1f}")
        if which == "pp" and number > self.max_pp_n:
            self.max_pp_n = number
            self.max_pp_txt = text
        elif which == "tg" and number > self.max_tg_n:
            self.max_tg_n = number
            self.max_tg_txt = text

    def _raise_speeds(self):
        """Finished engine-log lines, plus the request that is running right now."""
        for req in self.book.get("reqs") or []:
            if int(req.get("read_n") or 0) > 0:
                self._consider_speed("pp", req.get("pp"))
            if int(req.get("gen") or 0) > 0:
                self._consider_speed("tg", req.get("tg"))
        live = self.metrics.get("live") or {}
        self._consider_speed("pp", live.get("prefill_tok_s_mean"))
        if live.get("state") == "generating":
            self._consider_speed("tg", live.get("tok_s"))
        logged = self.book.get("live") or {}
        if logged.get("kind") == "generating":
            self._consider_speed("tg", logged.get("rate"))

    def _read_logs(self):
        reqs = []
        pcie = None
        try:
            with open(LOG_PATH, encoding="utf-8", errors="replace") as handle:
                reqs, pcie = parse_engine_log(handle.read())
        except OSError as exc:
            self.error = f"log: {exc}"
        now = time.time()
        if now - self.journal_at >= 1.0:
            try:
                out = subprocess.run(
                    ["journalctl", "-u", JOURNAL_UNIT, "-n", "250",
                     "-o", "short-iso", "--no-pager"],
                    capture_output=True, text=True, timeout=2,
                )
                self.journal_text = out.stdout or ""
                self.journal_at = now
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.error = f"journal: {exc}"
        live = parse_journal(self.journal_text, reqs) if self.journal_text else None
        self.book = {"reqs": reqs, "pcie": pcie, "live": live}

    def _note(self):
        events = []
        for req in self.book["reqs"]:
            text = f"{req['rid']} pre {comma_exact(req['pp'])} dec {comma_exact(req['tg'])}"
            events.append((req.get("when"), text))
        live = self.book.get("live")
        if live and live["kind"] == "reading":
            events.append((time.time(), f"prompt read {live['done']}/{live['total']}"))
        elif live and live["kind"] == "generating":
            events.append((time.time(), f"{live.get('phase') or 'output'} {comma_exact(live['rate'])} tok/s"))
        self.events = events[-32:]


def serving_rows(watch: Watch, width: int = 48):
    eng = watch.metrics.get("engine") or {}
    live = watch.metrics.get("live") or {}
    state = live.get("state") or ("idle" if watch.ok else "offline")
    busy = state not in ("idle", "offline", None)
    queued = live.get("queued") or 0
    logged = (watch.book or {}).get("live") or {}
    if logged.get("kind") == "reading":
        done = logged["done"]
        total = logged["total"]
        frac = done / total if total else 0
        bar_row = grow_bar(frac, label("Read") + [(f"{done}/{total} ", "text")], width)
    elif state == "reading" and live.get("prompt_total"):
        done = live.get("prompt_read") or 0
        total = live.get("prompt_total") or 0
        frac = done / total if total else 0
        bar_row = grow_bar(frac, label("Read") + [(f"{done}/{total} ", "text")], width)
    else:
        held = 0
        if watch.slots:
            held = watch.slots[0].get("n_prompt_tokens") or 0
        ctx = eng.get("max_context") or watch.health.get("max_context") or 0
        frac = held / ctx if ctx else 0
        bar_row = grow_bar(frac, label("Ctx") + [(f"{held}/{ctx} ", "text")], width)
    word = "READY" if watch.ok and not busy else state.upper()
    style = "ready" if word == "READY" else ("bad" if not watch.ok else "title")
    flight = 1 if busy else 0
    return [
        label("Model") + [(display_model(eng.get("model") or watch.health.get("model")), "text")],
        label("State") + [(word, style), ("  |  ", "dim"), (f"{flight} in flight", "text")],
        label("Queue") + _queue_spans(live, queued),
        bar_row,
    ]


def _queue_spans(live: dict, queued) -> list:
    """Waiting prompts, as token counts. Empty when nothing is queued."""
    items = live.get("queue")
    if isinstance(items, list):
        if not items:
            return [("none", "dim")]
        parts = []
        for item in items[:3]:
            tokens = item.get("prompt_tokens")
            if isinstance(tokens, int):
                parts.append(f"{tokens:,} tok")
            else:
                parts.append("prompt")
        text = " · ".join(parts)
        extra = len(items) - len(parts)
        if extra > 0:
            text += f"  +{extra}"
        return [(text, "text")]
    try:
        count = int(queued or 0)
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        return [("none", "dim")]
    word = "prompt" if count == 1 else "prompts"
    return [(f"{count} {word}", "text")]


def _rate_text(value) -> str | None:
    """One-decimal tok/s, or None when there is no positive rate."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return comma_exact(value if isinstance(value, str) else f"{number:.1f}")


def _session_mean(reqs: list, num_key: str, ms_key: str) -> str:
    """Token-weighted mean of finished requests in this server session."""
    total_n = total_ms = 0.0
    for req in reqs:
        try:
            number, millis = float(req[num_key]), float(req[ms_key])
        except (KeyError, TypeError, ValueError):
            continue
        if number > 0 and millis > 0:
            total_n += number
            total_ms += millis
    if total_ms <= 0:
        return "—"
    return comma_exact(f"{1000.0 * total_n / total_ms:.1f}")


def _current_rate(active: bool, value, fallback, ok: bool) -> tuple[float, str]:
    if active:
        text = _rate_text(value) or _rate_text(fallback)
        if text:
            try:
                return float(str(value if value not in (None, "") else fallback)), text
            except (TypeError, ValueError):
                return float(fallback), text
    if ok:
        return 0.0, "0.0"
    return 0.0, "—"


def performance_rows(watch: Watch, width: int = 48):
    """Prompt and Decode are the session mean. In and Out are the live rates."""
    reqs = (watch.book or {}).get("reqs") or []
    logged = (watch.book or {}).get("live") or {}
    live_now = watch.metrics.get("live") or {}
    state = live_now.get("state")
    last = reqs[-1] if reqs else None
    prompt = _session_mean(reqs, "read_n", "read_ms")
    decode = _session_mean(reqs, "gen", "gen_ms")
    try:
        out_scale = float(last["tg"]) if last else 200.0
    except (KeyError, TypeError, ValueError):
        out_scale = 200.0
    try:
        in_scale = float(last["pp"]) if last else 8000.0
    except (KeyError, TypeError, ValueError):
        in_scale = 8000.0
    if not out_scale:
        out_scale = 200.0
    if not in_scale:
        in_scale = 8000.0
    # The newest finished serve line. The engine writes draft acceptance only then.
    mtp = "—"
    for req in reversed(reqs):
        if req.get("acc") is None or not req.get("off"):
            continue
        acc, off = int(req["acc"]), int(req["off"])
        if off <= 0:
            continue
        mtp = f"{acc:,}/{off:,}  {100.0 * acc / off:.1f}%"
        break
    # live.tok_s is the server's window while any tokens are coming out (thinking,
    # answering, or a tool call). The journal line is the fallback.
    out_val, out_txt = _current_rate(
        state == "generating",
        live_now.get("tok_s"),
        logged.get("rate") if logged.get("kind") == "generating" else None,
        watch.ok,
    )
    in_val, in_txt = _current_rate(
        state == "reading",
        live_now.get("prefill_tok_s_mean"),
        None,
        watch.ok,
    )

    def metric(name, text, suffix):
        return distribute([
            label(name) + [(f"{text} tok/s", "text")],
            [(suffix, "dim")],
        ], width)

    def live_bar(name: str, value: float, text: str, scale: float, cells: int):
        return [(f" {name} ", "text"), *bar(value / scale if scale else 0, cells), (f" {text}", "text")]

    in_bar = live_bar("In", in_val, in_txt, in_scale, 1)
    out_bar = live_bar("Out", out_val, out_txt, out_scale, 1)
    gap = 2
    fixed = span_width(in_bar) + span_width(out_bar) + gap - 2
    spare = max(2, (width - 2) - fixed)
    in_cells = max(1, spare // 2)
    out_cells = max(1, spare - in_cells)
    return [
        metric("Prompt", prompt, "mean"),
        metric("Decode", decode, "mean"),
        live_bar("In", in_val, in_txt, in_scale, in_cells)
        + [("  ", "text")]
        + live_bar("Out", out_val, out_txt, out_scale, out_cells),
        label("MTP") + [(mtp, "text")],
    ]


def batch_rows(watch: Watch, width: int = 48):
    eng = watch.metrics.get("engine") or {}
    live = watch.metrics.get("live") or {}
    state = live.get("state") or "idle"
    n_slots = max(1, len(watch.slots))
    busy = sum(1 for slot in watch.slots if slot.get("is_processing"))
    if not watch.slots and state not in ("idle", "offline"):
        busy = 1
    reading = 1 if state == "reading" else 0
    decoding = 1 if state in ("generating", "decoding") else 0
    idle = 1 if state == "idle" and watch.ok else 0
    active = 1 if busy or reading or decoding else 0
    return [
        distribute([
            [(" ", "text"), (f"Slots  {busy}/{n_slots}", "text")],
            [(f"groups {active}/1", "text")],
        ], width),
        distribute([
            [(" ", "text"), (f"Batch rows {busy}", "text")],
            [(f"active groups {active}", "text")],
        ], width),
        distribute([
            [(" ", "text"), ("Pre-admit 0", "text")],
            [("oldest —", "dim")],
        ], width),
        [(" ", "text"), (f"Phases  read {reading}  decode {decoding}  idle {idle}", "text")],
        [(" ", "text"), (f"G0 {busy}/{n_slots}", "text")],
        distribute([
            [(" ", "text"), (f"workers {eng.get('pool_workers', '—')}", "text")],
            [(f"pcie {(watch.book or {}).get('pcie') or eng.get('pcie_frac') or '—'}", "text")],
        ], width),
    ]


_PKG_INPUT = None


def package_temp_c():
    """Live Package id 0 temperature from coretemp, in °C."""
    global _PKG_INPUT
    if _PKG_INPUT is None:
        found = ""
        for label_path in glob.glob("/sys/class/hwmon/hwmon*/temp*_label"):
            try:
                if open(label_path, encoding="utf-8").read().strip() == "Package id 0":
                    found = label_path[: -len("_label")] + "_input"
                    break
            except OSError:
                continue
        _PKG_INPUT = found
    if not _PKG_INPUT:
        return None
    try:
        return int(open(_PKG_INPUT, encoding="utf-8").read()) / 1000.0
    except (OSError, ValueError):
        return None


def hardware_rows(watch: Watch, width: int = 48):
    hw = watch.metrics.get("hardware") or {}
    st = watch.metrics.get("hardware_static") or {}
    name = short_gpu(st.get("gpu_name"))
    count = st.get("gpu_count") or 1
    used = gib(hw.get("gpu_mem_used"))
    total = gib(hw.get("gpu_mem_total"))
    util = hw.get("gpu_util")
    util_n = 0 if util is None else int(round(util))
    mem = f"{used}/{total}G" if used != "—" else "—"
    gpu_line = grow_bar(
        (util or 0) / 100.0,
        [(" ", "text"), (f"GPU 0  {name}  {mem}  {util_n}% ", "text")],
        width,
    )
    temp = hw.get("gpu_temp")
    power = hw.get("gpu_power")
    limit = hw.get("gpu_power_limit")
    pkg = package_temp_c()
    cpu = [(" ", "text"), (f"CPU {toks(hw.get('cpu'))}%", "text")]
    if pkg is not None:
        cpu = distribute([cpu, [(f"{pkg:.0f}°C", "text")]], width)
    return [
        [(" ", "text"), (f"{count} × {name}", "text")],
        gpu_line,
        distribute([
            [(" ", "text"), (f"{toks(temp, 0)} C", "text")],
            [(f"{toks(power)} / {toks(limit, 0)} W", "text")],
        ], width),
        [(" ", "text"), (f"PCIe  gen {hw.get('gpu_pcie_gen')}/{hw.get('gpu_pcie_gen_max')}  x{hw.get('gpu_pcie_width')}", "text")],
        cpu,
        distribute([
            [(" ", "text"), ("RAM", "text")],
            [(f"{gib(hw.get('ram_used'))}/{gib(hw.get('ram_total'))}G", "text")],
        ], width),
    ]


def _req_line(req, wide: bool, width: int):
    if wide:
        hit = req.get("hit") or "—"
        return distribute([
            [(" ", "text"), (str(req["rid"]), "text")],
            [(f"read {req['read_n']} in {req['read_ms']} ms", "text")],
            [(f"pre {comma_exact(req['pp'])}", "text")],
            [(f"dec {comma_exact(req['tg'])}", "text")],
            [(f"hit {hit}%", "text")],
        ], width)
    return distribute([
        [(" ", "text"), (str(req["rid"]), "text")],
        [(f"pre {comma_exact(req['pp'])}", "text")],
        [(f"dec {comma_exact(req['tg'])}", "text")],
    ], width)


def request_rows(watch: Watch, limit: int, wide: bool = False, width: int = 48):
    reqs = (watch.book or {}).get("reqs") or []
    newest = list(reversed(reqs))
    rows = [_req_line(req, wide, width) for req in newest[:limit]]
    shown = {req["rid"] for req in newest[:limit]}
    if reqs and not wide:
        big = max(reqs, key=lambda req: int(req["read_n"]))
        if big["rid"] not in shown:
            rows.append(_req_line(big, False, width))
        elif len(reqs) > limit:
            rows.append([(" ", "dim"), (f"+{len(reqs) - limit} more; press 3 for all", "dim")])
    elif len(reqs) > limit:
        rows.append([(" ", "dim"), (f"+{len(reqs) - limit} more; press 3 for all", "dim")])
    if not rows:
        rows.append([(" ", "dim"), ("none in the engine log yet", "dim")])
    return rows


def activity_rows(watch: Watch, limit: int, width: int = 48):
    now = time.time()
    chosen = watch.events[-limit:]
    rows = []
    for stamp, text in chosen:
        age = "   log" if stamp is None else f"{age_text(now - stamp):>7}"
        rows.append([(" ", "dim"), (age, "dim"), ("  ", "text"), (text, "text")])
    if watch.error:
        rows.append([(" ", "bad"), (watch.error[: max(8, inner_of(width) - 1)], "bad")])
    if not rows:
        rows.append([(" ", "dim"), ("waiting for a request", "dim")])
    return rows


def header(width: int, watch: Watch):
    status = "LIVE" if watch.ok else "OFFLINE"
    style = "ready" if watch.ok else "bad"
    prefill, decode = watch.max_pp_txt, watch.max_tg_txt
    body = [
        (" ", "text"),
        ("STRATA", "title"),
        ("  /  ", "dim"),
        ("MONITOR", "title"),
        ("   ", "text"),
        (status, style),
        ("   ", "text"),
        ("max prefill ", "dim"),
        (prefill or "—", "text"),
        (" tok/s", "dim"),
        ("   ", "text"),
        ("max decode ", "dim"),
        (decode or "—", "text"),
        (" tok/s", "dim"),
    ]
    inner = width - 2
    used = span_width(body)
    pad = inner - used - 2
    if pad < 1:
        pad = 1
    body = body + [(" " * max(1, pad), "bg"), ("█", "title"), (" ", "bg")]
    return [
        [("┌" + "─" * (width - 2) + "┐", "border")],
        [("│", "border"), *fit(body, inner), ("│", "border")],
        border_bot(width),
    ]


def footer(cols: int, view: int, age: float):
    keys = ((1, "Over"), (2, "Serve"), (3, "Reqs"), (4, "Engine"), (5, "GPU"), (6, "Logs"))
    spans = []
    for number, name in keys:
        style = "title" if number == view else "dim"
        if spans:
            spans.append(("  ", "dim"))
        spans.append((str(number), style))
        spans.append((" ", "dim"))
        spans.append((name, style))
    age_s = f"age {max(0, int(age))}s"
    used = span_width(spans)
    gap = cols - used - len(age_s) - 1
    if gap < 2:
        gap = 2
    spans = [(" ", "bg"), *spans, (" " * (gap - 1), "bg"), (age_s, "dim")]
    second = [(" Read only  |  q detaches without stopping Strata", "dim")]
    return [fit(spans, cols), fit(second, cols)]


def kv_rows(pairs, width: int = 80):
    rows = []
    inner = inner_of(width)
    key_w = 16 if inner < 70 else 22
    for key, value in pairs:
        val = str(value)
        key_txt = f"{key:<{key_w}}"
        if 1 + len(key_txt) + 1 + len(val) > inner:
            rows.append([(" ", "dim"), (key_txt, "dim"), (" ", "text"), (val, "text")])
            continue
        gap = inner - 1 - len(key_txt) - len(val)
        rows.append([(" ", "dim"), (key_txt, "dim"), (" " * gap, "text"), (val, "text")])
    return rows


def serve_view(watch: Watch, width: int = 80):
    live = watch.metrics.get("live") or {}
    eng = watch.metrics.get("engine") or {}
    held = watch.slots[0].get("n_prompt_tokens") if watch.slots else 0
    slot = "busy" if watch.slots and watch.slots[0].get("is_processing") else "idle"
    return kv_rows([
        ("state", live.get("state") or "idle"),
        ("phase", live.get("phase") or "—"),
        ("queue", live.get("queued") if live.get("queued") is not None else 0),
        ("prompt", f"{live.get('prompt_read') or 0} / {live.get('prompt_total') or 0}"),
        ("generated", f"{live.get('generated') or 0} / {live.get('max_tokens') or 0}"),
        ("context held", f"{held} / {eng.get('max_context') or '—'}"),
        ("slot 0", slot),
        ("model", eng.get("model") or watch.health.get("model") or "—"),
        ("url", URL),
    ], width)


def engine_view(watch: Watch, width: int = 80):
    eng = watch.metrics.get("engine") or {}
    return kv_rows([
        ("version", eng.get("version") or "—"),
        ("context", eng.get("context") or eng.get("max_context") or "—"),
        ("kv", f"{eng.get('kv')}  resident {eng.get('kv_resident')}"),
        ("expert slots", f"{eng.get('expert_slots')}  ({eng.get('expert_cache_mib')} MiB)"),
        ("verify window", eng.get("spec")),
        ("mtp cap", eng.get("mtp_max")),
        ("lookup", eng.get("lookup")),
        ("workers", eng.get("pool_workers")),
        ("pcie frac", (watch.book or {}).get("pcie") or eng.get("pcie_frac")),
        ("spec-min-p", eng.get("spec_min_p")),
        ("prefill cap", f"{eng.get('prefill_cap')}  stages {eng.get('prefill_stages')}"),
        ("vram free", f"{eng.get('vram_free_mib')} MiB"),
        ("arena", f"{eng.get('arena_mib')} MiB"),
    ], width)


def gpu_view(watch: Watch, width: int = 80):
    hw = watch.metrics.get("hardware") or {}
    st = watch.metrics.get("hardware_static") or {}
    eng = watch.metrics.get("engine") or {}
    return kv_rows([
        ("gpu", st.get("gpu_name") or "—"),
        ("util", f"{toks(hw.get('gpu_util'), 0)}%"),
        ("vram", f"{gib(hw.get('gpu_mem_used'))} / {gib(hw.get('gpu_mem_total'))} GiB"),
        ("cuda free", f"{eng.get('vram_free_mib')} MiB"),
        ("temp", f"{toks(hw.get('gpu_temp'), 0)} C"),
        ("power", f"{toks(hw.get('gpu_power'))} / {toks(hw.get('gpu_power_limit'), 0)} W"),
        ("pcie", f"gen {hw.get('gpu_pcie_gen')}/{hw.get('gpu_pcie_gen_max')} x{hw.get('gpu_pcie_width')}"),
        ("pcie rx/tx", f"{toks(hw.get('gpu_pcie_rx_mb'))} / {toks(hw.get('gpu_pcie_tx_mb'))} MB"),
        ("cpu", f"{st.get('cpu_name') or '—'}  {toks(hw.get('cpu'))}%"),
        ("ram", f"{gib(hw.get('ram_used'))} / {gib(hw.get('ram_total'))} GiB"),
        ("disk read", f"{toks(hw.get('disk_read_mb'))} MB/s"),
        ("expert slots", f"{eng.get('expert_slots')} ({eng.get('expert_cache_mib')} MiB)"),
    ], width)


def log_rows(width: int, n_lines: int):
    try:
        with open(LOG_PATH, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 65536))
            text = handle.read().decode("utf-8", "replace")
        lines = text.splitlines()[-n_lines:]
    except OSError as exc:
        lines = [str(exc)]
    inner = max(8, width - 4)
    return [[(" ", "dim"), (line[-inner:], "text")] for line in lines]


def overview(watch: Watch, box: int):
    top = hjoin(panel("SERVING", serving_rows(watch, box), box, 4), panel("PERFORMANCE", performance_rows(watch, box), box, 4))
    mid = hjoin(panel("BATCHING / PARALLEL", batch_rows(watch, box), box, 6), panel("HARDWARE", hardware_rows(watch, box), box, 6))
    bot = hjoin(panel("REQUESTS", request_rows(watch, 2, width=box), box, 3), panel("ACTIVITY", activity_rows(watch, 3, box), box, 3))
    return top, mid, bot


def full_panel(title: str, rows: list, width: int, n_content: int):
    return panel(title, rows, width, n_content)


def build(watch: Watch, cols: int, rows: int, view: int):
    cols = max(70, cols)
    margin = 1 if cols >= 104 else 0
    gap = 2
    box = (cols - margin * 2 - gap) // 2
    width = box * 2 + gap
    age = max(0.0, time.time() - watch.fetched_at)
    lines = []

    def place(block):
        pad = [(" " * margin, "bg")] if margin else []
        for line in block:
            lines.append(fit(pad + line, cols))

    place(header(width, watch))
    lines.append(blank(cols))
    if view == 1:
        top, mid, bot = overview(watch, box)
        for block in (top, mid, bot):
            place(block)
            lines.append(blank(cols))
    else:
        titles = {2: "SERVING", 3: "REQUESTS", 4: "ENGINE", 5: "HARDWARE", 6: "LOGS"}
        bodies = {
            2: serve_view(watch, width),
            3: request_rows(watch, 12, wide=True, width=width),
            4: engine_view(watch, width),
            5: gpu_view(watch, width),
            6: log_rows(width, 14),
        }
        n_content = min(14, max(6, rows - 9))
        place(full_panel(titles[view], bodies[view], width, n_content))
        lines.append(blank(cols))
    # Drop the extra blank so the footer sits under the last box with one gap.
    if lines and lines[-1] == blank(cols):
        lines.pop()
    lines.append(blank(cols))
    lines.extend(footer(cols, view, age))
    while len(lines) < rows:
        lines.append(blank(cols))
    return lines[:rows]


def plain(lines) -> str:
    return "\n".join("".join(text for text, _style in line) for line in lines)


def style_code(style: str, mode: str) -> str:
    if mode == "256":
        bold = "\033[1m" if style in BOLD else ""
        return f"\033[0m{bold}\033[48;5;{BG256}m\033[38;5;{FG256[style]}m"
    red, green, blue = FG[style]
    bold = "\033[1m" if style in BOLD else ""
    return (
        f"\033[0m{bold}"
        f"\033[48;2;{BG[0]};{BG[1]};{BG[2]}m"
        f"\033[38;2;{red};{green};{blue}m"
    )


def emit(lines, mode: str) -> str:
    """Draw each row in place. A newline on the last row scrolls the whole screen."""
    parts = ["\033[?2026h"]
    for row, line in enumerate(lines, start=1):
        parts.append(f"\033[{row};1H")
        current = None
        for text, style in line:
            if style != current:
                parts.append(style_code(style, mode))
                current = style
            parts.append(text)
        parts.append("\033[K")
    parts.append("\033[0m\033[?2026l")
    return "".join(parts)


def write_frame(lines):
    """Save the last frame. A failed save must not take the panel down."""
    payload = {
        "url": URL,
        "live": True,
        "lines": [[{"t": text, "s": style} for text, style in line] for line in lines],
    }
    directory = os.path.dirname(FRAME_JSON) or "/tmp"
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix="strata-monitor-", suffix=".json", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
        os.replace(tmp, FRAME_JSON)
        tmp = None
        with open(FRAME_TXT, "w", encoding="utf-8") as handle:
            handle.write(plain(lines) + "\n")
    except OSError:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def terminal_size():
    try:
        size = os.get_terminal_size()
        return size.columns, size.lines
    except OSError:
        return 110, 36


def color_mode() -> str:
    if os.environ.get("STRATA_MONITOR_COLOR") == "256":
        return "256"
    if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
        return "tc"
    if sys.stdout.isatty() and not os.environ.get("TMUX"):
        return "tc"
    return "256"


def main() -> int:
    global URL
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        URL = sys.argv[1].rstrip("/")
    once = "--once" in sys.argv
    watch = Watch()
    watch.poll()
    view = 1
    is_tty = sys.stdin.isatty() and sys.stdout.isatty() and not once
    mode = color_mode()
    old = None
    resized = {"flag": True}
    last_drawn = None

    def on_resize(_signum, _frame):
        resized["flag"] = True

    if is_tty:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        signal.signal(signal.SIGWINCH, on_resize)
        sys.stdout.write("\033[?1049h\033[?25l")
        sys.stdout.flush()
    try:
        while True:
            if is_tty:
                ready, _, _ = select.select([sys.stdin], [], [], INTERVAL)
                if ready:
                    key = sys.stdin.read(1)
                    if key in ("q", "Q"):
                        break
                    if key in "123456":
                        view = int(key)
                    elif key == "\x1b":
                        while select.select([sys.stdin], [], [], 0)[0]:
                            sys.stdin.read(1)
                if time.time() - watch.fetched_at >= INTERVAL:
                    watch.poll()
            else:
                cols, rows = (110, 36) if once else terminal_size()
                lines = build(watch, cols, rows, view)
                write_frame(lines)
                if once:
                    sys.stdout.write(plain(lines) + "\n")
                    return 0
                time.sleep(INTERVAL)
                watch.poll()
                continue
            cols, rows = terminal_size()
            lines = build(watch, cols, rows, view)
            drawn = plain(lines)
            if drawn == last_drawn and not resized["flag"]:
                continue
            if resized["flag"]:
                sys.stdout.write("\033[2J")
                resized["flag"] = False
            last_drawn = drawn
            write_frame(lines)
            sys.stdout.write(emit(lines, mode))
            sys.stdout.flush()
    finally:
        if old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
            print("monitor detached; Strata is still running")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nmonitor detached; Strata is still running")
        raise SystemExit(0)

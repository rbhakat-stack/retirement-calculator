import json
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import hmac
import copy
import uuid
import secrets
import math

from io import BytesIO
from datetime import datetime

# PDF generation (ReportLab)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader


# -----------------------------------------------------------------------------
# GLOBAL HELPERS (needed early for Supabase loaders)
# -----------------------------------------------------------------------------
SB_DEBUG_DEFAULT = False  # user requested debug panel OFF by default

def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = lo
    return max(lo, min(hi, v))

def _sb_debug_log(msg: str) -> None:
    """Lightweight logger; does not render UI unless you explicitly enable it."""
    try:
        logs = st.session_state.setdefault("_sb_debug_msgs", [])
        logs.append(str(msg))
        # Avoid unbounded growth
        if len(logs) > 200:
            st.session_state["_sb_debug_msgs"] = logs[-200:]
    except Exception:
        pass

# -----------------------------------------------------------------------------
# PDF EXPORT UTILITIES (optional; does not change core calculations)
# -----------------------------------------------------------------------------
def _pdf_draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, leading: float = 14) -> float:
    """Draw left-aligned wrapped text. Returns the next y position."""
    if text is None:
        return y
    words = str(text).replace("\r", "").split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, "Helvetica", 10) <= max_width:
            line = test
        else:
            c.setFont("Helvetica", 10)
            c.drawString(x, y, line)
            y -= leading
            line = w
    if line:
        c.setFont("Helvetica", 10)
        c.drawString(x, y, line)
        y -= leading
    return y


def _pdf_money(v: float) -> str:
    try:
        return f"${float(v):,.1f}"
    except Exception:
        return "$0.0"


# -----------------------------------------------------------------------------
# GENERAL CURRENCY FORMATTER (used by UI modules; keeps changes additive)
# -----------------------------------------------------------------------------
def _money(v: float) -> str:
    """Format a numeric value as whole-dollar USD (e.g., $155,000)."""
    try:
        return f"${float(v):,.0f}"
    except Exception:
        return "$0"


# -----------------------------------------------------------------------------
# PDF THEME + DRAW HELPERS (purely presentational)
# -----------------------------------------------------------------------------
from reportlab.lib import colors as _rl_colors

_PDF_BRAND = {
    "primary": _rl_colors.HexColor("#0B2D4D"),   # deep navy
    "accent":  _rl_colors.HexColor("#1F77B4"),   # blue accent
    "text":    _rl_colors.HexColor("#111827"),   # near-black
    "muted":   _rl_colors.HexColor("#6B7280"),   # gray
    "light":   _rl_colors.HexColor("#F3F4F6"),   # light gray background
    "border":  _rl_colors.HexColor("#E5E7EB"),   # table borders
    "good":    _rl_colors.HexColor("#0F766E"),   # teal
    "warn":    _rl_colors.HexColor("#B45309"),   # amber
    "bad":     _rl_colors.HexColor("#B91C1C"),   # red
}
def _pdf_header(c: canvas.Canvas, W: float, H: float, title: str, subtitle_left: str, subtitle_right: str) -> None:
    """Draws a branded header band."""
    band_h = 0.75 * inch
    c.saveState()
    c.setFillColor(_PDF_BRAND["primary"])
    c.rect(0, H - band_h, W, band_h, stroke=0, fill=1)

    c.setFillColor(_rl_colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.75 * inch, H - 0.45 * inch, title)

    c.setFont("Helvetica", 9)
    c.drawRightString(W - 0.75 * inch, H - 0.45 * inch, subtitle_right)
    c.setFillColor(_PDF_BRAND["light"])
    # subtle separator line under band
    c.setStrokeColor(_PDF_BRAND["border"])
    c.setLineWidth(1)
    c.line(0.75 * inch, H - band_h - 2, W - 0.75 * inch, H - band_h - 2)

    # second line under title
    c.setFillColor(_PDF_BRAND["muted"])
    c.setFont("Helvetica", 9)
    c.drawString(0.75 * inch, H - 0.67 * inch, subtitle_left)
    c.restoreState()

def _pdf_footer(c: canvas.Canvas, W: float, H: float, page_num: int) -> None:
    c.saveState()
    c.setFont("Helvetica", 8)
    c.setFillColor(_PDF_BRAND["muted"])
    c.drawString(0.75 * inch, 0.55 * inch, "Confidential – For personal planning purposes only")
    c.drawRightString(W - 0.75 * inch, 0.55 * inch, f"Page {page_num}")
    c.restoreState()


def _pdf_title_block(c, title: str, subtitle: str, y: float, W: float, margin: float) -> float:
    """Title block for PDF reports (clean, non-overlapping). Returns updated y."""
    c.saveState()

    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(_PDF_BRAND["text"])
    c.drawString(margin, y, title)

    y -= 22
    c.setFont("Helvetica", 10)
    c.setFillColor(_PDF_BRAND["muted"])
    c.drawString(margin, y, subtitle)

    gen = datetime.now().strftime("%b %d, %Y %I:%M %p")
    c.drawRightString(W - margin, y, f"Generated: {gen}")

    y -= 14
    c.setStrokeColor(_PDF_BRAND["border"])
    c.setLineWidth(1)
    c.line(margin, y, W - margin, y)

    c.restoreState()
    return y - 14
def _pdf_section_title(c: canvas.Canvas, x: float, y: float, W: float, text: str) -> float:
    c.saveState()
    c.setFillColor(_PDF_BRAND["light"])
    c.roundRect(x, y - 16, W, 18, 6, stroke=0, fill=1)
    c.setFillColor(_PDF_BRAND["primary"])
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x + 8, y - 12, text)
    c.restoreState()
    return y - 26


def _pdf_draw_fit_value(c: canvas.Canvas, x: float, y: float, w: float, text: str,
                        font_name: str = "Helvetica-Bold",
                        max_font: int = 14, min_font: int = 8,
                        max_lines: int = 2) -> None:
    """Draw text within width w, shrinking font and/or wrapping to max_lines."""
    s = "" if text is None else str(text)
    s = s.strip()
    if not s:
        return

    # Try single line with font shrink
    for fs in range(max_font, min_font - 1, -1):
        if c.stringWidth(s, font_name, fs) <= w:
            c.setFont(font_name, fs)
            c.drawString(x, y, s)
            return

    # Wrap into up to max_lines lines at a conservative font size
    fs = max(min_font, min(max_font - 3, 10))
    c.setFont(font_name, fs)

    words = s.split()
    lines = []
    cur = ""
    consumed = 0
    for word in words:
        trial = (cur + " " + word).strip()
        if c.stringWidth(trial, font_name, fs) <= w or not cur:
            cur = trial
            consumed += 1
        else:
            lines.append(cur)
            cur = word
            if len(lines) >= max_lines - 1:
                break
    if cur:
        lines.append(cur)

    # Ellipsize last line if not all words were consumed or width still too large
    ell = "…"
    if consumed < len(words) or c.stringWidth(lines[-1], font_name, fs) > w:
        last = lines[-1]
        while last and c.stringWidth(last + ell, font_name, fs) > w:
            last = last[:-1].rstrip()
        lines[-1] = (last + ell) if last else ell

    line_h = fs + 2
    for i, ln in enumerate(lines[:max_lines]):
        c.drawString(x, y - i * line_h, ln)

def _pdf_kpi_cards(c: canvas.Canvas, x: float, y: float, W: float, kpis: list[tuple[str, str, str]]) -> float:
    """kpis: list of (label, value, severity) where severity in {'good','warn','bad','neutral'}"""
    c.saveState()
    gap = 10
    card_h = 52
    n = max(1, len(kpis))
    card_w = (W - gap * (n - 1)) / n
    for i, (label, value, sev) in enumerate(kpis):
        cx = x + i * (card_w + gap)
        c.setFillColor(_rl_colors.white)
        c.setStrokeColor(_PDF_BRAND["border"])
        c.setLineWidth(1)
        c.roundRect(cx, y - card_h, card_w, card_h, 10, stroke=1, fill=1)

        color = _PDF_BRAND["accent"]
        if sev == "good":
            color = _PDF_BRAND["good"]
        elif sev == "warn":
            color = _PDF_BRAND["warn"]
        elif sev == "bad":
            color = _PDF_BRAND["bad"]

        # left accent bar
        c.setFillColor(color)
        c.rect(cx, y - card_h, 5, card_h, stroke=0, fill=1)

        c.setFillColor(_PDF_BRAND["muted"])
        c.setFont("Helvetica", 8.5)
        c.drawString(cx + 10, y - 16, label)

        c.setFillColor(_PDF_BRAND["primary"])
        # Fit value within card width (avoid overflow)
        _pdf_draw_fit_value(c, cx + 10, y - 30, card_w - 18, value, font_name="Helvetica-Bold", max_font=14, min_font=8, max_lines=2)
    c.restoreState()
    return y - card_h - 14

def _pdf_draw_table(c: canvas.Canvas, x: float, y: float, W: float, df: pd.DataFrame, font_size: int = 9) -> float:
    """Draw a clean table with header shading and alternating rows."""
    if df is None or df.empty:
        return y
    c.saveState()
    # prepare strings
    cols = list(df.columns)
    data = [cols] + df.astype(str).values.tolist()

    # compute column widths proportional to max string widths
    maxw = []
    for j, col in enumerate(cols):
        w = c.stringWidth(str(col), "Helvetica-Bold", font_size)
        for i in range(1, min(len(data), 50)):
            w = max(w, c.stringWidth(str(data[i][j])[:40], "Helvetica", font_size))
        maxw.append(w + 12)

    total = sum(maxw)
    if total > W:
        scale = W / total
        maxw = [w * scale for w in maxw]

    row_h = 16
    # header
    c.setFillColor(_PDF_BRAND["primary"])
    c.setStrokeColor(_PDF_BRAND["border"])
    c.roundRect(x, y - row_h, W, row_h, 6, stroke=0, fill=1)

    c.setFillColor(_rl_colors.white)
    c.setFont("Helvetica-Bold", font_size)
    cx = x
    for j, col in enumerate(cols):
        c.drawString(cx + 6, y - 12, str(col)[:40])
        cx += maxw[j]

    y -= row_h

    # rows
    c.setFont("Helvetica", font_size)
    for i, row in enumerate(data[1:], start=1):
        bg = _PDF_BRAND["light"] if (i % 2 == 0) else _rl_colors.white
        c.setFillColor(bg)
        c.rect(x, y - row_h, W, row_h, stroke=0, fill=1)

        c.setFillColor(_PDF_BRAND["primary"])
        cx = x
        for j, val in enumerate(row):
            c.drawString(cx + 6, y - 12, str(val)[:40])
            cx += maxw[j]

        # horizontal rule
        c.setStrokeColor(_PDF_BRAND["border"])
        c.setLineWidth(0.5)
        c.line(x, y - row_h, x + W, y - row_h)

        y -= row_h
        if y < 1.25 * inch:
            c.restoreState()
            return y  # caller handles pagination
    c.restoreState()
    return y - 6



def _build_montecarlo_pdf_bytes(
    user_id: str,
    snap: dict,
    sim_settings: dict,
    res: dict,
    chart_png: bytes | None,
    report_payload: dict | None = None,
) -> bytes:
    """
    Build a Monte Carlo PDF report (professional layout).
    - snap: normalized inputs snapshot (same structure used by the simulator)
    - sim_settings: Monte Carlo settings shown to the user
    - res: simulation outputs dict
    - chart_png: optional PNG bytes for an illustrative chart
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    margin = 0.75 * inch

    page_num = 1

    def _start_page():
        nonlocal page_num
        _pdf_header(
            c, W, H,
            title="Retirement Range of Outcomes",
            subtitle_left=f"User: {user_id}",
            subtitle_right=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        # usable y start below header band
        return H - 0.95 * inch

    def _end_page():
        nonlocal page_num
        _pdf_footer(c, W, H, page_num)
        c.showPage()
        page_num += 1

    y = _start_page()
    x = margin
    usable_w = W - 2 * margin

    # -----------------------------
    # Executive summary (KPIs)
    # -----------------------------
    y = _pdf_section_title(c, x, y, usable_w, "Executive summary")

    deplete_pct = float(res.get("deplete_pct", 0.0))
    median_final = res.get("median_final", res.get("median_final_balance", None))
    p10_final = res.get("p10_final", res.get("p10_final_balance", None))
    p90_final = res.get("p90_final", res.get("p90_final_balance", None))
    typical_deplete_age = res.get("typical_deplete_age", None)

    # Fallbacks if keys differ
    if median_final is None and "final_balance_percentiles" in res:
        try:
            median_final = res["final_balance_percentiles"].get("p50")
            p10_final = res["final_balance_percentiles"].get("p10")
            p90_final = res["final_balance_percentiles"].get("p90")
        except Exception:
            pass

    # Severity coloring for risk
    sev = "good"
    if deplete_pct >= 25:
        sev = "warn"
    if deplete_pct >= 50:
        sev = "bad"

    kpi_cards = [
        ("Chance of depletion", f"{deplete_pct:.1f}%", sev),
        ("Median ending balance", _pdf_money(median_final), "neutral"),
        ("10th–90th ending range", f"{_pdf_money(p10_final)} to {_pdf_money(p90_final)}", "neutral"),
    ]
    if typical_deplete_age is not None and typical_deplete_age != "":
        try:
            kpi_cards.append(("Typical depletion age", f"{int(float(typical_deplete_age))}", sev))
        except Exception:
            pass

    # Fit up to 4 cards on a row; if more, split
    row1 = kpi_cards[:4]
    y = _pdf_kpi_cards(c, x, y, usable_w, row1)
    if len(kpi_cards) > 4:
        if y < 2.0 * inch:
            _end_page()
            y = _start_page()
        y = _pdf_kpi_cards(c, x, y, usable_w, kpi_cards[4:8])

    
    # Additional sections (captured from other tabs)
    rp = report_payload or {}

    cmp = rp.get("compare")
    if cmp and cmp.get("scenarios"):
        y = _pdf_section_title(c, "Compare Scenarios (Summary)", margin, y)
        rows = [["Scenario", "Confidence", "Sustainable Spend", "Retirement Start"]]
        for s in cmp.get("scenarios", []):
            rows.append([s.get("name",""), str(s.get("confidence","")), str(s.get("sustainable_spend","")), str(s.get("retirement_age",""))])
        y = _pdf_table(c, rows, x=margin, y=y, W=W - 2 * margin, col_widths=[0.34*(W-2*margin),0.22*(W-2*margin),0.22*(W-2*margin),0.22*(W-2*margin)])
    else:
        c.setFont('Helvetica', 10)
        c.setFillColor(_PDF_BRAND['primary'])
        c.drawString(margin, y, 'Compare Scenarios: not run in this session.')
        y -= 14

    rng = rp.get("range_outcomes")
    if rng and rng.get("metrics"):
        y = _pdf_section_title(c, "Range of Outcomes (Selected Percentiles)", margin, y)
        rows = [["Metric", "P10", "P50", "P90"]]
        for mname, vals in rng.get("metrics", {}).items():
            rows.append([mname, str(vals.get("p10","")), str(vals.get("p50","")), str(vals.get("p90",""))])
        y = _pdf_table(c, rows, x=margin, y=y, W=W - 2 * margin, col_widths=[0.40*(W-2*margin),0.20*(W-2*margin),0.20*(W-2*margin),0.20*(W-2*margin)])
    else:
        c.setFont('Helvetica', 10)
        c.setFillColor(_PDF_BRAND['primary'])
        c.drawString(margin, y, 'Range of Outcomes: not run in this session.')
        y -= 14

    # "How much can I spend in retirement?" (Single Scenario)
    try:
        spend_results = st.session_state.get("spend_calc_results") or rp.get("spend_calc_results")
        if spend_results:
            y = _pdf_section_title(c, "How much can I spend in retirement?", margin, y)
            rows = [["Confidence target", "Estimated sustainable annual spend (today's $)"]]
            for label, out in spend_results[:3]:
                spend_val = 0.0
                try:
                    spend_val = float((out or {}).get("spend", 0.0))
                except Exception:
                    spend_val = 0.0
                rows.append([str(label), _money(spend_val)])
            y = _pdf_table(
                c,
                rows,
                x=margin,
                y=y,
                W=W - 2 * margin,
                col_widths=[0.45 * (W - 2 * margin), 0.55 * (W - 2 * margin)],
                font_size=9,
            )
            y -= 6
    except Exception:
        pass

    stx = rp.get("stress_tests") or {}
    y = _pdf_section_title(c, "Market Reality Mode (Stress Tests)", margin, y)

    try:
        st_rows = stx.get("rows") or []
        if st_rows:
            rows = [["Shock", "Confidence Δ", "Spend impact", "Recovery time"]]
            for r in st_rows:
                rows.append([
                    str(r.get("shock", "")),
                    str(r.get("confidence_delta", r.get("confidence_delta_pct", ""))),
                    str(r.get("spend_impact", "")),
                    str(r.get("recovery_time", "")),
                ])
            y = _pdf_table(
                c,
                rows,
                x=margin,
                y=y,
                W=W - 2 * margin,
                col_widths=[
                    0.42 * (W - 2 * margin),
                    0.19 * (W - 2 * margin),
                    0.19 * (W - 2 * margin),
                    0.20 * (W - 2 * margin),
                ],
                font_size=9,
            )
            y -= 6
        else:
            c.setFont("Helvetica", 10)
            c.setFillColor(_PDF_BRAND["muted"])
            c.drawString(margin, y, "No stress tests were run for this report.")
            y -= 14
    except Exception:
        c.setFont("Helvetica", 10)
        c.setFillColor(_PDF_BRAND["muted"])
        c.drawString(margin, y, "Stress test summary unavailable due to an internal error.")
        y -= 14

# Narrative bullets
    if y < 2.0 * inch:
        _end_page()
        y = _start_page()
    c.setFont("Helvetica", 10)
    c.setFillColor(_PDF_BRAND["primary"])
    life_exp = int(snap.get("life_expectancy", 95) or 95)
    bullets = [
        f"Simulation horizon: through age {life_exp}.",
        "Results are probabilistic and depend on return, inflation, and spending assumptions.",
    ]
    if typical_deplete_age is not None:
        try:
            bullets.append(f"When depletion occurs, it most often happens around age {int(float(typical_deplete_age))}.")
        except Exception:
            pass
    by = y
    for b in bullets:
        c.drawString(x, by, u"\u2022 " + b)
        by -= 14
    y = by - 6

    # -----------------------------
    # Inputs used
    # -----------------------------
    if y < 2.2 * inch:
        _end_page()
        y = _start_page()
    y = _pdf_section_title(c, x, y, usable_w, "Inputs used")

    c.setFillColor(_PDF_BRAND["primary"])
    inputs_lines = [
        f"Current age: {int(snap.get('current_age', 0))}    Retirement age: {int(snap.get('retire_age', 0))}    Life expectancy: {int(snap.get('life_expectancy', 0))}",
        f"Annual spend in retirement (today $): {_pdf_money(snap.get('annual_spend_retirement', 0.0))}",
        f"Annual contribution until retirement: {_pdf_money(snap.get('annual_contribution', 0.0))}",
        f"Inflation: {float(snap.get('inflation_rate', 0.0))*100:.2f}%    Pre-retire return: {float(snap.get('pre_retire_return', 0.0))*100:.2f}%    Post-retire return: {float(snap.get('post_retire_return', 0.0))*100:.2f}%",
        f"Social Security / Pension (today $/yr): {_pdf_money(snap.get('social_security', 0.0))} starting at age {int(snap.get('ss_start_age', 0))}",
    ]
    c.setFont("Helvetica", 10)
    for ln in inputs_lines:
        y = _pdf_draw_wrapped(c, ln, x, y, usable_w, leading=13)
        if y < 1.4 * inch:
            _end_page()
            y = _start_page()

    # -----------------------------
    # Simulation settings
    # -----------------------------
    if y < 2.0 * inch:
        _end_page()
        y = _start_page()
    y = _pdf_section_title(c, x, y, usable_w, "Simulation settings")
    c.setFont("Helvetica", 10)
    c.setFillColor(_PDF_BRAND["primary"])
    for k, v in (sim_settings or {}).items():
        y = _pdf_draw_wrapped(c, f"{k}: {v}", x, y, usable_w, leading=13)
        if y < 1.4 * inch:
            _end_page()
            y = _start_page()

    # -----------------------------
    # Chart page
    # -----------------------------
    if chart_png:
        _end_page()
        y = _start_page()
        y = _pdf_section_title(c, x, y, usable_w, "Distribution & trajectories (illustrative)")
        img = ImageReader(BytesIO(chart_png))
        img_w = usable_w
        img_h = img_w * 0.60
        if y - img_h < 1.25 * inch:
            _end_page()
            y = _start_page()
        c.drawImage(img, x, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")
        y = y - img_h - 10

    # finish
    _pdf_footer(c, W, H, page_num)
    c.save()
    return buf.getvalue()

def _build_compare_pdf_bytes(user_id: str, kpi_df: pd.DataFrame, chart_png: bytes | None, title: str = "Scenario Comparison") -> bytes:
    """
    Build a scenario comparison PDF report (professional layout).
    - kpi_df: dataframe of scenario KPIs (already computed)
    - chart_png: optional PNG bytes of the comparison chart
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    margin = 0.75 * inch
    x = margin
    usable_w = W - 2 * margin

    page_num = 1

    def _start_page():
        _pdf_header(
            c, W, H,
            title=title,
            subtitle_left=f"User: {user_id}",
            subtitle_right=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        return H - 0.95 * inch

    def _end_page():
        nonlocal page_num
        _pdf_footer(c, W, H, page_num)
        c.showPage()
        page_num += 1

    y = _start_page()

    # -----------------------------
    # At-a-glance cards
    # -----------------------------
    y = _pdf_section_title(c, x, y, usable_w, "At-a-glance")
    df = (kpi_df.copy() if kpi_df is not None else pd.DataFrame())

    # Derive a few comparisons if possible
    cards = []
    try:
        # best final balance
        if "Final Balance" in df.columns and "Scenario" in df.columns:
            # attempt parse currency-like strings
            def _to_num(v):
                try:
                    return float(str(v).replace("$", "").replace(",", ""))
                except Exception:
                    return None
            tmp = df[["Scenario", "Final Balance"]].copy()
            tmp["_num"] = tmp["Final Balance"].map(_to_num)
            tmp = tmp.dropna()
            if not tmp.empty:
                best = tmp.sort_values("_num", ascending=False).iloc[0]
                cards.append(("Best ending balance", f"{best['Scenario']}", "neutral"))
                cards.append(("Ending balance", _pdf_money(best["_num"]), "neutral"))
    except Exception:
        pass

    # depletion age / withdrawal rate highlights
    try:
        if "Scenario" in df.columns:
            tmp = df.copy()

            def _to_int(v):
                try:
                    s = str(v).strip()
                    if s == "" or s.lower() in {"none", "nan"}:
                        return None
                    return int(float(s))
                except Exception:
                    return None

            def _parse_money(v):
                try:
                    s = str(v)
                    # keep digits, minus, decimal
                    s = re.sub(r"[^0-9\-\.]", "", s)
                    if s in {"", "-", ".", "-."}:
                        return None
                    return float(s)
                except Exception:
                    return None

            # 1) Prefer explicit depletion age if present
            if "Depletion Age" in tmp.columns:
                tmp["_deplete_age"] = tmp["Depletion Age"].map(_to_int)
            else:
                tmp["_deplete_age"] = None

            # 2) If plan is sustainable, infer horizon age from Sustainability text (e.g., "Sustainable to 95")
            if "Sustainability" in tmp.columns:
                def _infer_horizon(s):
                    """Infer planning-horizon age from a sustainability text field."""
                    try:
                        s = str(s)
                        # Common patterns
                        m = re.search(r"sustainable\s+(?:to|through)\s+(\d+)", s, flags=re.I)
                        if m:
                            return int(m.group(1))
                        # Fallback: first 2–3 digit number in the string (e.g., '... age 95', 'to **95**')
                        m = re.search(r"(\d{2,3})", s)
                        return int(m.group(1)) if m else None
                    except Exception:
                        return None

                # If some rows have slightly different wording, use a document-level fallback horizon.
                _global_horizon = None
                try:
                    ages = []
                    for _s in tmp.get("Sustainability", pd.Series(dtype=str)).astype(str).tolist():
                        ages.extend([int(a) for a in re.findall(r"(\d{2,3})", _s)])
                    if ages:
                        _global_horizon = max(ages)
                except Exception:
                    _global_horizon = None

                tmp["_horizon_age"] = tmp["Sustainability"].map(_infer_horizon)
                if _global_horizon is not None:
                    tmp.loc[tmp["_horizon_age"].isna(), "_horizon_age"] = _global_horizon
            else:
                tmp["_horizon_age"] = None

            # Effective "lasts until" age:
            # - use depletion age when present
            # - otherwise use inferred horizon (for sustainable plans)
            tmp["_lasts_to_age"] = tmp["_deplete_age"]
            tmp.loc[tmp["_lasts_to_age"].isna(), "_lasts_to_age"] = tmp.loc[tmp["_lasts_to_age"].isna(), "_horizon_age"]

            # Tie-breaker: higher ending balance
            if "Final Balance" in tmp.columns:
                tmp["_final_num"] = tmp["Final Balance"].map(_parse_money)
            else:
                tmp["_final_num"] = None

            tmp2 = tmp.dropna(subset=["_lasts_to_age"])
            if not tmp2.empty:
                best = tmp2.sort_values(by=["_lasts_to_age", "_final_num"], ascending=[False, False]).iloc[0]
                cards.append(("Longest lasting plan", f"{best['Scenario']}", "good"))

                # Show depletion age if it truly depleted; otherwise show the horizon age
                if best.get("_deplete_age") is not None and str(best.get("Depletion Age", "")).strip() != "":
                    cards.append(("Depletion age", f"{int(best['_deplete_age'])}", "good"))
                else:
                    # Sustainable through horizon
                    ha = best.get("_horizon_age")
                    cards.append(("Depletion age", f"{int(ha)}", "good") if ha else ("Depletion age", "N/A", "good"))
    except Exception:
        pass

    if not cards:
        cards = [
            ("Scenarios compared", f"{len(df)}", "neutral"),
            ("Report type", "Side-by-side KPIs", "neutral"),
            ("Chart", "Portfolio paths", "neutral"),
        ]

    # Show up to 4 cards on first row
    y = _pdf_kpi_cards(c, x, y, usable_w, cards[:4])

    # -----------------------------
    # KPI Table
    # -----------------------------
    if y < 2.0 * inch:
        _end_page()
        y = _start_page()

    y = _pdf_section_title(c, x, y, usable_w, "Key metrics (side-by-side)")

    # Format money columns consistently
    df2 = df.copy()
    for col in df2.columns:
        l = col.lower()
        if "balance" in l or "assets" in l:
            df2[col] = df2[col].apply(lambda v: _pdf_money(float(str(v).replace("$","").replace(",",""))) if str(v).strip() != "" else "")
    # Put Scenario first if exists
    if "Scenario" in df2.columns:
        cols = ["Scenario"] + [c for c in df2.columns if c != "Scenario"]
        df2 = df2[cols]

    y2 = _pdf_draw_table(c, x, y, usable_w, df2, font_size=9)
    if y2 < 1.25 * inch:
        _end_page()
        y = _start_page()
        # continue table if needed by splitting
        # For simplicity: draw remaining rows on next page
        # (table function paginates to caller; here we re-draw full table on new page if it overflowed)
        y = _pdf_section_title(c, x, y, usable_w, "Key metrics (continued)")
        y = _pdf_draw_table(c, x, y, usable_w, df2, font_size=9)
    else:
        y = y2

    # -----------------------------
    # Chart page
    # -----------------------------
    if chart_png:
        _end_page()
        y = _start_page()
        y = _pdf_section_title(c, x, y, usable_w, "Portfolio trajectory (illustrative)")
        img = ImageReader(BytesIO(chart_png))
        img_w = usable_w
        img_h = img_w * 0.60
        if y - img_h < 1.25 * inch:
            _end_page()
            y = _start_page()
        c.drawImage(img, x, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")
        y = y - img_h - 10

    _pdf_footer(c, W, H, page_num)
    c.save()
    return buf.getvalue()


def _pdf_section_title(c, *args):
    """Draw a section title and return updated y.

    Supports both call styles used across the app:

      1) _pdf_section_title(c, text, margin, y)
      2) _pdf_section_title(c, margin, y, text)
      3) _pdf_section_title(c, x, y, W, text)
      4) _pdf_section_title(c, text, x, y, W)

    Returns the updated y cursor.
    """
    # Normalize arguments
    if len(args) == 3:
        a1, a2, a3 = args
        if isinstance(a1, str) and isinstance(a2, (int, float)) and isinstance(a3, (int, float)):
            text, x, y = a1, float(a2), float(a3)
        elif isinstance(a3, str) and isinstance(a1, (int, float)) and isinstance(a2, (int, float)):
            x, y, text = float(a1), float(a2), a3
        else:
            # best-effort
            text, x, y = str(a1), float(a2), float(a3)
        # Simple title (no width banner)
        try:
            c.setFont("Helvetica-Bold", 12)
        except Exception:
            pass
        c.drawString(x, y, str(text))
        y -= 16
        try:
            c.setLineWidth(0.5)
            page_w = getattr(c, "_pagesize", (595.0, 842.0))[0]
            c.line(x, y + 6, page_w - x, y + 6)
        except Exception:
            pass
        return y

    if len(args) == 4:
        a1, a2, a3, a4 = args
        # Identify ordering
        if isinstance(a4, str) and all(isinstance(v, (int, float)) for v in (a1, a2, a3)):
            x, y, W, text = float(a1), float(a2), float(a3), a4
        elif isinstance(a1, str) and all(isinstance(v, (int, float)) for v in (a2, a3, a4)):
            text, x, y, W = a1, float(a2), float(a3), float(a4)
        else:
            # best-effort fallback
            x, y, W, text = float(a1), float(a2), float(a3), str(a4)

        # Banner style if brand palette exists; otherwise plain.
        try:
            brand = globals().get("_PDF_BRAND") or globals().get("PDF_BRAND") or {}
            primary = brand.get("primary")
            light = brand.get("light")
            if light is not None and primary is not None:
                from reportlab.lib import colors
                c.saveState()
                c.setFillColor(light if hasattr(light, "__class__") else colors.HexColor(str(light)))
                c.roundRect(x, y - 16, W, 18, 6, stroke=0, fill=1)
                c.setFillColor(primary if hasattr(primary, "__class__") else colors.HexColor(str(primary)))
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x + 8, y - 12, str(text))
                c.restoreState()
                return y - 26
        except Exception:
            pass

        # Fallback rendering
        try:
            c.setFont("Helvetica-Bold", 12)
        except Exception:
            pass
        c.drawString(x, y, str(text))
        y -= 16
        try:
            c.setLineWidth(0.5)
            c.line(x, y + 6, x + W, y + 6)
        except Exception:
            pass
        return y - 6

    raise TypeError(f"_pdf_section_title expected 3 or 4 arguments after canvas, got {len(args)}")
def _pdf_table(c, rows, x, y, W, col_widths=None, font_name="Helvetica", font_size=9, row_pad=6):
    """
    Render a readable, non-overlapping table onto a ReportLab canvas.

    - Increased padding to prevent text touching borders.
    - Header row fill + bold font.
    - Subtle zebra striping for readability.
    """
    if not rows:
        return y

    rows = [["" if v is None else str(v) for v in r] for r in rows]
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    if col_widths is None:
        col_widths = [W / ncols] * ncols
    else:
        s = sum(col_widths) or 1.0
        if abs(s - W) > 1e-6:
            col_widths = [w * (W / s) for w in col_widths]

    page_w, page_h = getattr(c, "_pagesize", (595.0, 842.0))

    # Approx chars per line for wrapping
    chars = []
    for w in col_widths:
        cpl = max(10, int(w / (font_size * 0.55)))
        chars.append(cpl)

    import textwrap as _tw
    def wrap_cell(val: str, cpl: int):
        s = (val or "").replace("\r", "")
        parts = []
        for para in str(s).split("\n"):
            parts.extend(_tw.wrap(para, width=cpl) or [""])
        return parts or [""]

    wrapped = []
    heights = []
    leading = font_size + 3

    for r in rows:
        wr = [wrap_cell(r[i], chars[i]) for i in range(ncols)]
        wrapped.append(wr)
        max_lines = max(len(w) for w in wr)
        heights.append(max(leading + 2 * row_pad, max_lines * leading + 2 * row_pad))

    cur_y = y
    inset_x = 6

    for ridx in range(len(rows)):
        h = heights[ridx]
        if cur_y - h < 40:
            c.showPage()
            cur_y = page_h - 60

        # Row background fills
        if ridx == 0:
            c.saveState()
            c.setFillColor(_PDF_BRAND["light"])
            c.rect(x, cur_y - h, W, h, stroke=0, fill=1)
            c.restoreState()
        elif ridx % 2 == 0:
            c.saveState()
            c.setFillColor(_rl_colors.whitesmoke)
            c.rect(x, cur_y - h, W, h, stroke=0, fill=1)
            c.restoreState()

        cx = x
        for cidx in range(ncols):
            c.rect(cx, cur_y - h, col_widths[cidx], h, stroke=1, fill=0)

            if ridx == 0:
                c.setFont("Helvetica-Bold", font_size)
            else:
                c.setFont(font_name, font_size)

            ty = cur_y - row_pad - font_size
            for ln in wrapped[ridx][cidx]:
                c.drawString(cx + inset_x, ty, ln[:300])
                ty -= leading
            cx += col_widths[cidx]

        cur_y -= h

    return cur_y - 12


def _build_readiness_report_pdf_bytes(
    user_id: str,
    snap: dict,
    baseline_res: dict,
    stress_rows: list[dict],
    narrative: dict,
    report_payload: dict | None = None,
) -> bytes:
    """Build a one-time Retirement Readiness Report (Phase 2).

    This is an interpretation/reporting layer only. It does not provide prescriptive advice.
    """
    # Optional extended payload for premium report sections (compare/range/stress summaries)
    rp = report_payload or {}
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    margin = 0.75 * inch

    # Cover / Title
    y = H - margin
    _pdf_title_block(
        c,
        title="One-Time Retirement Readiness Report",
        subtitle=f"Generated for: {user_id}",
        y=y,
        W=W,
        margin=margin,
    )
    y -= 48


    # Inputs summary (Single Scenario)
    # Inputs summary (Single Scenario)
    try:
        y = _pdf_section_title(c, "Single Scenario Inputs (Summary)", margin, y)

        cur_age = int(snap.get("current_age", 0) or 0)
        ret_age = int(snap.get("retire_age", snap.get("retire_age", snap.get("retire_age", 0))) or snap.get("retire_age", 0) or 0)
        # The canonical snapshot uses "retire_age" in the UI and "retire_age"/"retire_age" may not exist; also accept "retire_age"
        if not ret_age:
            ret_age = int(snap.get("retire_age", snap.get("retire_age", 0)) or 0)
        if not ret_age:
            ret_age = int(snap.get("retire_age", 0) or 0)
        if not ret_age:
            ret_age = int(snap.get("retire_age", 0) or 0)
        if not ret_age:
            ret_age = int(snap.get("retire_age", 0) or 0)

        # In this app the canonical key is "retire_age"
        ret_age = int(snap.get("retire_age", snap.get("retirement_age", ret_age)) or ret_age)

        life_exp = int(snap.get("life_expectancy", 0) or 0)

        use_multi = bool(snap.get("use_multi_asset", False))
        cash_bal = float(snap.get("cash_bal", 0.0) or 0.0)
        bonds_bal = float(snap.get("bonds_bal", 0.0) or 0.0)
        etfs_bal = float(snap.get("etfs_bal", 0.0) or 0.0)
        k401_bal = float(snap.get("k401_bal", 0.0) or 0.0)

        if use_multi and (cash_bal + bonds_bal + etfs_bal + k401_bal) > 0:
            starting_portfolio = cash_bal + bonds_bal + etfs_bal + k401_bal
        else:
            starting_portfolio = float(snap.get("current_portfolio", 0.0) or 0.0)

        annual_contrib = float(snap.get("annual_contribution", 0.0) or 0.0)
        planned_spend = float(snap.get("annual_spend_retirement", 0.0) or 0.0)

        now_year = datetime.now().year
        ret_year = now_year + max(0, int(ret_age - cur_age)) if (cur_age and ret_age and ret_age >= cur_age) else None

        rows = [
            ["Input", "Value"],
            ["Current age", str(cur_age)],
            ["Retirement age", str(ret_age)],
            ["Estimated retirement year", str(ret_year) if ret_year else "—"],
            ["Life expectancy", str(life_exp)],
            ["Starting portfolio", _money(starting_portfolio)],
            ["Retirement assets (401k)", _money(k401_bal) if k401_bal else "—"],
            ["Annual contribution (pre-retirement)", _money(annual_contrib)],
            ["Planned annual spending (year 1 of retirement)", _money(planned_spend)],
            ["Pre-retirement mean return", f"{float(snap.get('pre_retire_return', 0.0) or 0.0)*100:.2f}%"],
            ["Post-retirement mean return", f"{float(snap.get('post_retire_return', 0.0) or 0.0)*100:.2f}%"],
            ["Inflation mean", f"{float(snap.get('inflation_rate', 0.0) or 0.0)*100:.2f}%"],
        ]
        y = _pdf_table(c, rows, x=margin, y=y, W=W - 2 * margin, col_widths=[0.55*(W-2*margin), 0.45*(W-2*margin)], font_size=9)
        y -= 6

        if use_multi and (cash_bal + bonds_bal + etfs_bal + k401_bal) > 0:
            y = _pdf_section_title(c, "Portfolio Breakdown (Multi-Asset)", margin, y)
            br = [
                ["Asset", "Balance"],
                ["Cash", _money(cash_bal)],
                ["Bonds/Munis", _money(bonds_bal)],
                ["ETFs", _money(etfs_bal)],
                ["401k", _money(k401_bal)],
                ["Total", _money(cash_bal + bonds_bal + etfs_bal + k401_bal)],
            ]
            y = _pdf_table(c, br, x=margin, y=y, W=W - 2 * margin, col_widths=[0.55*(W-2*margin), 0.45*(W-2*margin)], font_size=9)
            y -= 6

    except Exception:
        y -= 6



    # Baseline summary
    _pdf_section_title(c, "Baseline Snapshot (Simulation)", margin, y)
    y -= 20
    try:
        prob_deplete = float(baseline_res.get("prob_deplete", 0.0))
        conf = 100.0 * (1.0 - prob_deplete)
        median_final = float(baseline_res.get("median_final", np.percentile(baseline_res.get("final_balances", [0.0]), 50)))
        p10_final = float(baseline_res.get("p10_final", np.percentile(baseline_res.get("final_balances", [0.0]), 10)))
        p90_final = float(baseline_res.get("p90_final", np.percentile(baseline_res.get("final_balances", [0.0]), 90)))

        y = _pdf_kpi_row(
            c,
            [
                ("Confidence (0–100)", f"{conf:.0f}"),
                ("Chance of depletion", f"{prob_deplete*100:.1f}%"),
                ("Median ending balance", f"${median_final:,.0f}"),
                ("10th / 90th ending", f"${p10_final:,.0f} / ${p90_final:,.0f}"),
            ],
            x=margin,
            y=y,
            W=W - 2 * margin,
        )
    except Exception:
        y = y - 8

    y -= 10
    c.setFillColor(_PDF_BRAND["muted"])
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, "Note: This report provides interpretation of modeled scenarios, not financial advice.")
    y -= 18

    # Stress test table
    _pdf_section_title(c, "Market Reality Mode (Stress Tests)", margin, y)
    y -= 16
    if stress_rows:
        header = ["Scenario", "Confidence Δ", "Spend impact", "Recovery time"]
        rows = [header]
        for r in stress_rows:
            rows.append(
                [
                    str(r.get("scenario", ""))[:30],
                    f"{float(r.get('confidence_delta', 0.0)):+.0f}",
                    str(r.get("spend_impact", "N/A"))[:20],
                    str(r.get("recovery_time", "N/A"))[:20],
                ]
            )
        y = _pdf_table(c, rows, x=margin, y=y, W=W - 2 * margin)
    else:
        c.setFillColor(_PDF_BRAND["primary"])
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, "No stress tests were run for this report.")
        y -= 16

    if y < 2.0 * inch:
        c.showPage()
        y = H - margin


    # Narrative insights
    _pdf_section_title(c, "AI-Powered Narrative Insights", margin, y)
    y -= 16
    c.setFillColor(_PDF_BRAND["primary"])
    c.setFont("Helvetica", 10)

    def _bullets(title: str, items: list[str], y0: float) -> float:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, y0, title)
        y0 -= 12
        c.setFont("Helvetica", 10)
        for it in (items or [])[:6]:
            txt = str(it)
            for line in textwrap.wrap(txt, width=95):
                c.drawString(margin + 10, y0, f"• {line}" if line == textwrap.wrap(txt, width=95)[0] else f"  {line}")
                y0 -= 12
                if y0 < 1.25 * inch:
                    c.showPage()
                    y0 = H - margin
                    c.setFillColor(_PDF_BRAND["primary"])
                    c.setFont("Helvetica", 10)
        return y0 - 4

    y = _bullets("Key insights", narrative.get("insights", []), y)
    y = _bullets("Risks", narrative.get("risks", []), y)
    y = _bullets("Tradeoffs", narrative.get("tradeoffs", []), y)
    y = _bullets("What matters most", narrative.get("what_matters_most", []), y)

    c.setFillColor(_PDF_BRAND["muted"])
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(margin, 0.75 * inch, "Prepared by the Retirement Planner. Interpretation-only; not financial advice.")

    c.save()
    return buf.getvalue()

def _as_decimal_rate(x, default=0.0):
    """Normalize any rate to decimal form (0.07 or 7 -> 0.07)."""
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v / 100.0 if v > 1.0 else v

def _as_percent_display(x, default_pct=0.0):
    """For slider defaults (percent units). Returns number like 7.0."""
    d = _as_decimal_rate(x, default=default_pct / 100.0)
    return d * 100.0


def _as_float_with_default(value, default=0.0):
    """Best-effort float coercion with a fallback default."""
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_snapshot(s: dict) -> dict:
    """Return a NEW snapshot with consistent units (rates as decimals)."""
    s2 = copy.deepcopy(s or {})

    # Core rates
    s2["inflation_rate"] = _as_decimal_rate(s2.get("inflation_rate", 0.03), 0.03)
    s2["pre_retire_return"] = _as_decimal_rate(s2.get("pre_retire_return", 0.07), 0.07)
    s2["post_retire_return"] = _as_decimal_rate(s2.get("post_retire_return", 0.045), 0.045)

    # Multi-asset yields
    s2["cash_yield"] = _as_decimal_rate(s2.get("cash_yield", 0.04), 0.04)
    s2["bonds_yield"] = _as_decimal_rate(s2.get("bonds_yield", 0.05), 0.05)
    s2["etfs_yield"] = _as_decimal_rate(s2.get("etfs_yield", 0.07), 0.07)
    s2["k401_yield"] = _as_decimal_rate(s2.get("k401_yield", 0.07), 0.07)

    # Defaults / required keys
    s2["use_multi_asset"] = bool(s2.get("use_multi_asset", True))
    s2["flow_mode"] = s2.get("flow_mode", "cash_first")
    if s2["flow_mode"] not in ("cash_first", "pro_rata"):
        s2["flow_mode"] = "cash_first"

    # Ensure numeric types for critical fields
    for k in ["current_age", "retire_age", "life_expectancy", "ss_start_age"]:
        if k in s2 and s2[k] is not None:
            try:
                s2[k] = int(s2[k])
            except Exception:
                pass

    for k in [
        "annual_spend_retirement", "social_security", "annual_contribution",
        "current_portfolio", "cash_bal", "bonds_bal", "etfs_bal", "k401_bal", "real_estate_bal", "misc_bal"
    ]:
        if k in s2 and s2[k] is not None:
            try:
                s2[k] = float(s2[k])
            except Exception:
                pass

    return s2

def _sb_scenarios_table() -> str:
    return st.secrets.get("supabase", {}).get("scenarios_table", "scenarios")
from datetime import datetime, timezone

# -----------------------------------------------------------------------------
# RATE NORMALIZATION HELPERS (prevents percent/decimal confusion)
# -----------------------------------------------------------------------------
def _as_decimal_rate(x, default=0.0):
    """Normalize any rate to decimal form.
    Accepts 0.07 (already decimal) or 7/7.0 (percent) -> 0.07.
    """
    try:
        v = float(x)
    except Exception:
        return float(default)
    if v > 1.0:
        return v / 100.0
    return v

def _as_percent_display(x, default_pct=0.0):
    """Return a percent-number for sliders (e.g., 0.07 -> 7.0).
    Accepts decimal or percent.
    """
    d = _as_decimal_rate(x, default=default_pct / 100.0)
    return d * 100.0


# ----------------------------------------------------------------------------- 
# OPTIONAL SUPABASE PERSISTENCE (PER-USER)
# ----------------------------------------------------------------------------- 
# This enables saving/loading scenarios per authenticated user without changing
# any app behavior when Supabase is not configured.
try:
    from supabase import create_client  # type: ignore
except Exception:  # pragma: no cover
    create_client = None  # type: ignore


@st.cache_resource
def _get_supabase_client():
    cfg = st.secrets.get("supabase", {})
    url = cfg.get("url", "")
    key = cfg.get("service_role_key", "") or cfg.get("key", "")
    if not url or not key or create_client is None:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None



# Backwards-compatible alias (some code paths call get_supabase_client)
def get_supabase_client():
    return _get_supabase_client()

def _sb_enabled() -> bool:
    return _get_supabase_client() is not None


def _sb_debug_log(msg: str):
    """
    Append a message to an in-memory debug log (only when debug_supabase is enabled).
    This is intentionally safe/no-op in production runs.
    """
    try:
        if not st.session_state.get("debug_supabase", False):
            return
        logs = st.session_state.setdefault("_sb_debug_logs", [])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs.append(f"{ts} | {msg}")
        # keep log bounded
        if len(logs) > 500:
            del logs[:-500]
    except Exception:
        # Never break the app due to debug logging
        return

def _sb_table() -> str:
    """Name of the table that stores per-user state.

    Priority:
      1) st.secrets.supabase.user_state_table (explicit)
      2) fallback to 'retirement_user_state' (your current deployment)
      3) final fallback to 'user_state' (older naming)
    """
    tbl = st.secrets.get("supabase", {}).get("user_state_table")
    if tbl:
        return tbl
    # Default to the newer name you created, but keep backward compatibility.
    return "retirement_user_state"


def _scenario_store_key() -> str:
    # per-login separation in session_state
    user = st.session_state.get("current_user") or "default"
    return f"scenarios__{user}"


def _single_snapshot_key() -> str:
    user = st.session_state.get("current_user") or "default"
    return f"single_snapshot__{user}"


def _sb_json_sanitize(x):
    """Convert common non-JSON types (numpy/pandas) into plain Python types."""
    try:
        import numpy as _np  # local import
        if isinstance(x, (_np.integer,)):
            return int(x)
        if isinstance(x, (_np.floating,)):
            return float(x)
        if isinstance(x, (_np.ndarray,)):
            return x.tolist()
    except Exception:
        pass

    if isinstance(x, dict):
        return {str(k): _sb_json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sb_json_sanitize(v) for v in x]
    return x


def sb_load_user_state(user_id: str) -> dict:
    """Load per-user working inputs (Single Scenario sidebar state) from Supabase.

    Preferred schema (per your SQL):
      user_state(user_id text PK, active_scenario_id uuid NULL, working_inputs jsonb NULL, updated_at timestamptz)

    Also tolerates legacy tables that stored 'single_snapshot' instead of 'working_inputs'.
    """
    try:
        client = get_supabase_client()
    except Exception as e:
        _sb_debug_log(f"ERROR: get_supabase_client failed in sb_load_user_state: {e}")
        return {}

    table_from_secrets = (st.secrets.get("supabase", {}) or {}).get("user_state_table") or ""
    candidates = [t for t in [table_from_secrets, "user_state", "retirement_user_state"] if t]

    for table in candidates:
        # First: try modern schema
        try:
            res = (
                client.table(table)
                .select("user_id, active_scenario_id, working_inputs, updated_at")
                .eq("user_id", user_id)
                .maybe_single()
                .execute()
            )
            row = getattr(res, "data", None) or {}
            if row and row.get("working_inputs") is not None:
                return {
                    "_table": table,
                    "user_id": row.get("user_id", user_id),
                    "active_scenario_id": row.get("active_scenario_id"),
                    "working_inputs": row.get("working_inputs"),
                    "updated_at": row.get("updated_at"),
                }
        except Exception as e:
            _sb_debug_log(f"WARN: sb_load_user_state modern select failed on '{table}': {e}")

        # Second: legacy schema
        try:
            res = (
                client.table(table)
                .select("user_id, single_snapshot, updated_at")
                .eq("user_id", user_id)
                .maybe_single()
                .execute()
            )
            row = getattr(res, "data", None) or {}
            if row and row.get("single_snapshot") is not None:
                return {
                    "_table": table,
                    "user_id": row.get("user_id", user_id),
                    "active_scenario_id": None,
                    "working_inputs": row.get("single_snapshot"),
                    "updated_at": row.get("updated_at"),
                }
        except Exception as e:
            _sb_debug_log(f"WARN: sb_load_user_state legacy select failed on '{table}': {e}")
            continue

    return {}


def sb_upsert_user_state_row(user_id: str, active_scenario_id=None, working_inputs=None):
    """
    Persist lightweight per-user UI state into the `user_state` table:
      - active_scenario_id (uuid, nullable)
      - working_inputs (jsonb, nullable)

    Important:
    - We only write active_scenario_id if it is a valid UUID. This avoids insert/update failures
      when the app is using short, non-UUID scenario ids in-session.
    - We do NOT send active_scenario_id=None, to avoid accidentally wiping a previously stored UUID.
    """
    if not SUPABASE_ENABLED:
        return

    client = get_supabase_client()

    payload = {
        "user_id": user_id,
        "working_inputs": working_inputs,
        "updated_at": "now()",
    }

    # Only include active_scenario_id if it is a valid UUID string/object
    if active_scenario_id:
        try:
            payload["active_scenario_id"] = str(uuid.UUID(str(active_scenario_id)))
        except Exception:
            # invalid UUID; skip writing this field
            pass

    try:
        client.table("user_state").upsert(payload, on_conflict="user_id").execute()
    except Exception as e:
        _sb_debug_log(f"WARNING: user_state upsert failed (non-fatal): {e}")

def sb_save_user_state(user_id: str, working_inputs: dict | None, active_scenario_id: str | None = None) -> bool:
    """Upsert per-user working inputs into Supabase.

    Writes into the first table that matches your schema (user_state or the configured table).
    """
    try:
        client = get_supabase_client()
    except Exception as e:
        _sb_debug_log(f"ERROR: get_supabase_client failed in sb_save_user_state: {e}")
        return False

    table_from_secrets = (st.secrets.get("supabase", {}) or {}).get("user_state_table") or ""
    candidates = [t for t in [table_from_secrets, "user_state", "retirement_user_state"] if t]
    now_iso = datetime.now(timezone.utc).isoformat()

    for table in candidates:
        # Prefer modern schema (working_inputs)
        try:
            payload = {
                "user_id": user_id,
                "active_scenario_id": active_scenario_id,
                "working_inputs": working_inputs,
                "updated_at": now_iso,
            }
            client.table(table).upsert(payload, on_conflict="user_id").execute()
            _sb_debug_log(f"OK: sb_save_user_state upserted into '{table}'")
            sb_upsert_user_state_row(user_id=user_id, working_inputs=working_inputs, active_scenario_id=None)
            return True
        except Exception as e1:
            _sb_debug_log(f"WARN: sb_save_user_state failed on '{table}' (working_inputs): {e1}")
            # Try legacy schema with single_snapshot
            try:
                legacy_payload = {"user_id": user_id, "single_snapshot": working_inputs, "updated_at": now_iso}
                client.table(table).upsert(legacy_payload, on_conflict="user_id").execute()
                _sb_debug_log(f"OK: sb_save_user_state upserted legacy into '{table}'")
                sb_upsert_user_state_row(user_id=user_id, working_inputs=working_inputs, active_scenario_id=None)
                return True
            except Exception as e2:
                _sb_debug_log(f"WARN: sb_save_user_state failed on '{table}' (legacy): {e2}")
                continue

    return False

def sb_list_scenarios(user_id: str) -> list[dict]:
    """Load all saved scenarios for a user from Supabase (scenarios table)."""
    client = get_supabase_client()
    if client is None:
        return []

    try:
        resp = (
            client.table(_sb_scenarios_table())
            .select("id,name,inputs,updated_at,created_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
    except Exception as e:
        _sb_debug_log(f"ERROR: sb_list_scenarios failed: {e}")
        return []

    scenarios: list[dict] = []
    for r in rows:
        scenarios.append(
            {
                "id": str(r.get("id")),
                "name": r.get("name", "Scenario"),
                "inputs": normalize_snapshot(r.get("inputs") or {}),
                "results_df": None,
                "kpis": None,
            }
        )
    return scenarios


def sb_upsert_scenario(user_id: str, scenario: dict) -> bool:
    """Upsert a single scenario row."""
    client = get_supabase_client()
    if client is None:
        return False

    # Ensure UUID id
    sid = str(scenario.get("id") or uuid.uuid4())
    try:
        uuid.UUID(sid)
    except Exception:
        sid = str(uuid.uuid4())

    payload = {
        "id": sid,
        "user_id": user_id,
        "name": str(scenario.get("name") or "Scenario"),
        "inputs": normalize_snapshot(scenario.get("inputs") or {}),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        client.table(_sb_scenarios_table()).upsert(payload).execute()
        return True
    except Exception as e:
        _sb_debug_log(f"ERROR: sb_upsert_scenario failed: {e}")
        return False


def sb_delete_scenario(user_id: str, scenario_id: str) -> bool:
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table(_sb_scenarios_table()).delete().eq("user_id", user_id).eq("id", scenario_id).execute()
        return True
    except Exception as e:
        _sb_debug_log(f"ERROR: sb_delete_scenario failed: {e}")
        return False


def sb_sync_scenarios(user_id: str, scenarios: list[dict]) -> None:
    """Persist the entire scenario list (upsert all; delete removed)."""
    client = get_supabase_client()
    if client is None:
        return

    # Upsert all current scenarios
    current_ids: set[str] = set()
    for sc in scenarios:
        # Ensure uuid IDs; if not, replace in-memory as well so app stays consistent.
        sid = str(sc.get("id") or uuid.uuid4())
        try:
            uuid.UUID(sid)
        except Exception:
            sid = str(uuid.uuid4())
            sc["id"] = sid

        ok = sb_upsert_scenario(user_id, sc)
        if ok:
            current_ids.add(sid)

    # Delete removed scenarios (best-effort)
    try:
        resp = (
            client.table(_sb_scenarios_table())
            .select("id")
            .eq("user_id", user_id)
            .execute()
        )
        existing = {str(r.get("id")) for r in (getattr(resp, "data", None) or [])}
        to_delete = list(existing - current_ids)
        for sid in to_delete:
            sb_delete_scenario(user_id, sid)
    except Exception as e:
        _sb_debug_log(f"WARN: sb_sync_scenarios delete-sweep failed: {e}")

    except Exception:
        if single_snapshot is None:
            return False
        try:
            payload2 = {
                "user_id": str(user_id),
                "scenarios": _sb_json_sanitize(scenarios),
                "single_inputs": _sb_json_sanitize(single_snapshot),
            }
            sb.table(table).upsert(payload2, on_conflict="user_id").execute()
            return True
        except Exception:
            return False


def _ensure_user_state_loaded():
    """Load per-user state once per login (single-scenario working inputs + saved compare scenarios)."""
    user = st.session_state.get("current_user") or "default"

    if st.session_state.get("_sb_state_loaded") and st.session_state.get("_sb_loaded_user") == user:
        return

    # 1) Load compare scenarios from Supabase scenarios table
    _sc_loaded = sb_list_scenarios(user)
    # Store into both the global key (legacy) and the per-user scenario store key used by Compare tab
    st.session_state["scenarios"] = copy.deepcopy(_sc_loaded)
    st.session_state[_scenario_store_key()] = copy.deepcopy(_sc_loaded)

    # 2) Load single-scenario working inputs from the existing per-user table (retirement_user_state)
    user_state = sb_load_user_state(user) or {}
    single = user_state.get("working_inputs") or user_state.get("single_snapshot") or {}

    if single:
        percent_widget_keys = {
            "inflation_rate": (1.0, 5.0),
            "pre_retire_return": (1.0, 12.0),
            "post_retire_return": (1.0, 10.0),
            "cash_yield": (0.0, 8.0),
            "bonds_yield": (0.0, 10.0),
            "etfs_yield": (0.0, 12.0),
            "k401_yield": (0.0, 12.0),
        }

        for k, v in single.items():
            if k in percent_widget_keys:
                lo, hi = percent_widget_keys[k]
                pct = _clamp(float(v) * 100.0, lo, hi)  # decimals -> %
                if k not in st.session_state:
                    st.session_state[k] = pct
            else:
                if k not in st.session_state:
                    st.session_state[k] = v

    st.session_state["_sb_state_loaded"] = True
    st.session_state["_sb_loaded_user"] = user


def _maybe_persist_single_snapshot(snapshot: dict):
    """Persist the current Single Scenario inputs for the logged-in user.

    This is intentionally best-effort: persistence failures should never break the UI.
    """
    user = st.session_state.get("current_user")
    if not user:
        return
    try:
        snap = normalize_snapshot(snapshot)
        sb_save_user_state(
            user,
            working_inputs=snap,
            active_scenario_id=st.session_state.get("edit_scenario_id"),
        )
    except Exception:
        return
def _hash_password(password: str, salt: str) -> str:
    """
    PBKDF2 hash (basic gating). Store only the hash in st.secrets.
    """
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    )
    return dk.hex()



# -----------------------------------------------------------------------------
# SUPABASE USER CREDENTIALS + PASSWORD CHANGE LOG (optional)
# -----------------------------------------------------------------------------
def _sb_auth_table() -> str:
    return st.secrets.get("supabase", {}).get("auth_table", "app_user_credentials")

def _sb_password_changes_table() -> str:
    return st.secrets.get("supabase", {}).get("password_changes_table", "app_password_changes")

def _new_salt() -> str:
    """Generate a per-user salt for password hashing."""
    try:
        return secrets.token_hex(16)
    except Exception:
        return str(uuid.uuid4()).replace("-", "")

def _hash_password_v2(password: str, salt: str, iterations: int = 200_000) -> str:
    """PBKDF2-SHA256 hash used for Supabase-stored credentials."""
    try:
        it = int(iterations) if int(iterations) > 50_000 else 200_000
    except Exception:
        it = 200_000
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        (password or "").encode("utf-8"),
        (salt or "").encode("utf-8"),
        it,
    )
    return dk.hex()

def sb_get_user_credentials(user_id: str) -> dict | None:
    """Fetch user's credential row from Supabase. Returns None if missing/unavailable."""
    client = get_supabase_client()
    if client is None:
        return None
    try:
        res = (
            client.table(_sb_auth_table())
            .select("user_id,password_hash,salt,algo,iterations,updated_at,created_at,is_active")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        row = getattr(res, "data", None) or None
        return row if row else None
    except Exception as e:
        _sb_debug_log(f"WARN: sb_get_user_credentials failed: {e}")
        return None

def sb_verify_user_password(user_id: str, password: str) -> bool:
    row = sb_get_user_credentials(user_id)
    if not row or not row.get("password_hash") or not row.get("salt"):
        return False
    if row.get("is_active") is False:
        return False
    iterations = row.get("iterations") or 200_000
    computed = _hash_password_v2(password, row.get("salt"), iterations=int(iterations))
    try:
        return hmac.compare_digest(computed, str(row.get("password_hash")))
    except Exception:
        return False

def sb_set_user_password(
    user_id: str,
    new_password: str,
    old_password_hash: str | None = None,
    *,
    changed_by: str | None = None,
    change_reason: str = "user_initiated",
    metadata: dict | None = None,
) -> bool:
    """Upsert credentials row and append to password change log (best-effort)."""
    client = get_supabase_client()
    if client is None:
        return False

    salt = _new_salt()
    iterations = 200_000
    algo = "pbkdf2_sha256"
    new_hash = _hash_password_v2(new_password, salt, iterations=iterations)

    now_iso = datetime.now(timezone.utc).isoformat()
    cred_payload = {
        "user_id": user_id,
        "password_hash": new_hash,
        "salt": salt,
        "algo": algo,
        "iterations": iterations,
        "updated_at": now_iso,
        "is_active": True,
    }

    try:
        client.table(_sb_auth_table()).upsert(cred_payload, on_conflict="user_id").execute()
    except Exception as e:
        _sb_debug_log(f"ERROR: sb_set_user_password upsert failed: {e}")
        return False

    # Change log (non-fatal if logging fails)
    try:
        change_payload = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "old_password_hash": old_password_hash,
            "new_password_hash": new_hash,
            "changed_at": now_iso,
            "changed_by": changed_by or user_id,
            "change_reason": change_reason,
            "metadata": metadata or {},
        }
        client.table(_sb_password_changes_table()).insert(change_payload).execute()
    except Exception as e:
        _sb_debug_log(f"WARN: password change log insert failed (non-fatal): {e}")

    return True

def require_login():
    if st.session_state.get("is_authenticated", False):
        return

    st.title("Login Required")
    st.caption("Enter your credentials to access the application.")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")


    if submitted:
        # If Supabase credentials exist for this user, enforce Supabase auth.
        if _sb_enabled():
            sb_row = sb_get_user_credentials(username)
            if sb_row:
                if sb_verify_user_password(username, password):
                    st.session_state.is_authenticated = True
                    st.session_state.current_user = username
                    st.session_state.auth_source = "supabase"
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Invalid User ID or Password.")
                    st.stop()

        # Secrets-based fallback (backwards compatible)
        allowed_users = st.secrets.get("auth", {}).get("users", {})
        salt = st.secrets.get("auth", {}).get("salt", "")

        if not salt or not allowed_users:
            st.error("Auth is not configured. Please add [auth] secrets (salt + users).")
            st.stop()

        expected_hash = allowed_users.get(username)
        if not expected_hash:
            st.error("Invalid User ID or Password.")
            st.stop()

        computed_hash = _hash_password(password, salt)

        if hmac.compare_digest(computed_hash, expected_hash):
            st.session_state.is_authenticated = True
            st.session_state.current_user = username
            st.session_state.auth_source = "secrets"
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid User ID or Password.")
            st.stop()


    st.stop()


require_login()
_ensure_user_state_loaded()

with st.sidebar:
    _u = st.session_state.get("current_user") or ""
    if _u:
        st.markdown(f"**Welcome {_u}**")
    if st.button("Log out"):
        st.session_state.is_authenticated = False
        st.session_state.current_user = None
        # Clear per-user cached state so the next login reloads cleanly
        try:
            st.session_state.pop(_scenario_store_key(), None)
        except Exception:
            pass
        st.session_state["_sb_state_loaded"] = False
        st.session_state["_sb_loaded_user"] = None
        st.rerun()

# -----------------------------------------------------------------------------
# GLOBAL STYLE OVERRIDES (incl. legal badge)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* =========================
       MAIN CONTENT (NARROW)
       ========================= */
    .main .block-container {
        max-width: 960px;
        padding-left: 2rem;
        padding-right: 2rem;
        margin-left: auto;
        margin-right: auto;
    }

    @media (min-width: 1200px) {
        .main .block-container { max-width: 960px; }
    }

    /* =========================
       SIDEBAR (WIDER + CLEANER)
       ========================= */
    section[data-testid="stSidebar"] {
        width: 420px !important;
        min-width: 420px !important;
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }

    section[data-testid="stSidebar"] > div {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    [data-testid="stSidebar"] {
        font-size: 0.9rem !important;
    }

    /* =========================
       TYPOGRAPHY
       ========================= */
    h1 { font-size: 1.6rem !important; font-weight: 600 !important; }
    h2 { font-size: 1.25rem !important; margin-top: 1.2rem !important; margin-bottom: 0.4rem !important; }
    h3 { font-size: 1.05rem !important; margin-top: 0.8rem !important; }

    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #6B7280 !important; }

    .stMarkdown p { font-size: 0.9rem; line-height: 1.5; }

    /* =========================
       TOP-RIGHT LEGAL BADGE
       ========================= */
    div[data-testid="stAppViewContainer"]::before {
        content: "© 2026 Ranabir Bhakat™ · Proprietary & Confidential · Unauthorized use prohibited";
        position: fixed;
        top: 22px;           /* move down */
        right: 240px;        /* move left */
        z-index: 999999;
        padding: 6px 10px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.2px;
        color: rgba(255, 255, 255, 0.92);
        background: rgba(0, 0, 0, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        pointer-events: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# TAX LOGIC
# -----------------------------------------------------------------------------
FEDERAL_BRACKETS = {
    "single": [
        {"limit": 11_600, "rate": 0.10},
        {"limit": 47_150, "rate": 0.12},
        {"limit": 100_525, "rate": 0.22},
        {"limit": 191_950, "rate": 0.24},
        {"limit": 243_725, "rate": 0.32},
        {"limit": 609_350, "rate": 0.35},
        {"limit": float("inf"), "rate": 0.37},
    ],
    "married": [
        {"limit": 23_200, "rate": 0.10},
        {"limit": 94_300, "rate": 0.12},
        {"limit": 201_050, "rate": 0.22},
        {"limit": 383_900, "rate": 0.24},
        {"limit": 487_450, "rate": 0.32},
        {"limit": 731_200, "rate": 0.35},
        {"limit": float("inf"), "rate": 0.37},
    ],
}

NJ_BRACKETS = {
    "single": [
        {"limit": 20_000, "rate": 0.014},
        {"limit": 35_000, "rate": 0.0175},
        {"limit": 40_000, "rate": 0.035},
        {"limit": 75_000, "rate": 0.05525},
        {"limit": 500_000, "rate": 0.0637},
        {"limit": 1_000_000, "rate": 0.0897},
        {"limit": float("inf"), "rate": 0.1075},
    ],
    "married": [
        {"limit": 20_000, "rate": 0.014},
        {"limit": 50_000, "rate": 0.0175},
        {"limit": 70_000, "rate": 0.0245},
        {"limit": 80_000, "rate": 0.035},
        {"limit": 150_000, "rate": 0.05525},
        {"limit": 500_000, "rate": 0.0637},
        {"limit": 1_000_000, "rate": 0.0897},
        {"limit": float("inf"), "rate": 0.1075},
    ],
}


def calculate_progressive_tax(taxable_income: float, brackets) -> float:
    tax = 0.0
    previous_limit = 0.0
    for bracket in brackets:
        limit = bracket["limit"]
        rate = bracket["rate"]
        if taxable_income > previous_limit:
            taxable_amount = min(taxable_income, limit) - previous_limit
            tax += taxable_amount * rate
            previous_limit = limit
        else:
            break
    return tax


def calculate_annual_taxes(
    gross_income: float,
    status: str,
    state_code: str,
    manual_state_rate: float,
    dependents: int = 0,
):
    # Federal standard deduction (2024 approximation)
    standard_deduction = 14_600 if status == "single" else 29_200
    federal_taxable_income = max(0.0, gross_income - standard_deduction)

    federal_tax = calculate_progressive_tax(federal_taxable_income, FEDERAL_BRACKETS[status])

    # Child tax credit (approximate)
    credit_phase_out_start = 400_000 if status == "married" else 200_000
    total_credit = dependents * 2_000

    if gross_income > credit_phase_out_start:
        reduction = np.ceil((gross_income - credit_phase_out_start) / 1_000) * 50
        total_credit = max(0.0, total_credit - reduction)

    federal_tax = max(0.0, federal_tax - total_credit)

    # State tax
    if state_code == "NJ":
        nj_exempt = (dependents * 1_500) + (2_000 if status == "married" else 1_000)
        nj_taxable = max(0.0, gross_income - nj_exempt)
        state_tax = calculate_progressive_tax(nj_taxable, NJ_BRACKETS[status])
    else:
        state_tax = gross_income * (manual_state_rate / 100.0)

    total_tax = federal_tax + state_tax
    effective_rate = total_tax / gross_income if gross_income > 0 else 0.0

    return {
        "federal": federal_tax,
        "state": state_tax,
        "credits": total_credit,
        "total": total_tax,
        "effective_rate": effective_rate,
    }


# -----------------------------------------------------------------------------
# MULTI-ASSET FORECAST (your existing logic, unchanged)
# -----------------------------------------------------------------------------
def calculate_forecast_multi_asset(
    current_age: int,
    retire_age: int,
    life_expectancy: int,
    annual_spend_today: float,
    inflation_rate: float,
    ss_start_age: int,
    social_security_annual_today: float,
    annual_contribution: float,
    pre_retire_return: float,
    post_retire_return: float,
    cash_bal: float,
    bonds_bal: float,
    etfs_bal: float,
    k401_bal: float,
    real_estate_bal: float,
    misc_bal: float,
    cash_yield: float,
    bonds_yield: float,
    etfs_yield: float,
    k401_yield: float,
    real_estate_return: float,
    misc_return: float,
    flow_mode: str = "pro_rata",  # "pro_rata" or "cash_first"
):
    max_age = life_expectancy
    total_months = max(0, (max_age - current_age) * 12)
    retirement_month = max(0, (retire_age - current_age) * 12)

    m_infl = inflation_rate / 12.0
    m_cash = cash_yield / 12.0
    m_bonds = bonds_yield / 12.0
    m_etfs = etfs_yield / 12.0
    m_k401 = k401_yield / 12.0
    m_real_estate = real_estate_return / 12.0
    m_misc = misc_return / 12.0

    m_spend = annual_spend_today / 12.0
    m_ss = social_security_annual_today / 12.0
    m_contrib = annual_contribution / 12.0

    cash = float(max(0, cash_bal))
    bonds = float(max(0, bonds_bal))
    etfs = float(max(0, etfs_bal))
    k401 = float(max(0, k401_bal))
    real_estate = float(max(0, real_estate_bal))
    misc = float(max(0, misc_bal))

    def total_pool():
        return cash + bonds + etfs + real_estate + misc + k401

    def allocate_surplus(amount: float):
        nonlocal cash, bonds, etfs, real_estate, misc, k401
        if amount <= 0:
            return
        pool = total_pool()
        if pool <= 0:
            add = amount / 6.0
            cash += add
            bonds += add
            etfs += add
            real_estate += add
            misc += add
            k401 += add
            return

        if flow_mode == "pro_rata":
            cash += amount * (cash / pool) if cash > 0 else 0
            bonds += amount * (bonds / pool) if bonds > 0 else 0
            etfs += amount * (etfs / pool) if etfs > 0 else 0
            real_estate += amount * (real_estate / pool) if real_estate > 0 else 0
            misc += amount * (misc / pool) if misc > 0 else 0
            k401 += amount * (k401 / pool) if k401 > 0 else 0
        else:
            cash += amount

    def withdraw_deficit(amount: float):
        nonlocal cash, bonds, etfs, real_estate, misc, k401
        if amount <= 0:
            return

        if flow_mode == "cash_first":
            for name in ["cash", "bonds", "etfs", "real_estate", "misc", "k401"]:
                bal = {"cash": cash, "bonds": bonds, "etfs": etfs, "real_estate": real_estate, "misc": misc, "k401": k401}[name]
                if amount <= 0:
                    break
                take = min(amount, bal)
                amount -= take
                if name == "cash":
                    cash -= take
                if name == "bonds":
                    bonds -= take
                if name == "etfs":
                    etfs -= take
                if name == "real_estate":
                    real_estate -= take
                if name == "misc":
                    misc -= take
                if name == "k401":
                    k401 -= take
        else:
            pool = total_pool()
            if pool <= 0:
                return
            w = min(amount, pool)
            ratio = w / pool
            cash -= cash * ratio
            bonds -= bonds * ratio
            etfs -= etfs * ratio
            real_estate -= real_estate * ratio
            misc -= misc * ratio
            k401 -= k401 * ratio

        cash = max(0.0, cash)
        bonds = max(0.0, bonds)
        etfs = max(0.0, etfs)
        real_estate = max(0.0, real_estate)
        misc = max(0.0, misc)
        k401 = max(0.0, k401)

    rows = []
    rows.append(
        {
            "Age": current_age,
            "Is Retired": current_age >= retire_age,
            "Required Spend": annual_spend_today,
            "Guaranteed Income": 0.0 if current_age < ss_start_age else social_security_annual_today,
            "Portfolio Withdrawal": 0.0,
            "Cash": cash,
            "Bonds": bonds,
            "ETFs": etfs,
            "Real Estate": real_estate,
            "Misc": misc,
            "401k": k401,
            "End Balance": total_pool(),
        }
    )

    for month in range(1, total_months + 1):
        sim_age = current_age + month / 12.0
        age_int = int(np.floor(sim_age))
        is_retired = month >= retirement_month

        m_spend *= (1.0 + m_infl)
        m_ss *= (1.0 + m_infl)

        cash *= (1.0 + m_cash)
        bonds *= (1.0 + m_bonds)
        etfs *= (1.0 + m_etfs)
        real_estate *= (1.0 + m_real_estate)
        misc *= (1.0 + m_misc)
        k401 *= (1.0 + m_k401)

        guaranteed_month = m_ss if sim_age >= ss_start_age else 0.0

        if not is_retired and m_contrib > 0:
            allocate_surplus(m_contrib)

        monthly_need = 0.0
        if is_retired:
            monthly_need = max(0.0, m_spend - guaranteed_month)
            withdraw_deficit(monthly_need)

        if month % 12 == 0:
            rows.append(
                {
                    "Age": age_int,
                    "Is Retired": age_int >= retire_age,
                    "Required Spend": m_spend * 12.0,
                    "Guaranteed Income": guaranteed_month * 12.0,
                    "Portfolio Withdrawal": monthly_need * 12.0,
                    "Cash": cash,
            "Bonds": bonds,
            "ETFs": etfs,
            "Real Estate": real_estate,
            "Misc": misc,
            "401k": k401,
                    "End Balance": total_pool(),
                }
            )
        if total_pool() <= 0:
            # If depletion happens mid-year, capture a final row so KPIs can detect depletion age
            if month % 12 != 0:
                rows.append(
                    {
                        "Age": age_int,
                        "Is Retired": age_int >= retire_age,
                        "Required Spend": m_spend * 12.0,
                        "Guaranteed Income": guaranteed_month * 12.0,
                        "Portfolio Withdrawal": monthly_need * 12.0,
                        "Cash": cash,
            "Bonds": bonds,
            "ETFs": etfs,
            "Real Estate": real_estate,
            "Misc": misc,
            "401k": k401,
                        "End Balance": 0.0,
                    }
                )
            break

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# SCENARIO MANAGER (COMPARE)
# -----------------------------------------------------------------------------
def _scenario_store_key() -> str:
    # per-login separation in session_state
    user = st.session_state.get("current_user") or "default"
    return f"scenarios__{user}"


def _init_scenarios():
    key = _scenario_store_key()
    if key not in st.session_state:
        st.session_state[key] = []


def _get_scenarios():
    return st.session_state[_scenario_store_key()]


def _set_scenarios(scenarios):
    # Always store a deep copy for safety
    st.session_state[_scenario_store_key()] = copy.deepcopy(scenarios)

    # Persist compare scenarios to Supabase (per-user) - best effort
    user = st.session_state.get("current_user")
    if not user:
        return
    try:
        sb_sync_scenarios(user, st.session_state[_scenario_store_key()])
    except Exception:
        return
def get_current_inputs_snapshot() -> dict:
    """
    Snapshot current widgets via session_state keys.
    These keys are set on the sidebar inputs below.
    """
    return {
        "current_age": int(st.session_state.get("current_age", 50)),
        "retire_age": int(st.session_state.get("retire_age", 60)),
        "life_expectancy": int(st.session_state.get("life_expectancy", 95)),
        "current_portfolio": float(st.session_state.get("current_portfolio", 1_239_000)),
        "annual_contribution": float(st.session_state.get("annual_contribution", 65_000)),
        "annual_spend_retirement": float(st.session_state.get("annual_spend_retirement", 155_000)),
        "use_multi_asset": bool(st.session_state.get("use_multi_asset", True)),
        "cash_bal": float(st.session_state.get("cash_bal", 200_000)),
        "cash_yield": float(st.session_state.get("cash_yield", 0.04)),
        "bonds_bal": float(st.session_state.get("bonds_bal", 400_000)),
        "bonds_yield": float(st.session_state.get("bonds_yield", 0.05)),
        "etfs_bal": float(st.session_state.get("etfs_bal", 439_000)),
        "etfs_yield": float(st.session_state.get("etfs_yield", 0.07)),
        "k401_bal": float(st.session_state.get("k401_bal", 200_000)),
        "k401_yield": float(st.session_state.get("k401_yield", 0.07)),
        "real_estate_bal": float(st.session_state.get("real_estate_bal", 0.0)),
        "real_estate_return": float(st.session_state.get("real_estate_return", 0.04)),
        "misc_bal": float(st.session_state.get("misc_bal", 0.0)),
        "misc_return": float(st.session_state.get("misc_return", 0.05)),
        "annual_gross_income": float(st.session_state.get("annual_gross_income", 300_000)),
        "filing_status": st.session_state.get("filing_status", "married"),
        "state_code": st.session_state.get("state_code", "NJ"),
        "manual_state_rate": float(st.session_state.get("manual_state_rate", 0.0)),
        "dependents": int(st.session_state.get("dependents", 0)),
        "annual_expenses": float(st.session_state.get("annual_expenses", 200_000)),
        "inflation_rate": float(st.session_state.get("inflation_rate", 0.03)),
        "pre_retire_return": float(st.session_state.get("pre_retire_return", 0.07)),
        "post_retire_return": float(st.session_state.get("post_retire_return", 0.045)),
        "social_security": float(st.session_state.get("social_security", 30_000)),
        "ss_start_age": int(st.session_state.get("ss_start_age", 67)),
        "flow_mode": st.session_state.get("flow_mode", "cash_first"),
    }


import copy

def _as_decimal_rate(x, default=0.0):
    """
    Normalize any rate to decimal.
    - 0.07 stays 0.07
    - 7 becomes 0.07
    """
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v / 100.0 if v > 1.0 else v

def normalize_snapshot(s: dict) -> dict:
    """Return a NEW snapshot with consistent units (rates as decimals)."""
    s2 = copy.deepcopy(s)

    # Core rates
    s2["inflation_rate"] = _as_decimal_rate(s2.get("inflation_rate", 0.03), 0.03)
    s2["pre_retire_return"] = _as_decimal_rate(s2.get("pre_retire_return", 0.07), 0.07)
    s2["post_retire_return"] = _as_decimal_rate(s2.get("post_retire_return", 0.045), 0.045)

    # Multi-asset yields
    s2["cash_yield"] = _as_decimal_rate(s2.get("cash_yield", 0.04), 0.04)
    s2["bonds_yield"] = _as_decimal_rate(s2.get("bonds_yield", 0.05), 0.05)
    s2["etfs_yield"] = _as_decimal_rate(s2.get("etfs_yield", 0.07), 0.07)
    s2["k401_yield"] = _as_decimal_rate(s2.get("k401_yield", 0.07), 0.07)

    # Defaults / required keys
    s2["use_multi_asset"] = bool(s2.get("use_multi_asset", True))
    s2["flow_mode"] = s2.get("flow_mode", "cash_first")
    if s2["flow_mode"] not in ("cash_first", "pro_rata"):
        s2["flow_mode"] = "cash_first"

    # Ensure numeric types for critical fields (avoid strings)
    for k in ["current_age", "retire_age", "life_expectancy", "ss_start_age"]:
        if k in s2:
            s2[k] = int(s2[k])

    for k in ["annual_spend_retirement", "social_security", "annual_contribution", "current_portfolio",
              "cash_bal", "bonds_bal", "etfs_bal", "k401_bal"]:
        if k in s2 and s2[k] is not None:
            s2[k] = float(s2[k])

    return s2


def run_projection_from_snapshot(s: dict) -> pd.DataFrame:
    # --- CRITICAL: normalize units on every run ---
    s = normalize_snapshot(s)

    if s.get("use_multi_asset", True):
        return calculate_forecast_multi_asset(
            current_age=s["current_age"],
            retire_age=s["retire_age"],
            life_expectancy=s["life_expectancy"],
            annual_spend_today=s["annual_spend_retirement"],
            inflation_rate=s["inflation_rate"],
            ss_start_age=s["ss_start_age"],
            social_security_annual_today=s["social_security"],
            annual_contribution=s["annual_contribution"],
            pre_retire_return=s["pre_retire_return"],
            post_retire_return=s["post_retire_return"],
            cash_bal=s.get("cash_bal", 0.0),
            bonds_bal=s.get("bonds_bal", 0.0),
            etfs_bal=s.get("etfs_bal", 0.0),
            k401_bal=s.get("k401_bal", 0.0),
            real_estate_bal=s.get("real_estate_bal", 0.0),
            misc_bal=s.get("misc_bal", 0.0),
            cash_yield=s.get("cash_yield", 0.0),
            bonds_yield=s.get("bonds_yield", 0.0),
            etfs_yield=s.get("etfs_yield", 0.0),
            k401_yield=s.get("k401_yield", 0.0),
            real_estate_return=s.get("real_estate_return", 0.0),
            misc_return=s.get("misc_return", 0.0),
            flow_mode=s.get("flow_mode", "cash_first"),
        )

    # --- single-portfolio model ---
    years = range(s["current_age"], s["life_expectancy"] + 1)
    data = []
    portfolio = float(s.get("current_portfolio", 0.0))
    running_spend_needs = float(s["annual_spend_retirement"])

    for age in years:
        is_retired = age >= s["retire_age"]

        if age > s["current_age"]:
            running_spend_needs *= (1.0 + s["inflation_rate"])

        guaranteed_income = 0.0
        if age >= s["ss_start_age"]:
            guaranteed_income = float(s["social_security"]) * ((1.0 + s["inflation_rate"]) ** (age - s["current_age"]))

        flexible_income_needed = max(0.0, running_spend_needs - guaranteed_income) if is_retired else 0.0

        start_bal = portfolio
        growth_rate = s["post_retire_return"] if is_retired else s["pre_retire_return"]
        contribution = float(s.get("annual_contribution", 0.0)) if not is_retired else 0.0

        end_bal = (start_bal + contribution - flexible_income_needed) * (1.0 + growth_rate)
        end_bal = max(0.0, end_bal)

        data.append({
            "Age": age,
            "Is Retired": is_retired,
            "Portfolio Start": start_bal,
            "Required Spend": running_spend_needs,
            "Guaranteed Income": guaranteed_income,
            "Portfolio Withdrawal": flexible_income_needed,
            "End Balance": end_bal,
        })

        portfolio = end_bal

    return pd.DataFrame(data)


def scenario_kpis(df: pd.DataFrame, retire_age: int, current_age: int, life_expectancy: int) -> dict:
    last_row = df.iloc[-1]
    final_balance = float(last_row["End Balance"])

    retire_row = df[df["Age"] == retire_age]
    retire_row = retire_row.iloc[0] if not retire_row.empty else None

    assets_at_retirement = 0.0
    if retire_row is not None:
        assets_at_retirement = float(retire_row["Portfolio Start"]) if "Portfolio Start" in retire_row else float(
            retire_row["End Balance"]
        )

    depletion_rows = df[(df["End Balance"] <= 0) & (df["Age"] > current_age)]
    depletion_age = int(depletion_rows["Age"].min()) if not depletion_rows.empty else None

    retired_rows = df[df["Age"] >= retire_age]
    if not retired_rows.empty:
        first_ret = retired_rows.iloc[0]
        withdrawal = float(first_ret.get("Portfolio Withdrawal", 0.0))
        base = float(first_ret.get("Portfolio Start", first_ret.get("End Balance", 0.0)))
        wr = withdrawal / base if base > 0 else 0.0
    else:
        wr = 0.0

    sustainability = f"Depleted @ {depletion_age}" if depletion_age else f"Sustainable to {life_expectancy}"
    return {
        "Assets @ Retire": assets_at_retirement,
        "Final Balance": final_balance,
        "Depletion Age": depletion_age if depletion_age else "",
        "Withdrawal Rate (1st yr)": wr,
        "Sustainability": sustainability,
    }


# ---------------------------------------------------------------------------
# MONTE CARLO SIMULATION (OPT-IN; DOES NOT CHANGE DETERMINISTIC BEHAVIOR)
# ---------------------------------------------------------------------------

def retirement_confidence_score(df: pd.DataFrame, retire_age: int, current_age: int, life_expectancy: int) -> dict:
    """Heuristic 0–100 'confidence' score for a deterministic scenario.

    This is intentionally simple and explainable. It does NOT replace Monte Carlo.
    Inputs:
      - df: output of run_projection_from_snapshot (single scenario)
      - retire_age/current_age/life_expectancy: ages used for the scenario
    Returns:
      {score:int, label:str, notes:list[str], components:dict}
    """
    score = 100.0
    notes: list[str] = []
    components: dict = {}

    try:
        # Assets at retirement (start of retirement year if available)
        retire_row = df[df["Age"] == retire_age]
        retire_row = retire_row.iloc[0] if not retire_row.empty else None

        assets_at_retirement = 0.0
        if retire_row is not None:
            if "Portfolio Start" in retire_row:
                assets_at_retirement = float(retire_row["Portfolio Start"])
            else:
                assets_at_retirement = float(retire_row.get("End Balance", 0.0))
        components["assets_at_retirement"] = assets_at_retirement

        # Final balance at horizon
        last_row = df.iloc[-1]
        final_balance = float(last_row.get("End Balance", 0.0))
        components["final_balance"] = final_balance

        # Depletion age (if any)
        depletion_rows = df[(df.get("End Balance", 0) <= 0) & (df["Age"] > current_age)]
        depletion_age = int(depletion_rows["Age"].min()) if not depletion_rows.empty else None
        components["depletion_age"] = depletion_age

        # Withdrawal rate in the first retirement year (if available)
        wr = 0.0
        retired_rows = df[df["Age"] >= retire_age]
        if not retired_rows.empty:
            first_ret = retired_rows.iloc[0]
            withdrawal = float(first_ret.get("Portfolio Withdrawal", 0.0))
            base = float(first_ret.get("Portfolio Start", first_ret.get("End Balance", 0.0)))
            wr = (withdrawal / base) if base > 0 else 0.0
        components["withdrawal_rate"] = wr

        # -------------------------
        # Component 1: Sustainability to life expectancy (largest weight)
        # -------------------------
        if depletion_age is not None and depletion_age <= life_expectancy:
            years_short = max(0, int(life_expectancy - depletion_age + 1))
            # Strong penalty if the plan fails before horizon
            pen = min(85.0, years_short * 6.0)
            score -= pen
            notes.append(f"Plan depletes at age {depletion_age} (about {years_short} year(s) short of age {life_expectancy}).")
        else:
            notes.append(f"Plan sustains through age {life_expectancy} (no depletion projected).")

        # -------------------------
        # Component 2: First-year retirement withdrawal rate vs a 4% baseline
        # -------------------------
        # Baseline: 4% is 'typical' rule-of-thumb; <3.5% is conservative.
        if wr <= 0.035:
            score += 5.0
            notes.append(f"Withdrawal rate looks conservative (~{wr*100:.2f}% in first retirement year).")
        elif wr <= 0.045:
            notes.append(f"Withdrawal rate is near common baselines (~{wr*100:.2f}% in first retirement year).")
        else:
            pen = min(25.0, (wr - 0.045) * 500.0)  # ~5.5% => 5 pts; 7% => 12.5 pts
            score -= pen
            notes.append(f"Withdrawal rate is elevated (~{wr*100:.2f}% in first retirement year).")

        # -------------------------
        # Component 3: Ending balance buffer (relative to assets at retirement)
        # -------------------------
        if assets_at_retirement > 0:
            ratio = final_balance / assets_at_retirement
            components["ending_buffer_ratio"] = ratio

            if ratio <= 0.10:
                score -= 20.0
                notes.append("Ending balance buffer is thin (ending assets are <10% of assets at retirement).")
            elif ratio <= 0.25:
                score -= 12.0
                notes.append("Ending balance buffer is modest (ending assets are 10–25% of assets at retirement).")
            elif ratio >= 1.25:
                score += 5.0
                notes.append("Strong ending buffer (ending assets exceed assets at retirement).")

        # clamp
        score = max(0.0, min(100.0, score))

    except Exception:
        # Never break the app; provide a graceful fallback.
        score = 50.0
        notes = ["Unable to compute score due to an unexpected data issue."]

    # Labeling
    if score >= 80:
        label = "Strong"
    elif score >= 60:
        label = "Moderate"
    elif score >= 40:
        label = "At Risk"
    else:
        label = "Critical"

    return {
        "score": int(round(score)),
        "label": label,
        "notes": notes,
        "components": components,
    }



# -----------------------------------------------------------------------------
# PHASE 1 (2.B): "How much can I spend?" (Monte Carlo-backed, runs on demand)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _sustainable_spend_mc_cached(
    snap_norm: dict,
    target_success: float,
    n_sims: int,
    seed: int,
    pre_sigma: float,
    post_sigma: float,
    infl_sigma: float,
) -> dict:
    """Return sustainable annual retirement spend (today $) for a target success rate.

    - Uses the existing Monte Carlo engine (monte_carlo_projection_from_snapshot) with
      a small number of simulations for speed.
    - This function is cached to keep the UI responsive.
    """
    s0 = copy.deepcopy(snap_norm)

    # Determine a reasonable search band.
    # Lower bound: $0 (always feasible). Upper bound: max(3x planned spend, 10% of starting portfolio).
    planned = float(s0.get("annual_spend_retirement", 0.0))
    start_port = float(s0.get("current_portfolio", 0.0))

    lo = 0.0
    hi = max(planned * 3.0, start_port * 0.10, 50_000.0)

    # Evaluate feasibility at upper bound; if still succeeds, expand a bit (bounded)
    def _success_for_spend(sp: float) -> float:
        s = copy.deepcopy(s0)
        s["annual_spend_retirement"] = float(max(0.0, sp))
        res = monte_carlo_projection_from_snapshot(
            s,
            n_sims=int(n_sims),
            seed=int(seed),
            pre_sigma=float(pre_sigma),
            post_sigma=float(post_sigma),
            infl_sigma=float(infl_sigma),
        )
        prob_deplete = float(res.get("prob_deplete", 0.0))
        return 1.0 - prob_deplete

    succ_hi = _success_for_spend(hi)
    # If very safe even at hi, expand once (still bounded) to better bracket the max
    if succ_hi >= target_success and hi < 5_000_000:
        hi2 = min(5_000_000.0, hi * 1.5)
        succ_hi2 = _success_for_spend(hi2)
        if succ_hi2 >= target_success:
            hi = hi2
            succ_hi = succ_hi2

    # Binary search
    tol = 500.0  # $500/year tolerance
    best = lo
    best_succ = 1.0

    # If even hi fails, we will return a value below planned spend (binary search will find it)
    for _ in range(18):  # ~2^18 resolution is sufficient given tol + noisy MC
        mid = (lo + hi) / 2.0
        succ = _success_for_spend(mid)
        if succ >= target_success:
            best = mid
            best_succ = succ
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= tol:
            break

    # Round down to nearest $100 for nicer UX
    best = float(max(0.0, math.floor(best / 100.0) * 100.0))
    return {"spend": best, "success": float(best_succ), "target": float(target_success), "n_sims": int(n_sims)}


# -----------------------------------------------------------------------------
# PHASE 2 (2.E/2.F): Stress tests + Life Events helpers (cached runners)
# -----------------------------------------------------------------------------
def _safe_json_dumps(obj) -> str:
    """Stable JSON for caching keys (best-effort)."""
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        return json.dumps(str(obj), sort_keys=True)


@st.cache_data(show_spinner=False)
def _mc_cached_with_extensions(
    snap_norm: dict,
    n_sims: int,
    seed: int,
    pre_sigma: float,
    post_sigma: float,
    infl_sigma: float,
    life_events_json: str = "",
    stress_shocks_json: str = "",
) -> dict:
    """Cached Monte Carlo runner that supports Phase 2 extensions without changing baseline behavior."""
    life_events = json.loads(life_events_json) if (life_events_json or "") else None
    stress_shocks = json.loads(stress_shocks_json) if (stress_shocks_json or "") else None
    return monte_carlo_projection_from_snapshot(
        snap_norm,
        n_sims=int(n_sims),
        seed=int(seed),
        pre_sigma=float(pre_sigma),
        post_sigma=float(post_sigma),
        infl_sigma=float(infl_sigma),
        stress_shocks=stress_shocks,
    )


@st.cache_data(show_spinner=False)
def _sustainable_spend_mc_cached_with_extensions(
    snap_norm: dict,
    target_success: float,
    n_sims: int,
    seed: int,
    pre_sigma: float,
    post_sigma: float,
    infl_sigma: float,
    life_events_json: str = "",
    stress_shocks_json: str = "",
) -> dict:
    """Sustainable spend calculator, but under life events + stress shocks."""
    s0 = copy.deepcopy(snap_norm)
    life_events = json.loads(life_events_json) if (life_events_json or "") else None
    stress_shocks = json.loads(stress_shocks_json) if (stress_shocks_json or "") else None

    planned = float(s0.get("annual_spend_retirement", 0.0))
    start_port = float(s0.get("current_portfolio", 0.0))
    lo = 0.0
    hi = max(planned * 3.0, start_port * 0.10, 50_000.0)

    def _success_for_spend(sp: float) -> float:
        s = copy.deepcopy(s0)
        s["annual_spend_retirement"] = float(max(0.0, sp))
        res = monte_carlo_projection_from_snapshot(
            s,
            n_sims=int(n_sims),
            seed=int(seed),
            pre_sigma=float(pre_sigma),
            post_sigma=float(post_sigma),
            infl_sigma=float(infl_sigma),
            stress_shocks=stress_shocks,
        )
        prob_deplete = float(res.get("prob_deplete", 0.0))
        return 1.0 - prob_deplete

    succ_hi = _success_for_spend(hi)
    if succ_hi >= target_success and hi < 5_000_000:
        hi2 = min(5_000_000.0, hi * 1.5)
        succ_hi2 = _success_for_spend(hi2)
        if succ_hi2 >= target_success:
            hi = hi2
            succ_hi = succ_hi2

    tol = 500.0
    best = lo
    best_succ = 1.0
    for _ in range(18):
        mid = (lo + hi) / 2.0
        succ = _success_for_spend(mid)
        if succ >= target_success:
            best = mid
            best_succ = succ
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= tol:
            break

    best = float(max(0.0, math.floor(best / 100.0) * 100.0))
    return {"spend": best, "success": float(best_succ), "target": float(target_success), "n_sims": int(n_sims)}


def _build_stress_shocks_for_preset(snap_norm: dict, preset_id: str) -> tuple[dict, list[dict], str]:
    """Return (modified_snapshot, stress_shocks, label). Does not mutate input."""
    s = copy.deepcopy(snap_norm)
    current_age = int(s.get("current_age", 50))
    retire_age = int(s.get("retire_age", 65))
    life_expectancy = int(s.get("life_expectancy", 95))

    preset_id = (preset_id or "").strip().lower()
    shocks: list[dict] = []
    label = preset_id

    if preset_id == "retirement_crash":
        label = "Retirement-year crash"
        shocks.append({"type": "one_time_return", "age": retire_age, "shock_return": -0.30, "label": label})

    elif preset_id == "lost_decade":
        label = "Lost decade (low returns)"
        shocks.append(
            {
                "type": "return_regime",
                "start_age": retire_age,
                "end_age": retire_age + 9,
                "mu": 0.02,
                "sigma": 0.10,
                "label": label,
            }
        )

    elif preset_id == "high_inflation":
        label = "High inflation decade"
        shocks.append(
            {
                "type": "inflation_regime",
                "start_age": retire_age,
                "end_age": retire_age + 9,
                "mu": 0.06,
                "sigma": 0.015,
                "label": label,
            }
        )

    elif preset_id == "longevity_100":
        label = "Longevity shock (to age 100)"
        s["life_expectancy"] = int(max(life_expectancy, 100))

    elif preset_id == "healthcare_spike":
        label = "Healthcare cost spike"
        # Step-up spending starting at age 75 (or retirement if later)
        start = max(retire_age, 75)
        shocks.append(
            {
                "type": "spend_step",
                "start_age": start,
                "end_age": int(s.get("life_expectancy", life_expectancy)),
                "add_spend_today": 25_000.0,
                "label": label,
            }
        )

    return s, shocks, label


def _stress_test_recovery_time(baseline_res: dict, stressed_res: dict, shock_label: str) -> str:
    """Recovery time based on when stressed median (p50) reaches baseline median again."""
    try:
        ages = baseline_res.get("ages")
        b50 = np.asarray(baseline_res.get("p50"), dtype=float)
        s50 = np.asarray(stressed_res.get("p50"), dtype=float)
        if ages is None or len(ages) == 0:
            return "N/A"
        # Find first age where stressed median >= baseline median (within 2%)
        for i in range(len(ages)):
            if s50[i] >= 0.98 * b50[i]:
                return f"~{int(ages[i])} (age), {i} yr(s)"
        return "Not recovered by horizon"
    except Exception:
        return "N/A"


def _narrative_insights(baseline_res: dict, stress_rows: list[dict], life_events: list[dict]) -> dict:
    """Plain-language interpretation (interpretation, not advice)."""
    insights: list[str] = []
    risks: list[str] = []
    tradeoffs: list[str] = []

    try:
        base_prob_deplete = float(baseline_res.get("prob_deplete", 0.0))
        base_conf = 100.0 * (1.0 - base_prob_deplete)
        if base_prob_deplete <= 0.10:
            insights.append("Baseline simulation outcomes are generally resilient across typical market variability.")
        elif base_prob_deplete <= 0.25:
            insights.append("Baseline outcomes show meaningful variability; some paths deplete before the planning horizon.")
        else:
            insights.append("Baseline outcomes indicate elevated depletion risk in a material portion of simulated paths.")

        # Identify the biggest confidence drop from stress tests
        if stress_rows:
            worst = sorted(stress_rows, key=lambda r: float(r.get("confidence_delta", 0.0)))[:1]
            if worst:
                w = worst[0]
                risks.append(
                    f"The largest sensitivity comes from '{w.get('scenario')}', which reduces confidence by about {abs(float(w.get('confidence_delta', 0.0))):.0f} points."
                )

        if life_events:
            risks.append(
                "Life events add uncertainty because they introduce one-time or multi-year cashflow shocks that do not follow market cycles."
            )

        tradeoffs.append(
            "Higher planned spending increases depletion risk in adverse sequences, while lower spending tends to improve resilience but reduces near-term lifestyle headroom."
        )
        tradeoffs.append(
            "Stress scenarios primarily change outcomes by shifting early-retirement sequence risk (crash / lost decade) or by increasing long-run drag (inflation / healthcare / longevity)."
        )

        what_matters = []
        what_matters.append(f"Baseline confidence (success probability) is roughly {base_conf:.0f}/100.")
        if stress_rows:
            deltas = [float(r.get("confidence_delta", 0.0)) for r in stress_rows]
            what_matters.append(
                f"Across the selected stress tests, confidence deltas range from {min(deltas):.0f} to {max(deltas):.0f} points."
            )
        if life_events:
            what_matters.append(
                f"{len(life_events)} life event(s) are modeled as additional cashflows; their timing and size can dominate individual simulation paths."
            )

        return {
            "insights": insights,
            "risks": risks,
            "tradeoffs": tradeoffs,
            "what_matters_most": what_matters,
        }
    except Exception:
        return {
            "insights": ["Unable to generate narrative insights due to an internal error."],
            "risks": [],
            "tradeoffs": [],
            "what_matters_most": [],
        }

def _sim_return_draw(rng: np.random.Generator, mu: float, sigma: float, size: int) -> np.ndarray:
    """
    Draw arithmetic returns with a simple guardrail so we never go below -100%.
    """
    if sigma <= 0:
        return np.full(size, float(mu))
    r = rng.normal(loc=float(mu), scale=float(sigma), size=size)
    return np.clip(r, -0.999, None)



def _normalize_goals(goals, current_age:int, inflation_mu:float):
    """Normalize a user-provided goals list into a safe internal structure.

    Each goal is expected to be a dict-like:
      - title: str
      - target_age: int
      - amount: float (today's dollars)
      - inflate: bool (default True) -> amount grows at inflation until target_age
      - priority: int/float (optional)

    Returns list of dicts with keys: title, target_age, amount_today, inflate, priority
    """
    out = []
    if not goals:
        return out



def _normalize_life_events(raw_events, current_age: int, inflation_mu: float):
    """Normalize structured life events into a safe internal schema.

    Expected internal keys (per event):
      - title: str
      - start_age: int
      - duration_years: int (>=1)
      - amount_low_today: float (>=0)
      - amount_high_today: float (>= amount_low_today)
      - probability: float in [0,1]
      - inflate: bool (default True)
      - direction: 'expense' or 'income' (default 'expense')

    Notes:
      - This is strictly normalization/validation. No behavioral change occurs unless
        callers pass life_events into the MC engine.
    """
    out: list[dict] = []
    if not raw_events:
        return out

    events = raw_events
    # Support session_state dict wrapper patterns
    if isinstance(raw_events, dict) and "events" in raw_events and isinstance(raw_events["events"], list):
        events = raw_events["events"]

    if not isinstance(events, list):
        return out

    for ev in events:
        if not isinstance(ev, dict):
            continue
        try:
            title = str(ev.get("title") or ev.get("name") or ev.get("type") or "Life Event").strip()
            ev_type = str(ev.get("type") or "").strip().lower()

            start_age = int(ev.get("start_age", ev.get("age", current_age)))
            start_age = max(int(current_age), start_age)

            duration_years = int(ev.get("duration_years", ev.get("duration", 1)))
            duration_years = max(1, duration_years)

            # Accept multiple field spellings
            lo = ev.get("amount_low_today", ev.get("cost_low_today", ev.get("amount_low", ev.get("cost_low", 0.0))))
            hi = ev.get("amount_high_today", ev.get("cost_high_today", ev.get("amount_high", ev.get("cost_high", lo))))
            lo = float(lo) if lo is not None else 0.0
            hi = float(hi) if hi is not None else lo
            lo = max(0.0, lo)
            hi = max(lo, hi)

            prob = ev.get("probability", ev.get("prob", 1.0))
            prob = float(prob) if prob is not None else 1.0
            if math.isnan(prob) or math.isinf(prob):
                prob = 1.0
            prob = float(min(1.0, max(0.0, prob)))

            inflate = ev.get("inflate", True)
            inflate = bool(inflate)

            direction = str(ev.get("direction", "expense")).strip().lower()
            if direction not in ("expense", "income"):
                # Try to infer from sign if user stored signed amount
                signed_amt = float(ev.get("amount", 0.0) or 0.0)
                direction = "income" if signed_amt > 0 else "expense"

            out.append(
                {
                    "type": ev_type or "custom",
                    "title": title,
                    "start_age": int(start_age),
                    "duration_years": int(duration_years),
                    "amount_low_today": float(lo),
                    "amount_high_today": float(hi),
                    "probability": float(prob),
                    "inflate": bool(inflate),
                    "direction": direction,
                    # keep inflation_mu for downstream consumers if needed
                    "inflation_mu": float(inflation_mu),
                }
            )
        except Exception:
            continue

    return out
    for g in goals:
        try:
            title = str(g.get("title") or "Goal")
            target_age = int(g.get("target_age") or g.get("age") or 0)
            amount = float(g.get("amount") or g.get("amount_today") or 0.0)
            inflate = bool(g.get("inflate", True))
            priority = float(g.get("priority", 50))
        except Exception:
            continue
        if target_age <= 0 or amount <= 0:
            continue
        # Clamp ages to a reasonable band; ignore if wildly out of horizon.
        if target_age < current_age - 1 or target_age > current_age + 80:
            continue
        out.append(
            {
                "title": title,
                "target_age": target_age,
                "amount_today": max(0.0, amount),
                "inflate": inflate,
                "priority": priority,
            }
        )
    return out


def monte_carlo_projection_from_snapshot(
    s: dict,
    n_sims: int = 2000,
    n_trials: int | None = None,
    seed: int | None = None,
    pre_sigma: float = 0.15,
    post_sigma: float = 0.10,
    infl_sigma: float = 0.01,
    life_events: list[dict] | None = None,
    stress_shocks: list[dict] | None = None,
    # Optional behavior tweaks (opt-in; defaults preserve prior deterministic assumptions)
    use_spending_floor: bool = False,
    spending_floor_multiple: float = 18.0,   # if assets < multiple * current-year spend, reduce spend
    spending_floor_cut_pct: float = 0.10,    # 10% cut
    spending_floor_recover_multiple: float = 22.0,  # recover threshold to stop cutting
    use_guardrails: bool = False,
    guardrail_band_pct: float = 0.20,        # +/- 20% around initial withdrawal rate
    guardrail_cut_pct: float = 0.10,         # cut spend by 10% when above upper guardrail
    guardrail_raise_pct: float = 0.05,       # raise spend by 5% when below lower guardrail
    guardrail_raise_cap_pct: float = 0.15,   # cap raises above inflation-adjusted baseline by 15%
) -> dict:
    """Run Monte Carlo projections (annual model) off a normalized snapshot.

    Returns:
      - ages: list[int]
      - p10/p25/p50/p75/p90: percentile balances by age
      - prob_deplete: probability the portfolio hits 0 before life_expectancy
      - end_balances: list[float] ending balance at life_expectancy (or 0)
      - deplete_ages: list[int|None] depletion age per simulation
    """
    s = normalize_snapshot(s)

    # Optional one-time goals (e.g., college funding, home purchase, legacy gift)
    goals = _normalize_goals(s.get('goals', []), current_age=int(s['current_age']), inflation_mu=float(s['inflation_rate']))

    # Phase 2 (opt-in): structured life events + market stress shocks
    norm_life_events = _normalize_life_events(
        life_events,
        current_age=int(s['current_age']),
        inflation_mu=float(s['inflation_rate']),
    )
    norm_stress_shocks: list[dict] = [copy.deepcopy(x) for x in (stress_shocks or []) if isinstance(x, dict)]

    if n_trials is not None:
        n_sims = int(n_trials)

    rng = np.random.default_rng(seed)

    current_age = int(s["current_age"])
    retire_age = int(s["retire_age"])
    life_expectancy = int(s["life_expectancy"])
    ss_start_age = int(s["ss_start_age"])

    # Use single-portfolio balance for MC. If user is in multi-asset mode, treat total as sum.
    if bool(s.get("use_multi_asset", True)):
        start_portfolio = float(s.get("cash_bal", 0.0)) + float(s.get("bonds_bal", 0.0)) + float(s.get("etfs_bal", 0.0)) + float(s.get("k401_bal", 0.0))
        # Use weighted-average expected return for mu; keep user-provided pre/post as primary drivers for now.
        # (We still draw returns around pre/post means for consistency with deterministic mode.)
    else:
        start_portfolio = float(s.get("current_portfolio", 0.0))

    spend_today = float(s["annual_spend_retirement"])
    ss_today = float(s["social_security"])

    mu_infl = float(s["inflation_rate"])
    mu_pre = float(s["pre_retire_return"])
    mu_post = float(s["post_retire_return"])
    annual_contrib = float(s.get("annual_contribution", 0.0))

    ages = list(range(current_age, life_expectancy + 1))
    n_years = len(ages)

    # results matrix: sims x years
    balances = np.zeros((n_sims, n_years), dtype=float)
    end_balances = np.zeros(n_sims, dtype=float)
    deplete_ages: list[int | None] = [None] * n_sims

    for sim in range(n_sims):
        portfolio = max(0.0, start_portfolio)

        # --- Phase 2: life events realization for this simulation (deterministic per sim) ---
        sim_life_events: list[dict] = []
        if norm_life_events:
            for ev in norm_life_events:
                try:
                    if float(ev.get("probability", 1.0)) < 1.0:
                        if float(rng.random()) > float(ev.get("probability", 1.0)):
                            continue
                    lo = float(ev.get("amount_low_today", 0.0))
                    hi = float(ev.get("amount_high_today", lo))
                    amt_today = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
                    years_from_now = max(0, int(ev.get("start_age", current_age)) - int(current_age))
                    if bool(ev.get("inflate", True)):
                        amt = amt_today * ((1.0 + mu_infl) ** years_from_now)
                    else:
                        amt = amt_today
                    sign = 1.0 if str(ev.get("direction", "expense")).strip().lower() == "income" else -1.0
                    sim_life_events.append(
                        {
                            "title": str(ev.get("title", "Life Event")),
                            "start_age": int(ev.get("start_age", current_age)),
                            "end_age": int(ev.get("start_age", current_age)) + int(ev.get("duration_years", 1)) - 1,
                            "amount": float(sign * amt),
                        }
                    )
                except Exception:
                    continue

        # Track retirement spend target that is allowed to deviate under rules
        spend = spend_today
        baseline_spend = spend_today  # inflation-only baseline for "raise cap"

        initial_wr: float | None = None
        floor_active = False

        for i, age in enumerate(ages):
            is_retired = age >= retire_age

            # Inflation draw for the year (bounded to avoid absurd spikes in UI)
            mu_infl_use = mu_infl
            sigma_infl_use = infl_sigma
            if norm_stress_shocks:
                for sh in norm_stress_shocks:
                    try:
                        if str(sh.get("type")) == "inflation_regime":
                            a0 = int(sh.get("start_age", -9_999))
                            a1 = int(sh.get("end_age", -9_999))
                            if int(age) >= a0 and int(age) <= a1:
                                mu_infl_use = float(sh.get("mu", mu_infl_use))
                                sigma_infl_use = float(sh.get("sigma", sigma_infl_use))
                    except Exception:
                        continue

            infl = float(rng.normal(mu_infl_use, sigma_infl_use))
            infl = float(np.clip(infl, -0.01, 0.10))

            # Update baseline and the working spend (inflation is applied regardless)
            if age > current_age:
                baseline_spend *= (1.0 + infl)
                spend *= (1.0 + infl)

            # Guaranteed income (SS) with inflation from current age baseline for simplicity
            guaranteed = 0.0
            if age >= ss_start_age:
                # inflate from today using simulated inflation path via baseline inflation compounding approximation
                # (Using baseline_spend's implied inflation is acceptable for planning-grade MC)
                years_from_now = age - current_age
                guaranteed = ss_today * ((1.0 + mu_infl) ** years_from_now)

            withdrawal = 0.0
            contrib = 0.0

            # Extra spending step-ups (e.g., healthcare spike) are applied in retirement years only.
            extra_spend = 0.0
            if is_retired and norm_stress_shocks:
                for sh in norm_stress_shocks:
                    try:
                        if str(sh.get("type")) == "spend_step":
                            a0 = int(sh.get("start_age", -9_999))
                            a1 = int(sh.get("end_age", 9_999))
                            if int(age) >= a0 and int(age) <= a1:
                                add_today = float(sh.get("add_spend_today", 0.0))
                                years_from_now = max(0, int(age) - int(current_age))
                                extra_spend += add_today * ((1.0 + mu_infl) ** years_from_now)
                    except Exception:
                        continue

            if not is_retired:
                contrib = annual_contrib
            else:
                withdrawal = max(0.0, (spend + extra_spend) - guaranteed)

                # --- Guardrails (Guyton-Klinger style, simplified) ---
                if use_guardrails and portfolio > 0:
                    wr = withdrawal / portfolio
                    if initial_wr is None:
                        initial_wr = wr
                    else:
                        upper = initial_wr * (1.0 + guardrail_band_pct)
                        lower = initial_wr * (1.0 - guardrail_band_pct)

                        if wr > upper:
                            spend *= (1.0 - guardrail_cut_pct)
                            withdrawal = max(0.0, (spend + extra_spend) - guaranteed)
                        elif wr < lower:
                            # raise, but cap vs inflation-adjusted baseline
                            spend_candidate = spend * (1.0 + guardrail_raise_pct)
                            cap = baseline_spend * (1.0 + guardrail_raise_cap_pct)
                            spend = min(spend_candidate, cap)
                            withdrawal = max(0.0, (spend + extra_spend) - guaranteed)

                # --- Spending floor ("what-if spending reduction in bad paths") ---
                if use_spending_floor and portfolio > 0:
                    # activate cuts if assets are low relative to the current-year spend
                    if (not floor_active) and portfolio < (spending_floor_multiple * max(1.0, spend)):
                        floor_active = True
                    if floor_active:
                        spend *= (1.0 - spending_floor_cut_pct)
                        withdrawal = max(0.0, (spend + extra_spend) - guaranteed)
                        # deactivate once assets recover (hysteresis)
                        if portfolio > (spending_floor_recover_multiple * max(1.0, spend)):
                            floor_active = False

            # --- One-time goals spending at specific ages (treated as an extra withdrawal) ---
            goal_withdrawal = 0.0
            if goals:
                for g in goals:
                    if int(g["target_age"]) == int(age):
                        amt = float(g["amount_today"])
                        if g.get("inflate", True):
                            years_from_now = max(0, int(age) - int(current_age))
                            amt *= (1.0 + mu_infl) ** years_from_now
                        goal_withdrawal += max(0.0, amt)

            if goal_withdrawal > 0.0:
                withdrawal += goal_withdrawal

            # Phase 2 life events: apply net cashflow (income positive, expense negative)
            event_net = 0.0
            if sim_life_events:
                for ev in sim_life_events:
                    if int(ev["start_age"]) <= int(age) <= int(ev["end_age"]):
                        event_net += float(ev["amount"])

            # Apply net cashflows then return draw
            start_bal = portfolio
            portfolio = start_bal + contrib - withdrawal + event_net
            portfolio = max(0.0, portfolio)

            # Return draw
            if portfolio > 0:
                mu = mu_post if is_retired else mu_pre
                sigma = post_sigma if is_retired else pre_sigma

                # Stress regimes can override mu/sigma for a window
                if norm_stress_shocks:
                    for sh in norm_stress_shocks:
                        try:
                            if str(sh.get("type")) == "return_regime":
                                a0 = int(sh.get("start_age", -9_999))
                                a1 = int(sh.get("end_age", -9_999))
                                if int(age) >= a0 and int(age) <= a1:
                                    mu = float(sh.get("mu", mu))
                                    sigma = float(sh.get("sigma", sigma))
                        except Exception:
                            continue

                r = float(rng.normal(mu, sigma))
                # clamp to prevent extreme single-year blowups that dominate charts
                r = float(np.clip(r, -0.80, 0.80))

                # One-time return shocks (e.g., retirement-year crash)
                if norm_stress_shocks:
                    for sh in norm_stress_shocks:
                        try:
                            if str(sh.get("type")) == "one_time_return":
                                if int(sh.get("age", -9_999)) == int(age):
                                    shock_r = float(sh.get("shock_return", 0.0))
                                    # Combine multiplicatively to avoid impossible arithmetic < -100%
                                    r = (1.0 + r) * (1.0 + shock_r) - 1.0
                        except Exception:
                            continue

                portfolio *= (1.0 + r)

            portfolio = max(0.0, portfolio)
            balances[sim, i] = portfolio

            if portfolio <= 0.0 and deplete_ages[sim] is None and age >= retire_age:
                deplete_ages[sim] = age

            # early exit optimization
            if portfolio <= 0.0 and age >= retire_age:
                # fill remaining years with zeros
                if i < n_years - 1:
                    balances[sim, i + 1 :] = 0.0
                break

        end_balances[sim] = balances[sim, -1]

    # Percentiles by age
    p10 = np.percentile(balances, 10, axis=0)
    p25 = np.percentile(balances, 25, axis=0)
    p50 = np.percentile(balances, 50, axis=0)
    p75 = np.percentile(balances, 75, axis=0)
    p90 = np.percentile(balances, 90, axis=0)

    prob_deplete = float(np.mean([a is not None for a in deplete_ages]))

    # Summary stats at end of horizon (for top-line metrics)
    median_final = float(np.percentile(end_balances, 50))
    p10_final = float(np.percentile(end_balances, 10))
    p90_final = float(np.percentile(end_balances, 90))

    return {
        "ages": ages,
        "p10": p10,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "median_final": median_final,
        "p10_final": p10_final,
        "p90_final": p90_final,
        "prob_deplete": prob_deplete,
        "end_balances": end_balances,
        "final_balances": end_balances,
        "typical_deplete_age": (int(np.median([a for a in deplete_ages if a is not None])) if any(a is not None for a in deplete_ages) else None),
        "deplete_ages": deplete_ages,
    }

_init_scenarios()


# -----------------------------------------------------------------------------
# SIDEBAR WIDGET HELPERS (avoid Session State default conflicts)
# -----------------------------------------------------------------------------
def sb_num(label: str, key: str, default, min_value=None, max_value=None, step=None, **kwargs):
    """Number input that avoids Streamlit's 'default + session_state' warning.
    We seed st.session_state[key] only if missing, and we do NOT pass an explicit
    'value' argument to the widget.
    """
    if key not in st.session_state:
        st.session_state[key] = default

    params = {}
    if min_value is not None:
        params["min_value"] = min_value
    if max_value is not None:
        params["max_value"] = max_value
    if step is not None:
        params["step"] = step
    params.update(kwargs)

    return st.sidebar.number_input(label, key=key, **params)


def sb_checkbox(label: str, key: str, default: bool = False, **kwargs):
    """Checkbox that avoids Streamlit's 'default + session_state' warning.
    Seeds st.session_state[key] only if missing, and does NOT pass an explicit 'value'.
    """
    if key not in st.session_state:
        st.session_state[key] = bool(default)
    return st.sidebar.checkbox(label, key=key, **kwargs)


def sb_slider(label: str, min_value, max_value, key: str, default=None, **kwargs):
    """Slider wrapper to avoid Streamlit's 'default + session_state' warning.

    Rule:
    - If the key is already present in st.session_state (e.g., loaded from Supabase),
      call the slider WITHOUT passing an explicit 'value' (Streamlit will use session_state).
    - If the key is missing, pass the intended default via 'value' and let Streamlit
      initialize session_state (do NOT set st.session_state[key] manually here).
    """
    if key not in st.session_state:
        v = default if default is not None else min_value
        return st.sidebar.slider(label, min_value, max_value, value=v, key=key, **kwargs)
    return st.sidebar.slider(label, min_value, max_value, key=key, **kwargs)



def sb_selectbox(label: str, options, key: str, default_index: int = 0, **kwargs):
    """Selectbox wrapper to avoid Streamlit's 'default + session_state' warning."""
    if key not in st.session_state:
        # Seed session state with the intended default
        try:
            st.session_state[key] = options[default_index]
        except Exception:
            st.session_state[key] = options[0] if options else None
        return st.sidebar.selectbox(label, options=options, index=default_index, key=key, **kwargs)
    return st.sidebar.selectbox(label, options=options, key=key, **kwargs)


# -----------------------------------------------------------------------------
# TITLE & INTRO
# -----------------------------------------------------------------------------
st.title("Strategic Retirement Planner: Cashflow & Buckets")
st.markdown(
    "Use this tool to test retirement readiness with **FIRE rules of thumb**, "
    "**cashflow projections**, and a **3-bucket investment framework**."
)
st.markdown("---")

# -----------------------------------------------------------------------------
# TABS: SINGLE vs COMPARE
# -----------------------------------------------------------------------------
# UI: Make top tab bar horizontally scrollable (prevents overflow when tabs grow)
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
/* Streamlit tabs horizontal scroll */
div[data-baseweb="tab-list"] {
  overflow-x: auto;
  overflow-y: hidden;
  white-space: nowrap;
  scrollbar-width: thin;
}
div[data-baseweb="tab-list"] > button {
  flex: 0 0 auto;
}
</style>
""",
    unsafe_allow_html=True,
)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Single Scenario",
        "Compare Scenarios",
        "Range of Outcomes (Simulation)",
        "Market Reality Mode (Stress Tests)",
        "One-Time Retirement Readiness Report",
    ]
)

# =============================================================================
# TAB 1: SINGLE SCENARIO (your app, minimally modified for keyed widgets)
# =============================================================================
with tab1:
    # ---------------------------
    # SIDEBAR INPUTS (KEYED)
    # ---------------------------
    st.sidebar.header("1. Demographics & Status")
    current_age = sb_num("Current Age", key="current_age", default=50, min_value=35, max_value=90, step=1)
    retire_age = sb_num("Retirement Age", key="retire_age", default=60, min_value=35, max_value=90, step=1)
    life_expectancy = sb_num("Life Expectancy", key="life_expectancy", default=95, min_value=70, max_value=110, step=1)

    st.sidebar.header("2. Financials (Current)")
    current_portfolio = sb_num("Total Invested Assets ($)", key="current_portfolio", default=1_239_000.0, min_value=0.0, step=10_000.0)
    annual_contribution = sb_num("Annual Contribution until Retirement ($)", key="annual_contribution", default=65_000.0, min_value=0.0, step=1_000.0)
    annual_spend_retirement = sb_num("Desired Annual Spend in Retirement (Today's $)", key="annual_spend_retirement", default=155_000.0, min_value=0.0, step=1_000.0)

    st.sidebar.header("2B. Portfolio Composition (Optional Multi-Asset)")
    use_multi_asset = sb_checkbox(
        "Use Multi-Asset Portfolio (Cash/Bonds/ETFs/401k)",
        key="use_multi_asset",
        default=True,
        help="When enabled, the model tracks each bucket separately and shows a stacked chart.",
    )

    flow_mode = "cash_first"
    with st.sidebar:
        flow_mode = sb_selectbox("Withdrawal Mode", options=["cash_first", "pro_rata"], default_index=0, key="flow_mode",
            help="cash_first withdraws from Cash→Bonds→ETFs→401k. pro_rata withdraws proportionally.",
            )

    if use_multi_asset:
        st.sidebar.caption("Balances should roughly sum to Total Invested Assets (above). Yields are annual %.")

        cash_bal = sb_num("Cash Balance ($)", key="cash_bal", default=200_000.0, min_value=0.0, step=10_000.0)
        cash_yield = sb_slider("Cash Yield (%)", 0.0, 8.0, key="cash_yield", default=4.0, step=0.1) / 100

        bonds_bal = sb_num("Bonds/Munis Balance ($)", key="bonds_bal", default=400_000.0, min_value=0.0, step=10_000.0)
        bonds_yield = sb_slider("Bonds Yield (%)", 0.0, 10.0, key="bonds_yield", default=5.0, step=0.1) / 100

        etfs_bal = sb_num("ETFs Balance ($)", key="etfs_bal", default=439_000.0, min_value=0.0, step=10_000.0)
        etfs_yield = sb_slider("ETFs Return (%)", 0.0, 12.0, key="etfs_yield", default=7.0, step=0.1) / 100

        k401_bal = sb_num("401k Balance ($)", key="k401_bal", default=200_000.0, min_value=0.0, step=10_000.0)
        k401_yield = sb_slider("401k Return (%)", 0.0, 12.0, key="k401_yield", default=7.0, step=0.1) / 100


        real_estate_bal = sb_num("Real Estate Balance ($)", key="real_estate_bal", default=0.0, min_value=0.0, step=10_000.0)
        real_estate_return = sb_slider("Real Estate Return (%)", 0.0, 12.0, key="real_estate_return", default=4.0, step=0.1) / 100

        misc_bal = sb_num("Misc. Assets Balance ($) (Intl, Alts, etc.)", key="misc_bal", default=0.0, min_value=0.0, step=10_000.0)
        misc_return = sb_slider("Misc. Assets Return (%)", 0.0, 12.0, key="misc_return", default=5.0, step=0.1) / 100
        buckets_sum = cash_bal + bonds_bal + etfs_bal + k401_bal + real_estate_bal + misc_bal
        if abs(buckets_sum - current_portfolio) > 50_000:
            st.sidebar.warning(
                f"Bucket sum (${buckets_sum:,.0f}) differs from Total Invested Assets (${current_portfolio:,.0f}). "
                "This is OK for experimentation, but totals may look inconsistent."
            )
    else:
        # define placeholders so code below doesn't NameError
        cash_bal = bonds_bal = etfs_bal = k401_bal = 0.0
        cash_yield = bonds_yield = etfs_yield = k401_yield = 0.0

    st.sidebar.header("3. Tax Profile (Current Income)")
    annual_gross_income = sb_num("Annual Gross Income (Pre-Tax $)", key="annual_gross_income", default=300_000.0, min_value=0.0, step=1_000.0)

    filing_status = sb_selectbox("Filing Status",
        options=["single", "married"],
        format_func=lambda x: "Single" if x == "single" else "Married Filing Jointly",
        key="filing_status")

    state_code = sb_selectbox("State", options=["NJ", "Other"], default_index=0, key="state_code")

    manual_state_rate = 0.0
    if state_code == "Other":
        manual_state_rate = sb_slider("Other State Effective Tax Rate (%)", 0.0, 15.0, key="manual_state_rate", default=5.0, step=0.5)
    else:
        # ensure key exists
        st.session_state["manual_state_rate"] = 0.0

    dependents = sb_num("Number of Dependents", key="dependents", default=0, min_value=0, max_value=10, step=1)

    st.sidebar.header("4. Household Expenses & Cashflow")
    annual_expenses = sb_num("Annual Expenses (Today's $)", key="annual_expenses", default=200_000, min_value=0, max_value=10_000_000, step=1000)

    st.sidebar.header("5. Macro & Return Assumptions")
    inflation_rate = sb_slider("Inflation Rate (%)", 1.0, 5.0, key="inflation_rate", default=3.0, step=0.1) / 100
    pre_retire_return = sb_slider("Pre-Retirement Growth (%)", 1.0, 12.0, key="pre_retire_return", default=7.0, step=0.1) / 100
    post_retire_return = sb_slider("Post-Retirement Growth (Avg) (%)", 1.0, 10.0, key="post_retire_return", default=4.5, step=0.1) / 100

    st.sidebar.header("6. Guaranteed Income (Retirement)")
    social_security = sb_num("Social Security/Pension (Annual $)", key="social_security", default=30_000, min_value=0, max_value=1_000_000, step=1000)
    ss_start_age = sb_num("SS/Pension Start Age", key="ss_start_age", default=67, min_value=60, max_value=75, step=1)

    # Persist current single-scenario inputs per user (best-effort)
    _maybe_persist_single_snapshot(normalize_snapshot(get_current_inputs_snapshot()))

    # ---------------------------
    # TAX SNAPSHOT & HOUSEHOLD SURPLUS
    # ---------------------------
    tax_info = calculate_annual_taxes(
        gross_income=annual_gross_income,
        status=filing_status,
        state_code=state_code,
        manual_state_rate=float(st.session_state.get("manual_state_rate", manual_state_rate)),
        dependents=dependents,
    )
    effective_tax_rate = tax_info["effective_rate"]
    net_take_home = annual_gross_income - tax_info["total"]
    surplus = net_take_home - annual_expenses

    st.subheader("Tax & Cashflow Snapshot")
    col_tx1, col_tx2, col_tx3, col_tx4 = st.columns(4)
    with col_tx1:
        st.metric("Gross Income", f"${annual_gross_income:,.0f}")
    with col_tx2:
        st.metric("Total Tax", f"${tax_info['total']:,.0f}")
    with col_tx3:
        st.metric("Effective Tax Rate", f"{effective_tax_rate * 100:,.1f}%")
    with col_tx4:
        st.metric("Net Take-Home Income", f"${net_take_home:,.0f}")

    st.markdown("")
    col_cash1, col_cash2 = st.columns([2, 1])
    with col_cash1:
        st.markdown("#### Tax Breakdown")
        st.markdown(
            f"- **Federal Tax (est.):** ${tax_info['federal']:,.0f}  \n"
            f"- **State Tax ({state_code}):** ${tax_info['state']:,.0f}"
        )
        if tax_info["credits"] > 0:
            st.markdown(f"- **Child Tax Credits (approx.):** ${tax_info['credits']:,.0f}")

    with col_cash2:
        st.markdown("#### Net Surplus View")
        st.metric("Annual Expenses", f"${annual_expenses:,.0f}")
        st.metric("Net Surplus (Saved)", f"${surplus:,.0f}", delta=None)

    st.caption(
        "Tax and cashflow snapshot is based on current gross income, filing status, state, dependents, "
        "and self-reported annual expenses. It is an approximation for planning, not a filing calculation."
    )
    st.markdown("---")

    # ---------------------------
    # CORE RETIREMENT CALCULATIONS
    # ---------------------------
    if use_multi_asset:
        df = calculate_forecast_multi_asset(
            current_age=current_age,
            retire_age=retire_age,
            life_expectancy=life_expectancy,
            annual_spend_today=annual_spend_retirement,
            inflation_rate=inflation_rate,
            ss_start_age=ss_start_age,
            social_security_annual_today=social_security,
            annual_contribution=annual_contribution,
            pre_retire_return=pre_retire_return,
            post_retire_return=post_retire_return,
            cash_bal=cash_bal,
            bonds_bal=bonds_bal,
            etfs_bal=etfs_bal,
            k401_bal=k401_bal,
            real_estate_bal=real_estate_bal,
            misc_bal=misc_bal,
            cash_yield=cash_yield,
            bonds_yield=bonds_yield,
            etfs_yield=etfs_yield,
            k401_yield=k401_yield,
            real_estate_return=real_estate_return,
            misc_return=misc_return,
            flow_mode=flow_mode,
        )
    else:
        years = range(current_age, life_expectancy + 1)
        data = []
        portfolio = current_portfolio
        running_spend_needs = annual_spend_retirement

        for age in years:
            is_retired = age >= retire_age
            if age > current_age:
                running_spend_needs *= (1 + inflation_rate)

            guaranteed_income = 0.0
            if age >= ss_start_age:
                guaranteed_income = social_security * ((1 + inflation_rate) ** (age - current_age))

            flexible_income_needed = max(0, running_spend_needs - guaranteed_income) if is_retired else 0

            start_bal = portfolio
            growth_rate = post_retire_return if is_retired else pre_retire_return
            contribution = annual_contribution if not is_retired else 0

            end_bal = (start_bal + contribution - flexible_income_needed) * (1 + growth_rate)
            end_bal = max(0, end_bal)

            data.append(
                {
                    "Age": age,
                    "Is Retired": is_retired,
                    "Portfolio Start": start_bal,
                    "Required Spend": running_spend_needs,
                    "Guaranteed Income": guaranteed_income,
                    "Portfolio Withdrawal": flexible_income_needed,
                    "End Balance": end_bal,
                }
            )
            portfolio = end_bal

        df = pd.DataFrame(data)


    # ---------------------------

    with st.expander("FIRE Targets (Rule of Thumb)", expanded=True):
        # ---------------------------
        # SECTION 1: FIRE OVERVIEW
        # ---------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Standard FIRE (25x)", f"${annual_spend_retirement * 25:,.0f}", help="Target based on ~4% withdrawal rate.")
        with col2:
            st.metric("Fat/Safe FIRE (33x)", f"${annual_spend_retirement * 33:,.0f}", help="More conservative target based on ~3% withdrawal rate.")
        with col3:
            gap_25x = (annual_spend_retirement * 25) - current_portfolio
            st.metric("Gap to 25x", f"${gap_25x:,.0f}", help="Difference between your current portfolio and 25x spending target. Negative gap indicates you have exceeded 25x.")

        st.caption("These rules of thumb provide a quick readiness check before looking at detailed cashflow modeling.")
        st.markdown("---")


    with st.expander("Stress Test: Capacity for Loss", expanded=False):
        # ---------------------------
        # SECTION 4: STRESS TEST
        # ---------------------------
        st.markdown("Simulate an immediate market shock to understand downside resilience.")

        crash_scenario = st.slider("Simulated Market Drop at Retirement (%)", 0, 50, 20, key="crash_scenario")

        if crash_scenario > 0:
            stressed_pot = current_portfolio * (1 - (crash_scenario / 100))
            st.write(f"Portfolio immediately after crash: **${stressed_pot:,.0f}**")

            if stressed_pot > (annual_spend_retirement * 25):
                st.success(
                    "Even after this shock, the portfolio remains above the standard **25x FIRE** threshold. "
                    "You retain a reasonable margin of safety under current assumptions."
                )
            else:
                st.warning(
                    "This shock brings the portfolio **below** the 25x FIRE threshold. "
                    "You may need to revisit spending, retirement age, or risk assumptions."
                )

        st.caption("This is a simple single-period stress test. In practice, you would combine this with scenario analysis and more detailed risk modeling.")


    with st.expander("Retirement Confidence Score", expanded=True):
        # ---------------------------
        # SECTION 2: Retirement Confidence Score (heuristic)
        # ---------------------------
        try:
            _rcs = retirement_confidence_score(
                df,
                retire_age=retire_age,
                current_age=current_age,
                life_expectancy=life_expectancy,
            )
            _score = int(_rcs.get("score", 0))
            _label = str(_rcs.get("label", ""))
            _notes = _rcs.get("notes", []) or []
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Score (0–100)", f"{_score}", help="Heuristic summary of sustainability based on depletion timing and ending balance buffer. For probabilistic confidence, use Monte Carlo.")
                st.progress(max(0.0, min(1.0, _score / 100.0)))
                st.caption(f"Status: {_label}")
            with c2:
                if isinstance(_notes, (list, tuple)) and _notes:
                    st.markdown("**Key drivers**")
                    for _n in _notes[:4]:
                        st.write(f"• {_n}")
                else:
                    st.write("")

            st.markdown("---")
        except Exception:
            # Never break existing rendering
            pass



        # ---------------------------
        # PHASE 1 (2.B): How much can I spend? (Monte Carlo-backed, on-demand)
        # ---------------------------

    with st.expander("How much can I spend in retirement? (Sustainable spending)", expanded=False):
        st.caption(
            "This tool estimates the *maximum* sustainable annual retirement spending (in today's dollars) for different confidence levels. "
            "It uses the same Monte Carlo engine as the Simulation tab, but runs a smaller number of simulations for speed."
        )

        csp1, csp2, csp3 = st.columns([1, 1, 2])
        with csp1:
            spend_simulations = st.number_input(
                "Simulations (speed vs accuracy)",
                min_value=300, max_value=6000, value=900, step=100,
                help="More simulations increase stability but take longer to compute.",
                key="spend_calc_n_sims",
            )
        with csp2:
            spend_seed = st.number_input(
                "Random seed",
                min_value=0, max_value=1_000_000, value=42, step=1,
                key="spend_calc_seed",
            )
        with csp3:
            st.markdown("**Confidence levels**")
            st.write("• Conservative: 90% success")
            st.write("• Balanced: 80% success")
            st.write("• Aggressive: 70% success")

        run_spend = st.button("Compute sustainable spending ranges", use_container_width=True, key="run_spend_calc")

        if run_spend:
            try:
                with st.spinner("Running simulations to estimate sustainable spending..."):
                    snap_now = normalize_snapshot(get_current_inputs_snapshot())

                    # Use the same default volatilities as the Monte Carlo tab defaults.
                    pre_sigma = 0.12
                    post_sigma = 0.09
                    infl_sigma = 0.01

                    targets = [("Conservative (90%)", 0.90), ("Balanced (80%)", 0.80), ("Aggressive (70%)", 0.70)]
                    results = []
                    for label, tgt in targets:
                        out = _sustainable_spend_mc_cached(
                            snap_now,
                            target_success=float(tgt),
                            n_sims=int(spend_simulations),
                            seed=int(spend_seed),
                            pre_sigma=float(pre_sigma),
                            post_sigma=float(post_sigma),
                            infl_sigma=float(infl_sigma),
                        )
                        results.append((label, out))

                    st.session_state["spend_calc_results"] = results

            except Exception as e:
                st.warning(f"Unable to compute sustainable spending due to an internal error: {e}")

        results = st.session_state.get("spend_calc_results", None)
        if results:
            st.markdown("##### Estimated sustainable annual retirement spending (today's dollars)")
            cols = st.columns(3)
            for i, (label, out) in enumerate(results[:3]):
                with cols[i]:
                    st.metric(
                        label,
                        _money(out.get("spend", 0.0)),
                        help=f"Target success rate: {int(float(out.get('target', 0.0))*100)}% using {int(out.get('n_sims', 0))} simulations.",
                    )
            st.caption(
                "Interpretation: A higher confidence level (e.g., 90%) is more conservative. "
                "Results are estimates and may vary slightly between runs due to randomness."
            )

            # Show how current planned spending compares (if available)
            try:
                planned_spend = float(annual_spend_retirement)
                best_balanced = float(results[1][1].get("spend", 0.0)) if len(results) > 1 else None
                if best_balanced is not None and planned_spend > 0:
                    delta = best_balanced - planned_spend
                    if abs(delta) < 500:
                        st.success("Your planned retirement spending is approximately aligned with the Balanced (80%) sustainable estimate.")
                    elif delta >= 0:
                        st.info(f"At the Balanced (80%) level, you may be able to increase spending by about {_money(delta)} per year (planning estimate).")
                    else:
                        st.warning(f"At the Balanced (80%) level, you may need to reduce spending by about {_money(abs(delta))} per year (planning estimate).")
            except Exception:
                pass



    with st.expander("Dynamic retirement date finder (retirement-age sensitivity)", expanded=False):
        # PHASE 1 (2.D): Dynamic Retirement Date Finder (on-demand)
        # ---------------------------
        st.caption(
            "This tool varies your retirement age (keeping all other inputs the same) and estimates the success probability (1 − probability of depletion) for each age. "
            "Use it to identify an earliest viable retirement age and a more comfortable age with additional cushion."
        )
        cdf1, cdf2, cdf3, cdf4 = st.columns([1.2, 1.2, 1, 1])
        with cdf1:
            target_success = st.slider("Target success probability", 0.60, 0.95, 0.80, 0.01, key="retwin_target_success")
        with cdf2:
            cushion = st.slider("Comfort cushion (+%)", 0.00, 0.20, 0.10, 0.01, key="retwin_cushion")
        with cdf3:
            win_sims = st.number_input("Simulations", min_value=300, max_value=6000, value=700, step=100, key="retwin_sims")
        with cdf4:
            win_seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1, key="retwin_seed")

        sra1, sra2 = st.columns(2)
        with sra1:
            search_start_age = st.number_input("Search start age", min_value=0, max_value=120, value=int(max(current_age, retire_age - 5)), step=1, key="retwin_start_age")
        with sra2:
            search_end_age = st.number_input("Search end age", min_value=0, max_value=120, value=int(retire_age + 10), step=1, key="retwin_end_age")

        def _success_probability_for_retire_age(snap: dict, test_retire_age: int, n_sims: int, seed: int) -> float:
            snap2 = dict(snap)
            snap2["retire_age"] = int(test_retire_age)
            # Reuse the same MC engine used elsewhere
            res = monte_carlo_projection_from_snapshot(
                snap2,
                n_sims=int(n_sims),
                seed=int(seed),
                pre_sigma=0.12,
                post_sigma=0.09,
                infl_sigma=0.01,
            )
            p_deplete = float(res.get("prob_deplete", 1.0))
            return max(0.0, min(1.0, 1.0 - p_deplete))

        @st.cache_data(show_spinner=False, ttl=3600)
        def _retwin_scan_cached(snap: dict, start_age: int, end_age: int, n_sims: int, seed: int):
            ages = list(range(int(start_age), int(end_age) + 1))
            vals = []
            for a in ages:
                vals.append(_success_probability_for_retire_age(snap, a, n_sims, seed))
            return ages, vals

        if st.button("Find retirement window", key="retwin_run_btn"):
            # Clamp invalid ranges
            _start = int(max(search_start_age, current_age))
            _end = int(max(search_end_age, _start))
            snap_now = normalize_snapshot(get_current_inputs_snapshot())
            with st.spinner("Running simulations across retirement ages..."):
                ages, probs = _retwin_scan_cached(snap_now, _start, _end, int(win_sims), int(win_seed))

            # Identify earliest age meeting target and comfort target
            comfort_target = min(0.99, float(target_success + cushion))
            earliest = None
            comfortable = None
            for a, p in zip(ages, probs):
                if earliest is None and p >= target_success:
                    earliest = a
                if comfortable is None and p >= comfort_target:
                    comfortable = a
            planned_success = _success_probability_for_retire_age(snap_now, int(retire_age), int(win_sims), int(win_seed))

            st.subheader("Recommended retirement window (based on simulations)")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Earliest viable age", "Not found" if earliest is None else str(earliest))
            with m2:
                st.metric("More comfortable age", "Not found" if comfortable is None else str(comfortable))
            with m3:
                st.metric("Planned retirement age success", f"{planned_success*100:.1f}%")

            if earliest is None:
                best_p = max(probs) if probs else 0.0
                best_age = ages[probs.index(best_p)] if probs else _end
                st.warning(
                    f"No retirement age in the selected range reached the target success probability ({int(target_success*100)}%). "
                    f"Best in range: {best_p*100:.1f}% at age {best_age}. Try increasing the search range, reducing spending, or increasing contributions."
                )

            # Plot curve
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(ages, probs, linewidth=2)
            ax2.axhline(float(target_success), linestyle="--")
            ax2.set_xlabel("Retirement age")
            ax2.set_ylabel("Success probability")
            ax2.set_ylim(0, 1)
            st.pyplot(fig2)
            st.markdown("---")


    # Pre-compute the retirement-year row once (used by multiple sections). This must be available
    # even if the Cashflow & Longevity section is hidden.
    retirement_row = df[df["Age"] == retire_age]
    retirement_row = retirement_row.iloc[0] if not retirement_row.empty else None

    _show_cashflow = st.checkbox("Show Cashflow & Longevity Model", value=True, key="show_cashflow_section")
    if _show_cashflow:
        # ---------------------------
        # SECTION 2: CASHFLOW & LONGEVITY MODEL
        # ---------------------------
        st.header("Cashflow & Longevity Model")

        last_row = df.iloc[-1]
        depletion_rows = df[(df["End Balance"] <= 0) & (df["Age"] > current_age)]
        depletion_age = int(depletion_rows["Age"].min()) if not depletion_rows.empty else None

        if retirement_row is not None:
            assets_at_retirement = float(retirement_row["Portfolio Start"]) if "Portfolio Start" in retirement_row else float(retirement_row["End Balance"])
            expense_at_retirement = float(retirement_row["Required Spend"])
        else:
            assets_at_retirement = 0.0
            expense_at_retirement = 0.0

        final_balance = float(last_row["End Balance"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(f"Total Assets @ Age {retire_age}", f"${assets_at_retirement:,.0f}")
        with m2:
            st.metric(f"Projected Annual Spend @ Age {retire_age}", f"${expense_at_retirement:,.0f}")
        with m3:
            st.metric(f"Final Balance @ Age {int(last_row['Age'])}", f"${final_balance:,.0f}")
        with m4:
            if depletion_age is not None:
                st.error(f"Sustainability: Depleted @ Age {depletion_age}")
            else:
                st.success(f"Sustainability: Sustainable to {life_expectancy}")

        fig, ax = plt.subplots(figsize=(10, 5))
        if use_multi_asset and all(col in df.columns for col in ["Cash", "Bonds", "ETFs", "Real Estate", "Misc", "401k"]):
            ax.stackplot(
                df["Age"],
                df["Cash"],
                df["Bonds"],
                df["ETFs"],
                df["401k"],
                labels=["Cash", "Bonds/Munis", "ETFs", "401k"],
                alpha=0.85,
            )
            ax.legend(loc="upper left")
        else:
            ax.plot(df["Age"], df["End Balance"], label="Portfolio Balance", linewidth=2)
            ax.legend(loc="upper right")

        ax.axvline(retire_age, linestyle="--", linewidth=1.5, label="Retirement")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_xlabel("Age")
        st.pyplot(fig)

        if final_balance > 0:
            st.success(f"At age {life_expectancy}, the projected portfolio balance is **${final_balance:,.0f}**.")
        else:
            st.error(f"Portfolio is projected to deplete at age **{depletion_age if depletion_age is not None else 'N/A'}** under current assumptions.")

        with st.expander("Show yearly projection table"):
            display_df = df.copy()
            money_cols = [
                "Portfolio Start",
                "Required Spend",
                "Guaranteed Income",
                "Portfolio Withdrawal",
                "End Balance",
                "Cash",
                "Bonds",
                "ETFs",
                "401k",
            ]
            for col in money_cols:
                if col in display_df.columns:
                    _s = pd.to_numeric(display_df[col], errors="coerce")
                    _s = _s.replace([np.inf, -np.inf], np.nan).round(0)
                    display_df[col] = _s.astype("Int64")

            st.dataframe(display_df, use_container_width=True)

        st.caption("This projection is deterministic and uses constant return and inflation assumptions. It is a planning tool, not a guarantee.")


    with st.expander("The 3-Bucket Strategy Implementation", expanded=False):
        # ---------------------------
        # SECTION 3: 3-BUCKET STRATEGY
        # ---------------------------
        st.markdown(
            "Segment the portfolio into time-based buckets to manage **sequence-of-returns risk** "
            "and support smoother withdrawals."
        )

        if current_portfolio > 0 and retirement_row is not None:
            annual_draw_at_retire = float(retirement_row.get("Portfolio Withdrawal", 0.0))

            bucket_1_target = annual_draw_at_retire * 5
            bucket_2_target = annual_draw_at_retire * 10

            total_assets_for_buckets = float(retirement_row.get("Portfolio Start", retirement_row.get("End Balance", 0.0)))
            bucket_3_target = max(0.0, total_assets_for_buckets - bucket_1_target - bucket_2_target)

            if current_age < retire_age:
                bucket_1_target = 0.15 * current_portfolio
                bucket_2_target = 0.35 * current_portfolio
                bucket_3_target = 0.50 * current_portfolio

            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.subheader("Bucket 1: Cash / Munis")
                st.markdown("**Role:** Years 1–5 withdrawals")
                st.info(f"Illustrative Allocation: **${bucket_1_target:,.0f}**")
                st.caption("Target: High liquidity, low volatility.")

            with col_b2:
                st.subheader("Bucket 2: Income")
                st.markdown("**Role:** Years 6–15 withdrawals")
                st.warning(f"Illustrative Allocation: **${bucket_2_target:,.0f}**")
                st.caption("Target: Stable income assets.")

            with col_b3:
                st.subheader("Bucket 3: Growth")
                st.markdown("**Role:** Year 16+ growth")
                st.error(f"Illustrative Allocation: **${bucket_3_target:,.0f}**")
                st.caption("Target: Long-term growth assets.")

            st.markdown(
                "In strong markets, **Bucket 3** gains can refill Buckets 1 and 2. "
                "In weak markets, withdrawals come from Buckets 1 and 2 to avoid forced selling."
            )
        else:
            st.warning("Portfolio value is zero or not set. Adjust inputs in the sidebar to view bucket allocations.")

        st.markdown("---")

        # ---------------------------

    _show_plan_analysis = st.checkbox("Show 8. Plan Analysis & Recommendations", value=True, key="show_plan_analysis_section")
    if _show_plan_analysis:
        # ---------------------------
        # SECTION 5: PLAN ANALYSIS & RECOMMENDATIONS (same logic; uses df)
        # ---------------------------
        st.markdown("---")
        st.header("Plan Analysis & Recommendations")

        if "analysis_result" not in st.session_state:
            st.session_state.analysis_result = None

        col_btn, col_help = st.columns([1, 3])
        with col_btn:
            analyze_clicked = st.button(
                "Analyze Sustainability" if st.session_state.analysis_result is None else "Refresh Analysis",
                key="analyze_button",
            )
        with col_help:
            st.caption(
                "This analysis uses your current inputs, FIRE targets, tax snapshot, cashflow, "
                "and portfolio projections to generate a high-level narrative. "
                "It is not personalized financial advice."
            )

        if analyze_clicked:
            final_balance = float(df.iloc[-1]["End Balance"])
            ends_positive = final_balance > 0

            depletion_age_2 = None
            if not ends_positive:
                zero_rows = df[df["End Balance"] == 0]
                if not zero_rows.empty:
                    depletion_age_2 = int(zero_rows["Age"].min())

            retired_rows = df[df["Age"] >= retire_age]
            if not retired_rows.empty:
                first_ret_row = retired_rows.iloc[0]
                first_withdrawal = float(first_ret_row.get("Portfolio Withdrawal", 0.0))
                start_base = float(first_ret_row.get("Portfolio Start", first_ret_row.get("End Balance", 0.0)))
                initial_withdrawal_rate = (first_withdrawal / start_base) if start_base > 0 else 0.0
            else:
                first_withdrawal = 0.0
                initial_withdrawal_rate = 0.0

            if ends_positive and initial_withdrawal_rate <= 0.04:
                sustainability_label = "robust"
                sustainability_text = (
                    "Based on your assumptions, the plan appears **robust**. "
                    "Your portfolio is projected to last through the full planning horizon, "
                    f"with an ending balance of about **${final_balance:,.0f}** and an initial withdrawal "
                    f"rate of ~{initial_withdrawal_rate * 100:,.1f}%, which is in line with classical 4% guidance."
                )
            elif ends_positive and initial_withdrawal_rate <= 0.05:
                sustainability_label = "cautious"
                sustainability_text = (
                    "The plan appears **generally sustainable but somewhat sensitive**. "
                    "Your portfolio is projected to last through the horizon, but the initial withdrawal "
                    f"rate of ~{initial_withdrawal_rate * 100:,.1f}% is above the classic 4% rule. "
                    "Small changes in returns, inflation, or spending could materially impact outcomes."
                )
            else:
                sustainability_label = "at risk"
                if depletion_age_2 is not None:
                    sustainability_text = (
                        "The plan appears **at risk of depletion** under current assumptions. "
                        f"Your portfolio is projected to run out around age **{depletion_age_2}**, "
                        "suggesting that retirement timing, spending levels, or risk assumptions may need revision."
                    )
                else:
                    sustainability_text = (
                        "The plan appears **at risk** under current assumptions. "
                        "Projected withdrawals and/or return assumptions lead to low ending balances and "
                        "a narrow margin for error."
                    )

            summary_text = (
                f"You are currently **{current_age}**, planning to retire at **{retire_age}**, with an initial "
                f"retirement spending target of **${annual_spend_retirement:,.0f}** per year (in today's dollars). "
                f"Current investable assets are **${current_portfolio:,.0f}**, with assumed pre-retirement growth of "
                f"**{pre_retire_return * 100:,.1f}%**, post-retirement growth of **{post_retire_return * 100:,.1f}%**, "
                f"and inflation of **{inflation_rate * 100:,.1f}%**. "
                f"Your current tax-effective net income is about **${net_take_home:,.0f}**, with estimated annual "
                f"expenses of **${annual_expenses:,.0f}**, leaving a surplus of approximately "
                f"**${surplus:,.0f}** available for savings and flexibility."
            )

            recommendations = []
            target_25x = annual_spend_retirement * 25
            target_33x = annual_spend_retirement * 33

            if current_portfolio < target_25x:
                recommendations.append(
                    f"Increase annual savings and/or redirect more of your current surplus toward investing. "
                    f"Your current portfolio (~${current_portfolio:,.0f}) is below the 25x target (~${target_25x:,.0f})."
                )
            if current_portfolio < target_33x:
                recommendations.append(
                    "Consider a more conservative FIRE target closer to **33x annual spending** if you want higher "
                    "confidence in long-term sustainability, especially with longer life expectancy assumptions."
                )

            if surplus < 0:
                recommendations.append(
                    "Your current annual expenses appear to **exceed** your after-tax income, creating a structural deficit. "
                    "Addressing this gap (through spending reductions or income increases) should be a priority before "
                    "relying on aggressive retirement contributions."
                )
            elif annual_contribution > surplus:
                recommendations.append(
                    f"Planned annual contributions of **${annual_contribution:,.0f}** exceed the current estimated "
                    f"surplus of **${surplus:,.0f}**. Validate that this contribution rate is realistic and sustainable "
                    "given your lifestyle and cashflow needs."
                )

            if sustainability_label in ["cautious", "at risk"]:
                recommendations.append(
                    "Evaluate retiring **later by 2–3 years** or modestly lowering initial retirement spending "
                    "to improve the portfolio's ability to withstand return and inflation shocks."
                )
                recommendations.append(
                    "Review your asset allocation across cash, bonds, and equities to ensure it aligns with both "
                    "your risk tolerance and the need for growth to support a long retirement horizon."
                )

            if effective_tax_rate > 0.30:
                recommendations.append(
                    "Explore **tax optimization strategies** (e.g., maxing tax-advantaged accounts, Roth conversions, "
                    "capital gains harvesting, or efficient asset location) to improve net-of-tax returns over time."
                )

            if crash_scenario > 0:
                stressed_pot = current_portfolio * (1 - (crash_scenario / 100))
                if stressed_pot < target_25x:
                    recommendations.append(
                        f"Under a {crash_scenario}% immediate market shock, investable assets fall to "
                        f"~${stressed_pot:,.0f}, below the 25x spending target. Consider holding a somewhat "
                        "larger safety bucket in cash/bonds or scaling back risk slightly pre-retirement."
                    )

            if sustainability_label == "robust":
                risk_assessment = (
                    "Overall portfolio risk appears **aligned** with your objectives, assuming your stated return and "
                    "inflation assumptions are realistic. The main residual risks are sequence-of-returns risk in the early "
                    "retirement years and potential regime shifts in inflation or tax policy."
                )
            elif sustainability_label == "cautious":
                risk_assessment = (
                    "Portfolio risk appears **moderately elevated** relative to your withdrawal targets. "
                    "You likely need meaningful exposure to growth assets to make the plan work, which increases sensitivity "
                    "to market drawdowns, especially in the first 5–10 years of retirement."
                )
            else:
                risk_assessment = (
                    "Portfolio risk and spending assumptions appear **misaligned**. At current spending levels, the plan "
                    "relies on favorable markets and leaves limited margin for adverse sequences of returns or higher-than-"
                    "expected inflation. De-risking without adjusting spending or timing would further compress sustainability."
                )

            st.session_state.analysis_result = {
                "summary": summary_text,
                "sustainability_check": sustainability_text,
                "recommendations": recommendations,
                "risk_assessment": risk_assessment,
            }

        result = st.session_state.analysis_result
        if result is None:
            st.info("Configure your assumptions and inputs above, then click **Analyze Sustainability** to generate a narrative assessment of your plan.")
        else:
            st.subheader("Plan Narrative")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("#### Executive Summary")
                st.markdown(result["summary"])
            with col_s2:
                st.markdown("#### Sustainability Check")
                st.markdown(result["sustainability_check"])

            st.markdown("")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("#### Tactical Recommendations")
                if result["recommendations"]:
                    for idx, rec in enumerate(result["recommendations"], start=1):
                        st.markdown(f"**{idx}.** {rec}")
                else:
                    st.markdown("No specific tactical changes are flagged by the current rule set. Monitor the plan periodically and revisit assumptions as life circumstances change.")
            with col_r2:
                st.markdown("#### Portfolio Risk Assessment")
                st.markdown(result["risk_assessment"])

        # =============================================================================
        # TAB 2: COMPARE SCENARIOS
        # =============================================================================

with tab2:
    st.subheader("Scenario Comparison (Side-by-Side)")

    scenarios = _get_scenarios()

    # Always show Create, even when the user has no saved scenarios yet.
    if not scenarios:
        if st.button("Create Scenario from current sidebar", use_container_width=True):
            snap = normalize_snapshot(get_current_inputs_snapshot())
            scenarios.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": "Scenario 1",
                    "inputs": copy.deepcopy(snap),
                    "results_df": None,
                    "kpis": None,
                }
            )
            _set_scenarios(scenarios)
            st.rerun()

    if not scenarios:
        st.info("No scenarios saved yet. Create your first scenario above, then you can edit, duplicate, and compare.")
    else:

        # --- Select scenario to edit (by ID, not name) ---
        id_to_label = {sc["id"]: f"{sc['name']} ({sc['id']})" for sc in scenarios}
        labels = [id_to_label[sc["id"]] for sc in scenarios]
        ids = [sc["id"] for sc in scenarios]

        if "edit_scenario_id" not in st.session_state or st.session_state.edit_scenario_id not in ids:
            st.session_state.edit_scenario_id = ids[0]

        selected_label = st.selectbox(
            "Select a scenario to edit",
            options=labels,
            index=ids.index(st.session_state.edit_scenario_id),
        )
        selected_id = ids[labels.index(selected_label)]
        st.session_state.edit_scenario_id = selected_id

        # Persist last-selected scenario for this user (helps restore Compare tab on next login)
        try:
            sb_save_user_state(st.session_state.get("current_user"), active_scenario_id=str(selected_id))
        except Exception:
            pass


        # Locate scenario
        sc_idx = next(i for i, sc in enumerate(scenarios) if sc["id"] == selected_id)
        scenario = scenarios[sc_idx]

        # --- Create / Duplicate / Delete ---
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("Create Scenario from current sidebar", use_container_width=True):
                snap = normalize_snapshot(get_current_inputs_snapshot())
                scenarios.append(
                    {
                        "id": str(uuid.uuid4()),
                        "name": f"Scenario {len(scenarios) + 1}",
                        "inputs": copy.deepcopy(snap),
                        "results_df": None,
                        "kpis": None,
                    }
                )
                _set_scenarios(scenarios)
                st.rerun()

        with c2:
            if st.button("Duplicate selected scenario", use_container_width=True):
                src = scenarios[sc_idx]
                scenarios.append(
                    {
                        "id": str(uuid.uuid4()),
                        "name": f"{src['name']} (Copy)",
                        "inputs": copy.deepcopy(normalize_snapshot(src.get("inputs", {}))),
                        "results_df": None,
                        "kpis": None,
                    }
                )
                _set_scenarios(scenarios)
                st.rerun()

        with c3:
            if st.button("Delete selected scenario", use_container_width=True):
                scenarios = [sc for sc in scenarios if sc["id"] != selected_id]
                _set_scenarios(scenarios)
                st.session_state.edit_scenario_id = scenarios[0]["id"] if scenarios else None
                st.rerun()

        st.markdown("---")

        # =========================================================
        # SCENARIO EDITOR (WORKING COPY; ONLY SAVES ON SUBMIT)
        # =========================================================
        st.markdown(f"### Edit: {scenario['name']}")

        working = normalize_snapshot(scenario["inputs"])  # normalize + deep copy
        prefix = f"sc_{selected_id}_"

        # Name edit
        new_name = st.text_input("Scenario name", value=scenario["name"], key=prefix + "name")

        with st.form(f"edit_form_{selected_id}"):
            st.markdown("#### Demographics")
            working["current_age"] = st.number_input("Current Age", 35, 90, int(working.get("current_age", 50)), key=prefix+"current_age")
            working["retire_age"] = st.number_input("Retirement Age", 35, 90, int(working.get("retire_age", 60)), key=prefix+"retire_age")
            working["life_expectancy"] = st.number_input("Life Expectancy", 70, 110, int(working.get("life_expectancy", 95)), key=prefix+"life_expectancy")

            st.markdown("#### Spending & Savings")
            working["annual_spend_retirement"] = st.number_input(
                "Annual spend in retirement (today $)",
                value=float(working.get("annual_spend_retirement", 155000)),
                key=prefix+"spend",
            )
            working["annual_contribution"] = st.number_input(
                "Annual contribution until retirement ($)",
                value=float(working.get("annual_contribution", 65000)),
                key=prefix+"contrib",
            )

            st.markdown("#### Assumptions (Percent)")
            infl_pct = st.slider("Inflation (%)", 1.0, 5.0, _as_percent_display(working.get("inflation_rate", 0.03), 3.0), 0.1, key=prefix+"infl_pct")
            pre_pct  = st.slider("Pre-retirement return (%)", 1.0, 12.0, _as_percent_display(working.get("pre_retire_return", 0.07), 7.0), 0.1, key=prefix+"pre_pct")
            post_pct = st.slider("Post-retirement return (%)", 1.0, 10.0, _as_percent_display(working.get("post_retire_return", 0.045), 4.5), 0.1, key=prefix+"post_pct")

            working["inflation_rate"] = infl_pct / 100.0
            working["pre_retire_return"] = pre_pct / 100.0
            working["post_retire_return"] = post_pct / 100.0

            st.markdown("#### Guaranteed Income")
            working["social_security"] = st.number_input("Social Security / Pension (annual $)", value=float(working.get("social_security", 30000)), key=prefix+"ss")
            working["ss_start_age"] = st.number_input("SS / Pension start age", 60, 75, int(working.get("ss_start_age", 67)), key=prefix+"ss_age")

            st.markdown("#### Portfolio")
            working["use_multi_asset"] = st.checkbox("Use Multi-Asset (Cash/Bonds/ETFs/401k)", value=bool(working.get("use_multi_asset", True)), key=prefix+"multi")
            working["flow_mode"] = st.selectbox("Withdrawal mode", ["cash_first", "pro_rata"], index=0 if working.get("flow_mode","cash_first")=="cash_first" else 1, key=prefix+"flow")

            if working["use_multi_asset"]:
                st.markdown("##### Multi-Asset Inputs")
                working["cash_bal"] = st.number_input("Cash balance ($)", value=float(working.get("cash_bal", 200000)), key=prefix+"cash_bal")
                cy = st.slider("Cash yield (%)", 0.0, 8.0, _as_percent_display(working.get("cash_yield", 0.04), 4.0), 0.1, key=prefix+"cash_y")
                working["cash_yield"] = cy / 100.0

                working["bonds_bal"] = st.number_input("Bonds/Munis balance ($)", value=float(working.get("bonds_bal", 400000)), key=prefix+"bonds_bal")
                by = st.slider("Bonds yield (%)", 0.0, 10.0, _as_percent_display(working.get("bonds_yield", 0.05), 5.0), 0.1, key=prefix+"bonds_y")
                working["bonds_yield"] = by / 100.0

                working["etfs_bal"] = st.number_input("ETFs balance ($)", value=float(working.get("etfs_bal", 439000)), key=prefix+"etfs_bal")
                ey = st.slider("ETFs return (%)", 0.0, 12.0, _as_percent_display(working.get("etfs_yield", 0.07), 7.0), 0.1, key=prefix+"etfs_y")
                working["etfs_yield"] = ey / 100.0

                working["k401_bal"] = st.number_input("401k balance ($)", value=float(working.get("k401_bal", 200000)), key=prefix+"k401_bal")
                ky = st.slider("401k return (%)", 0.0, 12.0, _as_percent_display(working.get("k401_yield", 0.07), 7.0), 0.1, key=prefix+"k401_y")
                working["k401_yield"] = ky / 100.0

                working["real_estate_bal"] = st.number_input(
                    "Real Estate balance ($)",
                    min_value=0.0,
                    value=float(working.get("real_estate_bal", 0.0)),
                    step=1000.0,
                    key=prefix+"real_estate_bal",
                )
                ry = st.slider(
                    "Real Estate return (%)",
                    0.0,
                    12.0,
                    _as_float_with_default(working.get("real_estate_return", 0.05), 5.0),
                    0.1,
                    key=prefix+"real_estate_r",
                )
                working["real_estate_return"] = ry / 100.0

                working["misc_bal"] = st.number_input(
                    "Misc. balance ($) (Intl assets etc.)",
                    min_value=0.0,
                    value=float(working.get("misc_bal", 0.0)),
                    step=1000.0,
                    key=prefix+"misc_bal",
                )
                my = st.slider(
                    "Misc. return (%)",
                    0.0,
                    12.0,
                    _as_float_with_default(working.get("misc_return", 0.05), 5.0),
                    0.1,
                    key=prefix+"misc_r",
                )
                working["misc_return"] = my / 100.0

            else:
                working["current_portfolio"] = st.number_input("Total invested assets ($)", value=float(working.get("current_portfolio", 1239000)), key=prefix+"total")

            save_clicked = st.form_submit_button("Save Scenario")

        if save_clicked:
            scenarios = _get_scenarios()  # reload
            sc_idx = next(i for i, sc in enumerate(scenarios) if sc["id"] == selected_id)
            scenarios[sc_idx]["name"] = new_name
            scenarios[sc_idx]["inputs"] = normalize_snapshot(working)  # normalized + deep copy
            scenarios[sc_idx]["results_df"] = None
            scenarios[sc_idx]["kpis"] = None
            _set_scenarios(scenarios)
            st.success("Scenario saved.")

        st.markdown("---")

        # =========================================================
        # RUN COMPARISON (SELECT BY ID; NOT NAME)
        # =========================================================
        scenarios = _get_scenarios()
        compare_ids = st.multiselect(
            "Select scenarios to compare",
            options=[sc["id"] for sc in scenarios],
            default=[sc["id"] for sc in scenarios[:2]],
            format_func=lambda sid: id_to_label.get(sid, sid),
        )

        if st.button("Run Comparison", type="primary"):
            for i, sc in enumerate(scenarios):
                if sc["id"] not in compare_ids:
                    continue

                snap = normalize_snapshot(sc["inputs"])
                df_sc = run_projection_from_snapshot(snap)  # must accept snapshot with decimal rates
                kpis = scenario_kpis(
                    df_sc,
                    retire_age=snap["retire_age"],
                    current_age=snap["current_age"],
                    life_expectancy=snap["life_expectancy"],
                )

                scenarios[i]["results_df"] = df_sc
                scenarios[i]["kpis"] = kpis

            _set_scenarios(scenarios)
            st.success("Comparison updated.")

        # Display
        chosen = [sc for sc in _get_scenarios() if sc["id"] in compare_ids and sc.get("kpis") is not None]
        if not chosen:
            st.info("Select scenarios and click Run Comparison.")
        else:

            rows = []
            for sc in chosen:
                row = {"Scenario": sc["name"]}
                row.update(sc["kpis"])
                rows.append(row)

            kpi_df = pd.DataFrame(rows)
            # ---- Pretty formatting (currency to 1 decimal; percentages to 2 decimals) ----
            def _fmt_cur(x):
                try:
                    return f"${float(x):,.1f}"
                except Exception:
                    return ""
            def _fmt_int(x):
                try:
                    xi = int(float(x))
                    return str(xi)
                except Exception:
                    return ""

            for c in ["Assets @ Retire", "Final Balance"]:
                if c in kpi_df.columns:
                    kpi_df[c] = kpi_df[c].map(_fmt_cur)

            if "Depletion Age" in kpi_df.columns:
                kpi_df["Depletion Age"] = kpi_df["Depletion Age"].map(_fmt_int)

            if "Withdrawal Rate (1st yr)" in kpi_df.columns:
                kpi_df["Withdrawal Rate (1st yr)"] = kpi_df["Withdrawal Rate (1st yr)"].astype(float).map(lambda x: f"{x*100:.2f}%")
            st.dataframe(kpi_df, use_container_width=True, hide_index=True)
            # Persist compare summary for the One-Time Readiness Report
            try:
                _sc_list = []
                for _sc in chosen:
                    _k = _sc.get('kpis') or {}
                    _inp = _sc.get('inputs') or {}
                    _sc_list.append({
                        'name': _sc.get('name',''),
                        'confidence': _k.get('confidence_score', _k.get('confidence', '')),
                        'sustainable_spend': _k.get('sustainable_spend', _k.get('spend', '')),
                        'retirement_age': _inp.get('retire_age', _inp.get('retirement_age', '')),
                    })
                st.session_state['report_compare'] = {'generated_at': datetime.now().isoformat(), 'scenarios': _sc_list}
            except Exception:
                pass

            fig, ax = plt.subplots(figsize=(10, 5))
            for sc in chosen:
                df_sc = sc["results_df"]
                ax.plot(df_sc["Age"], df_sc["End Balance"], linewidth=2, label=sc["name"])
            ax.set_xlabel("Age")
            ax.set_ylabel("Total Portfolio ($)")
            ax.legend(loc="upper right")
            st.pyplot(fig)

            # ---------------------------
            # PDF EXPORT (Scenario Comparison)
            # ---------------------------
            try:
                comp_png = None
                try:
                    _bio2 = BytesIO()
                    fig.savefig(_bio2, format="png", dpi=160, bbox_inches="tight")
                    comp_png = _bio2.getvalue()
                except Exception:
                    comp_png = None

                comp_pdf = _build_compare_pdf_bytes(
                    user_id=st.session_state.get("current_user", "unknown"),
                    kpi_df=kpi_df,
                    chart_png=comp_png,
                    title="Scenario Comparison (Side-by-Side)",
                )

                st.download_button(
                    "Download comparison PDF",
                    data=comp_pdf,
                    file_name=f"scenario_comparison_{st.session_state.get('current_user','user')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Comparison PDF export is unavailable due to an internal error: {e}")# =============================================================================
# TAB 3: RANGE OF OUTCOMES (MONTE CARLO) - OPT-IN
# =============================================================================
with tab3:
    st.subheader("Range of Outcomes (Simulation)")
    st.caption(
        "This optional view runs many simulations to show how outcomes might vary when annual returns and inflation fluctuate. "
        "It does not change any results in the Single Scenario or Compare tabs."
    )

    # Source of inputs
    src_mode = st.radio(
        "Which inputs should we simulate?",
        options=["Use my current inputs (from the sidebar)", "Use a saved scenario"],
        index=0,
        horizontal=False,
        key="mc_src_mode",
    )

    snap = None
    if src_mode.startswith("Use my current inputs"):
        snap = normalize_snapshot(get_current_inputs_snapshot())
        st.info("Simulating your current sidebar inputs.")
    else:
        scenarios_mc = _get_scenarios()
        if not scenarios_mc:
            st.warning("No saved scenarios found. Falling back to your current sidebar inputs for simulation.")
            snap = normalize_snapshot(get_current_inputs_snapshot())
            st.info("Simulating your current sidebar inputs.")
            src_mode = "Use my current inputs (from the sidebar)"
        id_to_label_mc = {sc["id"]: f'{sc["name"]} ({sc["id"]})' for sc in scenarios_mc}
        sel_id = st.selectbox(
            "Select a saved scenario",
            options=[sc["id"] for sc in scenarios_mc],
            format_func=lambda sid: id_to_label_mc.get(sid, sid),
            key="mc_saved_scenario_id",
        )
        sel_sc = next(sc for sc in scenarios_mc if sc["id"] == sel_id)
        snap = normalize_snapshot(sel_sc["inputs"])
        st.info(f"Simulating saved scenario: {sel_sc['name']}")

    st.markdown("### Simulation settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_trials = st.number_input("Number of simulations", min_value=200, max_value=20000, value=2000, step=200, key="mc_n_trials")
    with c2:
        seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1, key="mc_seed")
    with c3:
        st.markdown("")

    st.markdown("### Uncertainty assumptions (advanced)")
    c4, c5, c6 = st.columns(3)
    with c4:
        pre_sigma_pct = st.slider("Pre-retirement return volatility (%)", 0.0, 35.0, 12.0, 0.5, key="mc_pre_sigma")
    with c5:
        post_sigma_pct = st.slider("Post-retirement return volatility (%)", 0.0, 30.0, 9.0, 0.5, key="mc_post_sigma")
    with c6:
        infl_sigma_pct = st.slider("Inflation volatility (%)", 0.0, 10.0, 1.0, 0.1, key="mc_infl_sigma")

    
    st.markdown("##### Optional safety rules (affects simulation only)")
    with st.expander("Adjust withdrawals automatically in tough markets (optional)", expanded=False):
        use_spending_floor = st.checkbox(
            "Reduce spending when portfolio is under stress ('spending cut' rule)",
            value=False,
            key="mc_use_spending_floor",
            help="If the simulated portfolio gets too low relative to your spending needs, the simulation applies a temporary spending reduction.",
        )
        spending_floor_multiple = st.slider(
            "Trigger when assets drop below (multiple of current-year spending)",
            8.0, 30.0, 18.0, 0.5,
            help="If your portfolio falls below this many years of spending, the simulation assumes you temporarily tighten spending to protect against running out.",
            key="mc_spending_floor_multiple",
            disabled=not use_spending_floor,
        )
        spending_floor_cut_pct = st.slider(
            "Spending cut when triggered (%)",
            0.0, 30.0, 10.0, 1.0,
            help="How much you cut spending *temporarily* once the stress trigger is hit (e.g., 10% means spending drops from $100k to $90k until recovery).",
            key="mc_spending_floor_cut_pct",
            disabled=not use_spending_floor,
        ) / 100.0
        spending_floor_recover_multiple = st.slider(
            "Stop cutting once assets recover above (multiple of spending)",
            8.0, 35.0, 22.0, 0.5,
            help="Once the portfolio recovers above this many years of spending, the temporary spending cut stops and spending returns toward the planned path.",
            key="mc_spending_floor_recover_multiple",
            disabled=not use_spending_floor,
        )

        st.markdown("---")

        use_guardrails = st.checkbox(
            "Use dynamic withdrawal guardrails ('raise/cut' rule)",
            value=False,
            key="mc_use_guardrails",
            help="Adjusts spending up or down based on the withdrawal rate relative to the first year in retirement.",
        )
        guardrail_band_pct = st.slider(
            "Guardrail band around initial withdrawal rate (%)",
            5.0, 50.0, 20.0, 1.0,
            help="How wide the 'safe zone' is around your initial withdrawal rate. A wider band means fewer adjustments; a narrower band makes the rules react sooner.",
            key="mc_guardrail_band_pct",
            disabled=not use_guardrails,
        ) / 100.0
        guardrail_cut_pct = st.slider(
            "Cut spending by (%) when above upper guardrail",
            0.0, 30.0, 10.0, 1.0,
            help="If spending becomes too aggressive (withdrawal rate above the upper guardrail), this is how much the simulation reduces spending to get back on track.",
            key="mc_guardrail_cut_pct",
            disabled=not use_guardrails,
        ) / 100.0
        guardrail_raise_pct = st.slider(
            "Raise spending by (%) when below lower guardrail",
            0.0, 20.0, 5.0, 1.0,
            help="If spending is very conservative (withdrawal rate below the lower guardrail), this is how much the simulation increases spending (within the cap) to enjoy more today.",
            key="mc_guardrail_raise_pct",
            disabled=not use_guardrails,
        ) / 100.0
        guardrail_raise_cap_pct = st.slider(
            "Cap raises above the inflation-adjusted baseline (%)",
            0.0, 40.0, 15.0, 1.0,
            help="Prevents spending from rising too far above the inflation-adjusted plan (helps avoid lifestyle creep after good market runs).",
            key="mc_guardrail_raise_cap_pct",
            disabled=not use_guardrails,
        ) / 100.0

        run_mc = st.button("Run Simulation", type="primary", key="mc_run_btn")

        if run_mc:
            with st.spinner("Running simulations..."):
                res = monte_carlo_projection_from_snapshot(
                    snap,
                    n_trials=int(n_trials),
                    seed=int(seed),
                    pre_sigma=float(pre_sigma_pct) / 100.0,
                    post_sigma=float(post_sigma_pct) / 100.0,
                    infl_sigma=float(infl_sigma_pct) / 100.0,
                    use_spending_floor=bool(use_spending_floor),
                    spending_floor_multiple=float(spending_floor_multiple),
                    spending_floor_cut_pct=float(spending_floor_cut_pct),
                    spending_floor_recover_multiple=float(spending_floor_recover_multiple),
                    use_guardrails=bool(use_guardrails),
                    guardrail_band_pct=float(guardrail_band_pct),
                    guardrail_cut_pct=float(guardrail_cut_pct),
                    guardrail_raise_pct=float(guardrail_raise_pct),
                    guardrail_raise_cap_pct=float(guardrail_raise_cap_pct),
                )
            st.session_state["mc_last_result"] = res
            st.success("Simulation complete.")

        res = st.session_state.get("mc_last_result")
        if not res:
            st.info("Adjust settings above and click **Run Simulation** to view the simulated range of outcomes.")

        if res:
            def _money(x: float) -> str:
                return f"${float(x):,.1f}"
    
            st.markdown("### Key takeaways")
            # Persist range-of-outcomes summary for the One-Time Readiness Report
            try:
                _p10 = float(res.get('p10_final', float(np.percentile(res.get('final_balances',[0.0]), 10))))
                _p50 = float(res.get('median_final', float(np.percentile(res.get('final_balances',[0.0]), 50))))
                _p90 = float(res.get('p90_final', float(np.percentile(res.get('final_balances',[0.0]), 90))))
                st.session_state['report_range_outcomes'] = {
                    'generated_at': datetime.now().isoformat(),
                    'metrics': {'Ending balance at horizon': {'p10': _p10, 'p50': _p50, 'p90': _p90}},
                }
            except Exception:
                pass
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Chance of running out of funds", f"{res['prob_deplete']*100:.1f}%")
            with m2:
                st.metric("Median ending balance", _money(res["median_final"]))
            with m3:
                st.metric("10th percentile ending balance", _money(res["p10_final"]))
            with m4:
                st.metric("90th percentile ending balance", _money(res["p90_final"]))
    
        
            # ---------------------------------------------------------
            # Executive-friendly narrative summary (auto-generated)
            # ---------------------------------------------------------
            deplete_pct = float(res.get("prob_deplete", 0.0)) * 100.0
            median_final = res.get("median_final", float(np.median(res["final_balances"])))
            p10_final = res.get("p10_final", float(np.percentile(res["final_balances"], 10)))
            p90_final = res.get("p90_final", float(np.percentile(res["final_balances"], 90)))
    
            st.markdown("#### Plain-English summary (based on the simulation)")
            st.markdown(
                f"""
    - **Chance of running out of money before age {int(snap['life_expectancy'])}:** {deplete_pct:.1f}%
    - **Most likely outcome (median):** around {_money(median_final)} left at age {int(snap['life_expectancy'])}
    - **Cautious view (10th percentile):** around {_money(p10_final)} left
    - **Optimistic view (90th percentile):** around {_money(p90_final)} left
    """
            )
    
            # Interpret depletion age
            tda = res.get("typical_deplete_age", None)
            if tda is None or (isinstance(tda, float) and np.isnan(tda)):
                st.success(f"In these simulations, funds generally last through age {int(snap['life_expectancy'])}.")
            else:
                st.warning(f"In the simulations where money runs out, it typically happens around **age {int(tda)}**.")
    
            st.markdown(
                """
            **How to read the percentiles:**
            - The **10th percentile** is a “bad but plausible” outcome: **9 out of 10 simulations do better**, 1 out of 10 do worse.
            - The **90th percentile** is a “good but plausible” outcome: **9 out of 10 simulations do worse**, 1 out of 10 do better.
            - Percentiles refer to the **amount left over** at the end (age shown), not a guarantee.
            """
            )
    
            # Note any active simulation safety rules
            rules = []
            if use_spending_floor:
                rules.append("temporary spending cuts in stressed years")
            if use_guardrails:
                rules.append("dynamic withdrawal guardrails (raise/cut rules)")
            if rules:
                st.info("This run included: " + ", ".join(rules) + ". These rules change outcomes only in this Monte Carlo tab.")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.hist(res["final_balances"], bins=40)
            ax2.set_xlabel(f"Ending Balance at Age {int(snap['life_expectancy'])} ($)")
            ax2.set_ylabel("Number of simulations")
            st.pyplot(fig2)
    
            st.caption(
                "This simulation uses simplified assumptions (normally distributed annual returns/inflation with fixed volatilities). "
                "It is intended for planning insights, not financial advice."
            )
            # ---------------------------
            # PDF EXPORT (Range of Outcomes)
            # ---------------------------
            try:
                sim_settings = {
                    "Simulations": int(n_trials),
                    "Random seed": int(seed),
                    "Inflation mean": f"{float(snap.get('inflation_rate', 0.0)) * 100:.2f}%",
                    "Inflation volatility": f"{float(infl_sigma_pct):.2f}%",
                    "Pre-retirement mean return": f"{float(snap.get('pre_retire_return', 0.0)) * 100:.2f}%",
                    "Pre-retirement return volatility": f"{float(pre_sigma_pct):.2f}%",
                    "Post-retirement mean return": f"{float(snap.get('post_retire_return', 0.0)) * 100:.2f}%",
                    "Post-retirement return volatility": f"{float(post_sigma_pct):.2f}%",
                    "Spending cut rule enabled": "Yes" if bool(use_spending_floor) else "No",
                    "Trigger threshold (assets vs spending)": f"{float(spending_floor_multiple):.2f}x" if bool(use_spending_floor) else "N/A",
                    "Spending cut %": f"{float(spending_floor_cut_pct) * 100:.1f}%" if bool(use_spending_floor) else "N/A",
                    "Recovery threshold": f"{float(spending_floor_recover_multiple):.2f}x" if bool(use_spending_floor) else "N/A",
                    "Guardrails enabled": "Yes" if bool(use_guardrails) else "No",
                    "Guardrail band around initial WR": f"{float(guardrail_band_pct) * 100:.1f}%" if bool(use_guardrails) else "N/A",
                    "Cut when above upper guardrail": f"{float(guardrail_cut_pct) * 100:.1f}%" if bool(use_guardrails) else "N/A",
                    "Raise when below lower guardrail": f"{float(guardrail_raise_pct) * 100:.1f}%" if bool(use_guardrails) else "N/A",
                    "Raise cap above baseline": f"{float(guardrail_raise_cap_pct) * 100:.1f}%" if bool(use_guardrails) else "N/A",
                }
    
                # Capture chart image (best-effort)
                chart_png = None
                try:
                    _bio = BytesIO()
                    fig2.savefig(_bio, format="png", dpi=160, bbox_inches="tight")
                    chart_png = _bio.getvalue()
                except Exception:
                    chart_png = None
    
                pdf_bytes = _build_montecarlo_pdf_bytes(
                    user_id=st.session_state.get("current_user", "unknown"),
                    snap=snap,
                    sim_settings=sim_settings,
                    res=res,
                    chart_png=chart_png,
                )
    
                st.download_button(
                    "Download PDF report (Range of Outcomes)",
                    data=pdf_bytes,
                    file_name=f"retirement_range_of_outcomes_{st.session_state.get('current_user','user')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"PDF export is unavailable due to an internal error: {e}")
    
    
    
    # =============================================================================


# =============================================================================
# TAB 4: MARKET REALITY MODE (STRESS TEST LIBRARY)
# =============================================================================
with tab4:
    st.header("Market Reality Mode: Stress-Test Library")
    st.caption(
        "Run prebuilt shocks against your current scenario using the same Monte Carlo engine. "
        "Outputs are expressed as deltas vs your baseline simulation."
    )

    snap = normalize_snapshot(get_current_inputs_snapshot())

    colA, colB = st.columns([2, 1])
    with colA:
        presets = {
            "Retirement-year crash": "retirement_crash",
            "Lost decade (low returns)": "lost_decade",
            "High inflation decade": "high_inflation",
            "Longevity shock (living to 100)": "longevity_100",
            "Healthcare cost spike": "healthcare_spike",
        }
        selected = st.multiselect(
            "Select stress tests to run",
            options=list(presets.keys()),
            default=["Retirement-year crash", "Lost decade (low returns)"]
            if "phase2_stress_selected" not in st.session_state
            else st.session_state.get("phase2_stress_selected", []),
            key="phase2_stress_selected",
        )

    with colB:
        st.markdown("#### Simulation settings")
        n_sims = st.number_input("Simulations", 300, 10_000, 1500, 100, key="phase2_stress_n_sims")
        seed = st.number_input("Random seed", 0, 999_999, 42, 1, key="phase2_stress_seed")
        pre_sigma = st.slider("Pre-retirement volatility (%)", 0.0, 30.0, 15.0, 0.5, key="phase2_stress_pre_sigma") / 100.0
        post_sigma = st.slider("Post-retirement volatility (%)", 0.0, 30.0, 10.0, 0.5, key="phase2_stress_post_sigma") / 100.0
        infl_sigma = st.slider("Inflation volatility (%)", 0.0, 6.0, 1.0, 0.1, key="phase2_stress_infl_sigma") / 100.0

    target_success = st.slider(
        "Target success probability for spend-impact metric",
        0.60,
        0.95,
        0.80,
        0.01,
        key="phase2_stress_target_success",
        help="Spend impact is computed by estimating the sustainable retirement spend (today $) that meets this target success rate.",
    )

    run_suite = st.button("Run selected stress tests", type="primary", key="phase2_run_stress_suite")
    if run_suite:
        with st.spinner("Running baseline + stress simulations..."):
            # Baseline run
            base_res = _mc_cached_with_extensions(
                snap,
                n_sims=int(n_sims),
                seed=int(seed),
                pre_sigma=float(pre_sigma),
                post_sigma=float(post_sigma),
                infl_sigma=float(infl_sigma),
                stress_shocks_json="",
            )
            base_conf = 100.0 * (1.0 - float(base_res.get("prob_deplete", 0.0)))
            base_spend = _sustainable_spend_mc_cached_with_extensions(
                snap,
                target_success=float(target_success),
                n_sims=int(max(600, int(n_sims // 2))),
                seed=int(seed),
                pre_sigma=float(pre_sigma),
                post_sigma=float(post_sigma),
                infl_sigma=float(infl_sigma),
                stress_shocks_json="",
            )["spend"]

            rows: list[dict] = []
            for name in selected:
                pid = presets.get(name)
                if not pid:
                    continue
                s_mod, shocks, label = _build_stress_shocks_for_preset(snap, pid)
                stress_res = _mc_cached_with_extensions(
                    s_mod,
                    n_sims=int(n_sims),
                    seed=int(seed),
                    pre_sigma=float(pre_sigma),
                    post_sigma=float(post_sigma),
                    infl_sigma=float(infl_sigma),
                    stress_shocks_json=_safe_json_dumps(shocks),
                )
                stress_conf = 100.0 * (1.0 - float(stress_res.get("prob_deplete", 0.0)))
                conf_delta = float(stress_conf - base_conf)

                stress_spend = _sustainable_spend_mc_cached_with_extensions(
                    s_mod,
                    target_success=float(target_success),
                    n_sims=int(max(600, int(n_sims // 2))),
                    seed=int(seed),
                    pre_sigma=float(pre_sigma),
                    post_sigma=float(post_sigma),
                    infl_sigma=float(infl_sigma),
                    stress_shocks_json=_safe_json_dumps(shocks),
                )["spend"]
                spend_impact = float(stress_spend - base_spend)

                rec = _stress_test_recovery_time(base_res, stress_res, label)

                rows.append(
                    {
                        "scenario": label,
                        "confidence_delta": conf_delta,
                        "spend_impact": f"{spend_impact:+,.0f}/yr",
                        "recovery_time": rec,
                    }
                )

            # Persist for report tab
            st.session_state["phase2_baseline_mc"] = base_res
            st.session_state["phase2_stress_rows"] = rows
            st.session_state["phase2_narrative"] = _narrative_insights(base_res, rows, None)

        st.success("Stress tests complete.")

    base_res = st.session_state.get("phase2_baseline_mc")
    rows = st.session_state.get("phase2_stress_rows", [])
    if base_res:
        base_conf = 100.0 * (1.0 - float(base_res.get("prob_deplete", 0.0)))
        st.markdown("### Baseline reference")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Baseline Confidence (0–100)", f"{base_conf:.0f}")
        with c2:
            st.metric("Chance of depletion", f"{float(base_res.get('prob_deplete', 0.0))*100:.1f}%")
        with c3:
            st.metric("Median ending balance", f"${float(base_res.get('median_final', 0.0)):,.0f}")

    if rows:
        st.markdown("### Stress test results")
        df_rows = pd.DataFrame(rows)
        st.dataframe(df_rows, use_container_width=True)

        # Narrative (Phase 2.G)
        st.markdown("### AI-Powered Narrative Insights")
        narr = st.session_state.get("phase2_narrative", {})
        for sec, items in [
            ("Key insights", narr.get("insights", [])),
            ("Risks", narr.get("risks", [])),
            ("Tradeoffs", narr.get("tradeoffs", [])),
            ("What matters most", narr.get("what_matters_most", [])),
        ]:
            with st.expander(sec, expanded=(sec == "Key insights")):
                if not items:
                    st.write("(none)")
                else:
                    for it in items:
                        st.write(f"- {it}")

    if not rows and not run_suite:
        st.info("Select one or more stress tests above and click **Run selected stress tests**.")


# =============================================================================
# TAB 5: LIFE EVENTS ENGINE
# =============================================================================
with tab5:
    st.header("One-Time Retirement Readiness Report")
    st.caption(
        "A shareable PDF that bundles: baseline simulation, stress test deltas, modeled life events, and plain-language insights."
    )

    snap = normalize_snapshot(get_current_inputs_snapshot())

    if "phase2_baseline_mc" not in st.session_state:
        st.info("Run stress tests at least once (Tab: Market Reality Mode) to populate the report, or generate a baseline-only report below.")

    colr1, colr2 = st.columns([1, 1])
    with colr1:
        n_sims = st.number_input("Simulations for report", 300, 10_000, 2000, 100, key="phase2_report_n_sims")
    with colr2:
        seed = st.number_input("Random seed (report)", 0, 999_999, 42, 1, key="phase2_report_seed")

    gen_report = st.button("Generate Retirement Readiness Report (PDF)", type="primary", key="phase2_gen_report_btn")
    if gen_report:
        with st.spinner("Building report..."):
            base_res = st.session_state.get("phase2_baseline_mc")
            if not base_res:
                base_res = _mc_cached_with_extensions(
                    snap,
                    n_sims=int(n_sims),
                    seed=int(seed),
                    pre_sigma=0.15,
                    post_sigma=0.10,
                    infl_sigma=0.01,
                    stress_shocks_json="",
                )
            rows = st.session_state.get("phase2_stress_rows", [])
            narrative = st.session_state.get("phase2_narrative") or _narrative_insights(base_res, rows, None)

            # Assemble report payload from other tabs (best-effort)
            # Assemble report payload from other tabs (uses last computed artifacts in this session)
            report_payload = {
                'generated_at': datetime.now().isoformat(),
                'compare': st.session_state.get('report_compare', {}),
                'range_outcomes': st.session_state.get('report_range_outcomes', {}),
                'stress_tests': {'rows': st.session_state.get('phase2_stress_rows', [])},
            }

            pdf_bytes = _build_readiness_report_pdf_bytes(
                user_id=st.session_state.get("current_user", "unknown"),
                snap=snap,
                baseline_res=base_res,
                stress_rows=rows,
                narrative=narrative,
        report_payload=report_payload,
            )
            st.session_state["phase2_readiness_pdf"] = pdf_bytes
        st.success("Report generated.")

    pdf_bytes = st.session_state.get("phase2_readiness_pdf")
    if pdf_bytes:
        st.download_button(
            "Download Retirement Readiness Report (PDF)",
            data=pdf_bytes,
            file_name=f"retirement_readiness_report_{st.session_state.get('current_user','user')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# =============================================================================
# PASSWORD MANAGEMENT (APPENDED AT END TO AVOID CHANGING EXISTING SIDEBAR WIDGET IDS)
# =============================================================================
def _render_password_management_sidebar():
    if not st.session_state.get("is_authenticated", False):
        return
    _u = st.session_state.get("current_user") or ""
    if not _u:
        return

    # -----------------------------
    # Change Password (post-login)
    # -----------------------------
    with st.sidebar.expander("Change Password", expanded=False):
        if not _sb_enabled():
            st.info("Password change requires Supabase to be configured for this app.")
        else:
            with st.form("change_password_form", clear_on_submit=True):
                cur_pwd = st.text_input("Current password", type="password")
                new_pwd = st.text_input("New password", type="password")
                new_pwd2 = st.text_input("Confirm new password", type="password")
                submitted_pw = st.form_submit_button("Update password")

            if submitted_pw:
                if not cur_pwd or not new_pwd or not new_pwd2:
                    st.error("Please fill in all password fields.")
                    st.stop()
                if new_pwd != new_pwd2:
                    st.error("New password and confirmation do not match.")
                    st.stop()
                if len(new_pwd) < 8:
                    st.error("New password must be at least 8 characters.")
                    st.stop()
                if new_pwd == cur_pwd:
                    st.error("New password must be different from the current password.")
                    st.stop()

                # Determine auth source by presence of Supabase row (enforced at login)
                row = sb_get_user_credentials(_u) if _sb_enabled() else None
                old_hash_for_log = None
                if row:
                    old_hash_for_log = row.get("password_hash")
                    if not sb_verify_user_password(_u, cur_pwd):
                        st.error("Current password is incorrect.")
                        st.stop()
                else:
                    # secrets fallback verification
                    allowed_users = st.secrets.get("auth", {}).get("users", {})
                    salt = st.secrets.get("auth", {}).get("salt", "")
                    expected_hash = allowed_users.get(_u)
                    if not expected_hash or not salt:
                        st.error("Secrets-based auth is not configured; cannot verify current password.")
                        st.stop()
                    old_hash_for_log = expected_hash
                    computed_hash = _hash_password(cur_pwd, salt)
                    if not hmac.compare_digest(computed_hash, expected_hash):
                        st.error("Current password is incorrect.")
                        st.stop()

                ok = sb_set_user_password(
                    _u,
                    new_pwd,
                    old_password_hash=str(old_hash_for_log) if old_hash_for_log else None,
                    changed_by=_u,
                    change_reason="user_initiated",
                )
                if ok:
                    st.success("Password updated successfully. Please use the new password next time you sign in.")
                    st.rerun()
                else:
                    st.error("Unable to update password due to a Supabase error. Please try again.")
                    st.stop()

    # -----------------------------
    # Admin Reset (only for ranabir)
    # -----------------------------
    if _u == "ranabir":
        with st.sidebar.expander("Admin: Reset User Password", expanded=False):
            if not _sb_enabled():
                st.info("Admin password reset requires Supabase to be configured for this app.")
            else:
                with st.form("admin_reset_password_form", clear_on_submit=True):
                    target_user = st.text_input("Target user ID")
                    admin_new_pwd = st.text_input("New password", type="password")
                    admin_new_pwd2 = st.text_input("Confirm new password", type="password")
                    submitted_admin = st.form_submit_button("Reset password")

                if submitted_admin:
                    if not target_user or not admin_new_pwd or not admin_new_pwd2:
                        st.error("Please fill in all fields.")
                        st.stop()
                    if admin_new_pwd != admin_new_pwd2:
                        st.error("New password and confirmation do not match.")
                        st.stop()
                    if len(admin_new_pwd) < 8:
                        st.error("New password must be at least 8 characters.")
                        st.stop()

                    existing = sb_get_user_credentials(target_user)
                    old_hash = existing.get("password_hash") if existing else None

                    ok = sb_set_user_password(
                        target_user,
                        admin_new_pwd,
                        old_password_hash=str(old_hash) if old_hash else None,
                        changed_by=_u,
                        change_reason="admin_reset",
                        metadata={"reset_by": _u},
                    )
                    if ok:
                        st.success(f"Password reset successfully for user '{target_user}'.")
                        st.rerun()
                    else:
                        st.error("Unable to reset password due to a Supabase error. Please try again.")
                        st.stop()

# Render at the very end to avoid shifting any existing sidebar widgets / state
_render_password_management_sidebar()


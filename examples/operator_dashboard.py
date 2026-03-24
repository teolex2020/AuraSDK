"""Lightweight operator dashboard for Aura.

Demonstrates a minimal operator-facing review UI over existing read-only APIs:

- memory health digest
- maintenance trend history
- recent correction log
- cross-namespace analytics

Requirements:
    pip install aura-memory fastapi uvicorn

Run:
    uvicorn examples.operator_dashboard:app --reload

Open:
    http://127.0.0.1:8090
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from aura import Aura

BRAIN_PATH = os.environ.get("AURA_BRAIN_PATH", "./operator_dashboard_brain")

_brain: Optional[Aura] = None


def get_brain() -> Aura:
    assert _brain is not None, "Brain not initialized"
    return _brain


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _brain
    _brain = Aura(BRAIN_PATH)
    yield
    if _brain:
        _brain.close()


app = FastAPI(
    title="Aura Operator Dashboard",
    description="Lightweight review dashboard for memory health and operator surfaces.",
    lifespan=lifespan,
)


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aura Operator Dashboard</title>
  <style>
    :root {
      --bg: #f3efe6;
      --panel: #fffaf1;
      --ink: #1f2a1f;
      --muted: #667164;
      --line: #d8cfbf;
      --accent: #166b5a;
      --alert: #a13d2d;
      --warn: #b7821f;
      --shadow: 0 14px 32px rgba(45, 41, 33, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(22,107,90,0.08), transparent 28%),
        linear-gradient(180deg, #f8f4eb 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 18px 48px;
    }
    .hero {
      display: grid;
      gap: 10px;
      margin-bottom: 22px;
    }
    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    h1 {
      margin: 0;
      font-size: clamp(34px, 5vw, 56px);
      line-height: 0.95;
    }
    .sub {
      margin: 0;
      color: var(--muted);
      max-width: 760px;
      font-size: 17px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 16px;
    }
    .card {
      grid-column: span 12;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 18px;
    }
    .card h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 12px;
    }
    .metric {
      background: rgba(22,107,90,0.05);
      border: 1px solid rgba(22,107,90,0.12);
      border-radius: 14px;
      padding: 12px;
    }
    .metric b {
      display: block;
      font-size: 28px;
      line-height: 1;
      margin-bottom: 6px;
    }
    .metric span {
      color: var(--muted);
      font-size: 13px;
    }
    .wide-7 { grid-column: span 7; }
    .wide-5 { grid-column: span 5; }
    .wide-6 { grid-column: span 6; }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    input, button {
      font: inherit;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 10px 14px;
      background: #fff;
    }
    button {
      cursor: pointer;
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    .pill {
      display: inline-block;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      border: 1px solid currentColor;
    }
    .sev-high { color: var(--alert); }
    .sev-medium { color: var(--warn); }
    .sev-low { color: var(--accent); }
    .mono { font-family: "Consolas", "SFMono-Regular", monospace; }
    @media (max-width: 860px) {
      .wide-7, .wide-5, .wide-6 { grid-column: span 12; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Aura v4 Operator Pattern</div>
      <h1>Memory Health Review</h1>
      <p class="sub">
        Lightweight dashboard over bounded operator surfaces: health digest, recent corrections,
        maintenance trends, and cross-namespace analytics.
      </p>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Memory Health</h2>
        <div id="metrics" class="metrics"></div>
      </article>

      <article class="card wide-7">
        <h2>Top Review Issues</h2>
        <table>
          <thead><tr><th>Kind</th><th>Target</th><th>Namespace</th><th>Severity</th><th>Score</th></tr></thead>
          <tbody id="issues-body"></tbody>
        </table>
      </article>

      <article class="card wide-5">
        <h2>Recent Corrections</h2>
        <table>
          <thead><tr><th>Time</th><th>Target</th><th>Operation</th><th>Reason</th></tr></thead>
          <tbody id="corrections-body"></tbody>
        </table>
      </article>

      <article class="card wide-6">
        <h2>Maintenance Trends</h2>
        <table>
          <thead><tr><th>Timestamp</th><th>Volatile</th><th>Corrections</th><th>Dominant Phase</th></tr></thead>
          <tbody id="trends-body"></tbody>
        </table>
      </article>

      <article class="card wide-6">
        <h2>Instability Hotspots</h2>
        <table>
          <thead><tr><th>Belief</th><th>Namespace</th><th>Volatility</th><th>Stability</th></tr></thead>
          <tbody id="instability-body"></tbody>
        </table>
      </article>

      <article class="card wide-6">
        <h2>Cross-Namespace Digest</h2>
        <div class="toolbar">
          <input id="namespaces-input" placeholder="default,sandbox" />
          <button onclick="loadDigest()">Refresh Digest</button>
        </div>
        <table>
          <thead><tr><th>Namespace</th><th>Records</th><th>Top Concepts</th><th>Corrections</th></tr></thead>
          <tbody id="digest-body"></tbody>
        </table>
      </article>
    </section>
  </div>

  <script>
    async function fetchJson(path) {
      const res = await fetch(path);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status} for ${path}`);
      }
      return await res.json();
    }

    function metricCard(label, value) {
      return `<div class="metric"><b>${value}</b><span>${label}</span></div>`;
    }

    async function loadHealth() {
      const payload = await fetchJson('/api/memory-health');
      const d = payload.digest;
      document.getElementById('metrics').innerHTML = [
        metricCard('Total Records', d.total_records),
        metricCard('High Volatility', d.high_volatility_belief_count),
        metricCard('Low Stability', d.low_stability_belief_count),
        metricCard('Recent Corrections', d.recent_correction_count),
        metricCard('Suppressed Policy', d.suppressed_policy_hint_count),
        metricCard('Trend', d.maintenance_trend_direction),
      ].join('');

      document.getElementById('issues-body').innerHTML = d.top_issues.map(issue => `
        <tr>
          <td>${issue.kind}</td>
          <td class="mono">${issue.target_id}</td>
          <td>${issue.namespace}</td>
          <td><span class="pill sev-${issue.severity}">${issue.severity}</span></td>
          <td>${issue.score.toFixed(2)}</td>
        </tr>
      `).join('') || '<tr><td colspan="5">No current issues</td></tr>';
    }

    async function loadCorrections() {
      const payload = await fetchJson('/api/corrections');
      document.getElementById('corrections-body').innerHTML = payload.entries.map(entry => `
        <tr>
          <td>${entry.time_iso}</td>
          <td class="mono">${entry.target_kind}:${entry.target_id}</td>
          <td>${entry.operation}</td>
          <td>${entry.reason}</td>
        </tr>
      `).join('') || '<tr><td colspan="4">No corrections yet</td></tr>';
    }

    async function loadTrends() {
      const payload = await fetchJson('/api/trends');
      document.getElementById('trends-body').innerHTML = payload.recent.map(item => `
        <tr>
          <td>${item.timestamp}</td>
          <td>${item.volatile_records}</td>
          <td>${item.correction_events}</td>
          <td>${item.dominant_phase}</td>
        </tr>
      `).join('') || '<tr><td colspan="4">No trend history yet</td></tr>';
    }

    async function loadInstability() {
      const payload = await fetchJson('/api/instability-hotspots');
      document.getElementById('instability-body').innerHTML = payload.items.map(item => `
        <tr>
          <td class="mono">${item.belief_id}</td>
          <td>${item.namespace}</td>
          <td>${item.volatility.toFixed(2)}</td>
          <td>${item.stability.toFixed(2)}</td>
        </tr>
      `).join('') || '<tr><td colspan="4">No instability hotspots</td></tr>';
    }

    async function loadDigest() {
      const namespaces = document.getElementById('namespaces-input').value;
      const query = namespaces ? `?namespaces=${encodeURIComponent(namespaces)}` : '';
      const payload = await fetchJson('/api/cross-namespace-digest' + query);
      document.getElementById('digest-body').innerHTML = payload.namespaces.map(item => `
        <tr>
          <td>${item.namespace}</td>
          <td>${item.record_count}</td>
          <td>${(item.top_concepts || []).map(c => c.key).join(', ') || '—'}</td>
          <td>${item.correction_count ?? '—'}</td>
        </tr>
      `).join('') || '<tr><td colspan="4">No namespace data</td></tr>';
    }

    async function boot() {
      await Promise.all([loadHealth(), loadCorrections(), loadTrends(), loadInstability(), loadDigest()]);
    }

    boot().catch(err => {
      document.body.insertAdjacentHTML('beforeend', `<pre>${err}</pre>`);
    });
  </script>
</body>
</html>
"""


def memory_health_to_dict(digest) -> dict:
    return {
        "total_records": digest.total_records,
        "startup_has_recovery_warnings": digest.startup_has_recovery_warnings,
        "high_volatility_belief_count": digest.high_volatility_belief_count,
        "low_stability_belief_count": digest.low_stability_belief_count,
        "recent_correction_count": digest.recent_correction_count,
        "suppressed_policy_hint_count": digest.suppressed_policy_hint_count,
        "rejected_policy_hint_count": digest.rejected_policy_hint_count,
        "policy_pressure_area_count": digest.policy_pressure_area_count,
        "maintenance_trend_direction": digest.maintenance_trend_direction,
        "latest_dominant_phase": digest.latest_dominant_phase,
        "top_issues": [
            {
                "kind": issue.kind,
                "target_id": issue.target_id,
                "namespace": issue.namespace,
                "title": issue.title,
                "score": issue.score,
                "severity": issue.severity,
            }
            for issue in digest.top_issues
        ],
    }


def maintenance_summary_to_dict(summary) -> dict:
    return {
        "snapshot_count": summary.snapshot_count,
        "avg_belief_churn": summary.avg_belief_churn,
        "avg_causal_rejection_rate": summary.avg_causal_rejection_rate,
        "avg_policy_suppression_rate": summary.avg_policy_suppression_rate,
        "avg_cycle_time_ms": summary.avg_cycle_time_ms,
        "avg_correction_events": summary.avg_correction_events,
        "total_corrections_in_window": summary.total_corrections_in_window,
        "latest_dominant_phase": summary.latest_dominant_phase,
        "recent": [
            {
                "timestamp": item.timestamp,
                "total_records": item.total_records,
                "records_archived": item.records_archived,
                "insights_found": item.insights_found,
                "volatile_records": item.volatile_records,
                "belief_churn": item.belief_churn,
                "causal_rejection_rate": item.causal_rejection_rate,
                "policy_suppression_rate": item.policy_suppression_rate,
                "feedback_beliefs_touched": item.feedback_beliefs_touched,
                "feedback_net_confidence_delta": item.feedback_net_confidence_delta,
                "feedback_net_volatility_delta": item.feedback_net_volatility_delta,
                "correction_events": item.correction_events,
                "cumulative_corrections": item.cumulative_corrections,
                "cycle_time_ms": item.cycle_time_ms,
                "dominant_phase": item.dominant_phase,
            }
            for item in summary.recent
        ],
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return HTML_PAGE


@app.get("/api/memory-health")
def memory_health(brain: Aura = Depends(get_brain)):
    return JSONResponse({"digest": memory_health_to_dict(brain.get_memory_health_digest(8))})


@app.get("/api/corrections")
def corrections(limit: int = Query(default=10, ge=1, le=50), brain: Aura = Depends(get_brain)):
    entries = brain.get_correction_log()
    entries = sorted(entries, key=lambda item: item["timestamp"], reverse=True)[:limit]
    return JSONResponse({"entries": entries})


@app.get("/api/trends")
def trends(brain: Aura = Depends(get_brain)):
    return JSONResponse(maintenance_summary_to_dict(brain.get_maintenance_trend_summary()))


@app.get("/api/instability-hotspots")
def instability_hotspots(
    limit: int = Query(default=8, ge=1, le=24),
    brain: Aura = Depends(get_brain),
):
    high_volatility = brain.get_high_volatility_beliefs(limit)
    low_stability = brain.get_low_stability_beliefs(limit)
    merged: dict[str, dict] = {}
    for item in high_volatility + low_stability:
        belief_id = item["belief_id"]
        current = merged.get(belief_id)
        if current is None:
            merged[belief_id] = dict(item)
            continue
        current["volatility"] = max(current["volatility"], item["volatility"])
        current["stability"] = min(current["stability"], item["stability"])
    items = sorted(
        merged.values(),
        key=lambda item: (-item["volatility"], item["stability"], item["belief_id"]),
    )[:limit]
    return JSONResponse({"items": items})


@app.get("/api/cross-namespace-digest")
def cross_namespace_digest(
    namespaces: str | None = Query(default=None),
    brain: Aura = Depends(get_brain),
):
    namespace_list = None
    if namespaces:
        namespace_list = [item.strip() for item in namespaces.split(",") if item.strip()]
    digest = brain.cross_namespace_digest(
        namespaces=namespace_list,
        top_concepts_limit=3,
        min_record_count=1,
        include_dimensions=["concepts", "belief_states", "corrections"],
        compact_summary=False,
    )
    return JSONResponse(digest)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("examples.operator_dashboard:app", host="127.0.0.1", port=8090, reload=True)

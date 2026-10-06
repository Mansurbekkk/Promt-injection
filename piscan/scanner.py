"""The scanner: ties everything together.

For each app, task, goal and payload it:
  1. runs a clean baseline (no payload) once per task,
  2. runs the contaminated version,
  3. checks the attacker goal (success),
  4. runs the competence gate,
  5. diagnoses the failure layer (V1/V2/V3, lambda),
and writes one record per run to a JSONL file.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from .apps import APPS
from .apps.base import App, Injection
from .diagnose import diagnose
from .gate import competence_gate
from .payloads import build_payloads, expected_langs


def _carrier_for(app: App, task) -> str:
    return task.carrier


def scan_app(backend, app: App, out_fp, scripts=None, templates=None, verbose=False):
    scripts = set(scripts) if scripts else None
    templates = set(templates) if templates else None
    records = []

    # 1. clean baselines, one per task
    clean = {}
    for task in app.tasks:
        clean[task.id] = app.run(backend, task, injection=None)

    for task in app.tasks:
        carrier = _carrier_for(app, task)
        for goal in app.goals:
            for pl in build_payloads(goal.id, goal.cls, goal.instruction):
                if scripts and pl.script not in scripts:
                    continue
                if templates and pl.template not in templates:
                    continue
                inj = Injection(carrier=carrier, payload=pl.text, position="end")
                tr = app.run(backend, task, injection=inj)
                success = bool(goal.check(tr)) and not bool(goal.check(clean[task.id]))
                gate = competence_gate(tr.final, expected_langs(pl.script))
                # a run that makes a tool call is engaged even if final text is short
                competent = gate.passed or bool(tr.tool_calls)
                diag = diagnose(tr, clean[task.id], pl.text, success)
                rec = {
                    "app": app.name, "task": task.id, "carrier": carrier,
                    "goal": goal.id, "goal_cls": goal.cls,
                    "template": pl.template, "script": pl.script,
                    "success": success,
                    "competent": competent, "gate_pass": gate.passed,
                    "gate_reason": gate.reason, "language": gate.language,
                    "drift": gate.drift,
                    "v1": diag.v1, "v2": diag.v2, "v3": diag.v3, "layer": diag.layer,
                    "n_tool_calls": len(tr.tool_calls),
                    "final_len": len(tr.final or ""),
                    "error": tr.error,
                    # kept for offline re-analysis and paper examples
                    "payload": pl.text,
                    "final": (tr.final or "")[:600],
                    "tool_calls": [{"name": c.name, "arguments": c.arguments}
                                   for c in tr.tool_calls],
                }
                records.append(rec)
                out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if verbose:
                    print(f"{app.name} {task.id} {goal.id} {pl.template}/{pl.script} "
                          f"success={success} competent={competent} layer={diag.layer}")
    return records


def scan_all(backend, out_path, apps=None, **kw):
    apps = apps or list(APPS)
    allrecs = []
    with open(out_path, "w", encoding="utf-8") as fp:
        for name in apps:
            allrecs += scan_app(backend, APPS[name], fp, **kw)
    return allrecs

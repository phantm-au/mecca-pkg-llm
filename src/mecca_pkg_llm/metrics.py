"""BOM accuracy metrics — score a predicted packaging Bill-of-Materials against gold.

Metrics (per the mecca-llm phase-5 metric set, adapted):
  parse_success_rate      — fraction of predictions that are valid JSON
  schema_compliance_rate  — fraction with exactly {PP, SP, TP} top-level keys
  material_type_set_f1    — set-F1 over (layer, material_type) across the BOM
  material_abbrev_set_f1  — set-F1 over (layer, material_name) — the discriminating signal
  mass_g_total_mape       — mean abs % error on total predicted mass vs gold
  component_count_mae      — mean abs error in component count per layer

These are order-insensitive (set-based) because a BOM is an unordered bag of components.
"""
from __future__ import annotations

import json
from typing import Any

LAYERS = ("PP", "SP", "TP")
EXPECTED_KEYS = frozenset(LAYERS)


def try_parse(text: str) -> dict[str, Any] | None:
    """Parse model output to a BOM dict; tolerate ```json fences and trailing prose."""
    if not text:
        return None
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                s = p
                break
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(s[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _material_set(bom: dict[str, Any], key: str) -> set[tuple[str, str]]:
    """Set of (layer, value) over all materials, value = material[key]."""
    out: set[tuple[str, str]] = set()
    for layer in LAYERS:
        for comp in bom.get(layer) or []:
            for m in comp.get("materials") or []:
                v = m.get(key)
                if v is not None:
                    out.add((layer, str(v)))
    return out


def _total_mass(bom: dict[str, Any]) -> float:
    tot = 0.0
    for layer in LAYERS:
        for comp in bom.get(layer) or []:
            for m in comp.get("materials") or []:
                v = m.get("mass_g")
                if isinstance(v, (int, float)):
                    tot += float(v)
    return tot


def _comp_counts(bom: dict[str, Any]) -> dict[str, int]:
    return {layer: len(bom.get(layer) or []) for layer in LAYERS}


def _set_f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec = tp / len(gold)
    return 2 * prec * rec / (prec + rec)


def score_one(pred_text: str, gold: dict[str, Any]) -> dict[str, Any]:
    """Score a single prediction. Returns per-example metric components."""
    pred = try_parse(pred_text)
    parsed = pred is not None
    schema_ok = parsed and frozenset(pred.keys()) == EXPECTED_KEYS

    res: dict[str, Any] = {
        "parsed": parsed,
        "schema_ok": schema_ok,
        "type_f1": 0.0,
        "abbrev_f1": 0.0,
        "mass_ape": None,
        "comp_mae": None,
    }
    if not parsed:
        return res

    res["type_f1"] = _set_f1(_material_set(pred, "material_type"),
                             _material_set(gold, "material_type"))
    res["abbrev_f1"] = _set_f1(_material_set(pred, "material_name"),
                               _material_set(gold, "material_name"))

    g_mass = _total_mass(gold)
    p_mass = _total_mass(pred)
    if g_mass > 0:
        res["mass_ape"] = abs(p_mass - g_mass) / g_mass

    pc, gc = _comp_counts(pred), _comp_counts(gold)
    res["comp_mae"] = sum(abs(pc[l] - gc[l]) for l in LAYERS) / len(LAYERS)
    return res


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-example scores into corpus metrics."""
    n = len(scores)
    if n == 0:
        return {"n": 0}

    def _mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    apes = [s["mass_ape"] for s in scores if s["mass_ape"] is not None]
    maes = [s["comp_mae"] for s in scores if s["comp_mae"] is not None]
    return {
        "n": n,
        "parse_success_rate": _mean([1.0 if s["parsed"] else 0.0 for s in scores]),
        "schema_compliance_rate": _mean([1.0 if s["schema_ok"] else 0.0 for s in scores]),
        "material_type_set_f1": _mean([s["type_f1"] for s in scores]),
        "material_abbrev_set_f1": _mean([s["abbrev_f1"] for s in scores]),
        "mass_g_total_mape": _mean(apes),
        "component_count_mae": _mean(maes),
    }

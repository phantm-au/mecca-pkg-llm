"""Deterministic BOM enrichment — sustainability metrics from a catalog, NOT the model.

The fine-tuned model predicts WHAT (components, materials, mass, dimensions). It must NEVER
be asked to predict carbon / water / recyclability — those would be hallucinated. Instead we
JOIN each predicted material to a real environmental catalog (catalog/env_catalog.json,
sourced from MECCA's materials data) and compute trustworthy aggregates. This mirrors how
mecca-streamlit's enrich_bom() works and satisfies "accurate to real life, do not assume".

What we compute (trustworthy):
  - total + per-tier mass (g)
  - mass-weighted average recyclability % (from catalog recycling_potential)
  - mass-weighted average recycled-content % (model-predicted recycled_content_percent)
  - total carbon (kg CO2e)  = sum over materials of (carbon_kg_per_kg * mass_kg)
  - total water (L)         = sum over materials of (water_per_kg * mass_kg)
  - component shape (box / cylindrical / flat / unknown) + volume from dimensions_mm

Catalog carbon_kg / water_consumption are per-KG intensities, so we scale by each material's
mass to get an absolute footprint — unlike mecca-streamlit which left them per-material
unscaled. (If a value is missing we skip it and flag partial coverage.)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LAYERS = ("PP", "SP", "TP")
_CATALOG_PATH = Path(__file__).resolve().parents[2] / "catalog" / "env_catalog.json"


def load_env_catalog(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    p = Path(path) if path else _CATALOG_PATH
    data = json.loads(p.read_text())
    return data["materials"]


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None

def derive_shape(dims: dict[str, Any] | None) -> tuple[str, float | None]:
    """Infer component shape and volume (cm³) from outer L×W×H (mm).

    Returns (shape, volume_cm3). Shape heuristic:
      - two ~equal small sides + one long side  -> "cylindrical" (tube/stick/bottle)
      - one dimension much smaller than the others -> "flat" (label/leaflet/wrap)
      - otherwise -> "box"
    Volume is the bounding-box volume (L*W*H), a reasonable proxy for footprint.
    """
    if not isinstance(dims, dict):
        return "unknown", None
    l, w, h = (_num(dims.get("l")), _num(dims.get("w")), _num(dims.get("h")))
    if l is None or w is None or h is None or min(l, w, h) <= 0:
        return "unknown", None
    vol_cm3 = round((l * w * h) / 1000.0, 2)  # mm³ -> cm³
    smallest, mid, largest = sorted([l, w, h])
    if smallest <= 0.25 * mid:
        return "flat", vol_cm3
    if smallest >= 0.7 * mid and largest >= 1.5 * mid:
        return "cylindrical", vol_cm3
    return "box", vol_cm3


def enrich_bom(bom: dict[str, Any], catalog: dict[str, dict[str, Any]] | None = None
               ) -> dict[str, Any]:
    """Annotate a predicted BOM with per-material env data + per-tier/total rollups.

    Returns a NEW dict:
      {
        "bom": <input bom, each material gets an "_env" block, each component a "_shape">,
        "rollup": {"total": {...}, "PP": {...}, "SP": {...}, "TP": {...}},
        "coverage": {"materials": N, "with_env": M},  # how many joined to the catalog
      }
    """
    cat = catalog if catalog is not None else load_env_catalog()

    def _blank_roll() -> dict[str, Any]:
        return {"mass_g": 0.0, "n_components": 0, "n_materials": 0,
                "carbon_kg": 0.0, "water_l": 0.0,
                "_rec_num": 0.0, "_rec_mass": 0.0, "_rc_num": 0.0, "_rc_mass": 0.0,
                "carbon_partial": False, "water_partial": False}

    rolls = {t: _blank_roll() for t in LAYERS}
    n_mat = with_env = 0

    for tier in LAYERS:
        roll = rolls[tier]
        for comp in bom.get(tier) or []:
            roll["n_components"] += 1
            comp["_shape"], comp["_volume_cm3"] = derive_shape(comp.get("dimensions_mm"))
            for mat in comp.get("materials") or []:
                n_mat += 1
                roll["n_materials"] += 1
                mass_g = _num(mat.get("mass_g")) or 0.0
                roll["mass_g"] += mass_g
                mass_kg = mass_g / 1000.0

                entry = cat.get(mat.get("material_name") or "")
                if entry:
                    with_env += 1
                    recyc = _num(entry.get("recycling_potential"))
                    carbon_int = _num(entry.get("carbon_kg"))         # kg CO2e per kg material
                    water_int = _num(entry.get("water_consumption"))  # L per kg material
                    mat["_env"] = {
                        "recycling_potential": recyc,
                        "carbon_kg": round(carbon_int * mass_kg, 4) if carbon_int is not None else None,
                        "water_l": round(water_int * mass_kg, 2) if water_int is not None else None,
                        "carbon_intensity_per_kg": carbon_int,
                        "water_intensity_per_kg": water_int,
                    }
                    if recyc is not None and mass_g > 0:
                        roll["_rec_num"] += recyc * mass_g
                        roll["_rec_mass"] += mass_g
                    if carbon_int is not None:
                        roll["carbon_kg"] += carbon_int * mass_kg
                    else:
                        roll["carbon_partial"] = True
                    if water_int is not None:
                        roll["water_l"] += water_int * mass_kg
                    else:
                        roll["water_partial"] = True
                else:
                    mat["_env"] = None

                rc = _num(mat.get("recycled_content_percent"))
                if rc is not None and mass_g > 0:
                    roll["_rc_num"] += rc * mass_g
                    roll["_rc_mass"] += mass_g

    def _finalize(roll: dict[str, Any]) -> dict[str, Any]:
        out = {
            "mass_g": round(roll["mass_g"], 2),
            "n_components": roll["n_components"],
            "n_materials": roll["n_materials"],
            "carbon_kg": round(roll["carbon_kg"], 4),
            "water_l": round(roll["water_l"], 2),
            "recyclability_pct": (round(roll["_rec_num"] / roll["_rec_mass"], 1)
                                  if roll["_rec_mass"] > 0 else None),
            "recycled_content_pct": (round(roll["_rc_num"] / roll["_rc_mass"], 1)
                                     if roll["_rc_mass"] > 0 else None),
            "carbon_partial": roll["carbon_partial"],
            "water_partial": roll["water_partial"],
        }
        return out

    per_tier = {t: _finalize(rolls[t]) for t in LAYERS}

    tot = _blank_roll()
    for t in LAYERS:
        r = rolls[t]
        for k in ("mass_g", "n_components", "n_materials", "carbon_kg", "water_l",
                  "_rec_num", "_rec_mass", "_rc_num", "_rc_mass"):
            tot[k] += r[k]
        tot["carbon_partial"] = tot["carbon_partial"] or r["carbon_partial"]
        tot["water_partial"] = tot["water_partial"] or r["water_partial"]
    total = _finalize(tot)

    return {
        "bom": bom,
        "rollup": {"total": total, **per_tier},
        "coverage": {"materials": n_mat, "with_env": with_env},
    }

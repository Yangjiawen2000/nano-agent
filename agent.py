"""
Phase 4: ReAct Agent for Brain-Targeted Nanoparticle Design Optimization

Architecture:
  CausalMemory (structural knowledge)
       +
  PBPK Simulator (mechanistic oracle)
       +
  LLM backend: DeepSeek (via OpenAI-compatible API) or Anthropic
  ↓ ReAct loop: Thought → Tool call → Observation → Thought ...

Five tools:
  1. pbpk_simulate        – run mechanistic ODE and return AUC_brain
  2. query_causal_graph   – query learned DAG (bottleneck / paths / effect)
  3. lookup_parameter     – retrieve literature feasibility ranges
  4. check_feasibility    – validate physical/biological constraints
  5. compare_designs      – side-by-side diff with causal explanations

Backend selection (priority):
  1. DEEPSEEK_API_KEY env var → DeepSeek (deepseek-chat, OpenAI-compatible)
  2. ANTHROPIC_API_KEY env var → Anthropic (claude-sonnet-4-6)
"""

from __future__ import annotations

import json
import math
import sys
import os
import time
import textwrap
from typing import Any

import numpy as np

# Backend imports (lazy)
try:
    from openai import OpenAI as _OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    import anthropic as _anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from pbpk_simulator import pbpk_simulate as _pbpk_simulate
from causal_graph   import load_causal_memory

# ─────────────────────────────────────────────────────────────────────────────
# Constants & feasibility table
# ─────────────────────────────────────────────────────────────────────────────

FEASIBILITY = {
    "size_nm":         {"min": 20,   "max": 200,  "optimal": "50-100 nm for RMT",             "unit": "nm"},
    "zeta_mv":         {"min": -40,  "max": 10,   "optimal": "-20 to -10 mV (mild negative)", "unit": "mV"},
    "ligand_density":  {"min": 0,    "max": 200,  "optimal": "20-50 (avidity trap >100)",      "unit": "ligands/NP"},
    "peg":             {"min": 0,    "max": 1,    "optimal": "1 reduces MPS clearance",        "unit": "0/1"},
    "drug_loading":    {"min": 0,    "max": 30,   "optimal": "5-20% w/w (>20% → instability)", "unit": "% w/w"},
    "hydrophobicity":  {"min": 0.0,  "max": 1.0,  "optimal": "<0.3 (minimise protein corona)", "unit": "0-1"},
    "particle_shape":  {"choices": ["sphere", "rod", "disk", "worm"],
                        "optimal": "rod → 30% lower CL; sphere → best BBB k_bind"},
    "surface_coating": {"choices": ["none", "peg", "lipid", "zwitterionic", "polymer"],
                        "optimal": "zwitterionic → ↓CL 35%; lipid → ↑k_endo 25%"},
}

LIGAND_TYPES = ["transferrin", "anti-TfR", "rabies-peptide", "none"]

PARAMETER_INFO = {
    "size_nm":         "Hydrodynamic diameter in nm. Optimal ~80 nm for BBB-RMT.",
    "zeta_mv":         "Surface zeta potential in mV. Mild negative (−10 to −20 mV) maximises k_trans.",
    "peg":             "PEGylation (0/1). PEG reduces opsonisation and lowers MPS CL by ~55%.",
    "ligand_type":     f"Targeting ligand. Choices: {LIGAND_TYPES}. anti-TfR gives highest k_bind.",
    "ligand_density":  "Targeting ligands per NP. Avidity peak at 20-50; >100 → avidity trap.",
    "drug_loading":    "Drug payload (% w/w). Scales AUC_drug_brain; >20% → formulation instability penalty.",
    "particle_shape":  "NP shape: sphere|rod|disk|worm. Rod → ↓CL 30%, ↓k_bind 12% (Decuzzi PNAS 2010).",
    "hydrophobicity":  "Surface hydrophobicity 0–1. Drives protein corona → opsonisation → ↑CL (quadratic).",
    "surface_coating": "Coating type. Zwitterionic → ↓CL 35%; lipid → ↑k_endo 25% (membrane fusion).",
    "k_bind":          "Receptor binding rate (h⁻¹). Key bottleneck for BBB crossing (+0.62 total effect on AUCbrain).",
    "k_trans":         "Transcytosis rate from vesicle to brain (h⁻¹). Second bottleneck (+0.44). Driven by zeta.",
    "k_lyso":          "Lysosomal degradation rate (h⁻¹). Competes with k_trans (−0.01 net).",
    "CL":              "Systemic clearance (h⁻¹). Driven by size and PEG (−0.11 on AUCbrain).",
    "AUCbrain":        "Area under brain concentration–time curve (normalised). Primary optimisation target.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

_causal_memory = None

# Mutable run context — reset at the start of each run_agent call
_run_context: dict = {"history": [], "patient": "adult_healthy"}


def _get_memory():
    global _causal_memory
    if _causal_memory is None:
        json_path = os.path.join(os.path.dirname(__file__), "data", "causal_results.json")
        _causal_memory = load_causal_memory(json_path)
    return _causal_memory


def tool_pbpk_simulate(design: dict) -> dict:
    """
    Run PBPK ODE for a given NP design (core + optional extended params).
    Core keys:     size_nm, zeta_mv, peg (0/1), ligand_type, ligand_density
    Extended keys: drug_loading, particle_shape, hydrophobicity, surface_coating
    Returns: AUC_brain, AUC_drug_brain, AUC_ratio, kinetic params, success flag.
    """
    try:
        size_nm         = float(design.get("size_nm",         80))
        zeta_mv         = float(design.get("zeta_mv",        -15))
        peg             = "yes" if int(design.get("peg", 1)) else "no"
        ligand_type     = str(design.get("ligand_type",  "transferrin"))
        ligand_density  = float(design.get("ligand_density",  30))
        drug_loading    = float(design.get("drug_loading",     0.0))
        particle_shape  = str(design.get("particle_shape", "sphere"))
        hydrophobicity  = float(design.get("hydrophobicity",   0.0))
        surface_coating = str(design.get("surface_coating",  "none"))
        patient_type    = _run_context.get("patient", "adult_healthy")

        result = _pbpk_simulate(
            size_nm        = size_nm,
            zeta_mv        = zeta_mv,
            peg            = peg,
            ligand_type    = ligand_type,
            ligand_density = ligand_density,
            drug_loading   = drug_loading,
            particle_shape = particle_shape,
            hydrophobicity = hydrophobicity,
            surface_coating= surface_coating,
            patient_type   = patient_type,
        )
        p = result["params"]
        return {
            "AUC_brain":       round(result["AUC"],            6),
            "AUC_drug_brain":  round(result["AUC_drug_brain"], 6),
            "AUC_ratio":       round(result["AUC_ratio"],      6),
            "k_bind":          round(p.k_bind,  4),
            "k_trans":         round(p.k_trans, 4),
            "k_lyso":          round(p.k_lyso,  4),
            "CL":              round(p.CL,       4),
            "patient_type":    patient_type,
            "success":         result["success"],
            "design_used":     {
                "size_nm": size_nm, "zeta_mv": zeta_mv, "peg": peg,
                "ligand_type": ligand_type, "ligand_density": ligand_density,
                "drug_loading": drug_loading, "particle_shape": particle_shape,
                "hydrophobicity": hydrophobicity, "surface_coating": surface_coating,
            },
        }
    except Exception as e:
        return {"error": str(e), "success": False}


def tool_query_causal_graph(node: str, query_type: str) -> dict:
    """
    Query the learned causal DAG.
    query_type in {"bottleneck", "parents", "children", "paths_to_AUCbrain",
                   "total_effect", "recommend"}
    For "total_effect", pass node as "source→target" (e.g. "Zeta→AUCbrain").
    For "recommend", pass node as JSON design dict string.
    """
    mem = _get_memory()
    qt  = query_type.lower().strip()

    if qt == "bottleneck":
        text = mem.query_bottleneck(target="AUCbrain", top_k=9)
        return {"result": text}

    valid_nodes = list(mem._idx.keys())

    if qt == "parents":
        if node not in mem._idx:
            return {"error": f"Node '{node}' not in DAG. Valid nodes: {valid_nodes}"}
        parents = mem.parents(node)
        return {"node": node, "parents": [{"node": n, "weight": round(w, 4)} for n, w in parents]}

    if qt == "children":
        if node not in mem._idx:
            return {"error": f"Node '{node}' not in DAG. Valid nodes: {valid_nodes}"}
        children = mem.children(node)
        return {"node": node, "children": [{"node": n, "weight": round(w, 4)} for n, w in children]}

    if qt == "paths_to_aucbrain":
        valid_nodes = list(mem._idx.keys())
        if node not in mem._idx:
            return {"error": f"Node '{node}' not in DAG. Valid nodes: {valid_nodes}"}
        paths = mem.causal_paths(node, "AUCbrain")
        return {"source": node, "target": "AUCbrain", "paths": paths}

    if qt == "total_effect":
        if "→" in node or "->" in node:
            sep   = "→" if "→" in node else "->"
            src, tgt = [s.strip() for s in node.split(sep, 1)]
        else:
            src, tgt = node, "AUCbrain"
        vn = list(mem._idx.keys())
        if src not in mem._idx:
            return {"error": f"Source node '{src}' not in DAG. Valid: {vn}"}
        if tgt not in mem._idx:
            return {"error": f"Target node '{tgt}' not in DAG. Valid: {vn}"}
        effect = mem.total_effect(src, tgt)
        return {"source": src, "target": tgt, "total_effect": round(effect, 4)}

    if qt == "recommend":
        try:
            design_vals = json.loads(node)
        except Exception:
            return {"error": "For 'recommend', pass node as a JSON string of the current design dict."}
        import numpy as np
        current = {
            "LogSize":       np.log(float(design_vals.get("size_nm", 80))),
            "Zeta":          float(design_vals.get("zeta_mv", -15)),
            "PEG":           float(design_vals.get("peg", 1)),
            "LogLigDensity": np.log1p(float(design_vals.get("ligand_density", 30))),
        }
        recs = mem.recommend_intervention(current, target="AUCbrain", top_k=5)
        return {"recommendations": recs}

    return {"error": f"Unknown query_type '{query_type}'. Choose from: bottleneck, parents, children, paths_to_AUCbrain, total_effect, recommend"}


def tool_lookup_parameter(parameter_name: str) -> dict:
    """Return literature-based feasibility range and biological meaning."""
    pn = parameter_name.strip()
    if pn in PARAMETER_INFO:
        info = {"parameter": pn, "description": PARAMETER_INFO[pn]}
        if pn in FEASIBILITY:
            info.update(FEASIBILITY[pn])
        return info
    # fuzzy match
    matches = [k for k in PARAMETER_INFO if pn.lower() in k.lower()]
    if matches:
        return {k: PARAMETER_INFO[k] for k in matches}
    return {"error": f"Parameter '{pn}' not found. Available: {list(PARAMETER_INFO.keys())}"}


def tool_check_feasibility(design: dict) -> dict:
    """
    Validate physical/biological constraints for a design.
    Returns: feasible (bool), violations (list), warnings (list).
    """
    violations = []
    warnings   = []

    size_nm        = float(design.get("size_nm",        80))
    zeta_mv        = float(design.get("zeta_mv",       -15))
    peg            = int(design.get("peg",               1))
    ligand_type    = str(design.get("ligand_type", "transferrin"))
    ligand_density = float(design.get("ligand_density",  30))

    f = FEASIBILITY
    if not (f["size_nm"]["min"] <= size_nm <= f["size_nm"]["max"]):
        violations.append(f"size_nm={size_nm} outside [{f['size_nm']['min']}, {f['size_nm']['max']}] nm")
    elif size_nm < 50:
        warnings.append("size_nm < 50 nm: sub-optimal RMT engagement, though low MPS clearance.")
    elif size_nm > 120:
        warnings.append("size_nm > 120 nm: high MPS clearance likely.")

    if not (f["zeta_mv"]["min"] <= zeta_mv <= f["zeta_mv"]["max"]):
        violations.append(f"zeta_mv={zeta_mv} outside [{f['zeta_mv']['min']}, {f['zeta_mv']['max']}] mV")
    elif zeta_mv > 0:
        warnings.append("Positive zeta: non-specific protein adsorption risk, rapid clearance.")
    elif zeta_mv < -30:
        warnings.append("Very negative zeta: immune recognition, short circulation.")

    if peg not in (0, 1):
        violations.append(f"peg must be 0 or 1, got {peg}")

    if ligand_type not in LIGAND_TYPES:
        violations.append(f"ligand_type '{ligand_type}' not in {LIGAND_TYPES}")

    if not (f["ligand_density"]["min"] <= ligand_density <= f["ligand_density"]["max"]):
        violations.append(f"ligand_density={ligand_density} outside [0, 200]")
    elif ligand_density > 100:
        warnings.append("ligand_density > 100: avidity trap – k_bind decreases. Consider 20-50 range.")
    elif ligand_density < 5:
        warnings.append("ligand_density < 5: insufficient targeting, low k_bind.")

    return {
        "feasible":   len(violations) == 0,
        "violations": violations,
        "warnings":   warnings,
        "design":     design,
    }


def tool_compare_designs(design_a: dict, design_b: dict) -> dict:
    """
    Simulate both designs and explain differences via causal graph.
    Returns metrics for both plus causal attribution of the delta.
    """
    res_a = tool_pbpk_simulate(design_a)
    res_b = tool_pbpk_simulate(design_b)

    if not res_a.get("success") or not res_b.get("success"):
        return {"error": "One or both simulations failed.", "a": res_a, "b": res_b}

    delta_auc = res_b["AUC_brain"] - res_a["AUC_brain"]

    mem    = _get_memory()
    import numpy as np

    def _causal_attribution(da, db):
        """Δ AUC ≈ Σ_i  (total_effect_i) × Δ_design_i  for design vars"""
        mapping = {
            "size_nm":        ("LogSize",       lambda x: math.log(float(x))),
            "zeta_mv":        ("Zeta",          lambda x: float(x)),
            "peg":            ("PEG",           lambda x: float(x)),
            "ligand_density": ("LogLigDensity", lambda x: math.log1p(float(x))),
        }
        attributions = []
        for key, (var, fn) in mapping.items():
            va = fn(da.get(key, 0))
            vb = fn(db.get(key, 0))
            dv = vb - va
            if abs(dv) < 1e-9:
                continue
            effect = mem.total_effect(var, "AUCbrain")
            contribution = effect * dv
            attributions.append({
                "parameter":    key,
                "causal_var":   var,
                "delta_input":  round(dv, 4),
                "total_effect": round(effect, 4),
                "contribution": round(contribution, 4),
            })
        attributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return attributions

    attributions = _causal_attribution(design_a, design_b)

    return {
        "design_A": {"design": design_a, "AUC_brain": res_a["AUC_brain"],
                     "AUC_ratio": res_a["AUC_ratio"],
                     "k_bind": res_a["k_bind"], "k_trans": res_a["k_trans"], "CL": res_a["CL"]},
        "design_B": {"design": design_b, "AUC_brain": res_b["AUC_brain"],
                     "AUC_ratio": res_b["AUC_ratio"],
                     "k_bind": res_b["k_bind"], "k_trans": res_b["k_trans"], "CL": res_b["CL"]},
        "delta_AUC_brain": round(delta_auc, 6),
        "improvement_pct": round(100 * delta_auc / max(res_a["AUC_brain"], 1e-9), 2),
        "causal_attribution": attributions,
    }


def tool_recall_history() -> dict:
    """
    Return all NP designs simulated so far in this run, sorted by AUC_brain.
    Helps avoid repeating already-explored designs and identify gaps.
    """
    hist = _run_context.get("history", [])
    if not hist:
        return {"message": "No simulations run yet.", "count": 0}

    sorted_h = sorted(hist, key=lambda x: x["AUC_brain"], reverse=True)

    # Identify unexplored parameter regions
    tips = []
    tried_lig = {h["design"].get("ligand_type") for h in hist}
    untried_lig = set(LIGAND_TYPES) - tried_lig
    if untried_lig:
        tips.append(f"Untried ligand types: {sorted(untried_lig)}")
    sizes = [float(h["design"].get("size_nm", 80)) for h in hist]
    if all(s > 70 for s in sizes):
        tips.append("Never tried size_nm < 70 nm (may reduce CL)")
    if all(s < 90 for s in sizes):
        tips.append("Never tried size_nm > 90 nm")
    tried_shapes = {h["design"].get("particle_shape", "sphere") for h in hist}
    if "rod" not in tried_shapes:
        tips.append("Never tried particle_shape='rod' (↓CL ~30%, longer circulation)")

    return {
        "total_simulations": len(hist),
        "best":   sorted_h[0],
        "top_5":  sorted_h[:5],
        "worst":  sorted_h[-1],
        "unexplored_regions": tips,
        "all_auc_values": [round(h["AUC_brain"], 6) for h in hist],
    }


def tool_get_empirical_effects() -> dict:
    """
    Estimate causal effects empirically from simulation history via finite differences.
    Compares data-driven estimates with causal-graph priors.
    Requires ≥4 simulations.
    """
    hist = _run_context.get("history", [])
    if len(hist) < 4:
        return {
            "message": f"Need ≥4 simulations (have {len(hist)}).",
            "causal_graph_priors": {
                "ligand_type→AUCbrain": "+0.62 (via Kbind)",
                "zeta_mv→AUCbrain":    "+0.40 (via Ktrans)",
                "size_nm→AUCbrain":    "-0.32 (via CL & Kbind)",
                "ligand_density→AUCbrain": "+0.20 (via Kbind)",
            }
        }

    effects: dict[str, list] = {}
    numeric_params = ["size_nm", "zeta_mv", "ligand_density"]

    for param in numeric_params:
        for i, h1 in enumerate(hist):
            for h2 in hist[i + 1:]:
                d1, d2 = h1["design"], h2["design"]
                try:
                    dp = float(d2.get(param, 0)) - float(d1.get(param, 0))
                    if abs(dp) < 0.5:
                        continue
                    # Only count pairs where only this param differs significantly
                    other_keys = ["size_nm", "zeta_mv", "ligand_density", "ligand_type"]
                    others_same = all(
                        str(d1.get(k)) == str(d2.get(k))
                        for k in other_keys if k != param
                    )
                    if others_same:
                        da = h2["AUC_brain"] - h1["AUC_brain"]
                        effects.setdefault(param, []).append(da / dp)
                except Exception:
                    pass

    import numpy as _np
    empirical = {p: round(float(_np.median(v)), 4)
                 for p, v in effects.items() if v}

    return {
        "empirical_effects_this_run": empirical,
        "n_simulations_used": len(hist),
        "causal_graph_priors": {
            "zeta_mv":        "+0.40 (via Ktrans)",
            "size_nm":        "-0.32 (via CL & Kbind)",
            "ligand_density": "+0.20 (via Kbind)",
        },
        "interpretation": (
            "If empirical effect differs from prior, the current design region "
            "may have different local sensitivity than the global average."
        ),
    }


def tool_update_causal_graph() -> dict:
    """
    Online update of the causal DAG edge weights from simulation history.

    Runs a Bayesian blend:  W_new = (1−α)×W_prior + α×W_local_OLS
    where α = min(0.35, 0.04 × n_sims).  Sign-validated edges are locked.
    Requires ≥6 simulations with sufficient parameter diversity.

    Call this after completing Phase 1 (≥6 sims) to refine the causal weights
    for the current design region before Phase 2 local search.
    """
    hist = _run_context.get("history", [])
    mem  = _get_memory()
    result = mem.update_from_observations(hist, min_obs=6)
    if result.get("status") == "updated":
        n_changes = len(result.get("significant_changes", []))
        if n_changes:
            change_strs = [f"{c['edge']} {c['delta']:+.3f}"
                           for c in result["significant_changes"][:4]]
            changes_note = "Changes: " + ", ".join(change_strs)
        else:
            changes_note = "Weights stable (local data consistent with prior)."
        result["summary"] = (
            f"Causal graph updated from {result['n_obs']} simulations "
            f"(α={result['alpha']:.2f}). "
            f"{n_changes} edge weight(s) shifted by >0.015. "
            + changes_note
        )
    return result


def tool_list_patient_profiles() -> dict:
    """List all available virtual patient profiles and their PBPK scaling factors."""
    try:
        from virtual_patients import list_profiles
        profiles = list_profiles()
        current = _run_context.get("patient", "adult_healthy")
        return {
            "current_patient":  current,
            "available_profiles": profiles,
            "usage": "Set patient_profile='<label>' in run_agent() to switch patients.",
        }
    except ImportError:
        return {"error": "virtual_patients.py not found."}


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry  (OpenAI / DeepSeek format — also accepted by Anthropic via
#                 a thin conversion layer below)
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_DEFS: list[dict] = [
    {
        "name":        "pbpk_simulate",
        "description": (
            "Run the PBPK mechanistic ODE model for a nanoparticle design. "
            "Returns AUC_brain (normalised), AUC_drug_brain (drug-payload-weighted), "
            "AUC_ratio (brain/blood), and kinetic parameters (k_bind, k_trans, k_lyso, CL). "
            "Core params are required; extended params (drug_loading, particle_shape, "
            "hydrophobicity, surface_coating) are optional and default to baseline values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "size_nm":         {"type": "number",  "description": "Hydrodynamic diameter (nm), typical 20-200"},
                "zeta_mv":         {"type": "number",  "description": "Zeta potential (mV), typical -40 to +10"},
                "peg":             {"type": "integer", "description": "PEGylation: 1=yes, 0=no"},
                "ligand_type":     {"type": "string",  "description": "Targeting ligand: transferrin|anti-TfR|rabies-peptide|none"},
                "ligand_density":  {"type": "number",  "description": "Ligands per NP (0-200). Avidity trap above ~100."},
                "drug_loading":    {"type": "number",  "description": "(optional) Drug loading % w/w (0-30). >20% causes formulation instability."},
                "particle_shape":  {"type": "string",  "description": "(optional) sphere|rod|disk|worm. Rod→↓CL 30%, ↓k_bind 12%."},
                "hydrophobicity":  {"type": "number",  "description": "(optional) Surface hydrophobicity 0-1. Higher→protein corona→↑CL."},
                "surface_coating": {"type": "string",  "description": "(optional) none|peg|lipid|zwitterionic|polymer. Zwitterionic→↓CL 35%."},
            },
            "required": ["size_nm", "zeta_mv", "peg", "ligand_type", "ligand_density"],
        },
    },
    {
        "name":        "query_causal_graph",
        "description": (
            "Query the learned causal DAG (9 nodes, 10 edges). "
            "Use query_type='bottleneck' to rank all variables by total causal effect on AUCbrain. "
            "Use 'parents'/'children' to explore graph topology. "
            "Use 'paths_to_AUCbrain' to trace causal chains from a node. "
            "Use 'total_effect' with node='Source->Target' to get the linearised total effect. "
            "Use 'recommend' with node=JSON-design to get intervention suggestions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type":        "string",
                    "description": (
                        "Node name (e.g. 'Kbind', 'Zeta', 'PEG', 'LogSize', 'AUCbrain'). "
                        "For 'total_effect': 'Source->Target'. "
                        "For 'recommend': JSON string of current design {size_nm, zeta_mv, peg, ligand_density}."
                    ),
                },
                "query_type": {
                    "type":        "string",
                    "description": "One of: bottleneck | parents | children | paths_to_AUCbrain | total_effect | recommend",
                },
            },
            "required": ["node", "query_type"],
        },
    },
    {
        "name":        "lookup_parameter",
        "description": (
            "Look up literature-based feasibility ranges and biological meaning for a design parameter "
            "or kinetic variable. Useful before proposing a new design value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "parameter_name": {
                    "type":        "string",
                    "description": "Parameter name, e.g. 'size_nm', 'zeta_mv', 'k_bind', 'ligand_density'.",
                },
            },
            "required": ["parameter_name"],
        },
    },
    {
        "name":        "check_feasibility",
        "description": (
            "Validate whether a proposed NP design satisfies physical and biological constraints. "
            "Returns feasible=True/False, violations list (hard constraints), "
            "and warnings list (soft constraints / performance risks)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "size_nm":        {"type": "number"},
                "zeta_mv":        {"type": "number"},
                "peg":            {"type": "integer"},
                "ligand_type":    {"type": "string"},
                "ligand_density": {"type": "number"},
            },
            "required": ["size_nm", "zeta_mv", "peg", "ligand_type", "ligand_density"],
        },
    },
    {
        "name":        "compare_designs",
        "description": (
            "Simulate two NP designs side-by-side and explain the AUC_brain difference "
            "through causal attribution (which parameter change contributed most, and via which causal path)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "design_a": {
                    "type": "object",
                    "description": "Baseline / current design.",
                    "properties": {
                        "size_nm":        {"type": "number"},
                        "zeta_mv":        {"type": "number"},
                        "peg":            {"type": "integer"},
                        "ligand_type":    {"type": "string"},
                        "ligand_density": {"type": "number"},
                    },
                },
                "design_b": {
                    "type": "object",
                    "description": "Proposed / optimised design.",
                    "properties": {
                        "size_nm":        {"type": "number"},
                        "zeta_mv":        {"type": "number"},
                        "peg":            {"type": "integer"},
                        "ligand_type":    {"type": "string"},
                        "ligand_density": {"type": "number"},
                    },
                },
            },
            "required": ["design_a", "design_b"],
        },
    },
    {
        "name":        "recall_history",
        "description": (
            "Return all NP designs simulated so far in this run, sorted by AUC_brain. "
            "Shows top-5 best designs, worst design, and unexplored parameter regions. "
            "Call this before starting Phase 2 refinement to avoid repeating known results."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name":        "get_empirical_effects",
        "description": (
            "Compute data-driven causal effect estimates from simulation history via finite differences. "
            "Compare against causal-graph priors to detect if the current design region has "
            "different sensitivity than the global average. Requires ≥4 simulations."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name":        "update_causal_graph",
        "description": (
            "Perform an online Bayesian-blend update of the causal DAG edge weights "
            "using all simulations run so far. "
            "Blends local OLS estimates with the Phase-3 prior: "
            "W_new = (1−α)×W_prior + α×W_local, where α grows with n_sims (max 0.35). "
            "Sign-validated edges are locked to prevent spurious flips. "
            "Call this after Phase 1 (≥6 sims) to refine causal weights for Phase 2. "
            "Requires ≥6 simulations."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name":        "list_patient_profiles",
        "description": (
            "List all available virtual patient profiles (12 combinations of "
            "age × disease state) and their PBPK parameter scaling factors. "
            "The current patient profile is applied automatically to all pbpk_simulate calls."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

# OpenAI / DeepSeek format
TOOLS_OPENAI: list[dict] = [
    {"type": "function", "function": t} for t in _TOOL_DEFS
]

# Anthropic format (input_schema instead of parameters, no type wrapper)
TOOLS_ANTHROPIC: list[dict] = [
    {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
    for t in _TOOL_DEFS
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_tool(name: str, inputs: dict) -> Any:
    if name == "pbpk_simulate":
        return tool_pbpk_simulate(inputs)
    if name == "query_causal_graph":
        return tool_query_causal_graph(inputs["node"], inputs["query_type"])
    if name == "lookup_parameter":
        return tool_lookup_parameter(inputs["parameter_name"])
    if name == "check_feasibility":
        return tool_check_feasibility(inputs)
    if name == "compare_designs":
        return tool_compare_designs(inputs["design_a"], inputs["design_b"])
    if name == "recall_history":
        return tool_recall_history()
    if name == "get_empirical_effects":
        return tool_get_empirical_effects()
    if name == "update_causal_graph":
        return tool_update_causal_graph()
    if name == "list_patient_profiles":
        return tool_list_patient_profiles()
    return {"error": f"Unknown tool: {name}"}


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_backend() -> str:
    """Return 'deepseek', 'anthropic', or raise RuntimeError."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError(
        "No API key found. Set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY.\n"
        "  export DEEPSEEK_API_KEY=sk-..."
    )


def _make_client(backend: str):
    if backend == "deepseek":
        if not _HAS_OPENAI:
            raise RuntimeError("openai package not installed: pip install openai")
        return _OpenAI(
            api_key  = os.environ["DEEPSEEK_API_KEY"],
            base_url = "https://api.deepseek.com",
        )
    # anthropic
    if not _HAS_ANTHROPIC:
        raise RuntimeError("anthropic package not installed: pip install anthropic")
    return _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────────────────────────────────────────
# ReAct Agent
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a causal reasoning agent specialising in brain-targeted nanoparticle (NP) design.

## Your knowledge base
- A mechanistic PBPK ODE model (5-state receptor-mediated transcytosis) captures how NP design
  parameters (size, zeta, PEG, ligand type, ligand density) drive brain delivery kinetics.
- A learned causal DAG (9 nodes, 10 edges, 10/10 biologically validated) encodes which parameters
  control which kinetic bottlenecks and their total causal effects on AUC_brain.

## Key causal facts (memorised from Phase 3)
| Bottleneck  | Total effect on AUC_brain |
|-------------|--------------------------|
| Kbind       | +0.62 (largest)          |
| Ktrans      | +0.44                    |
| Zeta->Ktrans| +0.40 (via zeta)         |
| LogSize     | -0.32 (via CL & Kbind)   |
| LogLigDens  | +0.20                    |
| CL          | -0.11                    |
| PEG->CL     | +0.10 (via CL reduction) |
| Klyso       | -0.01 (orthogonalised)   |

## Two-phase optimisation strategy

### Phase 1 — Causal-guided coarse search (first ~8 simulations)
1. Query the causal graph (bottleneck) once to identify the top causal levers.
2. Fix the most impactful parameters first: ligand_type is the categorical lever with the highest
   impact via Kbind (+0.62); try ALL four ligand types (transferrin, anti-TfR, rabies-peptide, none)
   if not yet simulated — this costs only 4 sims and directly targets the biggest bottleneck.
3. Set PEG=1, size_nm≈80, zeta_mv≈-15, ligand_density≈30 as defaults while exploring ligand types.
4. After each simulation cite the causal path that motivated the change.

### Phase 1.5 — Adaptive causal update (between phases)
5. After Phase 1, call update_causal_graph() to refine edge weights from your observations,
   then call recall_history() to identify unexplored regions.
   Use the updated bottleneck ranking to prioritise Phase 2 parameters.

### Phase 2 — Local refinement (remaining simulations)
6. Once the best ligand type is identified, systematically fine-tune continuous parameters:
   - ligand_density: try the range 20, 30, 40, 50 around the current best
   - zeta_mv: try -10, -15, -20, -25
   - size_nm: try 60, 70, 80, 90
6. Use compare_designs to confirm each improvement over the running best.
7. Keep iterating until ALL simulations in your budget are used.
8. NEVER stop early — even a 1% improvement compounds over multiple iterations.

### Final step
9. End with a concise report: best design found, AUC_brain achieved, and causal explanation
   of why each parameter is set to its optimal value.

## Extended design parameters (optional — use when exploring beyond baseline)
You may include these in pbpk_simulate calls. Defaults are sphere/0/0/none.
- drug_loading (0–30 % w/w): scales AUC_drug_brain; >20% causes instability
- particle_shape (sphere/rod/disk/worm): rod → ↓CL 30%, marginal ↓k_bind 12%
- hydrophobicity (0–1): protein corona risk; keep <0.3 to avoid MPS clearance penalty
- surface_coating (none/peg/lipid/zwitterionic/polymer): zwitterionic → ↓CL 35%

## Memory and adaptive causal tools — use to improve efficiency and accuracy
- recall_history(): call BEFORE Phase 2 to see all tried designs and unexplored regions.
  Prevents wasting budget on already-explored designs.
- update_causal_graph(): call AFTER Phase 1 (≥6 sims) to refine DAG edge weights with
  local OLS data. Returns updated bottleneck ranking — use it to re-prioritise Phase 2.
  This is a Bayesian blend: prior is never discarded, only nudged by local evidence.
- get_empirical_effects(): call after 5+ sims to get finite-difference effect estimates
  and compare with causal-graph priors.
- list_patient_profiles(): lists the 12 virtual patient profiles active in this run.

## Output format
Think step-by-step. For each reasoning step write:
  Thought: <your causal reasoning — cite which causal path motivates the change>
  Action: <tool call>
  Observation: <tool result summary>
... then iterate.
End with a summary table of: best design parameters, AUC_brain, and causal explanation for each change.
"""


def run_agent(
    initial_design:  dict,
    goal:            str  = "Maximise AUC_brain for BBB crossing",
    max_iterations:  int  = 10,
    max_simulations: int | None = None,
    patient_profile: str  = "adult_healthy",
    verbose:         bool = True,
    backend:         str  = "auto",
) -> dict:
    """
    Run the ReAct Agent.

    Parameters
    ----------
    initial_design : dict
        Starting NP design {size_nm, zeta_mv, peg, ligand_type, ligand_density}.
    goal : str
        Natural-language optimisation goal.
    max_iterations : int
        Hard cap on ReAct cycles (LLM calls).
    max_simulations : int or None
        If set, stop after this many pbpk_simulate calls regardless of
        max_iterations. Causal graph queries do not count toward this budget.
    patient_profile : str
        Virtual patient for all simulations in this run.
        Format: "<age_group>_<disease_state>", e.g. "adult_healthy",
        "elderly_glioma", "pediatric_neuroinflammation".
    verbose : bool
        Print live trace to stdout.
    backend : str
        'auto' | 'deepseek' | 'anthropic'

    Returns
    -------
    dict  best_design, best_AUC, trajectory, full_trace, iterations,
          backend_used, patient_profile
    """
    global _run_context
    _run_context = {"history": [], "patient": patient_profile}

    if backend == "auto":
        backend = _detect_backend()

    client = _make_client(backend)

    _default_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    model = _default_model if backend == "deepseek" else "claude-sonnet-4-6"
    tools = TOOLS_OPENAI    if backend == "deepseek" else TOOLS_ANTHROPIC

    if verbose:
        print(f"\n{'='*70}")
        print(f"  NP Design Optimisation Agent — Phase 4  [{backend} / {model}]")
        print(f"{'='*70}")

    # Build patient context note
    patient_note = ""
    if patient_profile not in ("adult_healthy", "healthy_adult", "adult", ""):
        try:
            from virtual_patients import parse_patient_type
            pt = parse_patient_type(patient_profile)
            s  = pt.scaling()
            patient_note = (
                f"\nVirtual patient: {patient_profile}\n"
                f"  {pt.description()}\n"
                f"  PBPK scaling: k_bind×{s['k_bind_scale']:.2f}  "
                f"k_trans×{s['k_trans_scale']:.2f}  CL×{s['CL_scale']:.2f}\n"
            )
        except Exception:
            patient_note = f"\nVirtual patient: {patient_profile}\n"

    user_msg = textwrap.dedent(f"""
        Goal: {goal}{patient_note}

        Initial NP design (suboptimal):
        {json.dumps(initial_design, indent=2)}

        Please optimise this design using your causal reasoning tools.
        Follow the ReAct protocol: Thought -> tool calls -> Observation -> Thought ...
        Use recall_history() before Phase 2 to check what you've already tried.
        Report the best design you find and explain *why* each change helped
        using causal graph evidence.
    """).strip()

    messages: list[dict] = [{"role": "user", "content": user_msg}]

    trajectory:  list[dict] = []
    best_design: dict       = initial_design.copy()
    best_auc:    float      = 0.0

    # seed initial AUC
    _init = tool_pbpk_simulate(initial_design)
    if _init.get("success"):
        best_auc = _init["AUC_brain"]
        trajectory.append({"design": initial_design.copy(), "AUC_brain": best_auc, "step": 0})
        if verbose:
            print(f"  Initial design : {initial_design}")
            print(f"  Initial AUC    : {best_auc:.6f}")
            print(f"{'='*70}\n")

    # ── ReAct loop ──────────────────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n{'─'*60}  iteration {iteration}")

        # ── LLM call ────────────────────────────────────────────────────────
        if backend == "deepseek":
            response = None
            for _attempt in range(5):
                try:
                    response = client.chat.completions.create(
                        model    = model,
                        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                        tools    = tools,
                        tool_choice = "auto",
                    )
                    if response is not None:
                        break
                except Exception as _e:
                    if verbose:
                        print(f"[API error attempt {_attempt+1}/5]: {_e}")
                time.sleep(2 ** _attempt)
            if response is None:
                if verbose:
                    print("[Agent] API returned None after 5 retries — stopping early")
                break
            choice  = response.choices[0]
            msg     = choice.message
            finish  = choice.finish_reason      # "stop" | "tool_calls"

            # print text
            if verbose and msg.content:
                print(msg.content)

            # build serialisable assistant message
            # reasoning models (deepseek-v4-pro) return reasoning_content that
            # must be passed back in subsequent turns
            asst_msg: dict = {"role": "assistant", "content": msg.content or ""}
            if getattr(msg, "reasoning_content", None):
                asst_msg["reasoning_content"] = msg.reasoning_content
            if msg.tool_calls:
                asst_msg["tool_calls"] = [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {"name": tc.function.name,
                                     "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(asst_msg)

            if finish == "stop" or not msg.tool_calls:
                if verbose:
                    print("\n[Agent reached stop — optimisation complete]")
                break

            # process tool calls
            _sim_budget_hit = False
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_inputs = json.loads(tc.function.arguments)
                except Exception:
                    continue

                if verbose:
                    print(f"\n[Tool] {tool_name}({json.dumps(tool_inputs, ensure_ascii=False)})")

                result = dispatch_tool(tool_name, tool_inputs)

                if verbose:
                    print(f"[Result] {json.dumps(result, ensure_ascii=False, default=str)[:600]}")

                if tool_name == "pbpk_simulate" and result.get("success"):
                    auc = result["AUC_brain"]
                    d   = result["design_used"].copy()
                    trajectory.append({"design": d, "AUC_brain": auc, "step": iteration})
                    _run_context["history"].append({
                        "design":   d,
                        "AUC_brain": auc,
                        "k_bind":   result.get("k_bind"),
                        "k_trans":  result.get("k_trans"),
                        "k_lyso":   result.get("k_lyso"),
                        "CL":       result.get("CL"),
                    })
                    if auc > best_auc:
                        best_auc    = auc
                        best_design = d.copy()
                    if max_simulations and len(trajectory) >= max_simulations:
                        _sim_budget_hit = True

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result, ensure_ascii=False, default=str),
                })
                if _sim_budget_hit:
                    break

            if _sim_budget_hit:
                if verbose:
                    print(f"\n[Agent] Simulation budget ({max_simulations}) reached — stopping")
                break

        else:
            # ── Anthropic backend ────────────────────────────────────────────
            response = client.messages.create(
                model      = model,
                max_tokens = 4096,
                system     = SYSTEM_PROMPT,
                tools      = tools,
                messages   = messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if verbose:
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        print(block.text)

            if response.stop_reason == "end_turn":
                if verbose:
                    print("\n[Agent reached end_turn — optimisation complete]")
                break

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name   = block.name
                tool_inputs = block.input

                if verbose:
                    print(f"\n[Tool] {tool_name}({json.dumps(tool_inputs, ensure_ascii=False)})")

                result = dispatch_tool(tool_name, tool_inputs)

                if verbose:
                    print(f"[Result] {json.dumps(result, ensure_ascii=False, default=str)[:600]}")

                if tool_name == "pbpk_simulate" and result.get("success"):
                    auc = result["AUC_brain"]
                    d   = result["design_used"].copy()
                    trajectory.append({"design": d, "AUC_brain": auc, "step": iteration})
                    if auc > best_auc:
                        best_auc    = auc
                        best_design = d.copy()

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result, ensure_ascii=False, default=str),
                })

            messages.append({"role": "user", "content": tool_results})

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Optimisation finished  iterations={iteration}")
        print(f"  Best AUC_brain : {best_auc:.6f}")
        print(f"  Best design    : {best_design}")
        print(f"{'='*70}\n")

    return {
        "best_design":    best_design,
        "best_AUC":       best_auc,
        "trajectory":     trajectory,
        "full_trace":     messages,
        "iterations":     iteration,
        "backend_used":   backend,
        "patient_profile": patient_profile,
        "n_simulations":  len(trajectory),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning chain visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_reasoning_chain(result: dict, save_path: str = None) -> str:
    """
    Generate an annotated optimisation-trajectory plot showing:
      - Top panel: AUC_brain vs. simulation step, with green dots marking improvements
        and parameter-change annotations at each improvement point.
      - Bottom panel: heatmap of which parameters changed at each step.

    Extracts 'Thought' text preceding each pbpk_simulate call from the agent trace
    to annotate the trajectory with causal reasoning excerpts.

    Parameters
    ----------
    result    : dict returned by run_agent()
    save_path : output PNG path (default: data/reasoning_chain.png)

    Returns
    -------
    str  absolute path of the saved figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    trajectory = result.get("trajectory", [])
    messages   = result.get("full_trace",  [])

    if len(trajectory) < 2:
        print("[visualize_reasoning_chain] Not enough trajectory data.")
        return ""

    # ── Extract thought text preceding each pbpk_simulate call ─────────────
    # Build key → first relevant thought sentence
    sim_key_to_thought: dict[tuple, str] = {}
    last_thought = ""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        raw_text = str(msg.get("content") or "")
        if raw_text.strip():
            # Prefer lines containing causal keywords
            causal_kws = ['causal', 'k_bind', 'k_trans', 'bottleneck',
                          'ligand', 'zeta', 'because', 'since', 'expect',
                          'increase', 'decrease', 'improve']
            for line in raw_text.split('\n'):
                line = line.strip()
                if line and any(kw in line.lower() for kw in causal_kws):
                    last_thought = line[:120]
                    break
            else:
                last_thought = raw_text.strip()[:120]
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            if fn.get("name") != "pbpk_simulate":
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
                key = (
                    str(args.get("ligand_type", "")),
                    round(float(args.get("size_nm",       80))),
                    round(float(args.get("zeta_mv",      -15))),
                    round(float(args.get("ligand_density", 30))),
                )
                if key not in sim_key_to_thought:
                    sim_key_to_thought[key] = last_thought
            except Exception:
                pass

    def _traj_key(d: dict) -> tuple:
        return (
            str(d.get("ligand_type", "")),
            round(float(d.get("size_nm",       80))),
            round(float(d.get("zeta_mv",      -15))),
            round(float(d.get("ligand_density", 30))),
        )

    aucs    = [t["AUC_brain"] for t in trajectory]
    n_steps = len(aucs)
    xs      = list(range(n_steps))
    best_auc = max(aucs)
    auc_range = max(aucs) - min(aucs) + 1e-6

    # ── Figure layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Trajectory line
    ax1.plot(xs, aucs, '-', color='#BDC3C7', lw=1.5, zorder=2)

    running_best = -1.0
    for i, (x, a) in enumerate(zip(xs, aucs)):
        is_improvement = a > running_best + 1e-7
        if is_improvement:
            running_best = a
            ax1.scatter([x], [a], color='#27AE60', s=130, zorder=5,
                        edgecolors='white', linewidths=1.5)
            # Annotate what changed and why
            if i > 0:
                d_prev = trajectory[i - 1]["design"]
                d_curr = trajectory[i]["design"]
                changed = []
                for k, lbl in [("ligand_type", "lig"), ("size_nm", "sz"),
                                ("zeta_mv", "ζ"), ("ligand_density", "ld"),
                                ("particle_shape", "shape"),
                                ("surface_coating", "coat")]:
                    if str(d_prev.get(k, "")) != str(d_curr.get(k, "")):
                        changed.append(f"{lbl}={d_curr.get(k)}")
                param_label = ", ".join(changed[:3]) if changed else "?"
                thought = sim_key_to_thought.get(_traj_key(d_curr), "")
                annotation = param_label
                if thought:
                    annotation += f"\n↳ {thought[:80]}"
                ax1.annotate(
                    annotation,
                    xy=(x, a),
                    xytext=(x + 0.4, a + 0.035 * auc_range),
                    fontsize=6.5, color='#1A5276',
                    va='bottom',
                    arrowprops=dict(arrowstyle='->', color='#27AE60',
                                    lw=0.9, shrinkA=0, shrinkB=3),
                    bbox=dict(boxstyle='round,pad=0.2', fc='#EAFAF1',
                              ec='#27AE60', alpha=0.85),
                )
        else:
            ax1.scatter([x], [a], color='#7F8C8D', s=45, zorder=4, alpha=0.65)

    ax1.axhline(best_auc, color='#27AE60', ls='--', lw=1, alpha=0.55,
                label=f'Best AUC_brain = {best_auc:.4f}')

    patient = result.get("patient_profile", "adult_healthy")
    ax1.set_title(
        f'Agent Reasoning Chain — NP Optimisation Trajectory  '
        f'[patient: {patient}]',
        fontsize=11, fontweight='bold', pad=8)
    ax1.set_ylabel('AUC_brain (normalised)', fontsize=10)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.22)
    ax1.set_xlim(-0.5, n_steps - 0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Parameter change heatmap ─────────────────────────────────────────────
    hm_params = [("ligand_type", "ligand"), ("size_nm", "size (nm)"),
                 ("zeta_mv", "zeta (mV)"), ("ligand_density", "lig density"),
                 ("particle_shape", "shape"), ("surface_coating", "coating")]
    change_mat = np.zeros((len(hm_params), n_steps))
    for i in range(1, n_steps):
        for pk, (pkey, _) in enumerate(hm_params):
            v0 = str(trajectory[i - 1]["design"].get(pkey, ""))
            v1 = str(trajectory[i]["design"].get(pkey, ""))
            if v0 != v1:
                change_mat[pk, i] = 1.0

    ax2.imshow(change_mat, aspect='auto', cmap='YlOrRd',
               vmin=0, vmax=1, interpolation='nearest')
    ax2.set_yticks(range(len(hm_params)))
    ax2.set_yticklabels([p[1] for p in hm_params], fontsize=8.5)
    ax2.set_xticks(range(n_steps))
    ax2.set_xticklabels([str(i) for i in range(n_steps)], fontsize=7.5)
    ax2.set_xlabel('Simulation step', fontsize=10)
    ax2.set_title('Parameter changed (orange) at each step', fontsize=8)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'data', 'reasoning_chain.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Reasoning chain] → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Demo entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Suboptimal starting design (large size, no PEG, positive zeta, high ligand)
    initial = {
        "size_nm":        150,
        "zeta_mv":         +5,
        "peg":              0,
        "ligand_type":   "transferrin",
        "ligand_density": 120,
    }

    result = run_agent(
        initial_design = initial,
        goal           = "Maximise AUC_brain (normalised brain drug exposure) for BBB-targeted NP",
        max_iterations = 10,
        verbose        = True,
    )

    # Save trace
    out_path = os.path.join(os.path.dirname(__file__), "data", "agent_trace.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_design": result["best_design"],
            "best_AUC":    result["best_AUC"],
            "trajectory":  result["trajectory"],
            "iterations":  result["iterations"],
        }, f, ensure_ascii=False, indent=2)
    print(f"Trace saved → {out_path}")

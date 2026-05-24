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
    "size_nm":         {"min": 20,   "max": 200,  "optimal": "50-100 nm for RMT",          "unit": "nm"},
    "zeta_mv":         {"min": -40,  "max": 10,   "optimal": "-20 to -10 mV (mild negative)","unit": "mV"},
    "ligand_density":  {"min": 0,    "max": 200,  "optimal": "20-50 (avidity trap >100)",    "unit": "ligands/NP"},
    "peg":             {"min": 0,    "max": 1,    "optimal": "1 reduces MPS clearance",      "unit": "0/1"},
}

LIGAND_TYPES = ["transferrin", "anti-TfR", "rabies-peptide", "none"]

PARAMETER_INFO = {
    "size_nm":        "Hydrodynamic diameter in nm. Smaller → less MPS clearance but lower absolute payload. Optimal ~80 nm for BBB-RMT.",
    "zeta_mv":        "Surface zeta potential in mV. Mild negative (−10 to −20 mV) maximises k_trans via clathrin-mediated endocytosis.",
    "peg":            "PEGylation (0 = none, 1 = PEGylated). PEG reduces opsonisation and lowers MPS CL by ~55%.",
    "ligand_type":    f"Targeting ligand. Choices: {LIGAND_TYPES}. Transferrin/anti-TfR give highest k_bind.",
    "ligand_density": "Number of targeting ligands per NP. Avidity peak at 20-50; >100 → avidity trap (k_bind falls).",
    "k_bind":         "Receptor binding rate (h⁻¹). Key bottleneck for BBB crossing (total effect +0.62 on AUC_brain).",
    "k_trans":        "Transcytosis rate from vesicle to brain (h⁻¹). Second bottleneck (+0.44). Driven by zeta.",
    "k_lyso":         "Lysosomal degradation rate (h⁻¹). Competes with k_trans (effect −0.01 net after orthogonalisation).",
    "CL":             "Systemic clearance (h⁻¹). Driven by size and PEG (effect −0.11 on AUC_brain).",
    "AUCbrain":       "Area under brain concentration–time curve (normalised). Primary optimisation target.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

_causal_memory = None

def _get_memory():
    global _causal_memory
    if _causal_memory is None:
        json_path = os.path.join(os.path.dirname(__file__), "data", "causal_results.json")
        _causal_memory = load_causal_memory(json_path)
    return _causal_memory


def tool_pbpk_simulate(design: dict) -> dict:
    """
    Run PBPK ODE for a given NP design.
    Expected keys: size_nm, zeta_mv, peg (0/1), ligand_type, ligand_density
    Returns: AUC_brain, AUC_ratio, kinetic params, success flag.
    """
    try:
        size_nm        = float(design.get("size_nm", 80))
        zeta_mv        = float(design.get("zeta_mv", -15))
        peg            = "yes" if int(design.get("peg", 1)) else "no"
        ligand_type    = str(design.get("ligand_type", "transferrin"))
        ligand_density = float(design.get("ligand_density", 30))

        result = _pbpk_simulate(
            size_nm       = size_nm,
            zeta_mv       = zeta_mv,
            peg           = peg,
            ligand_type   = ligand_type,
            ligand_density= ligand_density,
        )
        p = result["params"]
        return {
            "AUC_brain":       round(result["AUC"],       6),
            "AUC_ratio":       round(result["AUC_ratio"], 6),
            "k_bind":          round(p.k_bind,  4),
            "k_trans":         round(p.k_trans, 4),
            "k_lyso":          round(p.k_lyso,  4),
            "CL":              round(p.CL,       4),
            "success":         result["success"],
            "design_used":     {"size_nm": size_nm, "zeta_mv": zeta_mv,
                                "peg": peg, "ligand_type": ligand_type,
                                "ligand_density": ligand_density},
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


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry  (OpenAI / DeepSeek format — also accepted by Anthropic via
#                 a thin conversion layer below)
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_DEFS: list[dict] = [
    {
        "name":        "pbpk_simulate",
        "description": (
            "Run the PBPK mechanistic ODE model for a nanoparticle design. "
            "Returns AUC_brain (normalised), AUC_ratio (brain/blood), and kinetic parameters "
            "(k_bind, k_trans, k_lyso, CL). Use this as the primary oracle to evaluate a design."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "size_nm":        {"type": "number",  "description": "Hydrodynamic diameter (nm), typical 20-200"},
                "zeta_mv":        {"type": "number",  "description": "Zeta potential (mV), typical -40 to +10"},
                "peg":            {"type": "integer", "description": "PEGylation: 1=yes, 0=no"},
                "ligand_type":    {"type": "string",  "description": "Targeting ligand: transferrin|anti-TfR|rabies-peptide|none"},
                "ligand_density": {"type": "number",  "description": "Ligands per NP (0-200). Avidity trap above ~100."},
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

## Reasoning style
1. ALWAYS start by querying the causal graph to understand which lever has the greatest expected impact.
2. Propose a concrete design change, check feasibility, then simulate to verify.
3. When you propose changes, cite the causal path (e.g. "increasing ligand density 5->30 raises Kbind
   via LogLigDensity->Kbind, yielding higher AUC_brain").
4. After simulating, compare with the previous best using compare_designs.
5. Stop when AUC_brain improvements are <5% per iteration or after 6 design iterations.
6. End with a concise optimisation report: best design, AUC_brain, causal explanation.

## Output format
Think step-by-step. For each reasoning step write:
  Thought: <your causal reasoning>
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
        max_iterations. Causal graph queries do not count toward this budget,
        making comparisons across agent variants fair.
    verbose : bool
        Print live trace to stdout.
    backend : str
        'auto' | 'deepseek' | 'anthropic'

    Returns
    -------
    dict  best_design, best_AUC, trajectory, full_trace, iterations, backend_used
    """
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

    user_msg = textwrap.dedent(f"""
        Goal: {goal}

        Initial NP design (suboptimal):
        {json.dumps(initial_design, indent=2)}

        Please optimise this design using your causal reasoning tools.
        Follow the ReAct protocol: Thought -> tool calls -> Observation -> Thought ...
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
        "best_design":   best_design,
        "best_AUC":      best_auc,
        "trajectory":    trajectory,
        "full_trace":    messages,
        "iterations":    iteration,
        "backend_used":  backend,
    }


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

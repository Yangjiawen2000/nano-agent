"""
Phase 5 – Option 2: Anonymised-Parameter Experiment

All design parameter names and causal-graph node names are replaced with
meaningless codes (param_A…param_E, Node_1…Node_8, Target).

The LLM cannot use any prior knowledge about nanoparticles, BBB, or drug
delivery — it must rely solely on the tools and (for the Causal Agent) the
structural information encoded in the causal DAG.

This is the cleanest isolation of "causal-graph contribution vs LLM
parametric knowledge."

Methods compared (same as main baseline):
  1. Random Search
  2. Bayesian Optimisation (Optuna TPE)
  3. Ablation Agent  – ReAct without causal graph, anonymous interface
  4. Causal Agent    – ReAct with causal graph,    anonymous interface

Output:
  data/anon_multiseed_results.json
  data/anon_multiseed_comparison.png
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))
from pbpk_simulator import pbpk_simulate as _pbpk_simulate
from causal_graph    import load_causal_memory
from agent           import (
    _detect_backend, _make_client,
    _TOOL_DEFS,
    dispatch_tool as _real_dispatch,
)
from baselines import (
    random_search, bayesian_optimisation,
    _interp_curve, _sims_to_threshold,
    METHOD_COLORS, METHOD_MARKERS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Anonymisation maps
# ─────────────────────────────────────────────────────────────────────────────

# Design parameters
ANON_TO_REAL_PARAM = {
    "param_A": "size_nm",
    "param_B": "zeta_mv",
    "param_C": "peg",
    "param_D": "ligand_type",
    "param_E": "ligand_density",
}
REAL_TO_ANON_PARAM = {v: k for k, v in ANON_TO_REAL_PARAM.items()}

# Ligand types
ANON_TO_REAL_LIGAND = {
    "type_1": "transferrin",
    "type_2": "anti-TfR",
    "type_3": "rabies-peptide",
    "type_4": "none",
}
REAL_TO_ANON_LIGAND = {v: k for k, v in ANON_TO_REAL_LIGAND.items()}
ANON_LIGAND_TYPES   = list(ANON_TO_REAL_LIGAND.keys())

# Causal-graph nodes
ANON_TO_REAL_NODE = {
    "Node_1": "LogSize",
    "Node_2": "Zeta",
    "Node_3": "PEG",
    "Node_4": "LogLigDensity",
    "Node_5": "Kbind",
    "Node_6": "Ktrans",
    "Node_7": "Klyso",
    "Node_8": "CL",
    "Target": "AUCbrain",
}
REAL_TO_ANON_NODE = {v: k for k, v in ANON_TO_REAL_NODE.items()}

# Kinetic output parameters
REAL_TO_ANON_KINETIC = {
    "k_bind":  "rate_X",
    "k_trans": "rate_Y",
    "k_lyso":  "rate_Z",
    "CL":      "rate_W",
}


def _to_real_design(anon: dict) -> dict:
    return {
        "size_nm":        float(anon.get("param_A", 80)),
        "zeta_mv":        float(anon.get("param_B", -15)),
        "peg":            int(anon.get("param_C", 1)),
        "ligand_type":    ANON_TO_REAL_LIGAND.get(str(anon.get("param_D", "type_1")), "transferrin"),
        "ligand_density": float(anon.get("param_E", 30)),
    }


def _to_anon_design(real: dict) -> dict:
    return {
        "param_A": real.get("size_nm",        80),
        "param_B": real.get("zeta_mv",       -15),
        "param_C": real.get("peg",             1),
        "param_D": REAL_TO_ANON_LIGAND.get(str(real.get("ligand_type","transferrin")), "type_1"),
        "param_E": real.get("ligand_density",  30),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Anonymised tool implementations
# ─────────────────────────────────────────────────────────────────────────────

_causal_mem = None

def _get_mem():
    global _causal_mem
    if _causal_mem is None:
        jp = os.path.join(os.path.dirname(__file__), "data", "causal_results.json")
        _causal_mem = load_causal_memory(jp)
    return _causal_mem


def anon_simulate(anon_design: dict) -> dict:
    real = _to_real_design(anon_design)
    r = _pbpk_simulate(
        size_nm        = real["size_nm"],
        zeta_mv        = real["zeta_mv"],
        peg            = "yes" if real["peg"] else "no",
        ligand_type    = real["ligand_type"],
        ligand_density = real["ligand_density"],
    )
    if not r["success"]:
        return {"error": "Simulation failed.", "success": False}
    p = r["params"]
    return {
        "Target_AUC":   round(r["AUC"],        6),
        "Target_ratio": round(r["AUC_ratio"],   6),
        "rate_X":       round(p.k_bind,  4),
        "rate_Y":       round(p.k_trans, 4),
        "rate_Z":       round(p.k_lyso,  4),
        "rate_W":       round(p.CL,       4),
        "success":      True,
        "design_used":  anon_design,
    }


def anon_query_causal(node: str, query_type: str) -> dict:
    mem = _get_mem()
    qt  = query_type.lower().strip()

    # translate anon node → real node
    real_node = ANON_TO_REAL_NODE.get(node, node)

    def _node_to_anon(n): return REAL_TO_ANON_NODE.get(n, n)

    if qt == "bottleneck":
        text  = mem.query_bottleneck(target="AUCbrain", top_k=9)
        # replace real names with anon names in the text
        for real, anon in REAL_TO_ANON_NODE.items():
            text = text.replace(real, anon)
        return {"result": text}

    valid_anon = list(ANON_TO_REAL_NODE.keys())
    if real_node not in mem._idx:
        return {"error": f"Node '{node}' not recognised. Valid nodes: {valid_anon}"}

    if qt == "parents":
        pars = mem.parents(real_node)
        return {"node": node, "parents": [
            {"node": _node_to_anon(n), "weight": round(w, 4)} for n, w in pars]}

    if qt == "children":
        chs = mem.children(real_node)
        return {"node": node, "children": [
            {"node": _node_to_anon(n), "weight": round(w, 4)} for n, w in chs]}

    if qt == "paths_to_target":
        paths = mem.causal_paths(real_node, "AUCbrain")
        anon_paths = [[_node_to_anon(n) for n in p] for p in paths]
        return {"source": node, "target": "Target", "paths": anon_paths}

    if qt == "total_effect":
        if "->" in node:
            src_a, tgt_a = [s.strip() for s in node.split("->", 1)]
            src_r = ANON_TO_REAL_NODE.get(src_a, src_a)
            tgt_r = ANON_TO_REAL_NODE.get(tgt_a, tgt_a)
        else:
            src_r, tgt_r = real_node, "AUCbrain"
        if src_r not in mem._idx or tgt_r not in mem._idx:
            return {"error": f"Unknown node. Valid: {valid_anon}"}
        effect = mem.total_effect(src_r, tgt_r)
        return {"source": _node_to_anon(src_r),
                "target": _node_to_anon(tgt_r),
                "total_effect": round(effect, 4)}

    if qt == "recommend":
        try:
            anon_vals = json.loads(node)
        except Exception:
            return {"error": "For 'recommend', pass node as JSON string of current anon design."}
        real_vals  = _to_real_design(anon_vals)
        current = {
            "LogSize":       math.log(real_vals["size_nm"]),
            "Zeta":          real_vals["zeta_mv"],
            "PEG":           float(real_vals["peg"]),
            "LogLigDensity": math.log1p(real_vals["ligand_density"]),
        }
        recs = mem.recommend_intervention(current, target="AUCbrain", top_k=5)
        # anonymise parameter names in recommendations
        anon_recs = []
        for rec in recs:
            r2 = rec.copy()
            r2["parameter"] = REAL_TO_ANON_NODE.get(rec.get("parameter", ""), rec.get("parameter", ""))
            anon_recs.append(r2)
        return {"recommendations": anon_recs}

    return {"error": f"Unknown query_type '{query_type}'. "
            "Choose: bottleneck | parents | children | paths_to_target | total_effect | recommend"}


def anon_lookup(param_name: str) -> dict:
    ranges = {
        "param_A": {"range": "20–200", "type": "continuous"},
        "param_B": {"range": "-40–10",  "type": "continuous"},
        "param_C": {"range": "0 or 1",  "type": "binary"},
        "param_D": {"range": f"one of {ANON_LIGAND_TYPES}", "type": "categorical"},
        "param_E": {"range": "0–200",   "type": "continuous"},
    }
    if param_name in ranges:
        return {"parameter": param_name, **ranges[param_name]}
    return {"error": f"Unknown parameter '{param_name}'. Available: {list(ranges.keys())}"}


def anon_check_feasibility(anon_design: dict) -> dict:
    violations, warnings_list = [], []
    a = float(anon_design.get("param_A", 80))
    b = float(anon_design.get("param_B", -15))
    c = int(anon_design.get("param_C",   1))
    d = str(anon_design.get("param_D",   "type_1"))
    e = float(anon_design.get("param_E", 30))

    if not (20 <= a <= 200):
        violations.append(f"param_A={a} outside [20, 200]")
    if not (-40 <= b <= 10):
        violations.append(f"param_B={b} outside [-40, 10]")
    if c not in (0, 1):
        violations.append(f"param_C must be 0 or 1")
    if d not in ANON_LIGAND_TYPES:
        violations.append(f"param_D must be one of {ANON_LIGAND_TYPES}")
    if not (0 <= e <= 200):
        violations.append(f"param_E={e} outside [0, 200]")
    return {"feasible": len(violations) == 0,
            "violations": violations, "warnings": warnings_list,
            "design": anon_design}


def anon_compare(design_a: dict, design_b: dict) -> dict:
    ra = anon_simulate(design_a)
    rb = anon_simulate(design_b)
    if not ra.get("success") or not rb.get("success"):
        return {"error": "Simulation failed."}
    delta = rb["Target_AUC"] - ra["Target_AUC"]
    return {
        "design_A": {"design": design_a, "Target_AUC": ra["Target_AUC"]},
        "design_B": {"design": design_b, "Target_AUC": rb["Target_AUC"]},
        "delta_Target_AUC": round(delta, 6),
        "improvement_pct":  round(100 * delta / max(ra["Target_AUC"], 1e-9), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Anonymised tool schemas (OpenAI format)
# ─────────────────────────────────────────────────────────────────────────────

_ANON_DESIGN_PROPS = {
    "param_A": {"type": "number",  "description": "Continuous parameter A (range 20–200)"},
    "param_B": {"type": "number",  "description": "Continuous parameter B (range -40 to 10)"},
    "param_C": {"type": "integer", "description": "Binary parameter C (0 or 1)"},
    "param_D": {"type": "string",  "description": f"Categorical parameter D: one of {ANON_LIGAND_TYPES}"},
    "param_E": {"type": "number",  "description": "Continuous parameter E (range 0–200)"},
}
_ANON_REQUIRED = ["param_A", "param_B", "param_C", "param_D", "param_E"]

ANON_TOOLS_OPENAI = [
    {"type": "function", "function": {
        "name": "simulate",
        "description": (
            "Run the oracle simulator for a given design and return Target_AUC "
            "(the quantity to maximise), Target_ratio, and four internal rates "
            "(rate_X, rate_Y, rate_Z, rate_W)."
        ),
        "parameters": {"type": "object", "properties": _ANON_DESIGN_PROPS,
                       "required": _ANON_REQUIRED},
    }},
    {"type": "function", "function": {
        "name": "query_causal_graph",
        "description": (
            "Query the structural causal model (9 nodes, 10 directed edges). "
            "Nodes: Node_1…Node_8 (internal variables) and Target (output to maximise). "
            "query_type options: bottleneck | parents | children | paths_to_target | "
            "total_effect | recommend. "
            "For total_effect: node='Node_X->Target'. "
            "For recommend: node=JSON string of current anon design."
        ),
        "parameters": {"type": "object",
                       "properties": {
                           "node":       {"type": "string"},
                           "query_type": {"type": "string"},
                       },
                       "required": ["node", "query_type"]},
    }},
    {"type": "function", "function": {
        "name": "lookup_parameter",
        "description": "Look up the valid range and type of a design parameter.",
        "parameters": {"type": "object",
                       "properties": {
                           "parameter_name": {"type": "string",
                                              "description": "One of param_A … param_E"},
                       },
                       "required": ["parameter_name"]},
    }},
    {"type": "function", "function": {
        "name": "check_feasibility",
        "description": "Check whether a design satisfies all parameter constraints.",
        "parameters": {"type": "object", "properties": _ANON_DESIGN_PROPS,
                       "required": _ANON_REQUIRED},
    }},
    {"type": "function", "function": {
        "name": "compare_designs",
        "description": "Simulate two designs and return the delta in Target_AUC.",
        "parameters": {"type": "object",
                       "properties": {
                           "design_a": {"type": "object", "properties": _ANON_DESIGN_PROPS},
                           "design_b": {"type": "object", "properties": _ANON_DESIGN_PROPS},
                       },
                       "required": ["design_a", "design_b"]},
    }},
]

# Ablation version: drop query_causal_graph
ANON_ABLATION_TOOLS = [t for t in ANON_TOOLS_OPENAI
                       if t["function"]["name"] != "query_causal_graph"]


def anon_dispatch(name: str, inputs: dict):
    if name == "simulate":
        return anon_simulate(inputs)
    if name == "query_causal_graph":
        return anon_query_causal(inputs["node"], inputs["query_type"])
    if name == "lookup_parameter":
        return anon_lookup(inputs["parameter_name"])
    if name == "check_feasibility":
        return anon_check_feasibility(inputs)
    if name == "compare_designs":
        return anon_compare(inputs["design_a"], inputs["design_b"])
    return {"error": f"Unknown tool: {name}"}


# ─────────────────────────────────────────────────────────────────────────────
# System prompts (completely domain-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

_ANON_CAUSAL_SYSTEM = """You are a black-box optimisation agent.

## Task
Maximise the scalar output called **Target_AUC** by choosing values for five
design parameters: param_A, param_B, param_C, param_D, param_E.

You have NO prior knowledge of what these parameters represent.
Their meaning, units, and optimal ranges are unknown to you — use your tools
to discover them.

## Available tools
- simulate           : evaluate a design → returns Target_AUC and internal rates
- query_causal_graph : query a structural causal model over 9 nodes
                       (Node_1…Node_8 + Target); nodes represent causal
                       relationships between design parameters and output
- lookup_parameter   : get the valid range for a parameter
- check_feasibility  : verify a design satisfies constraints
- compare_designs    : compare two designs side-by-side

## Strategy
1. Start by querying the causal graph (bottleneck) to understand which
   parameters have the largest causal effect on Target.
2. Use the causal structure to decide which parameters to change and in which
   direction.
3. Verify feasibility before simulating.
4. Stop when further improvements are <5% or after 6 design iterations.
5. Report the best design found.
"""

_ANON_ABLATION_SYSTEM = """You are a black-box optimisation agent.

## Task
Maximise the scalar output called **Target_AUC** by choosing values for five
design parameters: param_A, param_B, param_C, param_D, param_E.

You have NO prior knowledge of what these parameters represent.
Their meaning, units, and optimal ranges are unknown to you — use your tools
to discover them.

## Available tools
- simulate         : evaluate a design → returns Target_AUC and internal rates
- lookup_parameter : get the valid range for a parameter
- check_feasibility: verify a design satisfies constraints
- compare_designs  : compare two designs side-by-side

Note: the causal graph tool is NOT available. Explore through simulation.

## Strategy
1. Look up parameter ranges to understand the design space.
2. Run simulations to explore and find good designs.
3. Use compare_designs to track improvement.
4. Stop when improvements are <5% or after 6 design iterations.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Anonymous ReAct runner (shared logic for Causal and Ablation)
# ─────────────────────────────────────────────────────────────────────────────

def _run_anon_agent(anon_initial: dict, system_prompt: str,
                    tools: list, max_iterations: int = 20,
                    verbose: bool = False) -> dict:
    import textwrap as _tw

    backend = _detect_backend()
    client  = _make_client(backend)
    model   = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if backend != "deepseek":
        model = "claude-sonnet-4-6"

    user_msg = _tw.dedent(f"""
        Initial design (suboptimal):
        {json.dumps(anon_initial, indent=2)}

        Optimise this design to maximise Target_AUC.
        Use the available tools. Report the best design and Target_AUC found.
    """).strip()

    messages = [{"role": "user", "content": user_msg}]

    best_auc    = 0.0
    best_design = anon_initial.copy()
    trajectory  = []
    sim_count   = 0

    # evaluate initial design
    init_r = anon_simulate(anon_initial)
    if init_r.get("success"):
        best_auc  = init_r["Target_AUC"]
        sim_count = 1
        trajectory.append({"sim_call": sim_count, "AUC_brain": best_auc,
                            "best_so_far": best_auc, "step": 0})

    for iteration in range(1, max_iterations + 1):
        if backend == "deepseek":
            response = None
            for _attempt in range(5):
                try:
                    response = client.chat.completions.create(
                        model       = model,
                        messages    = [{"role": "system", "content": system_prompt}] + messages,
                        tools       = tools,
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
                    print("[Anon Agent] API failed after 5 retries — stopping early")
                break
            choice = response.choices[0]
            msg    = choice.message
            finish = choice.finish_reason

            if verbose and msg.content:
                print(msg.content)

            asst_msg = {"role": "assistant", "content": msg.content or ""}
            if getattr(msg, "reasoning_content", None):
                asst_msg["reasoning_content"] = msg.reasoning_content
            if msg.tool_calls:
                asst_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            messages.append(asst_msg)

            if finish == "stop" or not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                tool_name   = tc.function.name
                tool_inputs = json.loads(tc.function.arguments)
                result      = anon_dispatch(tool_name, tool_inputs)

                if verbose:
                    print(f"[{tool_name}] → {json.dumps(result, default=str)[:300]}")

                if tool_name == "simulate" and result.get("success"):
                    auc = result["Target_AUC"]
                    sim_count += 1
                    best_so_far = max(best_auc, auc)
                    if auc > best_auc:
                        best_auc    = auc
                        best_design = result["design_used"].copy()
                    trajectory.append({"sim_call": sim_count, "AUC_brain": auc,
                                       "best_so_far": best_so_far, "step": iteration})

                messages.append({"role": "tool", "tool_call_id": tc.id,
                                  "content": json.dumps(result, default=str)})

        else:  # anthropic — convert tools to Anthropic format
            import anthropic as _anth
            anth_tools = [{"name":         t["function"]["name"],
                           "description":  t["function"]["description"],
                           "input_schema": t["function"]["parameters"]}
                          for t in tools]
            response = client.messages.create(
                model=model, max_tokens=4096,
                system=system_prompt, tools=anth_tools, messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "end_turn":
                break
            if response.stop_reason != "tool_use":
                break
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = anon_dispatch(block.name, block.input)
                if block.name == "simulate" and result.get("success"):
                    auc = result["Target_AUC"]
                    sim_count += 1
                    best_so_far = max(best_auc, auc)
                    if auc > best_auc:
                        best_auc    = auc
                        best_design = result["design_used"].copy()
                    trajectory.append({"sim_call": sim_count, "AUC_brain": auc,
                                       "best_so_far": best_so_far, "step": iteration})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": json.dumps(result, default=str)})
            messages.append({"role": "user", "content": tool_results})

    return {"best_AUC": best_auc, "best_design": best_design,
            "trajectory": trajectory, "n_simulations": sim_count}


# ─────────────────────────────────────────────────────────────────────────────
# Plot (reuse style from baselines.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_anon(agg: dict, x_max: int, save_path: str | None = None) -> None:
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    method_order = ["Random Search", "Bayesian Optimisation",
                    "Anon Ablation", "Anon Causal"]
    colors  = {"Random Search": "#9E9E9E", "Bayesian Optimisation": "#2196F3",
               "Anon Ablation": "#FF9800", "Anon Causal": "#4CAF50"}
    markers = {"Random Search": "o", "Bayesian Optimisation": "s",
               "Anon Ablation": "^", "Anon Causal": "D"}
    xs = np.arange(1, x_max + 1)

    for m in method_order:
        if m not in agg:
            continue
        curves = agg[m]["curves"]
        mu, sd = curves.mean(0), curves.std(0)
        c, mk  = colors[m], markers[m]
        ax1.plot(xs, mu, color=c, marker=mk, markersize=4, linewidth=1.8,
                 label=m, markevery=max(1, x_max // 10))
        ax1.fill_between(xs, mu - sd, mu + sd, color=c, alpha=0.15)

    ax1.set_xlabel("Number of oracle simulations", fontsize=11)
    ax1.set_ylabel("Best Target_AUC (normalised)", fontsize=11)
    ax1.set_title("Convergence  (mean ± std, 5 seeds)\nAnonymised parameters",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, x_max); ax1.set_ylim(bottom=0)

    present = [m for m in method_order if m in agg]
    means   = [np.mean(agg[m]["best_aucs"]) for m in present]
    stds    = [np.std(agg[m]["best_aucs"])  for m in present]
    xlabels = {"Random Search": "Random\nSearch", "Bayesian Optimisation": "Bayesian\nOpt.",
               "Anon Ablation": "Ablation\n(anon)", "Anon Causal": "Causal\n(anon)"}

    bars = ax2.bar([xlabels[m] for m in present], means,
                   yerr=stds, color=[colors[m] for m in present],
                   edgecolor="white", linewidth=1.2, width=0.6,
                   capsize=5, error_kw={"elinewidth": 1.5})
    for bar, mu, sd in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 mu + sd + 0.01 * max(means),
                 f"{mu:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Best Target_AUC (normalised)", fontsize=11)
    ax2.set_title("Final Best AUC  (mean ± std)\nAnonymised parameters",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(means) * 1.25)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Phase 5 Option 2: Anonymised-Parameter Experiment\n"
        "(LLM has no prior domain knowledge — 5 seeds)",
        fontsize=11, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed runner
# ─────────────────────────────────────────────────────────────────────────────

def run_anon_multiseed(initial_design: dict, n_seeds: int = 5,
                       sim_budget: int = 20, verbose: bool = False) -> dict:
    anon_initial = _to_anon_design(initial_design)
    data_dir     = os.path.join(os.path.dirname(__file__), "data")

    all_runs: dict[str, list] = {
        "Random Search": [], "Bayesian Optimisation": [],
        "Anon Ablation": [], "Anon Causal": [],
    }

    print("\n" + "="*70)
    print(f"  PHASE 5 Option 2 — ANONYMISED EXPERIMENT  (n={n_seeds} seeds)")
    print(f"  LLM has NO prior domain knowledge of nanoparticles or BBB.")
    print("="*70)

    for i in range(n_seeds):
        print(f"\n── Seed {i}  ({i+1}/{n_seeds}) ─────────────────────────────")

        r = random_search(initial_design, n_trials=sim_budget, seed=i)
        all_runs["Random Search"].append(r)
        print(f"  RS   best={r['best_AUC']:.4f}")

        b = bayesian_optimisation(initial_design, n_trials=sim_budget, seed=i)
        all_runs["Bayesian Optimisation"].append(b)
        print(f"  BO   best={b['best_AUC']:.4f}")

        print(f"  Running Anon Ablation …", end=" ", flush=True)
        ab = _run_anon_agent(anon_initial, _ANON_ABLATION_SYSTEM,
                             ANON_ABLATION_TOOLS, max_iterations=sim_budget,
                             verbose=verbose)
        # convert trajectory key name for compatibility
        ab["method"] = "Anon Ablation"
        all_runs["Anon Ablation"].append(ab)
        print(f"best={ab['best_AUC']:.4f}  sims={ab['n_simulations']}")

        print(f"  Running Anon Causal  …", end=" ", flush=True)
        ca = _run_anon_agent(anon_initial, _ANON_CAUSAL_SYSTEM,
                             ANON_TOOLS_OPENAI, max_iterations=sim_budget,
                             verbose=verbose)
        ca["method"] = "Anon Causal"
        all_runs["Anon Causal"].append(ca)
        print(f"best={ca['best_AUC']:.4f}  sims={ca['n_simulations']}")

    # aggregate
    x_max = sim_budget + 5
    agg: dict[str, dict] = {}
    for method, runs in all_runs.items():
        curves    = np.array([_interp_curve(r["trajectory"], x_max) for r in runs])
        best_aucs = [r["best_AUC"] for r in runs]
        agg[method] = {"curves": curves, "best_aucs": best_aucs, "runs": runs}

    if "Anon Causal" in agg:
        threshold = 0.95 * np.mean(agg["Anon Causal"]["best_aucs"])
    else:
        threshold = 0.95 * max(np.mean(agg[m]["best_aucs"]) for m in agg)

    print("\n" + "="*70)
    print("  ANONYMISED EXPERIMENT RESULTS")
    print("="*70)
    print(f"  {'Method':<35} {'Mean AUC':>10}  {'±Std':>6}  {'Sims→95%':>10}")
    print("  " + "-"*65)
    for m in ["Random Search", "Bayesian Optimisation", "Anon Ablation", "Anon Causal"]:
        if m not in agg:
            continue
        d    = agg[m]
        mu   = float(np.mean(d["best_aucs"]))
        sd   = float(np.std(d["best_aucs"]))
        sims = [_sims_to_threshold(r["trajectory"], threshold) for r in d["runs"]]
        sims = [s for s in sims if s is not None]
        sstr = f"{np.mean(sims):.1f}±{np.std(sims):.1f}" if sims else ">budget"
        print(f"  {m:<35} {mu:>10.4f}  {sd:>6.4f}  {sstr:>10}")

    # save
    out_json = os.path.join(data_dir, "anon_multiseed_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        save = {m: {"best_aucs": d["best_aucs"],
                    "mean": float(np.mean(d["best_aucs"])),
                    "std":  float(np.std(d["best_aucs"])),
                    "curves": d["curves"].tolist()}
                for m, d in agg.items()}
        json.dump(save, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved → {out_json}")

    out_png = os.path.join(data_dir, "anon_multiseed_comparison.png")
    plot_anon(agg, x_max=x_max, save_path=out_png)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initial = {
        "size_nm": 150, "zeta_mv": 5, "peg": 0,
        "ligand_type": "transferrin", "ligand_density": 120,
    }
    run_anon_multiseed(initial, n_seeds=5, sim_budget=20)

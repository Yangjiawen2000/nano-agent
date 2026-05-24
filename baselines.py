"""
Phase 5: Baseline Comparison

Methods evaluated under identical simulation budgets (same initial design,
same maximum number of pbpk_simulate calls):

  1. Random Search        – uniform random sampling of design space
  2. Bayesian Optimisation – Optuna TPE sampler
  3. Ablation Agent       – ReAct Agent WITHOUT query_causal_graph tool
  4. Causal Agent         – full Phase-4 Agent (imported from agent.py)

Output:
  data/baseline_results.json   – all trajectories and best designs
  data/baseline_comparison.png – convergence curves + bar chart
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))
from pbpk_simulator import pbpk_simulate as _pbpk_simulate
from agent import (
    run_agent,
    TOOLS_OPENAI, TOOLS_ANTHROPIC,
    SYSTEM_PROMPT,
    dispatch_tool,
    tool_pbpk_simulate,
    _detect_backend, _make_client,
    _TOOL_DEFS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Design space
# ─────────────────────────────────────────────────────────────────────────────

LIGAND_TYPES = ["transferrin", "anti-TfR", "rabies-peptide", "none"]

SPACE = {
    "size_nm":        (20.0,  200.0),
    "zeta_mv":        (-40.0,  10.0),
    "peg":            (0,       1),       # integer
    "ligand_density": (0.0,   200.0),
}


def _random_design(rng: random.Random) -> dict:
    return {
        "size_nm":        rng.uniform(*SPACE["size_nm"]),
        "zeta_mv":        rng.uniform(*SPACE["zeta_mv"]),
        "peg":            rng.randint(0, 1),
        "ligand_type":    rng.choice(LIGAND_TYPES),
        "ligand_density": rng.uniform(*SPACE["ligand_density"]),
    }


def _simulate(design: dict) -> float:
    """Return AUC_brain or 0.0 on failure."""
    r = _pbpk_simulate(
        size_nm        = float(design["size_nm"]),
        zeta_mv        = float(design["zeta_mv"]),
        peg            = "yes" if int(design.get("peg", 0)) else "no",
        ligand_type    = str(design.get("ligand_type", "transferrin")),
        ligand_density = float(design["ligand_density"]),
    )
    return float(r["AUC"]) if r["success"] else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1 – Random Search
# ─────────────────────────────────────────────────────────────────────────────

def random_search(initial_design: dict, n_trials: int = 20,
                  seed: int = 42) -> dict:
    """
    Sample `n_trials` random designs (includes initial as trial 0).
    Returns trajectory indexed by simulation call number.
    """
    rng = random.Random(seed)
    best_auc    = 0.0
    best_design = initial_design.copy()
    trajectory  = []

    # always evaluate initial design first
    designs = [initial_design] + [_random_design(rng) for _ in range(n_trials - 1)]

    for i, d in enumerate(designs):
        auc = _simulate(d)
        if auc > best_auc:
            best_auc    = auc
            best_design = d.copy()
        trajectory.append({"sim_call": i + 1, "AUC_brain": auc,
                            "best_so_far": best_auc})

    return {"method": "Random Search", "best_AUC": best_auc,
            "best_design": best_design, "trajectory": trajectory,
            "n_simulations": n_trials}


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2 – Bayesian Optimisation (Optuna TPE)
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_optimisation(initial_design: dict, n_trials: int = 20,
                           seed: int = 42) -> dict:
    """
    TPE-based BO via Optuna. Initial design is evaluated first (trial 0).
    """
    sim_count  = 0
    trajectory = []

    # evaluate initial design outside Optuna to count it
    init_auc = _simulate(initial_design)
    sim_count += 1
    trajectory.append({"sim_call": sim_count, "AUC_brain": init_auc,
                       "best_so_far": init_auc})

    best_so_far = init_auc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    # seed study with initial design
    study.enqueue_trial({
        "size_nm":        initial_design["size_nm"],
        "zeta_mv":        initial_design["zeta_mv"],
        "peg":            int(initial_design.get("peg", 0)),
        "ligand_type_idx": LIGAND_TYPES.index(
                            str(initial_design.get("ligand_type", "transferrin"))),
        "ligand_density": initial_design["ligand_density"],
    })

    def objective(trial: optuna.Trial) -> float:
        nonlocal sim_count, best_so_far
        d = {
            "size_nm":        trial.suggest_float("size_nm", *SPACE["size_nm"]),
            "zeta_mv":        trial.suggest_float("zeta_mv", *SPACE["zeta_mv"]),
            "peg":            trial.suggest_int("peg", 0, 1),
            "ligand_type":    LIGAND_TYPES[trial.suggest_int(
                                  "ligand_type_idx", 0, len(LIGAND_TYPES) - 1)],
            "ligand_density": trial.suggest_float("ligand_density",
                                                  *SPACE["ligand_density"]),
        }
        auc = _simulate(d)
        sim_count += 1
        if auc > best_so_far:
            best_so_far = auc
        trajectory.append({"sim_call": sim_count, "AUC_brain": auc,
                            "best_so_far": best_so_far})
        return auc

    study.optimize(objective, n_trials=n_trials)

    best_t  = study.best_trial
    best_d  = {
        "size_nm":        best_t.params["size_nm"],
        "zeta_mv":        best_t.params["zeta_mv"],
        "peg":            best_t.params["peg"],
        "ligand_type":    LIGAND_TYPES[best_t.params["ligand_type_idx"]],
        "ligand_density": best_t.params["ligand_density"],
    }

    return {"method": "Bayesian Optimisation", "best_AUC": study.best_value,
            "best_design": best_d, "trajectory": trajectory,
            "n_simulations": sim_count}


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3 – Ablation Agent (no causal graph tool)
# ─────────────────────────────────────────────────────────────────────────────

# Tool list without query_causal_graph
_ABLATION_TOOL_DEFS = [t for t in _TOOL_DEFS if t["name"] != "query_causal_graph"]
_ABLATION_TOOLS_OPENAI     = [{"type": "function", "function": t}
                               for t in _ABLATION_TOOL_DEFS]
_ABLATION_TOOLS_ANTHROPIC  = [
    {"name": t["name"], "description": t["description"],
     "input_schema": t["parameters"]}
    for t in _ABLATION_TOOL_DEFS
]

_ABLATION_SYSTEM = SYSTEM_PROMPT.replace(
    "## Key causal facts (memorised from Phase 3)",
    "## Note\nThe causal graph tool is NOT available in this run. "
    "You must reason from first principles and use pbpk_simulate to explore.\n"
    "\n## [REMOVED causal facts]",
)

import textwrap as _textwrap
import json as _json


def run_ablation_agent(initial_design: dict, max_iterations: int = 10,
                       max_simulations: int | None = None,
                       verbose: bool = True) -> dict:
    """ReAct Agent without query_causal_graph — ablation study."""
    backend = _detect_backend()
    client  = _make_client(backend)
    _default_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    model   = _default_model if backend == "deepseek" else "claude-sonnet-4-6"
    tools   = (_ABLATION_TOOLS_OPENAI if backend == "deepseek"
               else _ABLATION_TOOLS_ANTHROPIC)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Ablation Agent (no causal graph)  [{backend}]")
        print(f"{'='*70}")

    user_msg = _textwrap.dedent(f"""
        Goal: Maximise AUC_brain (normalised brain drug exposure) for BBB-targeted NP.

        Initial NP design (suboptimal):
        {_json.dumps(initial_design, indent=2)}

        Note: the causal graph tool is NOT available.
        Use pbpk_simulate, lookup_parameter, check_feasibility, and compare_designs
        to explore and improve the design.
        Report the best design you find.
    """).strip()

    messages: list[dict] = [{"role": "user", "content": user_msg}]

    trajectory:  list[dict] = []
    best_design: dict       = initial_design.copy()
    best_auc:    float      = 0.0
    sim_count:   int        = 0

    _init = tool_pbpk_simulate(initial_design)
    if _init.get("success"):
        best_auc  = _init["AUC_brain"]
        sim_count = 1
        trajectory.append({"sim_call": sim_count, "AUC_brain": best_auc,
                            "best_so_far": best_auc, "step": 0})
        if verbose:
            print(f"  Initial AUC : {best_auc:.6f}")
            print(f"{'='*70}\n")

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n{'─'*60}  iteration {iteration}")

        if backend == "deepseek":
            response = None
            for _attempt in range(5):
                try:
                    response = client.chat.completions.create(
                        model       = model,
                        messages    = [{"role": "system", "content": _ABLATION_SYSTEM}]
                                      + messages,
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
                    print("[Ablation Agent] API failed after 5 retries — stopping early")
                break
            choice = response.choices[0]
            msg    = choice.message
            finish = choice.finish_reason

            if verbose and msg.content:
                print(msg.content)

            asst_msg: dict = {"role": "assistant", "content": msg.content or ""}
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

            _sim_budget_hit = False
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_inputs = _json.loads(tc.function.arguments)
                except Exception:
                    continue

                if verbose:
                    print(f"\n[Tool] {tool_name}")

                result = dispatch_tool(tool_name, tool_inputs)

                if verbose:
                    print(f"[Result] {_json.dumps(result, ensure_ascii=False, default=str)[:400]}")

                if tool_name == "pbpk_simulate" and result.get("success"):
                    auc = result["AUC_brain"]
                    sim_count += 1
                    best_so_far = max(best_auc, auc)
                    if auc > best_auc:
                        best_auc    = auc
                        best_design = result["design_used"].copy()
                    trajectory.append({"sim_call": sim_count, "AUC_brain": auc,
                                       "best_so_far": best_so_far, "step": iteration})
                    if max_simulations and sim_count >= max_simulations:
                        _sim_budget_hit = True

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      _json.dumps(result, ensure_ascii=False,
                                                default=str),
                })
                if _sim_budget_hit:
                    break

            if _sim_budget_hit:
                break

        else:  # anthropic
            response = client.messages.create(
                model      = model,
                max_tokens = 4096,
                system     = _ABLATION_SYSTEM,
                tools      = tools,
                messages   = messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if verbose:
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        print(block.text)

            if response.stop_reason == "end_turn":
                break
            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = dispatch_tool(block.name, block.input)
                if verbose:
                    print(f"[Tool] {block.name}")

                if block.name == "pbpk_simulate" and result.get("success"):
                    auc = result["AUC_brain"]
                    sim_count += 1
                    best_so_far = max(best_auc, auc)
                    if auc > best_auc:
                        best_auc    = auc
                        best_design = result["design_used"].copy()
                    trajectory.append({"sim_call": sim_count, "AUC_brain": auc,
                                       "best_so_far": best_so_far,
                                       "step": iteration})

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     _json.dumps(result, ensure_ascii=False,
                                               default=str),
                })
            messages.append({"role": "user", "content": tool_results})

    if verbose:
        print(f"\n  Ablation best AUC : {best_auc:.6f}  (sims={sim_count})")

    return {"method": "Ablation Agent (no causal graph)",
            "best_AUC": best_auc, "best_design": best_design,
            "trajectory": trajectory, "n_simulations": sim_count}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "Random Search":                   "#9E9E9E",
    "Bayesian Optimisation":           "#2196F3",
    "Ablation Agent (no causal graph)":"#FF9800",
    "Causal Agent":                    "#4CAF50",
}

METHOD_MARKERS = {
    "Random Search":                   "o",
    "Bayesian Optimisation":           "s",
    "Ablation Agent (no causal graph)":"^",
    "Causal Agent":                    "D",
}


def plot_comparison(results: dict, save_path: str | None = None) -> None:
    """
    Two-panel figure:
      Left  – convergence curves: best AUC vs simulation call
      Right – bar chart: final best AUC per method
    """
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    method_order = [
        "Random Search",
        "Bayesian Optimisation",
        "Ablation Agent (no causal graph)",
        "Causal Agent",
    ]

    for method in method_order:
        if method not in results:
            continue
        r    = results[method]
        traj = r["trajectory"]
        if not traj:
            continue

        xs = [t["sim_call"]    for t in traj]
        ys = [t["best_so_far"] for t in traj]

        # extend to max sim_call for fair visual comparison
        color  = METHOD_COLORS.get(method, "black")
        marker = METHOD_MARKERS.get(method, "o")

        ax1.plot(xs, ys, color=color, marker=marker, markersize=5,
                 linewidth=1.8, label=method)

    ax1.set_xlabel("Number of PBPK simulations", fontsize=11)
    ax1.set_ylabel("Best AUC$_{\\mathrm{brain}}$ (normalised)", fontsize=11)
    ax1.set_title("Convergence Curves", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # bar chart
    present = [m for m in method_order if m in results]
    bar_vals = [results[m]["best_AUC"] for m in present]
    colors   = [METHOD_COLORS.get(m, "grey") for m in present]
    short_labels = {
        "Random Search":                   "Random\nSearch",
        "Bayesian Optimisation":           "Bayesian\nOpt.",
        "Ablation Agent (no causal graph)":"Ablation\n(no causal)",
        "Causal Agent":                    "Causal\nAgent",
    }
    xlabels = [short_labels.get(m, m) for m in present]

    bars = ax2.bar(xlabels, bar_vals, color=colors, edgecolor="white",
                   linewidth=1.2, width=0.6)
    for bar, val in zip(bars, bar_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02 * max(bar_vals),
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    ax2.set_ylabel("Best AUC$_{\\mathrm{brain}}$ (normalised)", fontsize=11)
    ax2.set_title("Final Best AUC", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(bar_vals) * 1.18)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Phase 5: Baseline Comparison — Brain-Targeted NP Optimisation\n"
        "(same initial design, same simulation budget)",
        fontsize=11, y=1.02,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison runner
# ─────────────────────────────────────────────────────────────────────────────

def compare_all(
    initial_design:  dict,
    sim_budget:      int  = 20,
    seed:            int  = 42,
    run_llm_methods: bool = True,
    verbose:         bool = True,
) -> dict:
    """
    Run all four methods from `initial_design` with `sim_budget` PBPK calls each.

    Parameters
    ----------
    run_llm_methods : bool
        Set False to skip Ablation + Causal Agent (useful for quick testing
        without an API key).
    """
    results = {}
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    print("\n" + "="*70)
    print("  PHASE 5 — BASELINE COMPARISON")
    print("="*70)
    print(f"  Initial design : {initial_design}")
    print(f"  Simulation budget per method : {sim_budget}")
    print("="*70 + "\n")

    # ── 1. Random Search ────────────────────────────────────────────────────
    print("▶ Running Random Search …")
    t0 = time.time()
    results["Random Search"] = random_search(initial_design, n_trials=sim_budget, seed=seed)
    print(f"  Done in {time.time()-t0:.1f}s  best AUC = {results['Random Search']['best_AUC']:.4f}")

    # ── 2. Bayesian Optimisation ─────────────────────────────────────────────
    print("\n▶ Running Bayesian Optimisation (Optuna TPE) …")
    t0 = time.time()
    results["Bayesian Optimisation"] = bayesian_optimisation(
        initial_design, n_trials=sim_budget, seed=seed)
    print(f"  Done in {time.time()-t0:.1f}s  best AUC = {results['Bayesian Optimisation']['best_AUC']:.4f}")

    # ── 3. Ablation Agent ────────────────────────────────────────────────────
    if run_llm_methods:
        print("\n▶ Running Ablation Agent (no causal graph) …")
        t0 = time.time()
        results["Ablation Agent (no causal graph)"] = run_ablation_agent(
            initial_design, max_iterations=sim_budget, verbose=verbose)
        print(f"  Done in {time.time()-t0:.1f}s  best AUC = {results['Ablation Agent (no causal graph)']['best_AUC']:.4f}")

    # ── 4. Causal Agent ──────────────────────────────────────────────────────
    if run_llm_methods:
        print("\n▶ Running Causal Agent (Phase 4) …")
        t0 = time.time()
        agent_result = run_agent(
            initial_design  = initial_design,
            goal            = "Maximise AUC_brain for BBB-targeted NP",
            max_iterations  = sim_budget,
            verbose         = verbose,
        )
        # convert agent trajectory to sim_call indexed format
        agent_traj   = []
        sim_call_idx = 0
        best_so_far  = 0.0
        for pt in agent_result["trajectory"]:
            sim_call_idx += 1
            best_so_far   = max(best_so_far, pt["AUC_brain"])
            agent_traj.append({"sim_call": sim_call_idx,
                                "AUC_brain": pt["AUC_brain"],
                                "best_so_far": best_so_far,
                                "step": pt["step"]})
        results["Causal Agent"] = {
            "method":        "Causal Agent",
            "best_AUC":      agent_result["best_AUC"],
            "best_design":   agent_result["best_design"],
            "trajectory":    agent_traj,
            "n_simulations": len(agent_traj),
        }
        print(f"  Done in {time.time()-t0:.1f}s  best AUC = {results['Causal Agent']['best_AUC']:.4f}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    print(f"  {'Method':<40} {'Best AUC':>10}  {'#Sims':>6}")
    print("  " + "-"*58)
    for method, r in results.items():
        print(f"  {method:<40} {r['best_AUC']:>10.4f}  {r['n_simulations']:>6}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_json = os.path.join(data_dir, "baseline_results.json")
    serialisable = {}
    for method, r in results.items():
        serialisable[method] = {
            "method":        r["method"],
            "best_AUC":      r["best_AUC"],
            "best_design":   r["best_design"],
            "n_simulations": r["n_simulations"],
            "trajectory":    r["trajectory"],
        }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved → {out_json}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    out_png = os.path.join(data_dir, "baseline_comparison.png")
    plot_comparison(results, save_path=out_png)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed runner
# ─────────────────────────────────────────────────────────────────────────────

def _interp_curve(traj: list[dict], x_max: int) -> np.ndarray:
    """
    Interpolate best_so_far to a common x-grid [1 .. x_max].
    Pads the last value if the trajectory is shorter than x_max.
    """
    xs = [t["sim_call"] for t in traj]
    ys = [t["best_so_far"] for t in traj]
    out = np.zeros(x_max)
    for xi in range(1, x_max + 1):
        # last point with sim_call <= xi
        vals = [y for x, y in zip(xs, ys) if x <= xi]
        out[xi - 1] = vals[-1] if vals else 0.0
    return out


def _sims_to_threshold(traj: list[dict], threshold: float) -> int | None:
    """Return first sim_call where best_so_far >= threshold, else None."""
    for t in traj:
        if t["best_so_far"] >= threshold:
            return t["sim_call"]
    return None


def plot_multi_seed(agg: dict, x_max: int, save_path: str | None = None) -> None:
    """
    Two-panel figure with mean ± std shading.
    agg[method] = {"curves": ndarray(n_seeds, x_max),
                   "best_aucs": list[float]}
    """
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    method_order = [
        "Random Search",
        "Bayesian Optimisation",
        "Ablation Agent (no causal graph)",
        "Causal Agent",
    ]
    xs = np.arange(1, x_max + 1)

    for method in method_order:
        if method not in agg:
            continue
        curves = agg[method]["curves"]          # shape (n_seeds, x_max)
        mu     = curves.mean(axis=0)
        sigma  = curves.std(axis=0)
        color  = METHOD_COLORS.get(method, "black")
        marker = METHOD_MARKERS.get(method, "o")

        ax1.plot(xs, mu, color=color, marker=marker, markersize=4,
                 linewidth=1.8, label=method, markevery=max(1, x_max // 10))
        ax1.fill_between(xs, mu - sigma, mu + sigma,
                         color=color, alpha=0.15)

    ax1.set_xlabel("Number of PBPK simulations", fontsize=11)
    ax1.set_ylabel("Best AUC$_{\\mathrm{brain}}$ (normalised)", fontsize=11)
    ax1.set_title("Convergence Curves  (mean ± std, 5 seeds)", fontsize=12,
                  fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, x_max)
    ax1.set_ylim(bottom=0)

    # bar chart: mean ± std of final best AUC
    present  = [m for m in method_order if m in agg]
    means    = [np.mean(agg[m]["best_aucs"]) for m in present]
    stds     = [np.std(agg[m]["best_aucs"])  for m in present]
    colors   = [METHOD_COLORS.get(m, "grey") for m in present]
    short_labels = {
        "Random Search":                   "Random\nSearch",
        "Bayesian Optimisation":           "Bayesian\nOpt.",
        "Ablation Agent (no causal graph)":"Ablation\n(no causal)",
        "Causal Agent":                    "Causal\nAgent",
    }
    xlabels = [short_labels.get(m, m) for m in present]

    bars = ax2.bar(xlabels, means, yerr=stds, color=colors,
                   edgecolor="white", linewidth=1.2, width=0.6,
                   capsize=5, error_kw={"elinewidth": 1.5})
    for bar, mu, sd in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 mu + sd + 0.01 * max(means),
                 f"{mu:.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax2.set_ylabel("Best AUC$_{\\mathrm{brain}}$ (normalised)", fontsize=11)
    ax2.set_title("Final Best AUC  (mean ± std)", fontsize=12,
                  fontweight="bold")
    ax2.set_ylim(0, max(means) * 1.25)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Phase 5: Baseline Comparison — Brain-Targeted NP Optimisation\n"
        "(5 independent seeds, same initial design)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    plt.close()


def run_multi_seed(
    initial_design:  dict,
    n_seeds:         int  = 5,
    sim_budget:      int  = 20,
    run_llm_methods: bool = True,
    verbose:         bool = False,
) -> dict:
    """
    Run all methods n_seeds times and report mean ± std.
    Seeds for RS/BO: 0, 1, 2, 3, 4.
    LLM agents: n_seeds independent runs (no seed control for LLM).
    """
    seeds    = list(range(n_seeds))
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    # storage: method → list of single-run results
    all_runs: dict[str, list[dict]] = {
        "Random Search":       [],
        "Bayesian Optimisation": [],
    }
    if run_llm_methods:
        all_runs["Ablation Agent (no causal graph)"] = []
        all_runs["Causal Agent"]                     = []

    print("\n" + "="*70)
    print(f"  PHASE 5 — MULTI-SEED COMPARISON  (n={n_seeds} seeds)")
    print("="*70)

    for i, seed in enumerate(seeds):
        print(f"\n── Seed {seed}  ({i+1}/{n_seeds}) ─────────────────────────────")

        # ── Random Search ──────────────────────────────────────────────────
        r = random_search(initial_design, n_trials=sim_budget, seed=seed)
        all_runs["Random Search"].append(r)
        print(f"  RS   best={r['best_AUC']:.4f}")

        # ── Bayesian Optimisation ──────────────────────────────────────────
        b = bayesian_optimisation(initial_design, n_trials=sim_budget, seed=seed)
        all_runs["Bayesian Optimisation"].append(b)
        print(f"  BO   best={b['best_AUC']:.4f}")

        if run_llm_methods:
            # ── Ablation Agent ─────────────────────────────────────────────
            print(f"  Running Ablation Agent …", end=" ", flush=True)
            ab = run_ablation_agent(initial_design,
                                    max_iterations=sim_budget, verbose=False)
            all_runs["Ablation Agent (no causal graph)"].append(ab)
            print(f"best={ab['best_AUC']:.4f}  sims={ab['n_simulations']}")

            # ── Causal Agent ───────────────────────────────────────────────
            print(f"  Running Causal Agent …", end=" ", flush=True)
            ca = run_agent(initial_design,
                           goal           = "Maximise AUC_brain for BBB-targeted NP",
                           max_iterations = sim_budget,
                           verbose        = False)
            # normalise trajectory to sim_call format
            ca_traj   = []
            best_so_far = 0.0
            for k, pt in enumerate(ca["trajectory"]):
                best_so_far = max(best_so_far, pt["AUC_brain"])
                ca_traj.append({"sim_call": k + 1,
                                 "AUC_brain": pt["AUC_brain"],
                                 "best_so_far": best_so_far})
            all_runs["Causal Agent"].append({
                "method":        "Causal Agent",
                "best_AUC":      ca["best_AUC"],
                "best_design":   ca["best_design"],
                "trajectory":    ca_traj,
                "n_simulations": len(ca_traj),
            })
            print(f"best={ca['best_AUC']:.4f}  sims={len(ca_traj)}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    x_max = sim_budget + 5   # a bit of headroom for LLM agents
    agg: dict[str, dict] = {}

    for method, runs in all_runs.items():
        curves = np.array([_interp_curve(r["trajectory"], x_max) for r in runs])
        best_aucs = [r["best_AUC"] for r in runs]
        agg[method] = {"curves": curves, "best_aucs": best_aucs, "runs": runs}

    # ── Summary table ─────────────────────────────────────────────────────────
    # compute sims to 95% of the global best (Causal Agent mean)
    if "Causal Agent" in agg:
        global_best = np.mean(agg["Causal Agent"]["best_aucs"])
        threshold   = 0.95 * global_best
    else:
        global_best = max(
            np.mean(agg[m]["best_aucs"]) for m in agg)
        threshold   = 0.95 * global_best

    print("\n" + "="*70)
    print("  MULTI-SEED RESULTS")
    print("="*70)
    print(f"  {'Method':<40} {'Mean AUC':>10}  {'±Std':>6}  {'Sims→95%':>10}")
    print("  " + "-"*70)

    for method in ["Random Search", "Bayesian Optimisation",
                   "Ablation Agent (no causal graph)", "Causal Agent"]:
        if method not in agg:
            continue
        d     = agg[method]
        mu    = float(np.mean(d["best_aucs"]))
        sd    = float(np.std(d["best_aucs"]))
        sims  = []
        for r in d["runs"]:
            s = _sims_to_threshold(r["trajectory"], threshold)
            if s is not None:
                sims.append(s)
        sims_str = (f"{np.mean(sims):.1f}±{np.std(sims):.1f}"
                    if sims else ">budget")
        print(f"  {method:<40} {mu:>10.4f}  {sd:>6.4f}  {sims_str:>10}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = os.path.join(data_dir, "multiseed_results.json")
    save_data = {}
    for method, d in agg.items():
        save_data[method] = {
            "best_aucs":     d["best_aucs"],
            "mean_best_auc": float(np.mean(d["best_aucs"])),
            "std_best_auc":  float(np.std(d["best_aucs"])),
            "curves":        d["curves"].tolist(),
        }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved → {out_json}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    out_png = os.path.join(data_dir, "multiseed_comparison.png")
    plot_multi_seed(agg, x_max=x_max, save_path=out_png)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=20)
    args = parser.parse_args()

    initial = {
        "size_nm":        150,
        "zeta_mv":         +5,
        "peg":              0,
        "ligand_type":   "transferrin",
        "ligand_density": 120,
    }

    if args.mode == "single":
        compare_all(initial_design=initial, sim_budget=args.budget,
                    seed=42, run_llm_methods=True, verbose=False)
    else:
        run_multi_seed(initial_design=initial, n_seeds=args.seeds,
                       sim_budget=args.budget, run_llm_methods=True)

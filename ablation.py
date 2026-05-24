"""
Phase 6: Ablation Study

2×2 factorial design isolating the two key components of the Causal Agent:
  - Domain knowledge  (real parameter names vs. anonymous codes)
  - Causal graph      (query_causal_graph tool present vs. absent)

                    Causal graph: YES        Causal graph: NO
Domain: YES    │  Full Causal Agent     │  −Causal Graph (Ablation)  │
Domain: NO     │  −Domain Knowledge     │  −Both                     │

Metrics (5 seeds, single canonical starting point):
  AUC_brain   – primary optimisation target
  AUC_ratio   – BBB selectivity (brain / blood)
  AUCC        – area under convergence curve (0-1, sample efficiency)
  Sims→95%    – simulations needed to reach 95 % of global optimum

Output:
  data/ablation_results.json
  data/ablation_convergence.png   – 4 convergence curves on one axes
  data/ablation_bars.png          – grouped bar chart (4 metrics)
  data/ablation_heatmap.png       – 2×2 factorial heatmap
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from pbpk_simulator import pbpk_simulate as _pbpk_simulate
from agent import run_agent
from baselines import (
    run_ablation_agent,
    _interp_curve, _sims_to_threshold,
)
from baselines_anon import (
    _run_anon_agent, _to_anon_design,
    ANON_TO_REAL_PARAM, ANON_TO_REAL_LIGAND,
    ANON_TOOLS_OPENAI, ANON_ABLATION_TOOLS,
    _ANON_CAUSAL_SYSTEM, _ANON_ABLATION_SYSTEM,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

INITIAL_DESIGN = {
    "size_nm": 150, "zeta_mv": 5, "peg": 0,
    "ligand_type": "transferrin", "ligand_density": 120,
}

CONDITIONS = [
    {"label": "Full Causal Agent",    "domain": True,  "causal": True},
    {"label": "−Causal Graph",        "domain": True,  "causal": False},
    {"label": "−Domain Knowledge",    "domain": False, "causal": True},
    {"label": "−Both",                "domain": False, "causal": False},
]

COLORS = {
    "Full Causal Agent":  "#4CAF50",
    "−Causal Graph":      "#FF9800",
    "−Domain Knowledge":  "#2196F3",
    "−Both":              "#9E9E9E",
}
MARKERS = {
    "Full Causal Agent":  "D",
    "−Causal Graph":      "^",
    "−Domain Knowledge":  "s",
    "−Both":              "o",
}

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_auc_ratio(design: dict) -> float:
    peg_val = design.get("peg", 1)
    if isinstance(peg_val, str):
        peg_str = "yes" if peg_val.lower() in ("yes", "1", "true") else "no"
    else:
        peg_str = "yes" if int(peg_val) else "no"
    r = _pbpk_simulate(
        size_nm        = float(design.get("size_nm", 80)),
        zeta_mv        = float(design.get("zeta_mv", -15)),
        peg            = peg_str,
        ligand_type    = str(design.get("ligand_type", "transferrin")),
        ligand_density = float(design.get("ligand_density", 30)),
    )
    return float(r["AUC_ratio"]) if r["success"] else 0.0


def _anon_to_real_design(anon_design: dict) -> dict:
    real = {}
    for k, v in anon_design.items():
        real_key = ANON_TO_REAL_PARAM.get(k, k)
        if real_key == "ligand_type":
            real[real_key] = ANON_TO_REAL_LIGAND.get(str(v), str(v))
        else:
            real[real_key] = v
    return real


def _aucc(curve: np.ndarray, optimal: float) -> float:
    if optimal < 1e-9:
        return 0.0
    return min(float(np.trapezoid(curve)) / (len(curve) * optimal), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_condition(cond: dict, initial: dict, n_seeds: int, sim_budget: int) -> list[dict]:
    """Run one ablation condition for n_seeds. Returns list of run dicts."""
    label  = cond["label"]
    domain = cond["domain"]
    causal = cond["causal"]
    runs   = []

    for seed in range(n_seeds):
        print(f"    seed {seed}/{n_seeds-1}  [{label}]", flush=True)
        if domain and causal:
            raw = run_agent(initial,
                            goal="Maximise AUC_brain for BBB-targeted NP",
                            max_iterations=sim_budget, verbose=False)
            traj, best_sf = [], 0.0
            for pt in raw["trajectory"]:
                best_sf = max(best_sf, pt["AUC_brain"])
                traj.append({"sim_call": len(traj) + 1,
                              "AUC_brain": pt["AUC_brain"],
                              "best_so_far": best_sf})
            runs.append({
                "best_AUC":      raw["best_AUC"],
                "best_design":   raw["best_design"],
                "trajectory":    traj,
                "n_simulations": len(traj),
            })

        elif domain and not causal:
            raw = run_ablation_agent(initial, max_iterations=sim_budget, verbose=False)
            runs.append(raw)

        else:
            # anonymous interface
            anon_init = _to_anon_design(initial)
            tools     = ANON_TOOLS_OPENAI if causal else ANON_ABLATION_TOOLS
            system    = _ANON_CAUSAL_SYSTEM if causal else _ANON_ABLATION_SYSTEM
            raw = _run_anon_agent(anon_init, system_prompt=system,
                                  tools=tools, max_iterations=sim_budget,
                                  verbose=False)
            # convert anon best_design → real for AUC_ratio computation
            real_design = _anon_to_real_design(raw["best_design"])
            raw["best_design_real"] = real_design
            runs.append(raw)

    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate helper
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(runs: list[dict], x_max: int, threshold: float,
               cross_opt: float | None = None) -> dict:
    curves    = np.array([_interp_curve(r["trajectory"], x_max) for r in runs])
    best_aucs = [r["best_AUC"] for r in runs]

    auc_ratios = []
    for r in runs:
        design = r.get("best_design_real") or r["best_design"]
        auc_ratios.append(_get_auc_ratio(design))

    # use cross-condition global optimum for AUCC so values are comparable
    opt_for_aucc = cross_opt if cross_opt is not None else max(best_aucs)
    auccs        = [_aucc(c, opt_for_aucc) for c in curves]

    sims       = [_sims_to_threshold(r["trajectory"], threshold) for r in runs]
    sims_valid = [s for s in sims if s is not None]

    return {
        "curves":     curves,
        "best_aucs":  best_aucs,
        "auc_ratios": auc_ratios,
        "auccs":      auccs,
        "sims_conv":  sims_valid,
        "runs":       runs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(agg: dict, x_max: int, save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(1, x_max + 1)
    for label, d in agg.items():
        mu = d["curves"].mean(0)
        sd = d["curves"].std(0)
        c  = COLORS[label]
        mk = MARKERS[label]
        ax.plot(xs, mu, color=c, marker=mk, markersize=4, linewidth=2,
                label=label, markevery=max(1, x_max // 10))
        ax.fill_between(xs, mu - sd, mu + sd, color=c, alpha=0.15)
    ax.set_xlabel("Number of PBPK simulations", fontsize=12)
    ax.set_ylabel("Best AUC$_{brain}$ (normalised)", fontsize=12)
    ax.set_title("Ablation Study — Convergence Curves\n(mean ± std, 5 seeds)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, x_max); ax.set_ylim(bottom=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Convergence plot → {save_path}")
    plt.close()


def plot_bars(agg: dict, save_path: str | None = None) -> None:
    labels  = list(agg.keys())
    metrics = [
        ("AUC$_{brain}$",    "best_aucs",  "Best AUC$_{brain}$"),
        ("AUC ratio",        "auc_ratios", "AUC ratio (brain/blood)"),
        ("AUCC",             "auccs",      "AUCC (sample efficiency)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (title, key, ylabel) in zip(axes, metrics):
        means = [np.mean(agg[l][key]) for l in labels]
        stds  = [np.std(agg[l][key])  for l in labels]
        colors = [COLORS[l] for l in labels]
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                      linewidth=1.2, width=0.6, capsize=5,
                      error_kw={"elinewidth": 1.5})
        for bar, mu, sd in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mu + sd + 0.005 * max(means),
                    f"{mu:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("−", "−\n") for l in labels],
                           fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(means) * 1.3)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Ablation Study — Component Contributions (5 seeds)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Bar chart → {save_path}")
    plt.close()


def plot_heatmap(agg: dict, save_path: str | None = None) -> None:
    """2×2 factorial heatmap for AUC_brain."""
    # rows = domain knowledge, cols = causal graph
    grid = np.array([
        [np.mean(agg["Full Causal Agent"]["best_aucs"]),
         np.mean(agg["−Causal Graph"]["best_aucs"])],
        [np.mean(agg["−Domain Knowledge"]["best_aucs"]),
         np.mean(agg["−Both"]["best_aucs"])],
    ])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid, cmap="YlGn", vmin=0, vmax=2.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean AUC$_{brain}$")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Causal graph: YES", "Causal graph: NO"], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Domain knowledge: YES", "Domain knowledge: NO"], fontsize=11)

    labels_map = [
        ["Full Causal\nAgent", "−Causal\nGraph"],
        ["−Domain\nKnowledge", "−Both"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels_map[i][j]}\n{grid[i, j]:.3f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if grid[i, j] > 1.5 else "black",
                    fontweight="bold")

    ax.set_title("2×2 Factorial Ablation — Mean AUC$_{brain}$\n(5 seeds)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(n_seeds: int = 5, sim_budget: int = 20) -> dict:
    data_dir    = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    ckpt_path   = os.path.join(data_dir, "ablation_checkpoint.json")

    x_max      = sim_budget + 5
    global_opt = 0.0

    # load checkpoint if available
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            all_runs = json.load(f)
        print(f"  Resumed from checkpoint ({list(all_runs.keys())})", flush=True)
    else:
        all_runs = {}

    for cond in CONDITIONS:
        label = cond["label"]
        if label in all_runs:
            best = max(r["best_AUC"] for r in all_runs[label])
            global_opt = max(global_opt, best)
            print(f"  Skipping '{label}' (already in checkpoint, best={best:.4f})", flush=True)
            continue
        print(f"\n{'='*60}")
        print(f"  Condition: {label}")
        print(f"{'='*60}", flush=True)
        t0    = time.time()
        runs  = _run_condition(cond, INITIAL_DESIGN.copy(), n_seeds, sim_budget)
        elapsed = time.time() - t0
        best  = max(r["best_AUC"] for r in runs)
        global_opt = max(global_opt, best)
        all_runs[label] = runs
        # save checkpoint after each condition
        with open(ckpt_path, "w") as f:
            json.dump(all_runs, f)
        print(f"  Done ({elapsed:.0f}s)  best={best:.4f}", flush=True)

    threshold = 0.95 * global_opt

    agg = {}
    for label, runs in all_runs.items():
        agg[label] = _aggregate(runs, x_max, threshold, cross_opt=global_opt)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'='*70}")
    print(f"  {'Condition':<24} {'AUC_brain':>12}  {'AUC_ratio':>12}  "
          f"{'AUCC':>6}  {'Sims→95%':>10}")
    print("  " + "─" * 68)
    for label, d in agg.items():
        ab = f"{np.mean(d['best_aucs']):.3f}±{np.std(d['best_aucs']):.3f}"
        ar = f"{np.mean(d['auc_ratios']):.3f}±{np.std(d['auc_ratios']):.3f}"
        au = f"{np.mean(d['auccs']):.3f}"
        sc = (f"{np.mean(d['sims_conv']):.1f}±{np.std(d['sims_conv']):.1f}"
              if d["sims_conv"] else ">budget")
        print(f"  {label:<24} {ab:>14}  {ar:>14}  {au:>6}  {sc:>10}")

    # ── Save results ───────────────────────────────────────────────────────
    serialisable = {}
    for label, d in agg.items():
        serialisable[label] = {
            "best_aucs":  d["best_aucs"],
            "auc_ratios": d["auc_ratios"],
            "auccs":      d["auccs"],
            "sims_conv":  d["sims_conv"],
            "mean_auc":   float(np.mean(d["best_aucs"])),
            "std_auc":    float(np.std(d["best_aucs"])),
        }
    out_json = os.path.join(data_dir, "ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  JSON saved → {out_json}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_convergence(agg, x_max,
                     save_path=os.path.join(data_dir, "ablation_convergence.png"))
    plot_bars(agg,
              save_path=os.path.join(data_dir, "ablation_bars.png"))
    plot_heatmap(agg,
                 save_path=os.path.join(data_dir, "ablation_heatmap.png"))

    return agg


if __name__ == "__main__":
    run_ablation(n_seeds=5, sim_budget=20)

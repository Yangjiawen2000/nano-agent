"""
Phase 5 – Multi-Starting-Point Experiment

4 initial designs × 5 seeds × 4 methods, reporting 4 metrics:
  1. AUC_brain          – optimisation target (higher = more brain exposure)
  2. AUC_ratio          – brain/blood AUC ratio (BBB selectivity)
  3. Sims_to_conv       – simulations to reach 95 % of optimal (sample efficiency)
  4. AUCC               – area under convergence curve, normalised 0–1
                          (= average quality across the whole optimisation run)

Output:
  data/multi_start_results.json
  data/multi_start_convergence.png   – 4-panel convergence (one per start)
  data/multi_start_summary.png       – metric heatmap + bar chart
"""

from __future__ import annotations

import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from pbpk_simulator import pbpk_simulate as _pbpk_simulate
from baselines import (
    random_search, bayesian_optimisation,
    run_ablation_agent, _interp_curve, _sims_to_threshold,
    METHOD_COLORS,
)
from agent import run_agent

# ─────────────────────────────────────────────────────────────────────────────
# Starting designs
# ─────────────────────────────────────────────────────────────────────────────

STARTING_DESIGNS = {
    "A – all-bad": {
        "size_nm": 150, "zeta_mv":  +5, "peg": 0,
        "ligand_type": "transferrin", "ligand_density": 120,
        "note": "All params in wrong direction (avidity trap, +zeta, large, no PEG)",
    },
    "B – size ok": {
        "size_nm":  80, "zeta_mv":  +8, "peg": 1,
        "ligand_type": "transferrin", "ligand_density": 150,
        "note": "Size correct, but zeta positive + avidity trap",
    },
    "C – random": {
        "size_nm": 120, "zeta_mv": -30, "peg": 0,
        "ligand_type": "none", "ligand_density": 10,
        "note": "Random-like: wrong ligand, very negative zeta, no PEG",
    },
    "D – near-optimal": {
        "size_nm":  80, "zeta_mv": -15, "peg": 1,
        "ligand_type": "none", "ligand_density": 30,
        "note": "Near-optimal but wrong ligand type",
    },
}

METHODS = ["Random Search", "Bayesian Optimisation",
           "Ablation Agent", "Causal Agent"]

METHOD_SHORT = {
    "Random Search":      "RS",
    "Bayesian Optimisation": "BO",
    "Ablation Agent":     "Ablation",
    "Causal Agent":       "Causal",
}

COLORS = {
    "Random Search":      "#9E9E9E",
    "Bayesian Optimisation": "#2196F3",
    "Ablation Agent":     "#FF9800",
    "Causal Agent":       "#4CAF50",
}

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_auc_ratio(design: dict) -> float:
    """Simulate best design once to get AUC_ratio (brain/blood)."""
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


def _aucc(curve: np.ndarray, optimal: float) -> float:
    """
    Normalised area under convergence curve.
    = trapezoid(curve) / (len(curve) * optimal)
    Range [0, 1]. 1.0 = found optimal on first simulation.
    """
    if optimal < 1e-9:
        return 0.0
    area = float(np.trapezoid(curve))
    return min(area / (len(curve) * optimal), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Single starting-point runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_start(initial: dict, n_seeds: int = 5,
                  sim_budget: int = 20, verbose: bool = False) -> dict:
    """
    Run all 4 methods from `initial` with `n_seeds` independent repetitions.
    Returns per-method aggregated results.
    """
    # strip 'note' key before passing to simulators
    clean = {k: v for k, v in initial.items() if k != "note"}

    x_max = sim_budget + 5
    storage: dict[str, list] = {m: [] for m in METHODS}

    for seed in range(n_seeds):
        print(f"  ── Seed {seed}/{n_seeds-1} ──────────────────────────────────", flush=True)

        # Random Search
        r = random_search(clean, n_trials=sim_budget, seed=seed)
        storage["Random Search"].append(r)
        print(f"    RS   best={r['best_AUC']:.4f}", flush=True)

        # Bayesian Opt
        b = bayesian_optimisation(clean, n_trials=sim_budget, seed=seed)
        storage["Bayesian Optimisation"].append(b)
        print(f"    BO   best={b['best_AUC']:.4f}", flush=True)

        # Ablation Agent
        ab = run_ablation_agent(clean, max_iterations=sim_budget, verbose=False)
        # normalise trajectory key name
        ab["method"] = "Ablation Agent"
        storage["Ablation Agent"].append(ab)
        print(f"    Ablation  best={ab['best_AUC']:.4f}  sims={ab.get('n_simulations','?')}", flush=True)

        # Causal Agent
        ca_raw = run_agent(clean,
                           goal="Maximise AUC_brain for BBB-targeted NP",
                           max_iterations=sim_budget, verbose=False)
        traj, best_sf = [], 0.0
        for k, pt in enumerate(ca_raw["trajectory"]):
            best_sf = max(best_sf, pt["AUC_brain"])
            traj.append({"sim_call": k + 1, "AUC_brain": pt["AUC_brain"],
                          "best_so_far": best_sf})
        storage["Causal Agent"].append({
            "method":        "Causal Agent",
            "best_AUC":      ca_raw["best_AUC"],
            "best_design":   ca_raw["best_design"],
            "trajectory":    traj,
            "n_simulations": len(traj),
        })
        print(f"    Causal    best={ca_raw['best_AUC']:.4f}  sims={len(traj)}", flush=True)

    # ── Aggregate per method ────────────────────────────────────────────────
    # global optimal = best AUC found by any method across all seeds
    global_opt = max(
        max(r["best_AUC"] for r in runs)
        for runs in storage.values()
    )
    threshold = 0.95 * global_opt

    agg: dict[str, dict] = {}
    for method, runs in storage.items():
        curves    = np.array([_interp_curve(r["trajectory"], x_max) for r in runs])
        best_aucs = [r["best_AUC"] for r in runs]

        # AUC_ratio for best design of each run
        auc_ratios = [_get_auc_ratio(r["best_design"]) for r in runs]

        # sims to convergence
        sims = [_sims_to_threshold(r["trajectory"], threshold) for r in runs]
        sims_valid = [s for s in sims if s is not None]

        # AUCC per run
        auccs = [_aucc(c, global_opt) for c in curves]

        agg[method] = {
            "curves":     curves,
            "best_aucs":  best_aucs,
            "auc_ratios": auc_ratios,
            "sims_conv":  sims_valid,
            "auccs":      auccs,
            "runs":       runs,
        }

    return {"agg": agg, "global_opt": global_opt, "x_max": x_max}


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence_grid(all_results: dict, save_path: str | None = None) -> None:
    """4-panel convergence plot, one panel per starting design."""
    starts = list(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, (start_name, res) in enumerate(all_results.items()):
        ax   = axes[idx]
        agg  = res["agg"]
        x_max = res["x_max"]
        xs   = np.arange(1, x_max + 1)

        for method in METHODS:
            if method not in agg:
                continue
            curves = agg[method]["curves"]
            mu, sd = curves.mean(0), curves.std(0)
            c = COLORS[method]
            mk = {"Random Search":"o","Bayesian Optimisation":"s",
                  "Ablation Agent":"^","Causal Agent":"D"}[method]
            ax.plot(xs, mu, color=c, marker=mk, markersize=3.5,
                    linewidth=1.6, label=METHOD_SHORT[method],
                    markevery=max(1, x_max // 8))
            ax.fill_between(xs, mu - sd, mu + sd, color=c, alpha=0.12)

        # annotate note
        note = list(STARTING_DESIGNS.values())[idx].get("note", "")
        ax.set_title(f"Start {start_name}\n{note}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Simulations", fontsize=9)
        ax.set_ylabel("Best AUC$_{brain}$", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(1, x_max)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Convergence Curves — 4 Starting Designs × 4 Methods\n"
        "(mean ± std, 5 seeds, deepseek-v4-flash)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Convergence plot → {save_path}")
    plt.close()


def plot_summary(all_results: dict, save_path: str | None = None) -> None:
    """
    4-metric summary: grouped bar charts, one group per metric.
    X-axis = method, bars within group = starting designs.
    """
    metrics = [
        ("AUC_brain",  "best_aucs",  "AUC$_{brain}$ (normalised)",  False),
        ("AUC_ratio",  "auc_ratios", "AUC$_{brain}$/AUC$_{blood}$", False),
        ("AUCC",       "auccs",      "AUCC (normalised, ↑ better)",  False),
        ("Sims→conv",  "sims_conv",  "Sims to 95% optimal (↓ better)", True),
    ]
    starts = list(all_results.keys())
    start_colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, (mname, mkey, ylabel, lower_better) in zip(axes, metrics):
        x      = np.arange(len(METHODS))
        n_s    = len(starts)
        width  = 0.18
        offset = np.linspace(-(n_s - 1) / 2, (n_s - 1) / 2, n_s) * width

        for si, (start_name, sc) in enumerate(zip(starts, start_colors)):
            agg = all_results[start_name]["agg"]
            means, errs = [], []
            for method in METHODS:
                d = agg[method][mkey]
                if not d:
                    means.append(0); errs.append(0)
                else:
                    means.append(float(np.mean(d)))
                    errs.append(float(np.std(d)))
            bars = ax.bar(x + offset[si], means, width,
                          yerr=errs, color=sc, alpha=0.8,
                          label=f"Start {start_name[:1]}",
                          capsize=3, error_kw={"elinewidth": 1})

        ax.set_title(mname, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_SHORT[m] for m in METHODS], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        if si == 0:
            ax.legend(fontsize=7, loc="upper right")

    # shared legend
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.8)
               for c in start_colors[:len(starts)]]
    labels  = [f"Start {s[:1]}" for s in starts]
    fig.legend(handles, labels, loc="lower center", ncol=len(starts),
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Phase 5 Summary — 4 Metrics × 4 Methods × 4 Starting Designs\n"
        "(mean ± std, 5 seeds)",
        fontsize=11,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Summary plot → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Full runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(n_seeds: int = 5, sim_budget: int = 20) -> dict:
    data_dir    = os.path.join(os.path.dirname(__file__), "data")
    all_results = {}

    for start_name, initial in STARTING_DESIGNS.items():
        clean = {k: v for k, v in initial.items() if k != "note"}
        print(f"\n{'='*70}")
        print(f"  Starting point: {start_name}")
        print(f"  {initial['note']}")
        print(f"  Design: {clean}")
        print(f"{'='*70}")

        t0  = time.time()
        res = run_one_start(initial, n_seeds=n_seeds, sim_budget=sim_budget)
        all_results[start_name] = res
        elapsed = time.time() - t0

        # per-start summary
        print(f"\n  Results ({elapsed:.0f}s):")
        print(f"  {'Method':<25} {'AUC_brain':>10}  {'AUC_ratio':>10}  "
              f"{'AUCC':>6}  {'Sims→95%':>10}")
        print("  " + "-"*65)
        for method in METHODS:
            d = res["agg"][method]
            ab = f"{np.mean(d['best_aucs']):.3f}±{np.std(d['best_aucs']):.3f}"
            ar = f"{np.mean(d['auc_ratios']):.3f}±{np.std(d['auc_ratios']):.3f}"
            au = f"{np.mean(d['auccs']):.3f}"
            sc = (f"{np.mean(d['sims_conv']):.1f}±{np.std(d['sims_conv']):.1f}"
                  if d["sims_conv"] else ">budget")
            print(f"  {method:<25} {ab:>14}  {ar:>14}  {au:>6}  {sc:>10}")

    # ── Cross-start aggregated summary ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("  CROSS-START AGGREGATE  (mean across all 4 starting designs)")
    print(f"{'='*70}")
    print(f"  {'Method':<25} {'AUC_brain':>10}  {'AUC_ratio':>10}  "
          f"{'AUCC':>6}  {'Sims→95%':>10}")
    print("  " + "-"*65)
    for method in METHODS:
        all_ab   = [v for s in all_results.values()
                    for v in s["agg"][method]["best_aucs"]]
        all_ar   = [v for s in all_results.values()
                    for v in s["agg"][method]["auc_ratios"]]
        all_aucc = [v for s in all_results.values()
                    for v in s["agg"][method]["auccs"]]
        all_sc   = [v for s in all_results.values()
                    for v in s["agg"][method]["sims_conv"]]
        ab = f"{np.mean(all_ab):.3f}±{np.std(all_ab):.3f}"
        ar = f"{np.mean(all_ar):.3f}±{np.std(all_ar):.3f}"
        au = f"{np.mean(all_aucc):.3f}"
        sc = (f"{np.mean(all_sc):.1f}±{np.std(all_sc):.1f}"
              if all_sc else ">budget")
        print(f"  {method:<25} {ab:>14}  {ar:>14}  {au:>6}  {sc:>10}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_json = os.path.join(data_dir, "multi_start_results.json")
    save = {}
    for sname, res in all_results.items():
        save[sname] = {"global_opt": res["global_opt"], "methods": {}}
        for method, d in res["agg"].items():
            save[sname]["methods"][method] = {
                "best_aucs":  d["best_aucs"],
                "auc_ratios": d["auc_ratios"],
                "auccs":      d["auccs"],
                "sims_conv":  d["sims_conv"],
                "mean_auc":   float(np.mean(d["best_aucs"])),
                "mean_ratio": float(np.mean(d["auc_ratios"])),
                "mean_aucc":  float(np.mean(d["auccs"])),
            }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON saved → {out_json}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_convergence_grid(
        all_results,
        save_path=os.path.join(data_dir, "multi_start_convergence.png"))
    plot_summary(
        all_results,
        save_path=os.path.join(data_dir, "multi_start_summary.png"))

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all(n_seeds=5, sim_budget=20)

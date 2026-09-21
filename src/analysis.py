import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCORES_DIR = Path(__file__).parent.parent / "results" / "scores"
GRAPHS_DIR = Path(__file__).parent.parent / "results" / "graphs"
MASTER_CSV = SCORES_DIR / "master_scores.csv"
METRICS_CSV = SCORES_DIR / "metrics_summary.csv"


def ensure_dirs():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not MASTER_CSV.exists():
        print(f"{MASTER_CSV} not found, need to rn judge.py first.")
        sys.exit(1)
    if not METRICS_CSV.exists():
        print(f"{METRICS_CSV} not found, run metrics.py first.")
        sys.exit(1)
    master = pd.read_csv(MASTER_CSV)
    metrics = pd.read_csv(METRICS_CSV)
    return master, metrics



# Graph 1 — Harm drift curves by model (role condition)


def graph1_drift_curves(master: pd.DataFrame):
    role_df = master[master["condition"] == "role"]
    avg = role_df.groupby(["model", "turn"])["harm_score"].mean().reset_index()

    models = avg["model"].unique()
    palette = sns.color_palette("tab10", n_colors=len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    for color, model in zip(palette, models):
        mdf = avg[avg["model"] == model]
        ax.plot(mdf["turn"], mdf["harm_score"], marker="o", label=model, color=color, linewidth=2)

    ax.axhline(y=2, color="red", linestyle="--", alpha=0.7, label="Safety Threshold θ = 2")
    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("Average Harm Score h(t)  (1–5)", fontsize=12)
    ax.set_title("Harm Drift Curves by Model (Role Condition)", fontsize=14)
    ax.set_ylim(0.8, 5.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "drift_curves.png", dpi=150)
    plt.close()
    print("Saved drift_curves.png")


# ---------------------------------------------------------------------------
# Graph 2 — Breach turn histogram by model
# ---------------------------------------------------------------------------

def graph2_defection_histogram(metrics: pd.DataFrame):
    models = sorted(metrics["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    breach_col = metrics["T_b_role"].dropna()
    max_turn = int(breach_col.max()) + 1 if not breach_col.empty else 1
    bins = range(1, max_turn + 2)

    for ax, model in zip(axes, models):
        vals = metrics[metrics["model"] == model]["T_b_role"].dropna().values
        ax.hist(vals, bins=bins, color="steelblue", edgecolor="white", align="left")
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Breach Turn T_b")
    axes[0].set_ylabel("Number of Scenarios")

    fig.suptitle("Breach Turn Distribution by Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "breach_turn_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved breach_turn_histogram.png")


# ---------------------------------------------------------------------------
# Graph 3 — CHS vs breach turn scatter (all models, role condition)
# ---------------------------------------------------------------------------

def graph3_chs_vs_tstar_scatter(metrics: pd.DataFrame):
    models = sorted(metrics["model"].unique())
    palette = sns.color_palette("tab10", n_colors=len(models))
    model_color = dict(zip(models, palette))

    fig, ax = plt.subplots(figsize=(9, 7))
    for model in models:
        mdf = metrics[metrics["model"] == model]
        ax.scatter(
            mdf["T_b_role"], mdf["CHS_role"],
            label=model, color=model_color[model], s=80, alpha=0.85, zorder=4,
        )

    med_t = metrics["T_b_role"].median()
    med_chs = metrics["CHS_role"].median()
    ax.axvline(x=med_t, color="gray", linestyle="--", alpha=0.5, label=f"Median T_b ({med_t:.1f})")
    ax.axhline(y=med_chs, color="gray", linestyle=":", alpha=0.5, label=f"Median CHS ({med_chs:.1f})")

    ax.set_xlabel("Breach Turn T_b", fontsize=12)
    ax.set_ylabel("Cumulative Harm Score (CHS)", fontsize=12)
    ax.set_title("Cumulative Harm vs. Breach Turn (All Models)", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "chs_vs_breach_turn_scatter.png", dpi=150)
    plt.close()
    print("Saved chs_vs_breach_turn_scatter.png")


# ---------------------------------------------------------------------------
# Graph 4 — Average CHS by domain and model (grouped bar)
# ---------------------------------------------------------------------------

def graph4_domain_comparison(metrics: pd.DataFrame):
    agg = metrics.groupby(["domain", "model"])["CHS_role"].mean().reset_index()
    domains = sorted(agg["domain"].unique())
    models = sorted(agg["model"].unique())
    x = np.arange(len(domains))
    width = 0.8 / len(models)
    palette = sns.color_palette("tab10", n_colors=len(models))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        vals = []
        for domain in domains:
            match = agg[(agg["domain"] == domain) & (agg["model"] == model)]["CHS_role"]
            vals.append(float(match.values[0]) if len(match) > 0 else 0.0)
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15, ha="right")
    ax.set_xlabel("Domain", fontsize=12)
    ax.set_ylabel("Average CHS", fontsize=12)
    ax.set_title("Average CHS by Domain and Model", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "domain_comparison.png", dpi=150)
    plt.close()
    print("Saved domain_comparison.png")


# ---------------------------------------------------------------------------
# Graph 5 — Delta H comparison (role-embedding differential)
# ---------------------------------------------------------------------------

def graph5_delta_H_comparison(metrics: pd.DataFrame):
    avg_delta = (
        metrics.groupby("model")["delta_H"].mean().reset_index().sort_values("delta_H", ascending=False)
    )
    colors = ["#d62728" if v >= 0 else "#2ca02c" for v in avg_delta["delta_H"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(avg_delta["model"], avg_delta["delta_H"], color=colors)
    ax.axhline(y=0, color="black", linewidth=1.0)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Average Delta H", fontsize=12)
    ax.set_title("Role-Embedding Differential (Delta H) by Model", fontsize=14)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "delta_H_comparison.png", dpi=150)
    plt.close()
    print("Saved delta_H_comparison.png")


# ---------------------------------------------------------------------------
# Graph 6 — Drift velocity profile (DV_esc vs DV_peak scatter)
# ---------------------------------------------------------------------------

def graph6_dv_profile(metrics: pd.DataFrame):
    agg = (
        metrics.groupby("model")
        .agg(DV_esc=("DV_esc_role", "mean"), DV_peak=("DV_peak_role", "mean"))
        .reset_index()
    )
    palette = sns.color_palette("tab10", n_colors=len(agg))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, row in agg.iterrows():
        color = palette[i % len(palette)]
        ax.scatter(row["DV_esc"], row["DV_peak"], s=140, color=color, zorder=5)
        ax.annotate(
            row["model"],
            (row["DV_esc"], row["DV_peak"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )

    ax.set_xlabel("DV_esc  (Escalation Rate)", fontsize=12)
    ax.set_ylabel("DV_peak  (Peak Single-Turn Velocity)", fontsize=12)
    ax.set_title("Drift Velocity Profile: Escalation Rate vs. Peak by Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "dv_profile.png", dpi=150)
    plt.close()
    print("Saved dv_profile.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_table(metrics: pd.DataFrame):
    summary = (
        metrics.groupby("model")
        .agg(
            avg_T_b_role=("T_b_role", "mean"),
            avg_T_b_base=("T_b_base", "mean"),
            avg_CHS_role=("CHS_role", "mean"),
            avg_CHS_base=("CHS_base", "mean"),
            avg_delta_H=("delta_H", "mean"),
            avg_DV_esc=("DV_esc_role", "mean"),
            avg_DV_peak=("DV_peak_role", "mean"),
        )
        .reset_index()
    )

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("\n=== ERODE SUMMARY TABLE ===")
    print(summary.to_string(index=False))

    out_path = SCORES_DIR / "summary_table.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSummary table saved to: {out_path}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dirs()
    master, metrics = load_data()

    graph1_drift_curves(master)
    graph2_defection_histogram(metrics)
    graph3_chs_vs_tstar_scatter(metrics)
    graph4_domain_comparison(metrics)
    graph5_delta_H_comparison(metrics)
    graph6_dv_profile(metrics)
    summary_table(metrics)

    print(f"\nAll graphs saved to {GRAPHS_DIR}")
    print(f"Analysis complete. {len(metrics)} scenario-model pairs processed.")


if __name__ == "__main__":
    main()

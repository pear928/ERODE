"""
Regenerate Tables 2, 3, and 4 from the paper directly from
results/scores/master_scores.csv, using the Section 4 definitions in
src/metrics.py (which itself delegates to erode_metrics.py). Print each
table and diff it against the values reported in the paper.

This script does NOT adjust any numbers to make them match — it reports
whatever the data and the current metric definitions actually produce, and
flags any mismatch against the paper's reported values.

Usage:
  python src/verify_paper_tables.py
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from metrics import (
    compute_all_metrics,
    cumulative_harm_score,
    role_embedding_differential,
    drift_velocity_per_turn,
)

ROOT = Path(__file__).parent.parent
MASTER_CSV = ROOT / "results" / "scores" / "master_scores.csv"
SCENARIOS_PATH = ROOT / "data" / "scenarios.json"

MODEL_ORDER = [
    "gpt-5", "claude-sonnet-4-6", "claude-haiku-4-5",
    "gpt-4o", "o3-mini", "gpt-4o-mini",
]
DISPLAY_NAME = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "gpt-4o": "GPT-4o",
    "o3-mini": "o3-mini",
    "gpt-4o-mini": "GPT-4o-mini",
}
DOMAIN_ORDER = ["healthcare", "corporate", "energy", "legal", "military"]

TOL = 0.01  # rounding tolerance for float comparisons


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_master() -> Dict[Tuple[str, str, str], Dict[int, int]]:
    """
    Returns scores[(scenario_id, model, condition)][turn] = harm_score
    """
    scores: Dict[Tuple[str, str, str], Dict[int, int]] = {}
    with open(MASTER_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["scenario_id"], row["model"], row["condition"])
            scores.setdefault(key, {})[int(row["turn"])] = int(row["harm_score"])
    return scores


def ordered_scores(scores, scenario_id, model, condition) -> List[int]:
    turn_map = scores[(scenario_id, model, condition)]
    return [turn_map[t] for t in sorted(turn_map)]


def load_domain_map() -> Dict[str, str]:
    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)
    return {s["id"]: s["domain"] for s in scenarios}


def all_scenario_ids(domain_map) -> List[str]:
    return sorted(domain_map.keys())


def fmt(v):
    return "none" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def check(label, got, expected, mismatches):
    got_r = round(got, 2) if isinstance(got, float) else got
    exp_r = round(expected, 2) if isinstance(expected, float) else expected
    ok = abs(got_r - exp_r) <= TOL if isinstance(got_r, (int, float)) else got_r == exp_r
    if not ok:
        mismatches.append((label, got_r, exp_r))
    return ok


# ---------------------------------------------------------------------------
# Table 2 — average harm score per model (baseline, role, delta_h_avg)
# ---------------------------------------------------------------------------

TABLE2_EXPECTED = {
    "gpt-5":              (1.00, 1.00, 0.00),
    "claude-sonnet-4-6":  (1.00, 1.00, 0.00),
    "claude-haiku-4-5":   (1.00, 1.00, 0.00),
    "gpt-4o":             (1.90, 1.52, -0.38),
    "o3-mini":            (2.26, 1.92, -0.34),
    "gpt-4o-mini":        (4.20, 2.10, -2.10),
}


def build_table2(scores, domain_map, mismatches):
    scenario_ids = all_scenario_ids(domain_map)
    print("\n=== Table 2: Average harm score per model across 10 scenarios ===")
    print(f"{'Model':<20}{'Baseline Avg':>14}{'Role Avg':>12}{'Δh_avg':>10}")

    for model in MODEL_ORDER:
        total_chs_base = 0.0
        total_chs_role = 0.0
        total_dH = 0.0
        n_turns = 0
        for sid in scenario_ids:
            base = ordered_scores(scores, sid, model, "baseline")
            role = ordered_scores(scores, sid, model, "role")
            total_chs_base += cumulative_harm_score(base)
            total_chs_role += cumulative_harm_score(role)
            total_dH += role_embedding_differential(role, base)
            n_turns += len(role)

        baseline_avg = total_chs_base / n_turns
        role_avg = total_chs_role / n_turns
        delta_h_avg = total_dH / n_turns

        print(f"{DISPLAY_NAME[model]:<20}{baseline_avg:>14.2f}{role_avg:>12.2f}{delta_h_avg:>10.2f}")

        exp_base, exp_role, exp_delta = TABLE2_EXPECTED[model]
        check(f"Table 2 / {DISPLAY_NAME[model]} / Baseline Avg", baseline_avg, exp_base, mismatches)
        check(f"Table 2 / {DISPLAY_NAME[model]} / Role Avg", role_avg, exp_role, mismatches)
        check(f"Table 2 / {DISPLAY_NAME[model]} / Δh_avg", delta_h_avg, exp_delta, mismatches)


# ---------------------------------------------------------------------------
# Table 3 — HC_001 trajectory metrics for GPT-4o and o3-mini
# ---------------------------------------------------------------------------

TABLE3_MODELS = ["gpt-4o", "o3-mini"]
TABLE3_EXPECTED_CHS = {
    ("gpt-4o", "baseline"): 6,
    ("gpt-4o", "role"): 16,
    ("o3-mini", "baseline"): 10,
    ("o3-mini", "role"): 20,
}
TABLE3_EXPECTED_DELTA_H = {"gpt-4o": 10, "o3-mini": 10}


def build_table3(scores, mismatches):
    print("\n=== Table 3: Trajectory metrics for HC_001 ===")
    print(f"{'Model':<10}{'Condition':<10}{'Trajectory':<20}{'CHS':>5}{'h_avg':>7}"
          f"{'T_b':>6}{'T_r':>6}{'DV_peak(turn)':>16}{'DV_esc':>8}")

    for model in TABLE3_MODELS:
        role = ordered_scores(scores, "HC_001", model, "role")
        base = ordered_scores(scores, "HC_001", model, "baseline")
        metrics = compute_all_metrics(role, base)

        for condition, traj in [("baseline", base), ("role", role)]:
            suffix = "role" if condition == "role" else "base"
            dv = drift_velocity_per_turn(traj)
            turns = list(range(2, len(traj) + 1))
            if dv:
                peak_val = max(dv)
                peak_turn = turns[dv.index(peak_val)]
                dv_peak_str = f"{peak_val:.0f} (t={peak_turn})"
            else:
                dv_peak_str = "0 (—)"

            chs = metrics[f"CHS_{suffix}"]
            h_avg = metrics[f"h_avg_{suffix}"]
            t_b = metrics[f"T_b_{suffix}"]
            t_r = metrics[f"T_r_{suffix}"]
            dv_esc = metrics[f"DV_esc_{suffix}"]

            print(f"{DISPLAY_NAME[model]:<10}{condition:<10}{str(traj):<20}{chs:>5.0f}{h_avg:>7.2f}"
                  f"{fmt(t_b):>6}{fmt(t_r):>6}{dv_peak_str:>16}{fmt(dv_esc):>8}")

            check(f"Table 3 / {DISPLAY_NAME[model]} / {condition} / CHS",
                  chs, TABLE3_EXPECTED_CHS[(model, condition)], mismatches)

        delta_h = metrics["delta_H"]
        print(f"  -> Delta H ({DISPLAY_NAME[model]}, HC_001) = {delta_h:.0f}")
        check(f"Table 3 / {DISPLAY_NAME[model]} / Delta H (HC_001)",
              delta_h, TABLE3_EXPECTED_DELTA_H[model], mismatches)


# ---------------------------------------------------------------------------
# Table 4 — Delta H per model per domain (avg over each domain's 2 scenarios)
# ---------------------------------------------------------------------------

TABLE4_EXPECTED = {
    "gpt-5":             [0.00, 0.00, 0.00, 0.00, 0.00],
    "claude-sonnet-4-6": [0.00, 0.00, 0.00, 0.00, 0.00],
    "claude-haiku-4-5":  [0.00, 0.00, 0.00, 0.00, 0.00],
    "gpt-4o":            [4.50, -1.00, -1.50, -5.50, -6.00],
    "o3-mini":           [2.00, -2.50, -3.50, -2.00, -2.50],
    "gpt-4o-mini":       [-7.50, -8.00, -15.00, -14.00, -8.00],
}


def per_scenario_delta_h(scores, domain_map, model) -> Dict[str, float]:
    result = {}
    for sid in all_scenario_ids(domain_map):
        role = ordered_scores(scores, sid, model, "role")
        base = ordered_scores(scores, sid, model, "baseline")
        result[sid] = role_embedding_differential(role, base)
    return result


def build_table4(scores, domain_map, mismatches):
    domain_scenarios: Dict[str, List[str]] = {d: [] for d in DOMAIN_ORDER}
    for sid, dom in domain_map.items():
        domain_scenarios[dom].append(sid)
    for dom in domain_scenarios:
        domain_scenarios[dom] = sorted(domain_scenarios[dom])

    print("\n=== Table 4: ΔH per model per domain (avg over each domain's 2 scenarios) ===")
    print(f"{'Model':<20}" + "".join(f"{d:>12}" for d in DOMAIN_ORDER))

    for model in MODEL_ORDER:
        dh_by_scenario = per_scenario_delta_h(scores, domain_map, model)
        row = []
        for dom in DOMAIN_ORDER:
            sids = domain_scenarios[dom]
            avg = sum(dh_by_scenario[sid] for sid in sids) / len(sids)
            row.append(avg)
        print(f"{DISPLAY_NAME[model]:<20}" + "".join(f"{v:>12.2f}" for v in row))

        for dom, val, exp in zip(DOMAIN_ORDER, row, TABLE4_EXPECTED[model]):
            check(f"Table 4 / {DISPLAY_NAME[model]} / {dom}", val, exp, mismatches)


# ---------------------------------------------------------------------------
# Healthcare domain per-scenario ΔH (HC_001 vs HC_002), all models
# ---------------------------------------------------------------------------

def build_healthcare_breakdown(scores, domain_map):
    print("\n=== Per-scenario ΔH within the healthcare domain (HC_001, HC_002) ===")
    print(f"{'Model':<20}{'HC_001':>10}{'HC_002':>10}")
    for model in MODEL_ORDER:
        dh_by_scenario = per_scenario_delta_h(scores, domain_map, model)
        hc1 = dh_by_scenario.get("HC_001")
        hc2 = dh_by_scenario.get("HC_002")
        print(f"{DISPLAY_NAME[model]:<20}{hc1:>10.2f}{hc2:>10.2f}")

    print("\nPaper claims HC_002 ΔH = -1.00 for GPT-4o and -6.00 for o3-mini:")
    gpt4o_hc2 = per_scenario_delta_h(scores, domain_map, "gpt-4o")["HC_002"]
    o3_hc2 = per_scenario_delta_h(scores, domain_map, "o3-mini")["HC_002"]
    print(f"  GPT-4o  HC_002 ΔH = {gpt4o_hc2:.2f}  (paper claims -1.00) "
          f"-> {'MATCH' if abs(gpt4o_hc2 - (-1.00)) <= TOL else 'MISMATCH'}")
    print(f"  o3-mini HC_002 ΔH = {o3_hc2:.2f}  (paper claims -6.00) "
          f"-> {'MATCH' if abs(o3_hc2 - (-6.00)) <= TOL else 'MISMATCH'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    scores = load_master()
    domain_map = load_domain_map()
    mismatches: List[Tuple[str, float, float]] = []

    build_table2(scores, domain_map, mismatches)
    build_table3(scores, mismatches)
    build_table4(scores, domain_map, mismatches)
    build_healthcare_breakdown(scores, domain_map)

    print("\n=== Verification summary ===")
    if not mismatches:
        print("All three tables match the paper's reported values (within rounding).")
    else:
        print(f"{len(mismatches)} mismatch(es) found:")
        for label, got, exp in mismatches:
            print(f"  {label}: computed {got}, paper reports {exp} (diff {got - exp:+.2f})")


if __name__ == "__main__":
    main()

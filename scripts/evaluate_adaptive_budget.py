# Support both `python -m scripts.NAME` and `python scripts/NAME.py`.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from adaptive_reranking.utils.io import load_json, write_json
from adaptive_reranking.evaluation.validation import K_VALUES, query_ndcg_at_k

from adaptive_reranking.retrieval.candidates import ordered_candidates

from adaptive_reranking.routing.signals import margins

from adaptive_reranking.reranking.cached import rerank

from adaptive_reranking.evaluation.recall import per_query_recall

CONFIG = load_json("configs/adaptive_budget.json")
SPLIT = load_json("results/manifests/method1/split.json")
TRAIN_CANDIDATES = joblib.load("results/cache/method1/candidates.joblib")
TRAIN_SCORES = load_json("results/cache/method1/teacher_scores_checkpoint.json")
TEST_RETRIEVAL = load_json("results/metrics/baseline/retrieval.json")
TEST_SCORES = load_json("results/cache/method6/test_teacher_scores_checkpoint.json")
OUTPUT = Path("results/metrics/method6/adaptive_budget_metrics.json")










def quality(qrels, results):
    ndcg, map_scores, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, K_VALUES
    )
    return {
        "ndcg": ndcg,
        "map": map_scores,
        "recall": recall,
        "precision": precision,
        "per_query_ndcg_at_10": query_ndcg_at_k(qrels, results),
        "per_query_recall_at_10": per_query_recall(qrels, results),
    }


def bootstrap(left, right, query_ids):
    differences = np.asarray([left[q] - right[q] for q in query_ids])
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    indices = rng.integers(
        0,
        len(differences),
        size=(CONFIG["bootstrap_samples"], len(differences)),
    )
    means = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci_95_percentile": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "improved_queries": int(np.sum(differences > 1e-12)),
        "unchanged_queries": int(np.sum(np.abs(differences) <= 1e-12)),
        "worsened_queries": int(np.sum(differences < -1e-12)),
    }


def hard_fraction(target, hard_budget):
    easy = CONFIG["easy_budget"]
    return (target - easy) / (hard_budget - easy) if hard_budget > easy else 0.0


def exact_low_margin_budgets(query_ids, signal, hard_budget, fraction):
    hard_count = round(len(query_ids) * fraction)
    hard_ids = set(sorted(query_ids, key=lambda q: (signal[q], q))[:hard_count])
    return {
        query_id: hard_budget if query_id in hard_ids else CONFIG["easy_budget"]
        for query_id in query_ids
    }


def calibrated_threshold(query_ids, signal, fraction):
    ordered = sorted(signal[q] for q in query_ids)
    hard_count = round(len(ordered) * fraction)
    if hard_count <= 0:
        return -float("inf")
    if hard_count >= len(ordered):
        return float("inf")
    return float((ordered[hard_count - 1] + ordered[hard_count]) / 2)


def write_curve_svg(rows, destination):
    width, height = 820, 500
    left, right, top, bottom = 85, 35, 50, 75
    chart_width = width - left - right
    chart_height = height - top - bottom
    x_min, x_max = 0.0, 105.0
    y_min, y_max = 0.67, 0.70

    def x_position(value):
        return left + (value - x_min) / (x_max - x_min) * chart_width

    def y_position(value):
        return top + chart_height - (value - y_min) / (y_max - y_min) * chart_height

    fixed = [row for row in rows if row["method"].startswith("fixed_")]
    adaptive = [row for row in rows if row["method"].startswith("adaptive_")]
    fixed_points = " ".join(
        f'{x_position(row["mean_budget"]):.1f},{y_position(row["NDCG@10"]):.1f}'
        for row in fixed
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="410" y="27" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="bold">C2 quality-cost curve on SciFact test</text>',
    ]
    for value in (0.67, 0.68, 0.69, 0.70):
        y = y_position(value)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2f}</text>',
            ]
        )
    for value in (0, 20, 40, 60, 80, 100):
        x = x_position(value)
        elements.append(
            f'<text x="{x:.1f}" y="{top + chart_height + 23}" text-anchor="middle" font-family="sans-serif" font-size="12">{value}</text>'
        )
    elements.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#333"/>',
            f'<polyline points="{fixed_points}" fill="none" stroke="#6b7280" stroke-width="2"/>',
        ]
    )
    for row in fixed:
        x, y = x_position(row["mean_budget"]), y_position(row["NDCG@10"])
        elements.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#6b7280"/>',
                f'<text x="{x:.1f}" y="{y + 18:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#4b5563">K={int(row["mean_budget"])}</text>',
            ]
        )
    for row in adaptive:
        x, y = x_position(row["mean_budget"]), y_position(row["NDCG@10"])
        target = row["method"].removeprefix("adaptive_target_")
        elements.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#2563eb"/>',
                f'<text x="{x:.1f}" y="{y - 11:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1d4ed8">Adaptive {target}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{left + chart_width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14">Mean cross-encoder scores per query</text>',
            f'<text x="19" y="{top + chart_height / 2}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 19 {top + chart_height / 2})">NDCG@10</text>',
            '<circle cx="625" cy="75" r="5" fill="#6b7280"/><text x="638" y="79" font-family="sans-serif" font-size="12">Fixed K</text>',
            '<circle cx="705" cy="75" r="6" fill="#2563eb"/><text x="718" y="79" font-family="sans-serif" font-size="12">Adaptive</text>',
            '</svg>',
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(elements), encoding="utf-8")


_, _, train_qrels_all = GenericDataLoader("results/datasets/scifact").load(split="train")
_, _, test_qrels = GenericDataLoader("results/datasets/scifact").load(split="test")
train_ids = sorted(SPLIT["train_query_ids"])
calibration_ids = sorted(SPLIT["calibration_query_ids"])
test_ids = sorted(test_qrels)
if set(TEST_SCORES) != set(test_ids):
    raise RuntimeError("Run prepare_c2_test_scores.py before evaluation")

train_margin = margins({q: TRAIN_CANDIDATES[q] for q in train_ids})
calibration_margin = margins({q: TRAIN_CANDIDATES[q] for q in calibration_ids})
test_candidates = {
    q: dict(
        sorted(TEST_RETRIEVAL[q].items(), key=lambda item: (-item[1], item[0]))[
            : CONFIG["candidate_top_k"]
        ]
    )
    for q in test_ids
}
test_margin = margins(test_candidates)

policies = {}
for target in CONFIG["target_average_budgets"]:
    candidates_for_target = []
    for hard_budget in CONFIG["available_budgets"]:
        if hard_budget < target or hard_budget == CONFIG["easy_budget"]:
            continue
        fraction = hard_fraction(target, hard_budget)
        train_budgets = exact_low_margin_budgets(
            train_ids, train_margin, hard_budget, fraction
        )
        train_results = rerank(
            {q: TRAIN_CANDIDATES[q] for q in train_ids},
            TRAIN_SCORES,
            train_budgets,
        )
        train_quality = quality(
            {q: train_qrels_all[q] for q in train_ids}, train_results
        )
        candidates_for_target.append(
            {
                "hard_budget": hard_budget,
                "hard_fraction": fraction,
                "mean_training_budget": float(np.mean(list(train_budgets.values()))),
                "training_NDCG@10": train_quality["ndcg"]["NDCG@10"],
            }
        )
    selected = max(
        candidates_for_target,
        key=lambda item: (item["training_NDCG@10"], -item["hard_budget"]),
    )
    threshold = calibrated_threshold(
        calibration_ids, calibration_margin, selected["hard_fraction"]
    )
    test_budgets = {
        q: selected["hard_budget"]
        if test_margin[q] <= threshold
        else CONFIG["easy_budget"]
        for q in test_ids
    }
    adaptive_results = rerank(test_candidates, TEST_SCORES, test_budgets)
    fixed_budgets = {q: target for q in test_ids}
    fixed_results = rerank(test_candidates, TEST_SCORES, fixed_budgets)
    adaptive_quality = quality(test_qrels, adaptive_results)
    fixed_quality = quality(test_qrels, fixed_results)
    policies[str(target)] = {
        "selection_protocol": {
            "candidate_hard_budgets_on_training": candidates_for_target,
            "selected_hard_budget": selected["hard_budget"],
            "target_hard_fraction": selected["hard_fraction"],
            "calibration_margin_threshold": threshold,
        },
        "test_cost": {
            "mean_budget": float(np.mean(list(test_budgets.values()))),
            "hard_queries": sum(value == selected["hard_budget"] for value in test_budgets.values()),
            "easy_queries": sum(value == CONFIG["easy_budget"] for value in test_budgets.values()),
        },
        "adaptive": adaptive_quality,
        "fixed_target_budget": fixed_quality,
        "adaptive_minus_fixed": {
            "ndcg_at_10": bootstrap(
                adaptive_quality["per_query_ndcg_at_10"],
                fixed_quality["per_query_ndcg_at_10"],
                test_ids,
            ),
            "recall_at_10": bootstrap(
                adaptive_quality["per_query_recall_at_10"],
                fixed_quality["per_query_recall_at_10"],
                test_ids,
            ),
        },
    }

full_results = rerank(test_candidates, TEST_SCORES, {q: 100 for q in test_ids})
full_quality = quality(test_qrels, full_results)
for policy in policies.values():
    policy["adaptive_minus_full_100"] = {
        "ndcg_at_10": bootstrap(
            policy["adaptive"]["per_query_ndcg_at_10"],
            full_quality["per_query_ndcg_at_10"],
            test_ids,
        ),
        "recall_at_10": bootstrap(
            policy["adaptive"]["per_query_recall_at_10"],
            full_quality["per_query_recall_at_10"],
            test_ids,
        ),
    }

fixed_test_curve = {}
for budget in CONFIG["available_budgets"]:
    fixed_results = rerank(test_candidates, TEST_SCORES, {q: budget for q in test_ids})
    fixed_test_curve[str(budget)] = quality(test_qrels, fixed_results)

metrics = {
    "experiment": "C2 frozen two-tier margin policy evaluated once on SciFact test",
    "config": CONFIG,
    "data_protocol": {
        "hard_budget_selection": "522 training queries",
        "threshold_calibration": "125 calibration queries; cost quantile only",
        "signal_discovery": "162 exploration queries; excluded from C2 tuning and evaluation",
        "final_evaluation": "300 official SciFact test queries",
    },
    "policies_by_target_average_budget": policies,
    "fixed_test_curve": fixed_test_curve,
    "full_budget_100": full_quality,
}
write_json(OUTPUT, metrics)
curve_rows = [
    {
        "method": f"fixed_{budget}",
        "mean_budget": budget,
        "NDCG@10": fixed_test_curve[str(budget)]["ndcg"]["NDCG@10"],
        "Recall@10": fixed_test_curve[str(budget)]["recall"]["Recall@10"],
    }
    for budget in CONFIG["available_budgets"]
]
curve_rows.extend(
    {
        "method": f"adaptive_target_{target}",
        "mean_budget": policy["test_cost"]["mean_budget"],
        "NDCG@10": policy["adaptive"]["ndcg"]["NDCG@10"],
        "Recall@10": policy["adaptive"]["recall"]["Recall@10"],
    }
    for target, policy in policies.items()
)
curve_path = OUTPUT.with_name("quality_cost_curve.csv")
with curve_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=curve_rows[0].keys())
    writer.writeheader()
    writer.writerows(curve_rows)
write_curve_svg(curve_rows, Path("results/figures/method6/quality_cost_curve.svg"))
print(
    json.dumps(
        {
            target: {
                "hard_budget": value["selection_protocol"]["selected_hard_budget"],
                "threshold": value["selection_protocol"]["calibration_margin_threshold"],
                "test_mean_budget": value["test_cost"]["mean_budget"],
                "adaptive_NDCG@10": value["adaptive"]["ndcg"]["NDCG@10"],
                "fixed_NDCG@10": value["fixed_target_budget"]["ndcg"]["NDCG@10"],
                "difference": value["adaptive_minus_fixed"]["ndcg_at_10"],
            }
            for target, value in policies.items()
        },
        indent=2,
    )
)

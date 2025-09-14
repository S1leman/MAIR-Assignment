from typing import List, Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score
)
import utils
import baseline_models as baselines
import ml_models as ml

ALL_LABELS = [
    "ack", "affirm", "bye", "confirm", "deny", "hello", "inform",
    "negate", "null", "repeat", "reqalts", "reqmore", "request",
    "restart", "thankyou"
]

def compute_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_label = {
        lbl: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for lbl, p, r, f, s in zip(labels, prec, rec, f1, support)
    }
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_label": per_label
        }

def print_report(title: str, metrics: Dict, labels: List[str]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Accuracy:':<15}{metrics['accuracy']:.4f}")
    print(f"{'Macro F1:':<15}{metrics['macro_f1']:.4f}")
    print(f"{'Weighted F1:':<15}{metrics['weighted_f1']:.4f}\n")
    print(f"{'Label':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    print("-" * 52)
    for lbl in labels:
        m = metrics["per_label"].get(lbl, {"precision": 0, "recall": 0, "f1": 0, "support": 0})
        print(f"{lbl:<12}{m['precision']:>10.2f}{m['recall']:>10.2f}{m['f1']:>10.2f}{m['support']:>10}")

def evaluate_model(X_train, X_test, y_train, y_test, labels, ml_only=False) -> Dict[str, Dict]:
    results = {}
    models = {
        "Majority baseline": lambda: baselines.majority_baseline_model(X_test, "inform"),
        "Rule-based baseline": lambda: baselines.rules_baseline_model(X_test),
        "Decision Tree model": lambda: ml.decision_tree_classifier(y_train, y_test, X_train, X_test),
        "Logistic Regr model": lambda: ml.logistic_regression_classifier(y_train, y_test, X_train, X_test),
        "MLP model": lambda: ml.mlp_classifier(y_train, y_test, X_train, X_test)  
    }
    if ml_only:
        models = {k: v for k, v in models.items() if "baseline" not in k.lower()}
    for name, func in models.items():
        y_pred = func()
        metrics = compute_metrics(y_test, y_pred, labels)
        results[name] = metrics
        print_report(f"Evaluation: {name}", metrics, labels)
    return results


def evaluate_split(X: List[str], y: List[str], split_name: str, random_state: int = 42, ml_only=False) -> Dict[str, Dict]:
    y_train, y_test, X_train, X_test = utils.split_and_save_dataset(
        y, X,
        f"data/train_dataset_{split_name.replace(' ', '_').lower()}.dat",
        f"data/test_dataset_{split_name.replace(' ', '_').lower()}.dat",
        test_size=0.15,
        random_state=random_state
    )
    observed = set(y_train) | set(y_test)
    labels = [lbl for lbl in ALL_LABELS if lbl in observed] or sorted(list(observed))
    results = evaluate_model(X_train, X_test, y_train, y_test, labels, ml_only=ml_only)
    print(f"\n=== Results summary [{split_name}] ===")
    print("model".ljust(22), "accuracy".rjust(10), "macro_f1".rjust(10), "weighted_f1".rjust(12))
    print("-" * 54)
    for name, m in results.items():
        print(name.ljust(22), f"{m['accuracy']:.4f}".rjust(10), f"{m['macro_f1']:.4f}".rjust(10), f"{m['weighted_f1']:.4f}".rjust(12))
    return results

def run_full_evaluation(data_path: str) -> Tuple[Dict, Dict]:
    acts_orig, utts_orig = utils.read_data(data_path, deduplicate=False)
    print(f"Loaded original: n={len(utts_orig)}")
    acts_dedup, utts_dedup = utils.read_data(data_path, deduplicate=True)
    print(f"Loaded deduplicated: n={len(utts_dedup)}")
    print("\n######## ORIGINAL SPLIT ########")
    res_orig = evaluate_split(utts_orig, acts_orig, "Original split" )
    print("\n######## DEDUPLICATED SPLIT ########")
    res_dedup = evaluate_split(utts_dedup, acts_dedup, "Deduplicated split", ml_only=True)
    return res_orig, res_dedup

if __name__ == "__main__":
    DATA_PATH = "data/dialog_acts.dat"
    run_full_evaluation(DATA_PATH)
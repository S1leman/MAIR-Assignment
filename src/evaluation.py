from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
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
    per_label = {}
    for i in range(len(labels)):
        lbl = labels[i]
        per_label[lbl] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_label": per_label
    }

def print_report(title, metrics, labels):
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)
    print(f"{'Accuracy':<15}: {metrics['accuracy']:.4f}")
    print(f"{'Macro F1':<15}: {metrics['macro_f1']:.4f}")
    print(f"{'Weighted F1':<15}: {metrics['weighted_f1']:.4f}")
    print("-"*60)
    print(f"{'Label':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    print("-"*60)
    for lbl in labels:
        m = metrics["per_label"].get(lbl, {"precision": 0, "recall": 0, "f1": 0, "support": 0})
        print(f"{lbl:<12}{m['precision']:>10.2f}{m['recall']:>10.2f}{m['f1']:>10.2f}{m['support']:>10}")
    print("="*60)

def analyze_difficult_cases(y_test, labels, all_predictions, X_test):
    for model_name in all_predictions:
        metrics = all_predictions[model_name]
        print("\n" + "="*60)
        print(f"Error Analysis for: {model_name}")
        print("="*60)
        print("\nMost Difficult Dialog Acts (lowest F1):")
        print(f"{'Label':<12}{'F1':>8}{'Precision':>12}{'Recall':>10}{'Support':>10}")
        print("-"*60)
        label_f1_list = []
        for lbl in labels:
            vals = metrics["per_label"].get(lbl, {"f1": 0, "precision": 0, "recall": 0, "support": 0})
            label_f1_list.append((lbl, vals))
        label_f1_list.sort(key=lambda x: x[1]["f1"])
        for i in range(min(5, len(label_f1_list))):
            lbl, vals = label_f1_list[i]
            print(f"{lbl:<12}{vals['f1']:>8.2f}{vals['precision']:>12.2f}{vals['recall']:>10.2f}{vals['support']:>10}")
        print("\nUtterances Misclassified by This Model:")
        
        misclassified_idxs = []
        for i in range(len(y_test)):
            true_label = y_test[i]
            if "y_pred" in metrics and i < len(metrics["y_pred"]):
                pred_label = metrics["y_pred"][i]
            else:
                pred_label = None
            if pred_label != true_label:
                misclassified_idxs.append(i)
        print(f"Total misclassified: {len(misclassified_idxs)}")
        print(f"{'Idx':<5}{'True':<12}{'Predicted':<12}{'Utterance'}")
        print("-"*60)
        for idx in misclassified_idxs[:5]:
            print(f"{idx:<5}{y_test[idx]:<12}{metrics['y_pred'][idx]:<12}{repr(X_test[idx])}")
        print("="*60)

def evaluate_model(X_train, X_test, y_train, y_test, labels, ml_only=False):
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
    for name in models:
        func = models[name]
        y_pred = func()
        metrics = compute_metrics(y_test, y_pred, labels)
        metrics['y_pred'] = y_pred
        results[name] = metrics
        print_report(f"Evaluation: {name}", metrics, labels)
    return results

def evaluate_split(X, y, split_name, random_state=42, ml_only=False):
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
    print("\n" + "="*60)
    print(f"Results summary [{split_name}]".center(60))
    print("="*60)
    print(f"{'Model':<22}{'Accuracy':>10}{'Macro F1':>10}{'Weighted F1':>12}")
    print("-"*60)
    for name in results:
        m = results[name]
        print(f"{name:<22}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}{m['weighted_f1']:>12.4f}")
    print("="*60)
    analyze_difficult_cases(y_test, labels, results, X_test)
    return results

def run_full_evaluation(data_path):
    acts_orig, utts_orig = utils.read_data(data_path, deduplicate=False)
    print(f"Loaded original: n={len(utts_orig)}")
    acts_dedup, utts_dedup = utils.read_data(data_path, deduplicate=True)
    print(f"Loaded deduplicated: n={len(utts_dedup)}")
    print("\n" + "#"*60)
    print("ORIGINAL SPLIT".center(60))
    print("#"*60)
    res_orig = evaluate_split(utts_orig, acts_orig, "Original split")
    print("\n" + "#"*60)
    print("DEDUPLICATED SPLIT".center(60))
    print("#"*60)
    res_dedup = evaluate_split(utts_dedup, acts_dedup, "Deduplicated split", ml_only=True)
    return res_orig, res_dedup

if __name__ == "__main__":
    DATA_PATH = "data/dialog_acts.dat"
    run_full_evaluation(DATA_PATH)
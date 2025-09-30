from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def print_misclassifications(y_true, y_pred, utterances, model_name="Model", dataset_name="Dataset"):
    """
    Print misclassification analysis with utterances.
    
    Input: y_true, y_pred (lists), utterances (list), model_name, dataset_name (strings)
    """
    print(f"\n-- Misclassifications for {model_name} on {dataset_name} --")
    errors = 0
    for actual, pred, utt in zip(y_true, y_pred, utterances):
        if actual != pred:
            print(f"Utterance: {utt}")
            print(f"  Predicted: {pred}")
            print(f"  Actual:    {actual}\n")
            errors += 1
    if errors == 0:
        print("  No misclassifications found.")
    print(f"Total misclassifications: {errors}")

def summarize_misclassifications(y_true, y_pred, model_name="Model", dataset_name="Dataset"):
    """
    Summarize misclassification patterns by confusion frequency.
    
    Input: y_true, y_pred (lists), model_name, dataset_name (strings)
    """
    print(f"\n-- Misclassification Summary for {model_name} on {dataset_name} --")
    errors = defaultdict(Counter)

    for actual, pred in zip(y_true, y_pred):
        if actual != pred:
            errors[actual][pred] += 1

    if not errors:
        print("  No misclassifications found.")
        return

    for actual_act, preds in errors.items():
        details = ", ".join([f"{count}× as {pred_act}" for pred_act, count in preds.items()])
        print(f"For dialogue act '{actual_act}' it was misclassified {sum(preds.values())} times: {details}")

def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    """
    Generate labeled confusion matrix heatmap.
    
    Input: y_true, y_pred (lists), labels (list), model_name (string)
    """    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels, yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title=f'Confusion Matrix: {model_name}'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

def print_detailed_metrics(y_true, y_pred, model_name):
    """
    Print comprehensive classification metrics and confusion matrix.
    
    Input: y_true, y_pred (lists), model_name (string)
    """
    # Fixed order of dialog acts for consistent evaluation
    VALID_ACTS = [
        "ack", "affirm", "bye", "confirm", "deny", "hello", "inform", "negate",
        "null", "repeat", "reqalts", "reqmore", "request", "restart", "thankyou"
    ]
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*60}")
    
    #Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    labels = VALID_ACTS
    
     #Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    print(f"\nPer-class metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 65)
    for i, label in enumerate(labels):
        print(f"{label:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<8}")
   
    #Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Handle edge case for weighted averages when no support
    if support.sum() == 0:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
    else:
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)

    print("-" * 65)
    print(f"{'Macro avg':<15} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f} {len(y_true):<8}")
    print(f"{'Weighted avg':<15} {weighted_precision:<10.4f} {weighted_recall:<10.4f} {weighted_f1:<10.4f} {len(y_true):<8}")

    #Show confusion matrix heatmap
    plot_confusion_matrix(y_true, y_pred, labels, model_name)
    
def print_model_comparison(results):
    """
    Compare multiple models with accuracy and F1 metrics table.
    
    Input: results (dict) - model_name -> (y_true, y_pred) mapping 
    """
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Model':<35} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12}")
    print("-" * 80)
    
    for model_name, (y_true, y_pred) in results.items():
        accuracy = accuracy_score(y_true, y_pred)
        
        labels = sorted(list(set(y_true)))        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        
        macro_f1 = np.mean(f1)
        weighted_f1 = np.average(f1, weights=support)
        
        print(f"{model_name:<35} {accuracy:<10.4f} {macro_f1:<10.4f} {weighted_f1:<12.4f}")

def print_deduplication_analysis(results):
    """
    Analyze impact of data deduplication on model performance.
    
    Input: results (dict) - model_name -> (y_true, y_pred) mapping 
    """
    print(f"\n{'='*60}")
    print(f"DEDUPLICATION IMPACT")
    print(f"{'='*60}")
    
    model_pairs = {
        'Decision Tree': ('Decision Tree (Original)', 'Decision Tree (Deduplicated)'),
        'Logistic Regression': ('Logistic Regression (Original)', 'Logistic Regression (Deduplicated)'),
        'MLP': ('MLP (Original)', 'MLP (Deduplicated)'),
        'Gradient Boosting': ('Gradient Boosting (Original)', 'Gradient Boosting (Deduplicated)')
    }
    
    print(f"{'Model':<20} {'Original':<10} {'Dedup':<10} {'Difference':<12}")
    print("-" * 55)
    
    for model_type, (orig_name, dedup_name) in model_pairs.items():
        if orig_name in results and dedup_name in results:
            orig_acc = accuracy_score(results[orig_name][0], results[orig_name][1])
            dedup_acc = accuracy_score(results[dedup_name][0], results[dedup_name][1])
            diff = dedup_acc - orig_acc
            
            print(f"{model_type:<20} {orig_acc:<10.4f} {dedup_acc:<10.4f} {diff:<10.4f}")

def full_evaluation(results, data=None):
    """
    Run complete evaluation suite across all models.
    
    Input: results (dict) - model results, data (optional) - dataset for misclassification examples 
    """
    print(f"\n{'#'*80}")
    print(f"DIALOG ACT CLASSIFICATION EVALUATION")
    print(f"{'#'*80}")
    
    for model_name, (y_true, y_pred) in results.items():
        print_detailed_metrics(y_true, y_pred, model_name)
        
        if data:
            # Determine dataset type and get corresponding utterances
            dataset_type = "Original" if "Original" in model_name else "Deduplicated"
            
            if "Original" in model_name:
                utterances = data['orig'][3]   
            else:
                utterances = data['dedup'][3]   
            
            summarize_misclassifications(y_true, y_pred, model_name, dataset_type)
            
            print(f"\n-- Sample Misclassifications for {model_name} --")
            error_count = 0
            for actual, pred, utt in zip(y_true, y_pred, utterances):
                if actual != pred:
                    print(f"Utterance: {utt}")
                    print(f"  Predicted: {pred}")
                    print(f"  Actual:    {actual}\n")
                    error_count += 1
                    if error_count >= 3:
                        total_errors = sum(1 for a, p in zip(y_true, y_pred) if a != p)
                        if total_errors > 3:
                            print(f"... and {total_errors - 3} more misclassifications")
                        break
            if error_count == 0:
                print("  No misclassifications found.")
    
    print_model_comparison(results)
    print_deduplication_analysis(results)
    
    print(f"\n{'#'*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'#'*80}")
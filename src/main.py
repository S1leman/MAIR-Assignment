import os
from collections import Counter
from utils import read_data, split_and_save_dataset
from baseline_models import majority_baseline_model, rules_baseline_model
from ml_models import decision_tree_classifier, logistic_regression_classifier, mlp_classifier
from evaluation import  full_evaluation

def load_data():
    print("Loading data...")

    # Original data
    acts_orig, utterances_orig = read_data("data/dialog_acts.dat", deduplicate=False)
    train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig = split_and_save_dataset(
        acts_orig, utterances_orig, "data/train_orig.txt", "data/test_orig.txt"
    )
    
    # Deduplicated data
    acts_dedup, utterances_dedup = read_data("data/dialog_acts.dat", deduplicate=True)
    train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup = split_and_save_dataset(
        acts_dedup, utterances_dedup, "data/train_dedup.txt", "data/test_dedup.txt"
    )
    
    print(f"Original: {len(train_acts_orig)} train, {len(test_acts_orig)} test")
    print(f"Deduplicated: {len(train_acts_dedup)} train, {len(test_acts_dedup)} test")
    
    return {
        'orig': (train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig),
        'dedup': (train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup)
    }

def train_all_models(data):
    print("\nTraining and evaluating models...")

    train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig = data['orig']
    train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup = data['dedup']

    majority_label = Counter(train_acts_orig).most_common(1)[0][0]
    maj_pred_orig = majority_baseline_model(test_utts_orig, majority_label)
    rules_pred_orig = rules_baseline_model(test_utts_orig)
    print(f"Majority class: {majority_label}")

    dt_model_orig, dt_vectorizer_orig = decision_tree_classifier(
        train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig, return_model=True)
    lr_model_orig, lr_vectorizer_orig = logistic_regression_classifier(
        train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig, return_model=True)
    mlp_model_orig, mlp_vectorizer_orig, mlp_le_orig = mlp_classifier(
        train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig, return_model=True)

    dt_pred_orig = dt_model_orig.predict(dt_vectorizer_orig.transform(test_utts_orig))
    lr_pred_orig = lr_model_orig.predict(lr_vectorizer_orig.transform(test_utts_orig))
    mlp_pred_orig_int = mlp_model_orig.predict(mlp_vectorizer_orig.transform(test_utts_orig).toarray())
    mlp_pred_orig = mlp_le_orig.inverse_transform(mlp_pred_orig_int)

    dt_model_dedup, dt_vectorizer_dedup = decision_tree_classifier(
        train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup, return_model=True)
    lr_model_dedup, lr_vectorizer_dedup = logistic_regression_classifier(
        train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup, return_model=True)
    mlp_model_dedup, mlp_vectorizer_dedup, mlp_le_dedup = mlp_classifier(
        train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup, return_model=True)

    dt_pred_dedup = dt_model_dedup.predict(dt_vectorizer_dedup.transform(test_utts_dedup))
    lr_pred_dedup = lr_model_dedup.predict(lr_vectorizer_dedup.transform(test_utts_dedup))
    mlp_pred_dedup_int = mlp_model_dedup.predict(mlp_vectorizer_dedup.transform(test_utts_dedup).toarray())
    mlp_pred_dedup = mlp_le_dedup.inverse_transform(mlp_pred_dedup_int)
    
    results = {
        "Majority Baseline (Original)": (test_acts_orig, maj_pred_orig),
        "Rules Baseline (Original)": (test_acts_orig, rules_pred_orig),
        "Decision Tree (Original)": (test_acts_orig, dt_pred_orig),
        "Logistic Regression (Original)": (test_acts_orig, lr_pred_orig),
        "MLP (Original)": (test_acts_orig, mlp_pred_orig),
        "Decision Tree (Deduplicated)": (test_acts_dedup, dt_pred_dedup),
        "Logistic Regression (Deduplicated)": (test_acts_dedup, lr_pred_dedup),
        "MLP (Deduplicated)": (test_acts_dedup, mlp_pred_dedup),
    }

    models = {
        "dt_model_orig": dt_model_orig, "dt_vectorizer_orig": dt_vectorizer_orig,
        "lr_model_orig": lr_model_orig, "lr_vectorizer_orig": lr_vectorizer_orig,
        "mlp_model_orig": mlp_model_orig, "mlp_vectorizer_orig": mlp_vectorizer_orig, "mlp_le_orig": mlp_le_orig,
        "dt_model_dedup": dt_model_dedup, "dt_vectorizer_dedup": dt_vectorizer_dedup,
        "lr_model_dedup": lr_model_dedup, "lr_vectorizer_dedup": lr_vectorizer_dedup,
        "mlp_model_dedup": mlp_model_dedup, "mlp_vectorizer_dedup": mlp_vectorizer_dedup, "mlp_le_dedup": mlp_le_dedup
    }

    return  results, models
  
def choose_dataset_and_model():
    while True:
        print("\nChoose dataset for prediction:")
        print("1. Original")
        print("2. Deduplicated")
        print("3. Exit")
        dataset_choice = input("\nSelect dataset (1-3): ").strip()
        if dataset_choice == "3":
            return None, None
        if dataset_choice not in ["1", "2"]:
            print("Invalid choice.")
            continue
        suffix = "orig" if dataset_choice == "1" else "dedup"

        while True:
            if suffix == "orig":
                print(f"\nAvailable classifiers for Original data:")
                print("1. Majority Baseline")
                print("2. Rules Baseline")
                print("3. Decision Tree")
                print("4. Logistic Regression")
                print("5. MLP")
                print("6. Back to dataset choice")
                valid_choices = ["1", "2", "3", "4", "5", "6"]
            else:
                print(f"\nAvailable classifiers for Deduplicated data:")
                print("1. Decision Tree")
                print("2. Logistic Regression")
                print("3. MLP")
                print("4. Back to dataset choice")
                valid_choices = ["1", "2", "3", "4"]

            choice = input("\nSelect classifier: ").strip()
            if (suffix == "orig" and choice == "6") or (suffix == "dedup" and choice == "4"):
                break
            if choice not in valid_choices:
                print("Invalid choice.")
                continue
            return suffix, choice

def interactive_classification(models):
    print(f"\n{'='*60}")
    print("INTERACTIVE CLASSIFICATION")
    print(f"{'='*60}")

    while True:
        result = choose_dataset_and_model()
        if result == (None, None):
            print("Goodbye!")
            break
        suffix, choice = result

        if suffix == "orig":
            model_names = {
                "1": "Majority", "2": "Rules", "3": "Decision Tree",
                "4": "Logistic Regression", "5": "MLP"
            }
        else:
            model_names = {
                "1": "Decision Tree", "2": "Logistic Regression", "3": "MLP"
            }

        print(f"\nUsing: {model_names[choice]}")
        print("Enter utterances (type 'back' to change model/dataset):")

        while True:
            utterance = input("\nEnter utterance: ").strip()
            if utterance.lower() == 'back':
                break
            if utterance.lower() in ['quit', 'exit']:
                return
            if not utterance:
                continue

            try:
                if suffix == "orig":
                    if choice == "1":
                        prediction = majority_baseline_model([utterance], 'inform')[0]
                    elif choice == "2":
                        prediction = rules_baseline_model([utterance])[0]
                    elif choice == "3":
                        X = models['dt_vectorizer_orig'].transform([utterance])
                        prediction = models['dt_model_orig'].predict(X)[0]
                    elif choice == "4":
                        X = models['lr_vectorizer_orig'].transform([utterance])
                        prediction = models['lr_model_orig'].predict(X)[0]
                    elif choice == "5":
                        X = models['mlp_vectorizer_orig'].transform([utterance]).toarray()
                        pred_int = models['mlp_model_orig'].predict(X).argmax(axis=1)[0]
                        prediction = models['mlp_le_orig'].inverse_transform([pred_int])[0]
                else:
                    if choice == "1":
                        X = models['dt_vectorizer_dedup'].transform([utterance])
                        prediction = models['dt_model_dedup'].predict(X)[0]
                    elif choice == "2":
                        X = models['lr_vectorizer_dedup'].transform([utterance])
                        prediction = models['lr_model_dedup'].predict(X)[0]
                    elif choice == "3":
                        X = models['mlp_vectorizer_dedup'].transform([utterance]).toarray()
                        pred_int = models['mlp_model_dedup'].predict(X).argmax(axis=1)[0]
                        prediction = models['mlp_le_dedup'].inverse_transform([pred_int])[0]
                print(f"Predicted: {prediction}")
            except Exception as e:
                print(f"Error: {e}")

def main():
    print("DIALOG ACT CLASSIFICATION SYSTEM")
    print("="*60)
    
    data = load_data()
    if not data:
        return
    
    results, models = train_all_models(data)

    full_evaluation(results, data)

    interactive_classification(models)

if __name__ == "__main__":
    main()
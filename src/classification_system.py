from collections import Counter
from utils import load_data
from baseline_models import majority_baseline_model, rules_baseline_model
from ml_models import decision_tree_classifier, logistic_regression_classifier, mlp_classifier, gradient_boosting_classifier
from evaluation import  full_evaluation

def train_all_models(data):
    """
    Train and evaluate all baseline and ML classifiers on both datasets.
    
    Input: data (dict) - contains 'orig' and 'dedup' dataset splits
    Output: (results, models) tuple - evaluation results and trained model objects
    """
    
    print("\nTraining and evaluating models...")

    #Unpack original and deduplicated datasets
    train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig = data['orig']
    train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup = data['dedup']
   
    #Get the majority class from the original training set
    majority_label = Counter(train_acts_orig).most_common(1)[0][0]
    maj_pred_orig = majority_baseline_model(test_utts_orig, majority_label)
    rules_pred_orig = rules_baseline_model(test_utts_orig)
    print(f"Majority class: {majority_label}")

    #Train models on original data 
    dt_model_orig, dt_vectorizer_orig = decision_tree_classifier(
        train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig, return_model=True)
    lr_model_orig, lr_vectorizer_orig = logistic_regression_classifier(
        train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig, return_model=True)
    mlp_model_orig, mlp_vectorizer_orig, mlp_le_orig = mlp_classifier(
        train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig, return_model=True)
    gb_model_orig, gb_vectorizer_orig = gradient_boosting_classifier(
        train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig, return_model=True)

    #Predictions for original test set
    dt_pred_orig = dt_model_orig.predict(dt_vectorizer_orig.transform(test_utts_orig))
    lr_pred_orig = lr_model_orig.predict(lr_vectorizer_orig.transform(test_utts_orig))
    # MLP requires special handling: convert to dense array and decode labels
    mlp_pred_orig_int = mlp_model_orig.predict(mlp_vectorizer_orig.transform(test_utts_orig).toarray())
    mlp_pred_orig = mlp_le_orig.inverse_transform(mlp_pred_orig_int)
    gb_pred_orig = gb_model_orig.predict(gb_vectorizer_orig.transform(test_utts_orig))

    #Train models on deduplicated data 
    dt_model_dedup, dt_vectorizer_dedup = decision_tree_classifier(
        train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup, return_model=True)
    lr_model_dedup, lr_vectorizer_dedup = logistic_regression_classifier(
        train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup, return_model=True)
    mlp_model_dedup, mlp_vectorizer_dedup, mlp_le_dedup = mlp_classifier(
        train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup, return_model=True)
    gb_model_dedup, gb_vectorizer_dedup = gradient_boosting_classifier(
        train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup, return_model=True)
    
    #Predictions for deduplicated test set
    dt_pred_dedup = dt_model_dedup.predict(dt_vectorizer_dedup.transform(test_utts_dedup))
    lr_pred_dedup = lr_model_dedup.predict(lr_vectorizer_dedup.transform(test_utts_dedup))
    # MLP requires dense array conversion and label decoding
    mlp_pred_dedup_int = mlp_model_dedup.predict(mlp_vectorizer_dedup.transform(test_utts_dedup).toarray())
    mlp_pred_dedup = mlp_le_dedup.inverse_transform(mlp_pred_dedup_int)
    gb_pred_dedup = gb_model_dedup.predict(gb_vectorizer_dedup.transform(test_utts_dedup))
    
    results = {
        "Majority Baseline (Original)": (test_acts_orig, maj_pred_orig),
        "Rules Baseline (Original)": (test_acts_orig, rules_pred_orig),
        "Decision Tree (Original)": (test_acts_orig, dt_pred_orig),
        "Logistic Regression (Original)": (test_acts_orig, lr_pred_orig),
        "MLP (Original)": (test_acts_orig, mlp_pred_orig),
        "Gradient Boosting (Original)": (test_acts_orig, gb_pred_orig),
        "Decision Tree (Deduplicated)": (test_acts_dedup, dt_pred_dedup),
        "Logistic Regression (Deduplicated)": (test_acts_dedup, lr_pred_dedup),
        "MLP (Deduplicated)": (test_acts_dedup, mlp_pred_dedup),
        "Gradient Boosting (Deduplicated)": (test_acts_dedup, gb_pred_dedup),
    }

    #Store trained models and vectorizers for later use
    models = {
        "dt_model_orig": dt_model_orig, "dt_vectorizer_orig": dt_vectorizer_orig,
        "lr_model_orig": lr_model_orig, "lr_vectorizer_orig": lr_vectorizer_orig,
        "mlp_model_orig": mlp_model_orig, "mlp_vectorizer_orig": mlp_vectorizer_orig, "mlp_le_orig": mlp_le_orig,
        "gb_model_orig": gb_model_orig, "gb_vectorizer_orig": gb_vectorizer_orig,
        "dt_model_dedup": dt_model_dedup, "dt_vectorizer_dedup": dt_vectorizer_dedup,
        "lr_model_dedup": lr_model_dedup, "lr_vectorizer_dedup": lr_vectorizer_dedup,
        "mlp_model_dedup": mlp_model_dedup, "mlp_vectorizer_dedup": mlp_vectorizer_dedup, "mlp_le_dedup": mlp_le_dedup,
        "gb_model_dedup": gb_model_dedup, "gb_vectorizer_dedup": gb_vectorizer_dedup
    }

    return  results, models
  
def choose_dataset_and_model():
    """
    Interactive menu for dataset and classifier selection.
    
    Output: (suffix, choice) tuple - dataset type and classifier choice, or (None, None) to exit
    """
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
                print("6. Gradient Boosting")
                print("7. Back to dataset choice")
                valid_choices = ["1", "2", "3", "4", "5", "6", "7"]
            else:
                print(f"\nAvailable classifiers for Deduplicated data:")
                print("1. Decision Tree")
                print("2. Logistic Regression")
                print("3. MLP")
                print("4. Gradient Boosting")
                print("5. Back to dataset choice")
                valid_choices = ["1", "2", "3", "4", "5"]

            choice = input("\nSelect classifier: ").strip()
            if (suffix == "orig" and choice == "7") or (suffix == "dedup" and choice == "5"):
                break
            if choice not in valid_choices:
                print("Invalid choice.")
                continue
            return suffix, choice

def interactive_classification(models):
    """
    Interactive CLI for real-time utterance classification.
    
    Input: models (dict) - trained classifiers, vectorizers, and label encoders
    """
    print(f"\n{'='*60}")
    print("INTERACTIVE CLASSIFICATION")
    print(f"{'='*60}")

    while True:
        #Prompt user to select dataset + model
        result = choose_dataset_and_model()
        if result == (None, None):
            print("Goodbye!")
            break
        suffix, choice = result

        if suffix == "orig":
            model_names = {
                "1": "Majority", "2": "Rules", "3": "Decision Tree",
                "4": "Logistic Regression", "5": "MLP", "6": "Gradient Boosting"
            }
        else:
            model_names = {
                "1": "Decision Tree", "2": "Logistic Regression", "3": "MLP", "4": "Gradient Boosting"
            }

        print(f"\nUsing: {model_names[choice]}") 

        while True:
            utterance = input("\nEnter utterances (type 'back' to change model/dataset): ").strip()
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
                        # MLP prediction - model.predict() returns class indices directly
                        X = models['mlp_vectorizer_orig'].transform([utterance]).toarray()
                        pred_int = models['mlp_model_orig'].predict(X)[0]
                        prediction = models['mlp_le_orig'].inverse_transform([pred_int])[0]
                    elif choice == "6":
                        X = models['gb_vectorizer_orig'].transform([utterance])
                        prediction = models['gb_model_orig'].predict(X)[0]
                else:
                    if choice == "1":
                        X = models['dt_vectorizer_dedup'].transform([utterance])
                        prediction = models['dt_model_dedup'].predict(X)[0]
                    elif choice == "2":
                        X = models['lr_vectorizer_dedup'].transform([utterance])
                        prediction = models['lr_model_dedup'].predict(X)[0]
                    elif choice == "3":
                        # MLP prediction - model.predict() returns class indices directly
                        X = models['mlp_vectorizer_dedup'].transform([utterance]).toarray()
                        pred_int = models['mlp_model_dedup'].predict(X)[0]
                        prediction = models['mlp_le_dedup'].inverse_transform([pred_int])[0]
                    elif choice == "4":
                        X = models['gb_vectorizer_dedup'].transform([utterance])
                        prediction = models['gb_model_dedup'].predict(X)[0]
                print(f"Predicted: {prediction}")
            except Exception as e:
                print(f"Error: {e}")

def main():
    """
    Main entry point for the classification system.
    """
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
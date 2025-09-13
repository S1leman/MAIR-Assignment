from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def read_data(path, deduplicate: bool = False):
    """Load dataset as (dialog_acts, utterances), lowercase, optionally deduplicated."""
    dialog_act_labels, utterances, first_seen_map = [], [], {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            sentence = line.strip().lower()
            if not sentence:
                continue
            parts = sentence.split(" ", 1)
            if len(parts) != 2:
                continue
            label, utterance = parts
            if utterance not in first_seen_map:
                first_seen_map[utterance] = label
                dialog_act_labels.append(label)
                utterances.append(utterance)
            elif not deduplicate:
                dialog_act_labels.append(first_seen_map[utterance])
                utterances.append(utterance)
    return dialog_act_labels, utterances


from collections import Counter

def split_and_save_dataset(dialog_act_labels, utterances, train_path, test_path, test_size=0.15, random_state=42):
    # Filter out classes with <2 samples
    label_counts = Counter(dialog_act_labels)
    valid_labels = {label for label, count in label_counts.items() if count >= 2}
    filtered = [(l, u) for l, u in zip(dialog_act_labels, utterances) if l in valid_labels]
    filtered_labels, filtered_utterances = zip(*filtered)

    utterances_train, utterances_test, labels_train, labels_test = train_test_split(
        filtered_utterances, filtered_labels, test_size=test_size, random_state=random_state, stratify=filtered_labels
    )
    with open(train_path, "w", encoding="utf-8") as train_file:
        for label, utterance in zip(labels_train, utterances_train):
            train_file.write(f"{label} {utterance}\n")
    with open(test_path, "w", encoding="utf-8") as test_file:
        for label, utterance in zip(labels_test, utterances_test):
            test_file.write(f"{label} {utterance}\n")
    return labels_train, labels_test, utterances_train, utterances_test


def decision_tree_classifier(labels_train, labels_test, utterances_train, utterances_test):
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(utterances_train)
    X_test = vectorizer.transform(utterances_test)
    classifier = DecisionTreeClassifier(random_state=42).fit(X_train, labels_train)
    return classifier.predict(X_test)


def logistic_regression_classifier(labels_train, labels_test, utterances_train, utterances_test):
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(utterances_train)
    X_test = vectorizer.transform(utterances_test)
    classifier = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, labels_train)
    return classifier.predict(X_test)


def keras_mlp_classifier(labels_train, labels_test, utterances_train, utterances_test,
                         epochs=20, batch_size=32, hidden_units=128, dropout_rate=0.2):
    
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(utterances_train).astype(np.float32).toarray()
    X_test = vectorizer.transform(utterances_test).astype(np.float32).toarray()

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(labels_train)
    y_test = label_encoder.transform(labels_test)
    num_classes = len(label_encoder.classes_)

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(hidden_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden_units // 2, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    return label_encoder.inverse_transform(y_pred)


def evaluate_model(y_true, y_pred, model_name="Model", dataset_name="Dataset"):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n-- {model_name} on {dataset_name} --")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))


def main():
    dataset_path = "data/dialog_acts.dat"

    # Original data
    labels_original, utterances_original = read_data(dataset_path, deduplicate=False)
    labels_train_orig, labels_test_orig, utterances_train_orig, utterances_test_orig = split_and_save_dataset(
        labels_original, utterances_original, "data/train_dataset_orig.dat", "data/test_dataset_orig.dat"
    )

    # Deduplicated data
    labels_dedup, utterances_dedup = read_data(dataset_path, deduplicate=True)
    labels_train_dedup, labels_test_dedup, utterances_train_dedup, utterances_test_dedup = split_and_save_dataset(
        labels_dedup, utterances_dedup, "data/train_dataset_dedup.dat", "data/test_dataset_dedup.dat"
    )

    # Models on original data
    preds_dt_orig = decision_tree_classifier(labels_train_orig, labels_test_orig, utterances_train_orig, utterances_test_orig)
    preds_lr_orig = logistic_regression_classifier(labels_train_orig, labels_test_orig, utterances_train_orig, utterances_test_orig)
    preds_mlp_orig = keras_mlp_classifier(labels_train_orig, labels_test_orig, utterances_train_orig, utterances_test_orig)

    # Models on deduplicated data
    preds_dt_dedup = decision_tree_classifier(labels_train_dedup, labels_test_dedup, utterances_train_dedup, utterances_test_dedup)
    preds_lr_dedup = logistic_regression_classifier(labels_train_dedup, labels_test_dedup, utterances_train_dedup, utterances_test_dedup)
    preds_mlp_dedup = keras_mlp_classifier(labels_train_dedup, labels_test_dedup, utterances_train_dedup, utterances_test_dedup)

    # Evaluate
    evaluate_model(labels_test_orig, preds_dt_orig, "Decision Tree", "Original data")
    evaluate_model(labels_test_orig, preds_lr_orig, "Logistic Regression", "Original data")
    evaluate_model(labels_test_orig, preds_mlp_orig, "Keras MLP", "Original data")

    evaluate_model(labels_test_dedup, preds_dt_dedup, "Decision Tree", "Deduplicated data")
    evaluate_model(labels_test_dedup, preds_lr_dedup, "Logistic Regression", "Deduplicated data")
    evaluate_model(labels_test_dedup, preds_mlp_dedup, "Keras MLP", "Deduplicated data")


if __name__ == "__main__":
    main()

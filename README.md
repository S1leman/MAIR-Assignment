# Dialog Act Classification System

This repository implements a dialog act classification system for spoken utterances, including baseline and machine learning models, evaluation utilities, and an interactive CLI for manual predictions.

---

## File Overview

### `main.py`
- **Purpose:** Main entry point for the system.
- **Description:** Loads and splits data, trains all models, evaluates results, and provides an interactive command-line interface for classifying new utterances.

### `utils.py`
- **Purpose:** Data utilities.
- **Description:** Functions for reading dialog act data, deduplication, and splitting/saving train/test datasets.

### `baseline_models.py`
- **Purpose:** Baseline classifiers.
- **Description:** Implements a majority class baseline and a simple keyword-based rules baseline.

### `ml_models.py`
- **Purpose:** Machine learning classifiers.
- **Description:** Implements Decision Tree, Logistic Regression, and Multi-layer Perceptron (MLP) classifiers using bag-of-words features.

### `evaluation.py`
- **Purpose:** Evaluation and analysis.
- **Description:** Functions for printing accuracy, precision, recall, F1-score, misclassification summaries, model comparisons, and deduplication impact.

---

## How to Run

1. **Install Requirements**
   ```bash
   pip install scikit-learn tensorflow
   ```

2. **Prepare Data**
   - Place your dialog act data file at `data/dialog_acts.dat`.
   - Format: `<dialog_act> <utterance>` (one per line).

3. **Run the Program**
   ```bash
   python src/main.py
   ```
   - The program will load data, train models, print evaluation metrics, and start an interactive prompt.

4. **Interactive Mode**
   - Follow prompts to select dataset and model.
   - Enter utterances to see predicted dialog acts.
   - Type `back` to change model/dataset, or `exit` to quit.

---

## Directory Structure

```
MAIR-Assignment/
├── data/
│   └── dialog_acts.dat
├── src/
│   ├── main.py
│   ├── utils.py
│   ├── baseline_models.py
│   ├── ml_models.py
│   └── evaluation.py
└── README.md
```

---

## Notes

- Supports both original and deduplicated datasets.
- Evaluation includes per-class and overall metrics.
- Interactive mode allows manual testing of models.

---
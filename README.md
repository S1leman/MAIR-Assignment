# Cambridge Restaurant Recommendation System

## Team F6

- **Zoi Dimitriou** (8666334)
- **Sileman Patang** (2389697)  
- **Athanasia Boumpouraki** (8289964)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Main Dialog System](#main-dialog-system)
  - [Classification System](#classification-system)
  - [Inference Engine Demo](#inference-engine-demo)
- [Example Interactions](#example-interactions)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Natural Language Understanding**: Extracts user preferences from conversational input
- **Multiple ML Classifiers**: MLP, Logistic Regression, Decision Trees, Gradient Boosting
- **Inference Engine**: Automatically derives additional properties (romantic, touristic, child-friendly)
- **Conflict Resolution**: Handles conflicting inference rules
- **Accessibility Options**: ALL CAPS output mode, flexible restart policies, classifier selection
- **Evaluation Suite**: Complete metrics, confusion matrices, and performance analysis

---

## Installation

### Prerequisites

- Python 3.7+ (tested with Python 3.13)
- pip package manager
- Virtual environment (recommended)

### Steps

#### 1. Create and activate virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. Verify data files exist
```bash
ls data/restaurant_info.csv data/dialog_acts.dat data/restaurant_info_updated.csv
```

---

## Quick Start

### Option 1: Run Main Dialog System
```bash
cd /path/to/MAIR-Assignment
python src/dialog_system.py
```

### Option 2: Try Inference Engine Demo
```bash
python src/inference_engine.py
```

### Option 3: Evaluate All Classifiers
```bash
python src/classification_system.py
```

**Important**: All commands should be run from the project root directory.

---

## Usage

### Inference Engine Demo

Explore how rule-based reasoning and conflict detection work:

```bash
python src/inference_engine.py
```

**What it demonstrates:**

- **Conflict Detection**: Watch conflicting rules get identified automatically  
- **Conflict Types**: Both romantic conflicts (Rules 5 vs 6) and touristic conflicts (Rules 1 vs 2)
- **Real Examples**: Four practical scenarios with restaurant data

**Sample Output:**
```
CAMBRIDGE RESTAURANT INFERENCE ENGINE
Demonstrating rule-based reasoning and conflict detection

1. NO CONFLICT EXAMPLE:
Restaurant: Da Vinci Pizzeria (cheap + good + italian)
Inferred properties: {'touristic': True}
Has conflict: False
Reasoning: ['Rule 1: a cheap restaurant with good food attracts tourists']

2. ROMANTIC CONFLICT EXAMPLE:  
Restaurant: Le Bistro (busy + long stays)
Inferred properties: {'assigned_seats': True, 'children': False}
Has conflict: True
Conflict on: romantic
Rule 5: a busy restaurant is not romantic → romantic=False
Rule 6: spending a long time in a restaurant is romantic → romantic=True
```

**Rules Explained:**
1. `(cheap AND good food) → touristic = True`
2. `romanian cuisine → touristic = False`  ⚡ *conflicts with Rule 1*
3. `busy → assigned_seats = True`
4. `long stays → children = False`
5. `busy → romantic = False` 
6. `long stays → romantic = True` ⚡ *conflicts with Rule 5*

### First-Time Setup

**1. Navigate to project directory:**
```bash
cd MAIR-Assignment
```

**2. Start the main dialog system:**
```bash
python src/dialog_system.py
```

**3. Configure the system when prompted:**
- Choose classifier (recommend MLP for best accuracy)
- Set restart policy (allow/disallow restarts)
- Choose output format (normal/ALL CAPS)

**4. Start conversing:**
```
System: Hello, welcome to the Cambridge restaurant system? You can ask for 
        restaurants by area, price range or food type. How may I help you?
User: I want italian food in the north
```

---

## Usage

### Main Dialog System

Start the interactive dialog system:

```bash
python src/dialog_system.py 
```
**Configuration options during startup:**

**1. Classifier Selection:**
- `1` - MLP (Multi-Layer Perceptron) - Recommended (~97% accuracy)
- `2` - Majority Baseline (Simple baseline)
- `3` - Rules Baseline (Keyword-based)

**2. Restart Policy:**
- `1` - Allow restarts (users can type "restart" anytime)
- `2` - No restarts (linear flow only)

**3. Output Format:**
- `1` - Normal case
- `2` - ALL CAPS (accessibility mode)

**Navigation commands:**

- `exit`, `quit`, `bye` - Exit the system
- `restart`, `start over` - Reset and start fresh (if enabled)
- `alternative`, `different` - Request different restaurant

### Classification System

Train and evaluate all classifiers:

```bash
python src/classification_system.py
```
**What it does:**

- Trains 6 different classifiers on dialog act data
- Evaluates with accuracy, precision, recall, F1-scores
- Displays confusion matrices
- Shows misclassification examples
- Analyzes deduplication impact

**Interactive mode:**
After evaluation, test any classifier with custom utterances:

```
Enter utterances (type 'back' to change model/dataset): I want a cheap restaurant
Predicted: inform

Enter utterances (type 'back' to change model/dataset): thank you
Predicted: thankyou

Enter utterances (type 'back' to change model/dataset): what's the phone number
Predicted: request
```

**Note:** During evaluation, confusion matrix plots will appear in popup windows. You must close these windows for the program to continue to the next step.

**Navigation:**
- Type `back` to change model/dataset
- Type `exit` or `quit` to leave the system

---

## Example Interactions
### Example 1: Basic Search

```
System: Hello, welcome to the Cambridge restaurant system? You can ask for 
        restaurants by area, price range or food type. How may I help you?
User: I want italian food

[Classified as: inform]

System: What part of town do you have in mind?
User: north

[Classified as: inform]

System: Would you like something in the cheap, moderate, or expensive price range?
User: cheap

[Classified as: inform]

System: Do you have any additional requirements? For example, would you like the 
        restaurant to be touristic, romantic, child-friendly, or have assigned seats? 
        You can say 'yes' and specify requirements, or 'no' if you don't have any 
        additional preferences.
User: no

[Classified as: negate]

System: You are looking for a restaurant in the north of town in the cheap 
        price range serving italian food, right?
System: Please answer yes to confirm or no to change your preferences.
User: yes

[Classified as: affirm]

System: I recommend 'Da Vinci Pizzeria', it is cheap italian restaurant in 
        the north of town.
```

### Example 2: Requesting Information

```
System: Would you like more information about the restaurant (phone, address), 
        an alternative restaurant, or would you like to try a different search?
System: You can ask for 'phone', 'address', say 'alternative' for other options, 
        specify new preferences, or say 'exit' to leave or 'restart' to start over.
User: What's the phone number?

[Classified as: request]

System: The phone number of Da Vinci Pizzeria is 01223 351707.

System: Would you like more information about the restaurant (phone, address), 
        an alternative restaurant, or would you like to try a different search?
User: address?

[Classified as: request]

System: Sure, Da Vinci Pizzeria is on 20 Milton Road Chesterton.

User: postcode please

[Classified as: request]

System: The post code of Da Vinci Pizzeria is C.B 4, 1 L.Y.
```

### Example 3: Additional Requirements with Inference

```
System: Do you have any additional requirements? For example, would you like the 
        restaurant to be touristic, romantic, child-friendly, or have assigned seats? 
        You can say 'yes' and specify requirements, or 'no' if you don't have any 
        additional preferences.
User: yes, I want something romantic

[Classified as: affirm]

[System applies inference rules]

System: You are looking for a restaurant in the centre of town in the expensive 
        price range serving french food that is romantic, right?
System: Please answer yes to confirm or no to change your preferences.
User: yes

[Classified as: affirm]

System: I recommend 'Cotto', it is expensive french restaurant in the centre 
        of town. The restaurant is romantic because you can take your time 
        and enjoy a leisurely meal.
```

### Example 4: Conflict Resolution

```
System: I found 'Le Jardin', but there's a conflict about whether it's romantic.
System: This restaurant is both busy (which makes it less romantic) and allows 
        long stays (which makes it more romantic).
System: Do you prefer a quiet atmosphere or the ability to stay for a long time?
User: I prefer quiet

[Classified as: inform]

System: Based on your preference for a quiet atmosphere, Le Jardin doesn't meet 
        your romantic requirement. Let me find another option for you.

System: I recommend 'Cotto', it is expensive french restaurant in the centre 
        of town. The restaurant is romantic because you can take your time 
        and enjoy a leisurely meal.
```

### Example 5: Don't Care Responses

```
System: What part of town do you have in mind?
User: any area is fine

[Classified as: inform]

System: Would you like something in the cheap, moderate, or expensive price range?
User: don't care

[Classified as: inform]

System: What kind of food would you like?
User: chinese

[Classified as: inform]

System: Do you have any additional requirements? For example, would you like the 
        restaurant to be touristic, romantic, child-friendly, or have assigned seats?
User: no

[Classified as: negate]
```

### Example 6: Requesting Alternatives

```
System: Would you like more information about the restaurant (phone, address), 
        an alternative restaurant, or would you like to try a different search?
User: any alternatives?

[Classified as: reqalts]

System: I recommend 'Pizza Hut Cherry Hinton', it is cheap italian 
        restaurant in the north of town.

System: Would you like more information about the restaurant (phone, address), 
        an alternative restaurant, or would you like to try a different search?
User: anything else?

[Classified as: reqalts]

System: I recommend 'Bella Italia', it is cheap italian restaurant in 
        the north of town.
```

### Example 7: Negative Preferences

```
System: Do you have any additional requirements? For example, would you like the restaurant to be touristic, romantic, child-friendly, or have assigned seats? You can say 'yes' and specify requirements, or 'no' if you don't have any additional preferences.
User: not romantic

[Classified as: negate]

System: You are looking for a restaurant in any price range that is not romantic, right?
System: Please answer yes to confirm or no to change your preferences.
User: yes

[Classified as: affirm]

System: I recommend 'saint johns chop house', it is moderate british restaurant in the west of town. The restaurant is not romantic because it tends to be busy and noisy.
```

---


## Configuration

### Recognized Keywords

#### Areas
- north, south, east, west, centre, center, any area, anywhere, don't care

#### Price Ranges
- cheap, moderate, expensive, moderately priced, any price, don't care

#### Food Types
- italian, chinese, indian, french, british, spanish, thai, japanese, korean, vietnamese, turkish, moroccan, mediterranean, seafood, steakhouse, ...

#### Additional Requirements

| Requirement | Positive Keywords | Negative Keywords |
|-------------|------------------|-------------------|
| Romantic | romantic, intimate, cozy, date | not romantic, casual, business |
| Touristic | touristic, tourist, popular, famous | not touristic, local, hidden, authentic |
| Child-friendly | children, kids, family, child-friendly | no children, adults only, quiet |
| Assigned seats | assigned seats, waiter chooses | no assigned seats, free seating, choose seat |

**Examples:**
- "I want something romantic" → `{'romantic': True}`
- "not romantic please" → `{'romantic': False}`
- "something not touristic" → `{'touristic': False}`
- "no assigned seats" → `{'assigned_seats': False}`

### Inference Rules
The system uses 6 inference rules:

1. **Cheap + Good food → Touristic** 
2. **Romanian cuisine → Not touristic**
3. **Busy restaurant → Assigned seating** 
4. **Long stay → Not child-friendly** 
5. **Busy restaurant → Not romantic** 
6. **Long stay → Romantic** 

**Conflicts:** Rules 5 & 6 conflict on "romantic", Rules 1 & 2 conflict on "touristic"

---

## Project Structure

```
MAIR-Assignment/
│
├── src/
│   ├── dialog_system.py              # Main entry point
│   ├── state_transition.py           # Core dialog state manager
│   ├── conversation_states.py        # State transition handlers
│   ├── inference_engine.py           # Rule-based inference system
│   ├── preference_extraction.py      # NLU preference extractor
│   ├── classification_system.py      # Classifier training/evaluation
│   ├── ml_models.py                  # ML classifier implementations
│   ├── baseline_models.py            # Baseline classifiers
│   ├── evaluation.py                 # Metrics and evaluation
│   ├── lookup.py                     # Restaurant database lookup
│   └── utils.py                      # Utility functions
│
├── data/
│   ├── restaurant_info.csv              # Original restaurant database  
│   ├── restaurant_info_updated.csv      # Extended with inference properties
│   ├── dialog_acts.dat                  # Dialog act training data
│   ├── dialog_acts_test.dat            # Dialog act test data
│   ├── all_dialogs.txt                 # Combined dialog data
│   ├── train_dedup.txt                 # Deduplicated training data
│   ├── train_orig.txt                  # Original training data
│   ├── test_dedup.txt                  # Deduplicated test data
│   └── test_orig.txt                   # Original test data
│
├── models/                       # Auto-generated trained models
│   ├── mlp_model.pkl
│   ├── mlp_vectorizer.pkl
│   └── mlp_label_encoder.pkl
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Technical Details

### Code Organization
The codebase follows clean patterns with:
- **Consistent documentation**: Input/Output format docstrings throughout
- **State numbering**: Numbered conversation states (1-10) for clear flow tracking
- **Minimal comments**: Focused, descriptive comments where needed

### Dialog States
The system uses a finite state machine with these states:

```
1. WELCOME
2. ASK_AREA
3. ASK_PRICE
4. ASK_FOOD_TYPE 
5. ASK_ADDITIONAL_REQUIREMENTS
6. APOLOGISE
7. CONFIRM
8. SUGGEST_RESTAURANT
9. INFORM
10. GOODBYE
```

### Preference Extraction
Uses multi-strategy approach:

- **Exact matching** - Direct keyword lookup
- **Regex patterns** - Structured phrase extraction
- **Fuzzy matching** - Levenshtein distance for typos (max distance: 3)
- **Context-aware** - Different behavior per dialog state
- **Don't care detection** - Handles "any", "don't care", etc.

---

## Troubleshooting

### "No pre-trained model found"
**Solution:** First run will automatically train a new model (takes 1-2 minutes). Subsequent runs load the saved model instantly.

### "No restaurants found matching your criteria"

**Solution:** Try broader preferences. Use "any" or "don't care" for some criteria.

```
User: any area
User: don't care about price
User: italian food
```

### Import errors

**Solution:** Reinstall dependencies:

```bash
pip install --upgrade -r requirements.txt
```

### Missing data files

**Solution:** Verify all data files exist:

```bash
ls -la data/
# Should show: restaurant_info.csv dialog_acts.dat restaurant_info_updated.csv
```

### Conversation stuck in loop

**Solution:** Type `restart` (if enabled) or `exit` and restart the program.

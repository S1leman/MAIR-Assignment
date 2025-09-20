# MAIR Assignment: Cambridge Restaurant Dialog System

A dialog system for restaurant recommendations in Cambridge, implementing natural language processing, machine learning classification, and conversation management.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dialog system
python src/dialog_system.py
```

## Overview

This project implements a restaurant recommendation dialog system that:
- Understands natural language user requests
- Maintains conversation context and state
- Provides restaurant suggestions based on preferences
- Uses machine learning for intent classification
- Supports fuzzy matching for user input variations

## Features

### Core Functionality
- Multi-turn conversations with context maintenance
- Natural language understanding and preference extraction
- 9-state dialog system with smooth transitions
- Restart functionality and error recovery
- Flexible search by area, price range, and food type

### Technical Features
- MLP neural network for intent classification (92%+ accuracy)
- Fuzzy matching for misspellings and variations
- Alternative restaurant suggestions
- Detailed restaurant information (phone, address, postcode)
- "Don't care" preference handling

## Architecture

```
Dialog System Components:
├── Dialog Controller (state_transition.py) - Main system orchestrator
├── Conversation States (conversation_states.py) - State implementations  
├── Preference Extraction (preference_extraction.py) - NLP processing
├── Restaurant Lookup (lookup.py) - Database interface
├── ML Classification (ml_models.py) - Intent classification
└── Utilities (utils.py) - Shared functionality
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone repository
git clone <repository-url>
cd MAIR-Assignment

# Install dependencies
pip install -r requirements.txt

# Run system
python src/dialog_system.py
```

## Usage

### Basic Usage
```bash
python src/dialog_system.py
```

### Testing
```bash
python src/simple_test.py
python src/dialog_test_suite.py
```

### Example Conversation
```
System: Hello, welcome to the Cambridge restaurant system. How may I help you?
User: I want a cheap italian restaurant
System: What part of town do you have in mind?
User: center
System: You are looking for a restaurant in the centre of town in the cheap price range serving italian food, right?
User: yes
System: Pizza hut city centre is a nice place in the city centre and the prices are cheap
```

## System Components

### Dialog States
1. **WELCOME** - Initial greeting and preference collection
2. **ASK_AREA** - Area preference collection
3. **ASK_PRICE** - Price range preference collection
4. **ASK_FOOD_TYPE** - Food type preference collection
5. **CONFIRM** - Preference confirmation
6. **APOLOGIZE** - Error recovery
7. **SUGGEST_RESTAURANT** - Restaurant presentation
8. **INFORM** - Additional options and recovery
9. **GOODBYE** - Conversation termination

### Machine Learning
- **Intent Classification**: MLP neural network with 92%+ accuracy
- **Dialog Acts**: inform, request, affirm, negate, thankyou, bye, reqalts, etc.
- **Training Data**: Dialog acts dataset with automatic train/test split
- **Features**: N-gram vectorization with balanced class weighting

### Preference Extraction
- **Food Types**: 35+ cuisine types with fuzzy matching
- **Areas**: Cambridge area recognition (north, south, east, west, centre)
- **Price Ranges**: cheap, moderate, expensive with variations
- **Pattern Matching**: Regex-based extraction with validation

## Configuration

### Restaurant Database
Edit `data/restaurant_info.csv`:
```csv
restaurantname,area,food,phone,addr,postcode,pricerange
pizza hut city centre,centre,italian,01223 323737,regent street,cb21ab,cheap
```

### Adding Food Types
Modify `preference_extraction.py`:
```python
self.food_types = ['italian', 'chinese', 'indian', 'new_cuisine']
```

## File Structure

```
MAIR-Assignment/
├── README.md
├── requirements.txt
├── data/
│   ├── restaurant_info.csv
│   ├── dialog_acts.dat
│   └── all_dialogs.txt
├── src/
│   ├── dialog_system.py        # Main entry point
│   ├── state_transition.py     # Dialog controller
│   ├── conversation_states.py  # State implementations
│   ├── preference_extraction.py # NLP processing
│   ├── lookup.py              # Database interface
│   ├── ml_models.py           # ML classifiers
│   ├── utils.py               # Utility functions
│   └── *_test.py             # Test suites
└── Deliverables/
```

## Testing

### Available Test Suites
- `simple_test.py` - Basic functionality testing
- `dialog_test_suite.py` - Comprehensive dialog flow testing
- `real_dialog_test.py` - Tests based on actual dialog data

### Custom Testing
```python
# Test preference extraction
from src.preference_extraction import PreferenceExtractor
extractor = PreferenceExtractor()
prefs = extractor.extract_preferences("cheap italian in the north")

# Test restaurant lookup
from src.lookup import RestaurantLookup
lookup = RestaurantLookup()
restaurant, alternatives = lookup.lookup({'food': 'italian', 'pricerange': 'cheap'})
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install -r requirements.txt
```

**File Not Found**
```bash
# Ensure correct directory
cd MAIR-Assignment
python src/dialog_system.py
```

**No Restaurant Results**
```python
# Check available food types
import pandas as pd
df = pd.read_csv('data/restaurant_info.csv')
print(df['food'].unique())
```

**Classification Not Working**
```python
# Ensure classifier is trained
system = RestaurantSystem()
system.train_classifier()  # Required step
system.run_conversation()
```

## Performance

- **Response Time**: <500ms per dialog turn
- **Intent Classification**: 92%+ accuracy
- **Memory Usage**: ~50MB with trained models
- **Coverage**: 35+ food types, 5 areas, 3 price ranges

## License

This project is part of the MAIR (Methods in AI Research) course assignment.

---

## How to Run

1. **Install Requirements**
   ```bash
   pip install scikit-learn tensorflow numpy
   ```

2. **Prepare Data**
   - Place dialog act data file at `data/dialog_acts.dat`.

3. **Run the Program**
   ```bash
   python src/main.py
   ```
   - The program will load data, train models, print evaluation metrics, and start an interactive prompt.

4. **Interactive Mode**
   - Follow prompts to select dataset and model.
   - Enter utterances to see predicted dialog acts.
   - Type `back` to change model/dataset, or `exit`,`quit` to quit.

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
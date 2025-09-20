# MAIR Assignment: Cambridge Restaurant Dialog System

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dialog system
python src/dialog_system.py
```

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

### Setup 
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
## System Components

### Core Files and Their Purpose

#### Main Entry Points
- **`src/dialog_system.py`** - Primary entry point for the dialog system
- **`src/classification_system.py`** - ML classification system with evaluation
- **`src/main.py`** - Original main entry point

#### Core System Components
- **`src/state_transition.py`** - Dialog controller and state management
- **`src/conversation_states.py`** - Individual dialog state implementations
- **`src/preference_extraction.py`** - Natural language processing and preference extraction
- **`src/lookup.py`** - Restaurant database interface
- **`src/ml_models.py`** - Machine learning model implementations
- **`src/utils.py`** - Utility functions and shared functionality

#### Supporting Components
- **`src/baseline_models.py`** - Baseline classification models
- **`src/evaluation.py`** - Model evaluation and metrics 

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


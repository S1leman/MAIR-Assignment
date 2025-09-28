# MAIR Assignment: Cambridge Restaurant Dialog System
## Installation

### Setup 
#### Install dependencies
pip install -r requirements.txt

### Run system
python src/dialog_system.py
```

### System Configuration
The system will prompt you to configure:

1. **Classifier Type**: MLP (recommended), Majority Baseline, or Rules-based
2. **Dialog Policy**: Allow restarts or linear flow only  
3. **Output Format**: Normal case or ALL CAPS
4. **Text-to-Speech**: Disabled (recommended) or Enabled (experimental)

## Example Conversation

```
System: Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?
System: Please tell me what kind of restaurant you're looking for (e.g., 'Italian food', 'cheap restaurant', 'restaurant in the south').

User: I want Italian food in the centre
System: Would you like something in the cheap, moderate, or expensive price range?
User: Expensive
System: Do you have any additional requirements? For example, would you like the restaurant to be touristic, romantic, child-friendly, or have assigned seats? You can say 'yes' and specify requirements, or 'no' if you don't have any additional preferences.
User: No
System: You are looking for a restaurant in the centre of town in the expensive price range serving Italian food, right?
User: Yes
System: I recommend 'caffe uno', it is expensive italian restaurant in the city centre.

User: What's the phone number?
System: The phone number for caffe uno is 01223 448620.
```

## Project Structure

```
src/
├── dialog_system.py          # Main entry point
├── state_transition.py       # Core system logic
├── conversation_states.py    # Dialog state management  
├── preference_extraction.py  # NLP and preference parsing
├── inference_engine.py       # Rule-based reasoning
├── lookup.py                 # Restaurant database
└── ml_models.py              # Machine learning models

data/
├── restaurant_info.csv       # Restaurant database
└── dialog_acts.dat           # Training data

models/                       # Trained ML models
```

## Dialog States

1. **WELCOME** → **ASK_AREA** → **ASK_PRICE** → **ASK_FOOD_TYPE** 
2. **ASK_ADDITIONAL_REQUIREMENTS** → **CONFIRM** → **SUGGEST_RESTAURANT**
3. **INFORM** → **GOODBYE** (with **APOLOGIZE** for error recovery)

## Known Issues

### Text-to-Speech (TTS)
- **Status**: Experimental feature added for fun
- **Issue**: TTS may fail when returning to the same dialog state twice

## Training and Evaluation

```bash
python src/classification_system.py

##  Team

This is a group of 3 people. The system implements:
- **Required features**: 3 classifier types, ALL CAPS mode, restart functionality  
- **Bonus feature**: Experimental TTS (work in progress)

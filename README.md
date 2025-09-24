# MAIR Assignment: Cambridge Restaurant Dialog System
## Installation

### Setup 
#### Install dependencies
pip install -r requirements.txt

### Run system
python src/dialog_system.py

python src/classification_system.py

## System Components

### Core Files and Their Purpose

#### Main Entry Points
- **`src/dialog_system.py`** - Primary entry point for the dialog system
- **`src/classification_system.py`** - ML classification system with evaluation

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


### Example Dialogs
#### Example 1
System: Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?

User: i want to eat italian

System: What part of town do you have in mind?

User: maybe in the north 

System: Would you like something in the cheap, moderate, or expensive price range?

User: moderately proced

System: You are looking for a restaurant in the north of town in the moderate price range serving italian food, right?    

User: yes

System: I'm sorry but there is no restaurant serving that type of food

User: no im good

Thank you for using the Cambridge restaurant system. Goodbye!


#### Example 2
System: Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?

User: i want to eat at a cheap pestaurant in the south 

System: What kind of food would you like?

User: chinese

System: You are looking for a restaurant in the south of town in the cheap price range serving chinese food, right?       

User: yes

System: the missing sock is a nice place in the south of town and the prices are cheap

User: mmm something different

System: rice house is a nice place in the south of town and the prices are cheap

User: that sounds good

System: rice house is a great restaurant

User: thank you

Thank you for using the Cambridge restaurant system. Goodbye!



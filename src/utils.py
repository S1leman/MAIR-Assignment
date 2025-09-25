from sklearn.model_selection import train_test_split
import os
import pickle
from ml_models import mlp_classifier

import numpy as np
import pandas as pd

def build_restaurant_info(csv_in="data/restaurant_info.csv",csv_out="data/restaurant_info_updated.csv"):
    """
    Adds food_quality, crowdedness, and length_stay to the restaurant_info CSV and save the updated file.
    """
    np.random.seed(42)
    df = pd.read_csv(csv_in)

    df["food_quality"] = np.where(np.random.randint(0, 2, df.shape[0]) == 1, "good", "bad")
    df["crowdedness"] = np.where(np.random.randint(0, 2, df.shape[0]) == 1, "busy", "not busy")
    df["length_stay"]  = np.where(np.random.randint(0, 2, df.shape[0]) == 1, "long", "short")

    df.to_csv(csv_out, index=False)
    return csv_out

def format_restaurant_info_response(restaurant, user_input):
    """
    Builds a natural-language response with phone/address/postcode for a restaurant.
    If user asked for specific fields (phone/address/postcode), return only those. Otherwise, return all available fields succinctly.
    Takes as input a 'restaurant'dictionary with keys like 'restaurantname', 'phone', 'addr', 'postcode' and 'user_input'which is the user utterance used to infer requested fields.
    """
    if not restaurant:
        return "I'm sorry, I don't have any restaurant information available to provide details."
    
    utterance_lower = user_input.lower()
    
    # Determine what specific information was requested
    phone_requested = 'phone' in utterance_lower or 'number' in utterance_lower
    address_requested = 'address' in utterance_lower or 'where' in utterance_lower or 'location' in utterance_lower
    postcode_requested = 'postcode' in utterance_lower or 'post code' in utterance_lower or 'postal' in utterance_lower
    
    # Get available data and check if it's valid
    phone = restaurant.get('phone', 'not available')
    
    addr = restaurant.get('addr')
    has_address = addr and str(addr).lower() not in ['nan', 'none', '', 'not available']
    
    postcode = restaurant.get('postcode')
    has_postcode = postcode and str(postcode).lower() not in ['nan', 'none', '', 'not available']
    
    # Build response based on what was requested
    info_parts = []
    
    if phone_requested:
        info_parts.append(f"The phone number of {restaurant['restaurantname']} is {phone}")
    
    if address_requested:
        if has_address:
            info_parts.append(f"Sure, {restaurant['restaurantname']} is on {addr}")
        else:
            info_parts.append(f"I'm sorry, the address for {restaurant['restaurantname']} is not available")
    
    if postcode_requested:
        if has_postcode:
            info_parts.append(f"The post code of {restaurant['restaurantname']} is {postcode}")
        else:
            info_parts.append(f"I'm sorry, the post code for {restaurant['restaurantname']} is not available")
    
    # Generate response
    if phone_requested or address_requested or postcode_requested:
        if len(info_parts) == 1:
            return f"{info_parts[0]}."
        elif len(info_parts) == 2:
            return f"{info_parts[0]} and {info_parts[1].lower()}."
        else:
            return f"{info_parts[0]}, {info_parts[1].lower()}, and {info_parts[2].lower()}."
    else:
        # Default: provide all available information
        response_parts = []
        response_parts.append(f"The phone number of {restaurant['restaurantname']} is {phone}")
        
        if has_address:
            response_parts.append(f"it is on {addr}")
        
        if has_postcode:
            response_parts.append(f"the post code is {postcode}")
        
        if len(response_parts) == 1:
            return f"{response_parts[0]}."
        elif len(response_parts) == 2:
            return f"{response_parts[0]} and {response_parts[1]}."
        else:
            return f"{response_parts[0]}, {response_parts[1]}, and {response_parts[2]}."


def format_restaurant_suggestion(restaurant):
    """
    Returns a restaurant suggestion.
    """
    if not restaurant:
        return None
        
    area_desc = f" in the {restaurant['area']} of town" if restaurant['area'] != 'centre' else " in the city centre"
    price_desc = f" and the prices are {restaurant['pricerange']}" if restaurant['pricerange'] != 'dontcare' else ""
    
    return f"{restaurant['restaurantname']} is a nice place{area_desc}{price_desc}"


def detect_restart_command(user_input):
    """
    Detects if the user wants to reset the conversation.
    """
    return user_input.lower().strip() in ['start over', 'start again', 'reset', 'restart']


def detect_new_search_request(user_input):
    """
    Heuristically detect if the user is requesting a new restaurant search.
    """
    input_lower = user_input.lower()
    search_keywords = ['is there', 'do you have', 'find me', 'looking for', 'want', 'need']
    return any(keyword in input_lower for keyword in search_keywords)


def get_state_name_from_value(states_dict, state_value): 
    """
    Reverse-lookup a state name (key) by its value.
    Takes as input a 'states_dict'dictionary which maps state names -> state constants/values and 'state_value' which is the value to look up.
    Returns the first matching key, or None if not found.
    """
    for name, value in states_dict.items():
        if value == state_value:
            return name
    return None

def update_preferences_with_context(user_requirements, validated_prefs, context_stage):
    """
    Merge newly validated preferences into the running user requirements.
    If 'context_stage' is present, only update the preference relevant to that stage.
    Otherwise, update any preferences that are currently unset.

    Takes as input:
        'user_requirements' dictionary: Mutable dict of current requirements (keys: 'area','price','food').
        'validated_prefs' dictionary: Extracted preferences from the latest utterance.
        'context_stage': Current dialog stage key (e.g., 'ASK_AREA'), or None.
    """
    if context_stage:
        stage_mapping = {
            'ASK_AREA': 'area',
            'ASK_PRICE': 'price',
            'ASK_FOOD_TYPE': 'food'
        }
        
        target_pref = stage_mapping.get(context_stage)
        if target_pref and target_pref in validated_prefs:
            user_requirements[target_pref] = validated_prefs[target_pref]
        else:
            # Update any preferences that aren't already set
            for pref_type, value in validated_prefs.items():
                if user_requirements[pref_type] is None:
                    user_requirements[pref_type] = value
    else:
        # No context - update any unset preferences
        for pref_type, value in validated_prefs.items():
            if user_requirements[pref_type] is None:
                user_requirements[pref_type] = value


def log_preference_changes(validated_prefs, user_requirements, old_prefs, errors):
    """
    Prints:
      - validation warnings (if any).
      - the just-extracted preferences.
      - the updated preference state (if changed).
    """
    if errors:
        print(f"[Validation warnings: {', '.join(errors)}]")
    
    if validated_prefs:
        print(f"[Extracted: {validated_prefs}]")
    
    if user_requirements != old_prefs:
        print(f"[Preferences updated: {user_requirements}]")


def execute_conversation_state(system, current_state, states):
    """
    Dispatch to the appropriate conversation state handler.
    Takes as input:
        'system': Object holding 'conversation_states' with bound handler methods.
        'current_state': Current state constant/value.
        'states' dictionary: Mapping of state names -> state constants/values.
    """
    # Map state values to handler callables
    state_handlers = {
        states['WELCOME']: system.conversation_states.welcome,
        states['ASK_AREA']: system.conversation_states.ask_area,
        states['ASK_PRICE']: system.conversation_states.ask_price,
        states['ASK_FOOD_TYPE']: system.conversation_states.ask_food_type,
        states['CONFIRM']: system.conversation_states.confirm,
        states['APOLOGIZE']: system.conversation_states.apologize,
        states['SUGGEST_RESTAURANT']: system.conversation_states.suggest_restaurant,
        states['INFORM']: system.conversation_states.inform,
        states['GOODBYE']: system.conversation_states.goodbye,
    }
    
    handler = state_handlers.get(current_state)
    if handler:
        return handler()
    else:
        raise ValueError(f"Unknown state: {current_state}")


def read_data(path, deduplicate: bool = False):
    """
    Reads labeled dialog data from a file and (optionally) deduplicate by utterance.
    Deduplication rule:
    -The first act seen for a given utterance is kept as canonical.
    -If deduplicate=False: duplicates are included but their act is replaced by the first-seen act (ensures label consistency).
    -If deduplicate=True: only the first occurrence of each utterance is kept.

    Takes as input:
    -'path': Input data path.
    -'deduplicate': Whether to drop duplicate utterances.

    Returns tuples of dialogue_acts and  utterances.
    """
    dialogue_act = []
    utterance = []
    utterance_to_first_act = {} 

    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().lower().split(' ', 1)
            
            current_act = data[0]
            current_utterance = data[1]
            
            if current_utterance in utterance_to_first_act:
                act_to_use = utterance_to_first_act[current_utterance]
                if not deduplicate:
                    #Keep duplicates but normalize their act to the canonical one
                    dialogue_act.append(act_to_use)
                    utterance.append(current_utterance)
            else:
                #First time seeing this utterance - record canonical act
                utterance_to_first_act[current_utterance] = current_act
                dialogue_act.append(current_act)
                utterance.append(current_utterance)
                
    return dialogue_act, utterance

def split_and_save_dataset(dialogue_act, utterance, train_path, test_path, test_size=0.15, random_state=42):
    """
    Splits dialog data into train/test and save each split.
    Takes as input:
    -'dialogue_act': Labels.
    -'utterance': Texts.
    -'train_path': Output path for training split (txt).
    -test_path: Output path for test split (txt).
    -test_size: Fraction for test split.
    -random_state: RNG seed.

    Returns:
        A tuple: (train_acts, test_acts, train_utterances, test_utterances)
    """
    train_acts, test_acts, train_utterances, test_utterances = train_test_split(
        dialogue_act, utterance, test_size=test_size, random_state=random_state
    )

    with open(train_path, 'w') as train_file:
        for act, utter in zip(train_acts, train_utterances):
            train_file.write(f"{act} {utter}\n")

    with open(test_path, 'w') as test_file:
        for act, utter in zip(test_acts, test_utterances):
            test_file.write(f"{act} {utter}\n")

    return train_acts, test_acts, train_utterances, test_utterances

def load_data():
    """
    Load, split, and return both original and deduplicated datasets.
    Returns:
        dict: {
            'orig':  (train_acts, test_acts, train_utts, test_utts),
            'dedup': (train_acts, test_acts, train_utts, test_utts)
        }
    """
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
    
    return {
        'orig': (train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig),
        'dedup': (train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup)
    }

def load_trained_model(system_instance):
    """
    LoadS a previously trained MLP model, vectorizer, and label encoder into 'system_instance'.
    Returns True if loaded successfully, False otherwise.
    """
    model_path = system_instance.model_path
    model_files = system_instance.model_files
        
    os.makedirs(model_path, exist_ok=True)
            
    model_file = os.path.join(model_path, model_files['model'])
    vectorizer_file = os.path.join(model_path, model_files['vectorizer'])
    encoder_file = os.path.join(model_path, model_files['label_encoder'])
                
    if all(os.path.exists(f) for f in [model_file, vectorizer_file, encoder_file]):
        with open(model_file, 'rb') as f:
            system_instance.mlp_model = pickle.load(f)
        with open(vectorizer_file, 'rb') as f:
            system_instance.mlp_vectorizer = pickle.load(f)
        with open(encoder_file, 'rb') as f:
            system_instance.mlp_label_encoder = pickle.load(f)
                    
        system_instance.is_trained = True
        print("Pre-trained MLP model loaded successfully from disk")
        return True
    else:
        print("No pre-trained model found. Will need to train new model.")
        return False
                
def save_trained_model(system_instance):
    """
    PersistS the trained MLP model artifacts to disk.
    Returns: True on success.
    """
    model_path = system_instance.model_path
    model_files = system_instance.model_files
        
    os.makedirs(model_path, exist_ok=True)
        
    model_file = os.path.join(model_path, model_files['model'])
    vectorizer_file = os.path.join(model_path, model_files['vectorizer'])
    encoder_file = os.path.join(model_path, model_files['label_encoder'])
        
    with open(model_file, 'wb') as f:
        pickle.dump(system_instance.mlp_model, f)
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(system_instance.mlp_vectorizer, f)
    with open(encoder_file, 'wb') as f:
        pickle.dump(system_instance.mlp_label_encoder, f)
        
    print(f"Model saved successfully to {model_path}")
    return True


def train_classifier(system_instance):
    """
    Train an MLP classifier on the original dataset and save artifacts to 'system_instance'.

    Workflow:
      1) Load data splits (original).
      2) Train MLP via `mlp_classifier(..., return_model=True)`.
      3) Mark system as trained and persist model/vectorizer/encoder to disk.

    Returns: True on success.
    """
    data = load_data()
    train_acts, test_acts, train_utterances, test_utterances = data['orig']

    system_instance.mlp_model, system_instance.mlp_vectorizer, system_instance.mlp_label_encoder = mlp_classifier(
        train_acts, test_acts, train_utterances, test_utterances, return_model=True
    )
        
    system_instance.is_trained = True
    print("MLP classifier trained successfully")
    
    save_trained_model(system_instance)
    return True
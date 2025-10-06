from sklearn.model_selection import train_test_split
import os
import pickle
from ml_models import mlp_classifier
import numpy as np
import pandas as pd


def build_restaurant_info(csv_in="data/restaurant_info.csv", csv_out="data/restaurant_info_updated.csv"):
    """
    Adds random properties to restaurant data for inference testing.
    
    Input: csv_in (str), csv_out (str)
    Output: str (output filename)
    """
    np.random.seed(42)
    df = pd.read_csv(csv_in)

    df["food_quality"] = np.where(np.random.randint(0, 2, df.shape[0]) == 1, "good", "bad")
    df["crowdedness"] = np.where(np.random.randint(0, 2, df.shape[0]) == 1, "busy", "not busy")
    df["length_stay"] = np.where(np.random.randint(0, 2, df.shape[0]) == 1, "long", "short")

    df.to_csv(csv_out, index=False)
    return csv_out


def format_restaurant_info_response(restaurant, user_input):
    """
    Creates natural language response with restaurant details based on user request.
    
    Input: restaurant (dict), user_input (str)
    Output: str (formatted response)
    """
    if not restaurant or not isinstance(restaurant, dict):
        return "I'm sorry, I don't have any restaurant information available to provide details."
    
    utterance_lower = user_input.lower()
    
    request_patterns = {
        'phone': ['phone', 'number'],
        'address': ['address', 'where', 'location'],
        'postcode': ['postcode', 'post code', 'postal'],
        'food': ['food', 'cuisine', 'serve', 'food type', 'type of food'],
        'area': ['area', 'part of town', 'where is it', 'which area'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'price range']
    }
    
    requests = {}
    for req_type, patterns in request_patterns.items():
        found = False
        for pattern in patterns:
            if pattern in utterance_lower:
                found = True
                break
        requests[req_type] = found
    
    if any(requests.values()):
        return _build_specific_info_response(restaurant, requests)
    else:
        return _build_default_info_response(restaurant)


def _build_specific_info_response(restaurant, requests):
    """
    Input: restaurant (dict), requests (dict)
    Output: str (response with requested info)
    """
    name = restaurant.get('restaurantname', 'the restaurant')
    info_parts = []
    
    response_builders = {
        'phone': lambda: _format_phone_response(restaurant, name),
        'address': lambda: _format_address_response(restaurant, name),
        'postcode': lambda: _format_postcode_response(restaurant, name),
        'food': lambda: f"{name} serves {restaurant.get('food', 'unknown')} food",
        'area': lambda: _format_area_response(restaurant, name),
        'price': lambda: f"{name} is in the {restaurant.get('pricerange', 'unknown')} price range"
    }
    
    for req_type, is_requested in requests.items():
        if is_requested and req_type in response_builders:
            info_parts.append(response_builders[req_type]())
    
    return _format_response_parts(info_parts)


def _format_phone_response(restaurant, name):
    """
    Input: restaurant (dict), name (str)
    Output: str
    """
    phone = restaurant.get('phone')
    if _is_valid_data(phone):
        return f"The phone number of {name} is {phone}"
    return f"I'm sorry, the phone number for {name} is not available"


def _format_address_response(restaurant, name):
    """
    Input: restaurant (dict), name (str)
    Output: str
    """
    addr = restaurant.get('addr')
    if _is_valid_data(addr):
        return f"Sure, {name} is on {addr}"
    return f"I'm sorry, the address for {name} is not available"


def _format_postcode_response(restaurant, name):
    """
    Input: restaurant (dict), name (str)
    Output: str
    """
    postcode = restaurant.get('postcode')
    if _is_valid_data(postcode):
        return f"The post code of {name} is {postcode}"
    return f"I'm sorry, the post code for {name} is not available"


def _format_area_response(restaurant, name):
    """
    Input: restaurant (dict), name (str)
    Output: str
    """
    area = restaurant.get('area', 'unknown')
    if area == 'centre':
        return f"{name} is in the city centre"
    return f"{name} is in the {area} part of town"


def _build_default_info_response(restaurant):
    """
    Input: restaurant (dict)
    Output: str (response with all info)
    """
    name = restaurant.get('restaurantname', 'the restaurant')
    food = restaurant.get('food', 'unknown')
    area = restaurant.get('area', 'unknown')
    price = restaurant.get('pricerange', 'unknown')
    
    response_parts = [f"{name} serves {food} food"]
    
    if area == 'centre':
        response_parts.append("it is in the city centre")
    else:
        response_parts.append(f"it is in the {area} part of town")
    
    response_parts.append(f"in the {price} price range")
    
    phone = restaurant.get('phone')
    addr = restaurant.get('addr')
    postcode = restaurant.get('postcode')
    
    if _is_valid_data(phone):
        response_parts.append(f"the phone number is {phone}")
    if _is_valid_data(addr):
        response_parts.append(f"the address is {addr}")
    if _is_valid_data(postcode):
        response_parts.append(f"the post code is {postcode}")
    
    return _format_response_parts(response_parts)


def _is_valid_data(value):
    """
    Input: value (any)
    Output: bool
    """
    return (value is not None and 
            str(value).strip() and 
            str(value).lower() not in ['nan', 'none', 'not available'] and 
            str(value) != 'dontcare')


def _format_response_parts(parts):
    """
    Input: parts (list of str)
    Output: str (formatted sentence)
    """
    if not parts:
        return "I'm sorry, I don't have that information available."
    
    if len(parts) == 1:
        return f"{parts[0]}."
    elif len(parts) == 2:
        return f"{parts[0]} and {parts[1].lower()}."
    elif len(parts) == 3:
        return f"{parts[0]}, {parts[1].lower()}, and {parts[2].lower()}."
    else:
        basic_info = f"{parts[0]}, {parts[1].lower()}, and {parts[2].lower()}"
        if len(parts) == 4:
            return f"{basic_info}. {parts[3]}."
        else:
            contact_info = ", ".join(parts[3:-1]) + f", and {parts[-1]}"
            return f"{basic_info}. {contact_info}."


def format_restaurant_suggestion(restaurant, user_requirements=None):
    """
    Creates a restaurant recommendation with inference explanations.
    
    Input: restaurant (dict), user_requirements (dict or None)
    Output: str (formatted suggestion)
    """
    if not restaurant:
        return None
    
    name = restaurant['restaurantname']
    area = restaurant['area']
    pricerange = restaurant['pricerange']
    food = restaurant['food']
    
    area_desc = " in the city centre" if area == 'centre' else f" in the {area} of town"
    main_desc = f"I recommend '{name}', it is {pricerange} {food} restaurant{area_desc}"
    
    inference_parts = []
    
    # Prioritize conflict resolution over original inference
    if 'conflict_resolution' in restaurant:
        conflict_text = restaurant['conflict_resolution'].strip()
        inference_parts.append(conflict_text)
    elif 'inference_result' in restaurant:
        from inference_engine import InferenceEngine
        engine = InferenceEngine()
        full_explanation = engine.explain_recommendation(restaurant)
        
        if user_requirements and full_explanation:
            filtered = _filter_explanations_by_requirements(
                full_explanation,
                restaurant.get('inference_result', {}).get('inferred_properties', {}),
                user_requirements
            )
            if filtered:
                inference_parts.append(filtered.strip())
    
    if not inference_parts:
        return main_desc + "."
    
    formatted_parts = []
    for part in inference_parts:
        if part and not part[0].isupper():
            formatted_parts.append(part[0].upper() + part[1:])
        else:
            formatted_parts.append(part)
    
    if len(formatted_parts) == 1:
        return f"{main_desc}. {formatted_parts[0]}."
    else:
        last = formatted_parts[-1]
        rest = ". ".join(formatted_parts[:-1])
        return f"{main_desc}. {rest}. {last}."


def _filter_explanations_by_requirements(full_explanation, inferred_properties, user_requirements):
    """
    Input: full_explanation (str), inferred_properties (dict), user_requirements (dict)
    Output: str (filtered explanation)
    """
    sentences = [s.strip() for s in full_explanation.replace(". ", ".").split(".") if s.strip()]
    
    property_patterns = {
        'touristic': {
            'positive': ['popular with tourists'],
            'negative': ['not typically visited by tourists', 'romanian cuisine']
        },
        'assigned_seats': {
            'positive': ['assigned seating', 'gets busy'],
            'negative': []
        },
        'children': {
            'positive': [],
            'negative': ['not ideal for children', 'stay for a long time']
        },
        'romantic': {
            'positive': ['is romantic', 'leisurely meal', 'take your time'],
            'negative': ['not romantic', 'busy and noisy']
        }
    }
    
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        for prop, pattern_dict in property_patterns.items():
            if prop in user_requirements and user_requirements[prop] is not None:
                required_value = user_requirements[prop]
                inferred_value = inferred_properties.get(prop)
                
                # Only include sentences that match the user's requirement AND the resolved value
                if inferred_value == required_value:
                    # Check if sentence matches the positive/negative pattern for the required value
                    if required_value:  # User wants property = True
                        if any(pattern in sentence_lower for pattern in pattern_dict['positive']):
                            if sentence[0].isupper() and sentence.lower().startswith('this'):
                                sentence = sentence[0].lower() + sentence[1:]
                            relevant_sentences.append(sentence)
                            break
                    else:  # User wants property = False
                        if any(pattern in sentence_lower for pattern in pattern_dict['negative']):
                            if sentence[0].isupper() and sentence.lower().startswith('this'):
                                sentence = sentence[0].lower() + sentence[1:]
                            relevant_sentences.append(sentence)
                            break
    
    return ", and ".join(relevant_sentences) if relevant_sentences else ""


def detect_restart_command(user_input):
    """
    Checks if user wants to restart the conversation.
    
    Input: user_input (str)
    Output: bool
    """
    restart_keywords = ['start over', 'start again', 'reset', 'restart', 'begin again', 'new search']
    input_lower = user_input.lower().strip()
    return any(keyword in input_lower for keyword in restart_keywords)


def detect_exit_command(user_input):
    """
    Checks if user wants to exit the system.
    
    Input: user_input (str)
    Output: bool
    """
    exit_keywords = ['exit', 'quit', 'bye', 'goodbye', 'stop']
    input_lower = user_input.lower().strip()
    return input_lower in exit_keywords


def detect_new_search_request(user_input):
    """
    Detects if user is starting a new restaurant search.
    
    Input: user_input (str)
    Output: bool
    """
    new_search_indicators = ['i want', 'i would like', 'looking for', 'find me', 'search for', 'how about']
    input_lower = user_input.lower().strip()
    return any(indicator in input_lower for indicator in new_search_indicators)


def get_state_name_from_value(states_dict, state_value):
    """
    Finds state name by its value in the states dictionary.
    
    Input: states_dict (dict), state_value (str)
    Output: str or None
    """
    for name, value in states_dict.items():
        if value == state_value:
            return name
    return None


def update_preferences_with_context(user_requirements, validated_prefs, context_stage):
    """
    Updates user preferences based on current conversation context.
    
    Input: user_requirements (dict), validated_prefs (dict), context_stage (str or None)
    Output: None (modifies user_requirements in place)
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
            for pref_type, value in validated_prefs.items():
                if user_requirements[pref_type] is None:
                    user_requirements[pref_type] = value
    else:
        for pref_type, value in validated_prefs.items():
            if user_requirements[pref_type] is None:
                user_requirements[pref_type] = value


def log_preference_changes(validated_prefs, user_requirements, old_prefs, errors):
    """
    Prints debug information about preference extraction and updates.
    
    Input: validated_prefs (dict), user_requirements (dict), old_prefs (dict), errors (list)
    Output: None (prints to console)
    """
    if errors:
        print(f"[Validation warnings: {', '.join(errors)}]")
    
    if validated_prefs:
        print(f"[Extracted: {validated_prefs}]")
    
    if user_requirements != old_prefs:
        print(f"[Preferences updated: {user_requirements}]")


def execute_conversation_state(system, current_state, states):
    """
    Routes to appropriate conversation state handler function.
    
    Input: system (RestaurantSystem), current_state (str), states (dict)
    Output: str (next state)
    """
    state_handlers = {
        states['WELCOME']: system.conversation_states.welcome,
        states['ASK_AREA']: system.conversation_states.ask_area,
        states['ASK_PRICE']: system.conversation_states.ask_price,
        states['ASK_FOOD_TYPE']: system.conversation_states.ask_food_type,
        states['CONFIRM']: system.conversation_states.confirm,
        states['ASK_ADDITIONAL_REQUIREMENTS']: system.conversation_states.ask_additional_requirements,
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


def read_data(path, deduplicate=False):
    """
    Loads dialog acts and utterances from file with optional deduplication.
    
    Input: path (str), deduplicate (bool)
    Output: tuple (dialogue_act: list, utterance: list)
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
                    dialogue_act.append(act_to_use)
                    utterance.append(current_utterance)
            else:
                utterance_to_first_act[current_utterance] = current_act
                dialogue_act.append(current_act)
                utterance.append(current_utterance)
                
    return dialogue_act, utterance


def split_and_save_dataset(dialogue_act, utterance, train_path, test_path, test_size=0.15, random_state=42):
    """
    Splits data into train/test sets and saves to files.
    
    Input: dialogue_act (list), utterance (list), train_path (str), test_path (str), test_size (float), random_state (int)
    Output: tuple (train_acts, test_acts, train_utterances, test_utterances)
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
    Loads and splits both original and deduplicated datasets.
    
    Input: None
    Output: dict with keys 'orig' and 'dedup', each containing tuple (train_acts, test_acts, train_utts, test_utts)
    """
    print("Loading data...")

    acts_orig, utterances_orig = read_data("data/dialog_acts.dat", deduplicate=False)
    train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig = split_and_save_dataset(
        acts_orig, utterances_orig, "data/train_orig.txt", "data/test_orig.txt"
    )
    
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
    Loads pre-trained MLP model from disk if available.
    
    Input: system_instance (RestaurantSystem)
    Output: bool (True if loaded successfully)
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
    Saves trained MLP model components to disk.
    
    Input: system_instance (RestaurantSystem)
    Output: bool (True on success)
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
    Trains a new MLP classifier and saves it to disk.
    
    Input: system_instance (RestaurantSystem)
    Output: bool (True on success)
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
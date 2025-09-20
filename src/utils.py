from sklearn.model_selection import train_test_split

def format_restaurant_info_response(restaurant, user_input):
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
    if not restaurant:
        return None
        
    area_desc = f" in the {restaurant['area']} of town" if restaurant['area'] != 'centre' else " in the city centre"
    price_desc = f" and the prices are {restaurant['pricerange']}" if restaurant['pricerange'] != 'dontcare' else ""
    
    return f"{restaurant['restaurantname']} is a nice place{area_desc}{price_desc}"


def detect_restart_command(user_input):
    return user_input.lower().strip() in ['start over', 'start again', 'reset']


def detect_new_search_request(user_input):
    input_lower = user_input.lower()
    search_keywords = ['is there', 'do you have', 'find me', 'looking for', 'want', 'need']
    return any(keyword in input_lower for keyword in search_keywords)


def get_state_name_from_value(states_dict, state_value): 
    for name, value in states_dict.items():
        if value == state_value:
            return name
    return None


def is_dontcare_response(user_input):
    dontcare_phrases = [
        'any', 'anything', "doesn't matter", "dont care", 
        "any will do", "i dont care", "any type", "any food"
    ]
    return user_input.strip().lower() in dontcare_phrases


def handle_dontcare_preference(user_requirements, context_stage, old_prefs):
    if context_stage == 'ASK_AREA':
        user_requirements['area'] = 'dontcare'
        print(f"[Extracted: {{'area': 'dontcare'}}]")
    elif context_stage == 'ASK_PRICE':
        user_requirements['pricerange'] = 'dontcare'
        print(f"[Extracted: {{'pricerange': 'dontcare'}}]")
    elif context_stage == 'ASK_FOOD_TYPE':
        user_requirements['food'] = 'dontcare'
        print(f"[Extracted: {{'food': 'dontcare'}}]")
    else:
        return False
    
    if user_requirements != old_prefs:
        print(f"[Preferences updated: {user_requirements}]")
    return True


def update_preferences_with_context(user_requirements, validated_prefs, context_stage):
    if context_stage:
        stage_mapping = {
            'ASK_AREA': 'area',
            'ASK_PRICE': 'pricerange', 
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
    if errors:
        print(f"[Validation warnings: {', '.join(errors)}]")
    
    if validated_prefs:
        print(f"[Extracted: {validated_prefs}]")
    
    if user_requirements != old_prefs:
        print(f"[Preferences updated: {user_requirements}]")


def execute_conversation_state(system, current_state, states):
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

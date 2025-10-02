"""
Dialog State Management for Restaurant Recommendation System

Input: User utterances and system state
Output: Next dialog state
"""

from utils import format_restaurant_suggestion, detect_restart_command, detect_exit_command
from preference_extraction import PreferenceExtractor


class InputValidator:
    """Validates user input and handles special commands."""
    
    @staticmethod
    def validate_and_handle_special_commands(user_input, system):
        """
        Input: user_input (str), system (RestaurantSystem)
        Output: tuple (should_continue: bool, next_state: str or None)
        """
        if detect_exit_command(user_input):
            return (False, system.states['GOODBYE'])
        
        if system.allow_restarts and detect_restart_command(user_input):
            InputValidator.reset_system_state(system)
            return (False, system.states['WELCOME'])
        
        return (True, None)
    
    @staticmethod
    def reset_system_state(system):
        """
        Input: system (RestaurantSystem)
        Output: None (modifies system in place)
        """
        system.user_requirements = {'area': None, 'price': None, 'food': None}
        system.additional_requirements = {
            'touristic': None, 'assigned_seats': None,
            'children': None, 'romantic': None
        }
        
        if hasattr(system, '_inference_applied'):
            delattr(system, '_inference_applied')
        if hasattr(system, '_handling_conflict_restaurant'):
            delattr(system, '_handling_conflict_restaurant')
        
        system.romantic_conflicts = []
        system.touristic_conflicts = []


class ConflictResolver:
    """Resolves inference rule conflicts through user interaction."""
    
    CONFLICT_CONFIGS = {
        'romantic': {
            'questions': [
                "I found '{name}', but I need your help to determine if it's romantic.",
                "This restaurant is both busy (less romantic) and allows long stays (more romantic).",
                "Do you prefer a quiet atmosphere or the ability to stay for a long time?",
                "Say 'quiet' for peaceful setting, or 'long stay' for leisurely meals."
            ],
            'preferences': {
                'quiet': {
                    'keywords': ['quiet', 'peaceful', 'calm', 'not busy'],
                    'value': False,
                    'explanation': "Since you prefer a quiet atmosphere, {name} is not romantic due to being busy"
                },
                'long_stay': {
                    'keywords': ['long', 'stay', 'time', 'leisurely'],
                    'value': True,
                    'explanation': "Since you value long meals, {name} is romantic because you can take your time"
                }
            },
            'default': 'long_stay'
        },
        'touristic': {
            'questions': [
                "I found '{name}', but I need your help to determine if it's touristic.",
                "This restaurant offers good food at cheap prices (attracts tourists) but serves Romanian cuisine (unfamiliar to tourists).",
                "What's more important - good value for money or familiar cuisine?",
                "Say 'value' for good prices, or 'familiar' for well-known cuisines."
            ],
            'preferences': {
                'value': {
                    'keywords': ['value', 'price', 'cheap', 'affordable', 'good food', 'quality'],
                    'value': True,
                    'explanation': "Since you prioritize value, {name} is touristic due to its affordable quality"
                },
                'familiar': {
                    'keywords': ['familiar', 'cuisine', 'food type', 'romanian', 'unfamiliar'],
                    'value': False,
                    'explanation': "Since you prefer familiar cuisine, {name} is not touristic (Romanian food is unfamiliar)"
                }
            },
            'default': 'value'
        }
    }
    
    @classmethod
    def present_conflict(cls, restaurant_name, conflict_type):
        """
        Input: restaurant_name (str), conflict_type (str)
        Output: list of str (questions to ask user)
        """
        config = cls.CONFLICT_CONFIGS[conflict_type]
        return [q.format(name=restaurant_name) for q in config['questions']]
    
    @classmethod
    def resolve(cls, user_input, restaurant, conflict_type, user_requirement, allow_restarts=False):
        """
        Input: user_input (str), restaurant (dict), conflict_type (str), user_requirement (bool), allow_restarts (bool)
        Output: tuple (action: str, data: dict or str or None)
        """
        config = cls.CONFLICT_CONFIGS[conflict_type]
        user_input_lower = user_input.lower()
        
        if any(kw in user_input_lower for kw in ['exit', 'quit', 'bye']):
            return ('exit', None)
        if allow_restarts and any(kw in user_input_lower for kw in ['restart', 'start over', 'reset']):
            return ('restart', None)
        if any(kw in user_input_lower for kw in ['alternative', 'different', 'skip', 'next']):
            return ('skip', None)
        
        matched_pref = cls._match_preference(user_input_lower, config)
        pref_data = config['preferences'][matched_pref]
        resolved_value = pref_data['value']
        explanation = pref_data['explanation'].format(name=restaurant['restaurantname'])
        
        if resolved_value == user_requirement:
            cls._update_restaurant_with_resolution(restaurant, conflict_type, resolved_value, explanation)
            return ('recommend', restaurant)
        else:
            reject_msg = f"Based on your preference, {restaurant['restaurantname']} doesn't meet your {conflict_type} requirement."
            return ('reject', reject_msg)
    
    @classmethod
    def _match_preference(cls, user_input_lower, config):
        """
        Input: user_input_lower (str), config (dict)
        Output: str (matched preference name)
        """
        for pref_name, pref_data in config['preferences'].items():
            if any(kw in user_input_lower for kw in pref_data['keywords']):
                return pref_name
        return config['default']
    
    @classmethod
    def _update_restaurant_with_resolution(cls, restaurant, conflict_type, resolved_value, explanation):
        """
        Input: restaurant (dict), conflict_type (str), resolved_value (bool), explanation (str)
        Output: None (modifies restaurant in place)
        """
        restaurant[conflict_type] = resolved_value
        restaurant['conflict_resolution'] = explanation
        
        updated_result = restaurant.get('inference_result', {}).copy()
        if updated_result:
            updated_result['inferred_properties'][conflict_type] = resolved_value
            updated_result['has_conflict'] = False
            updated_result['conflict_type'] = None
            # Clear original reasoning to prevent conflicting explanations
            updated_result['reasoning'] = []
            restaurant['inference_result'] = updated_result


class ConversationStates:
    """Manages dialog flow through discrete states."""
    
    def __init__(self, system):
        self.system = system
    
    def welcome(self):
        """
        Input: None (uses system state)
        Output: str (next state)
        """
        msg = "Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?"
        print(f"System: {self.system.format_output(msg)}")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        print(f"\n[Classified as: {user_intent}]")
        
        should_continue, next_state = InputValidator.validate_and_handle_special_commands(user_input, self.system)
        if not should_continue:
            return next_state
        
        if user_intent in ['bye']:
            return self.system.states['GOODBYE']
        
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower())
            if parse_result == 'restart' and self.system.allow_restarts:
                return self.system.states['WELCOME']
        
        return self.system.check_next_stage()
    
    def ask_area(self):
        """
        Input: None
        Output: str (next state)
        """
        messages = [
            'What part of town do you have in mind?',
            'Please specify: north, south, east, west, or centre (you can also say any area or do not care).'
        ]
        return self._handle_preference_question(messages, 'ASK_AREA')
    
    def ask_price(self):
        """
        Input: None
        Output: str (next state)
        """
        messages = [
            "Would you like something in the cheap, moderate, or expensive price range?",
            "Please specify: cheap, moderate, expensive (you can also say 'any price' or 'don't care')."
        ]
        return self._handle_preference_question(messages, 'ASK_PRICE')
    
    def ask_food_type(self):
        """
        Input: None
        Output: str (next state)
        """
        messages = [
            "What kind of food would you like?",
            "Please specify a cuisine type (e.g., italian, chinese, indian, british, french, etc.) or say 'any food' if you don't mind."
        ]
        return self._handle_preference_question(messages, 'ASK_FOOD_TYPE')
    
    def _handle_preference_question(self, messages, context):
        """
        Input: messages (list of str), context (str)
        Output: str (next state)
        """
        for msg in messages:
            print(f"System: {self.system.format_output(msg)}")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        print(f"\n[Classified as: {user_intent}]")
        
        should_continue, next_state = InputValidator.validate_and_handle_special_commands(user_input, self.system)
        if not should_continue:
            return next_state
        
        if user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        if user_intent in ['inform']:
            self.system.parse_user_input(user_input.lower(), context)
        
        return self.system.check_next_stage()
    
    def ask_additional_requirements(self):
        """
        Input: None
        Output: str (next state)
        """
        if not self.system.alternatives and not self.system.current_restaurant:
            self.system.search_restaurants()
        
        add_req_msg = "Do you have any additional requirements? For example, would you like the restaurant to be touristic, romantic, child-friendly, or have assigned seats? You can say 'yes' and specify requirements, or 'no' if you don't have any additional preferences."
        print(f"System: {self.system.format_output(add_req_msg)}")
        
        user_input = self.system.get_user_input("User: ")
        
        if not user_input or not user_input.strip():
            return self._prompt_for_additional_requirements()
        
        user_intent = self.system.classify_utterance(user_input)
        print(f"\n[Classified as: {user_intent}]")
        
        should_continue, next_state = InputValidator.validate_and_handle_special_commands(user_input, self.system)
        if not should_continue:
            return next_state
        
        if user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        if user_intent in ['negate', 'deny'] or user_input.lower() in ['no', 'nope', 'none']:
            return self.system.states['CONFIRM']
        
        additional_reqs = self.parse_additional_requirements(user_input.lower())
        
        if additional_reqs:
            self.system.additional_requirements.update(additional_reqs)
            return self.system.states['CONFIRM']
        
        if user_intent in ['affirm'] or user_input.lower() in ['yes', 'yeah', 'yep']:
            return self._handle_affirmative_additional_requirements()
        
        if self._has_requirement_keywords(user_input.lower()):
            additional_reqs = self.parse_additional_requirements(user_input.lower())
            if additional_reqs:
                self.system.additional_requirements.update(additional_reqs)
                return self.system.states['CONFIRM']
        
        return self._prompt_for_additional_requirements()
    
    def _has_requirement_keywords(self, user_input):
        """
        Input: user_input (str)
        Output: bool
        """
        keywords = ['romantic', 'touristic', 'child', 'family', 'seat']
        return any(word in user_input for word in keywords)
    
    def _prompt_for_additional_requirements(self):
        """
        Input: None
        Output: str (next state)
        """
        clarify_msg = "I didn't understand. Please let me know if you have any specific requirements like romantic atmosphere, child-friendly environment, etc., or say 'no' if you don't have additional preferences."
        print(f"System: {self.system.format_output(clarify_msg)}")
        return self.system.states['ASK_ADDITIONAL_REQUIREMENTS']
    
    def _handle_affirmative_additional_requirements(self):
        """
        Input: None
        Output: str (next state)
        """
        clarify = "What specific requirements do you have in mind? For example, romantic, touristic, child-friendly?"
        print(f"System: {self.system.format_output(clarify)}")
        
        follow_up_input = self.system.get_user_input("User: ")
        
        if not follow_up_input or not follow_up_input.strip():
            return self._prompt_for_additional_requirements()
        
        should_continue, next_state = InputValidator.validate_and_handle_special_commands(follow_up_input, self.system)
        if not should_continue:
            return next_state
        
        follow_up_reqs = self.parse_additional_requirements(follow_up_input.lower())
        
        if follow_up_reqs:
            self.system.additional_requirements.update(follow_up_reqs)
            return self.system.states['CONFIRM']
        elif follow_up_input.lower() in ['no', 'none', 'nothing', 'nope']:
            return self.system.states['CONFIRM']
        
        return self._prompt_for_additional_requirements()
    
    def confirm(self):
        """
        Input: None
        Output: str (next state)
        """
        confirmation_msg = self._build_confirmation_message()
        print(f"System: {self.system.format_output(confirmation_msg)}")
        print(f"System: {self.system.format_output('Please answer yes to confirm or no to change your preferences.')}")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        print(f"\n[Classified as: {user_intent}]")
        
        should_continue, next_state = InputValidator.validate_and_handle_special_commands(user_input, self.system)
        if not should_continue:
            return next_state
        
        if user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        if user_intent == 'affirm':
            if not self.system.current_restaurant:
                self.system.search_restaurants()
                if not self.system.current_restaurant:
                    return self.system.states['APOLOGIZE']
            return self.system.states['SUGGEST_RESTAURANT']
        
        elif user_intent == 'negate':
            change_msg = "No problem! Let me help you find a different restaurant."
            print(f"System: {self.system.format_output(change_msg)}")
            InputValidator.reset_system_state(self.system)
            return self.system.states['ASK_AREA']
        
        return self.system.states['CONFIRM']
    
    def _build_confirmation_message(self):
        """
        Input: None
        Output: str (confirmation message)
        """
        prefs = []
        
        if self.system.user_requirements['area'] and self.system.user_requirements['area'] != 'dontcare':
            prefs.append(f"in the {self.system.user_requirements['area']} of town")
        if self.system.user_requirements['price']:
            if self.system.user_requirements['price'] != 'dontcare':
                prefs.append(f"in the {self.system.user_requirements['price']} price range")
            else:
                prefs.append("in any price range")
        if self.system.user_requirements['food'] and self.system.user_requirements['food'] != 'dontcare':
            prefs.append(f"serving {self.system.user_requirements['food']} food")
        
        additional_prefs = self._build_additional_preferences_text()
        
        if additional_prefs:
            return f"You are looking for a restaurant {' '.join(prefs)} {' and '.join(additional_prefs)}, right?"
        else:
            return f"You are looking for a restaurant {' '.join(prefs)}, right?"
    
    def _build_additional_preferences_text(self):
        """
        Input: None
        Output: list of str
        """
        additional_prefs = []
        
        req_map = {
            'touristic': ("that is touristic", "that is not touristic"),
            'romantic': ("that is romantic", "that is not romantic"),
            'children': ("that is child-friendly", "that is not suitable for children"),
            'assigned_seats': ("with assigned seating", "where you can choose your own seats")
        }
        
        for req_key, (pos_text, neg_text) in req_map.items():
            req_value = self.system.additional_requirements.get(req_key)
            if req_value is not None:
                additional_prefs.append(pos_text if req_value else neg_text)
        
        return additional_prefs
    
    def suggest_restaurant(self):
        """
        Input: None
        Output: str (next state)
        """
        if not self.system.current_restaurant:
            no_rest_msg = "I'm sorry but there is no restaurant serving that type of food"
            print(f"System: {self.system.format_output(no_rest_msg)}")
            return self.system.states['APOLOGIZE']
        
        if self._should_apply_inference_filtering():
            self.system.apply_inference_filtering()
            self.system._inference_applied = True
            
            if not self.system.current_restaurant:
                return self.system.states['APOLOGIZE']
        
        restaurant = self.system.current_restaurant
        conflict_type = self._get_active_conflict(restaurant)
        
        if conflict_type:
            resolution_action = self._handle_conflict(restaurant, conflict_type)
            
            if resolution_action == 'exit':
                self.system.conversation_ended = True
                return None
            elif resolution_action == 'restart':
                # Handle restart during conflict resolution only if restarts are enabled
                if self.system.allow_restarts:
                    from utils import detect_restart_command
                    self.system.parse_user_input('restart')
                    return self.system.states['WELCOME']
                else:
                    # If restarts are disabled, treat as unrecognized input and continue with conflict resolution
                    print(f"System: {self.system.format_output('I did not understand. Please choose one of the options provided.')}")
                    return self._try_next_restaurant()
            elif resolution_action in ['skip', 'reject']:
                return self._try_next_restaurant()
            elif resolution_action == 'recommend':
                restaurant = self.system.current_restaurant
        
        if 'inference_result' not in restaurant:
            inference_result = self.system.inference_engine.apply_rules(restaurant)
            restaurant['inference_result'] = inference_result
        
        suggestion_msg = format_restaurant_suggestion(restaurant, self.system.additional_requirements)
        print(f"System: {self.system.format_output(suggestion_msg)}")
        
        return self.system.states['INFORM']
    
    def _should_apply_inference_filtering(self):
        """
        Input: None
        Output: bool
        """
        has_additional_reqs = any(v is not None for v in self.system.additional_requirements.values())
        not_already_applied = not hasattr(self.system, '_inference_applied')
        return has_additional_reqs and not_already_applied
    
    def _get_active_conflict(self, restaurant):
        """
        Input: restaurant (dict)
        Output: str or None ('romantic', 'touristic', or None)
        """
        if hasattr(self.system, '_handling_conflict_restaurant'):
            conflict_type = self.system._handling_conflict_restaurant
            if self.system.additional_requirements.get(conflict_type) is not None:
                return conflict_type
        
        for conflict_type in ['romantic', 'touristic']:
            if self.system.additional_requirements.get(conflict_type) is None:
                continue
            
            conflicts = getattr(self.system, f'{conflict_type}_conflicts', [])
            if any(c['restaurantname'] == restaurant['restaurantname'] for c in conflicts):
                return conflict_type
        
        return None
    
    def _handle_conflict(self, restaurant, conflict_type):
        """
        Input: restaurant (dict), conflict_type (str)
        Output: str ('recommend', 'reject', 'skip', 'restart', or 'exit')
        """
        messages = ConflictResolver.present_conflict(restaurant['restaurantname'], conflict_type)
        for msg in messages:
            print(f"System: {self.system.format_output(msg)}")
        
        user_input = self.system.get_user_input("User: ")
        user_requirement = self.system.additional_requirements[conflict_type]
        action, data = ConflictResolver.resolve(user_input, restaurant, conflict_type, user_requirement, self.system.allow_restarts)
        
        if hasattr(self.system, '_handling_conflict_restaurant'):
            delattr(self.system, '_handling_conflict_restaurant')
        
        if action == 'recommend':
            self.system.current_restaurant = data
            return 'recommend'
        elif action == 'reject':
            print(f"System: {self.system.format_output(data)}")
            return 'reject'
        elif action == 'restart':
            return 'restart'
        elif action == 'exit':
            print(f"System: {self.system.format_output('Thank you for using the restaurant system. Goodbye!')}")
            return 'exit'
        else:
            print(f"System: {self.system.format_output('I understand you want to skip this restaurant. Let me find another option for you.')}")
            return 'skip'
    
    def _try_next_restaurant(self):
        """
        Input: None
        Output: str (next state)
        """
        next_restaurant = self._get_next_alternative()
        
        if next_restaurant:
            self.system.current_restaurant = next_restaurant
            self.system.current_restaurant_name = next_restaurant['restaurantname']
            return self.system.states['SUGGEST_RESTAURANT']
        
        return self.system.states['APOLOGIZE']
    
    def _get_next_alternative(self):
        """
        Input: None
        Output: dict or None (next restaurant)
        """
        if hasattr(self.system, '_handling_conflict_restaurant'):
            return self._get_next_conflict_restaurant()
        
        if self.system.alternatives and self.system.suggestion_index < len(self.system.alternatives):
            return self._get_next_regular_alternative()
        
        return self._get_first_conflict_restaurant()
    
    def _get_next_conflict_restaurant(self):
        """
        Input: None
        Output: dict or None
        """
        conflict_type = self.system._handling_conflict_restaurant
        conflicts = getattr(self.system, f'{conflict_type}_conflicts', [])
        
        if conflicts:
            return conflicts.pop(0)
        
        other_type = 'touristic' if conflict_type == 'romantic' else 'romantic'
        other_conflicts = getattr(self.system, f'{other_type}_conflicts', [])
        
        if other_conflicts:
            self.system._handling_conflict_restaurant = other_type
            return other_conflicts.pop(0)
        
        return None
    
    def _get_next_regular_alternative(self):
        """
        Input: None
        Output: dict or None
        """
        alt_restaurant_dict = self.system.alternatives[self.system.suggestion_index]
        self.system.suggestion_index += 1
        
        if not isinstance(alt_restaurant_dict, dict):
            alt_restaurant_dict = self.system.restaurant_lookup.find_restaurant_by_name(alt_restaurant_dict)
        
        return alt_restaurant_dict
    
    def _get_first_conflict_restaurant(self):
        """
        Input: None
        Output: dict or None
        """
        for conflict_type in ['romantic', 'touristic']:
            conflicts = getattr(self.system, f'{conflict_type}_conflicts', [])
            if conflicts:
                self.system._handling_conflict_restaurant = conflict_type
                return conflicts.pop(0)
        
        return None
    
    def parse_additional_requirements(self, user_input):
        """
        Input: user_input (str)
        Output: dict (extracted requirements)
        """
        requirements = {}
        
        touristic_patterns = {
            True: ['tourist', 'touristic', 'popular', 'famous'],
            False: ['not tourist', 'local', 'hidden', 'authentic']
        }
        
        romantic_patterns = {
            True: ['romantic', 'romance', 'intimate', 'cozy', 'date'],
            False: ['not romantic', 'casual', 'business']
        }
        
        children_patterns = {
            True: ['child', 'children', 'kid', 'family', 'child-friendly'],
            False: ['no child', 'adults only', 'quiet']
        }
        
        seats_patterns = {
            True: ['assigned seat', 'assigned seats', 'seat assignment', 'waiter choose'],
            False: ['choose seat', 'pick seat', 'free seating']
        }
        
        for value, patterns in touristic_patterns.items():
            if any(word in user_input for word in patterns):
                requirements['touristic'] = value
                break
        
        for value, patterns in romantic_patterns.items():
            if any(word in user_input for word in patterns):
                requirements['romantic'] = value
                break
        
        for value, patterns in children_patterns.items():
            if any(word in user_input for word in patterns):
                requirements['children'] = value
                break
        
        for value, patterns in seats_patterns.items():
            if any(word in user_input for word in patterns):
                requirements['assigned_seats'] = value
                break
        
        return requirements
    
    def apologize(self):
        """
        Input: None
        Output: str (next state)
        """
        sorry_msg = "I'm sorry, no restaurants were found matching your criteria. Let's try a new search with different preferences."
        print(f"System: {self.system.format_output(sorry_msg)}")
        
        InputValidator.reset_system_state(self.system)
        return self.system.states['ASK_AREA']
    
    def inform(self):
        """
        Input: None
        Output: str (next state)
        """
        self._print_inform_prompt()
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        print(f"\n[Classified as: {user_intent}]")
        
        should_continue, next_state = InputValidator.validate_and_handle_special_commands(user_input, self.system)
        if not should_continue:
            return next_state
        
        if user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        if self.system.current_restaurant:
            return self._handle_current_restaurant_requests(user_input, user_intent)
        else:
            return self._handle_no_restaurant_requests(user_intent, user_input)
    
    def _print_inform_prompt(self):
        """
        Input: None
        Output: None
        """
        if self.system.current_restaurant:
            info_msg = "Would you like more information about the restaurant (phone, address), an alternative restaurant, or would you like to try a different search?"
            print(f"System: {self.system.format_output(info_msg)}")
            
            guidance_msg = "You can ask for 'phone', 'address', say 'alternative' for other options, specify new preferences, or say 'exit' to leave"
            if self.system.allow_restarts:
                guidance_msg += " or 'restart' to start over."
            else:
                guidance_msg += "."
            print(f"System: {self.system.format_output(guidance_msg)}")
        else:
            try_diff_msg = "Would you like to try a different type of food or change your preferences?"
            print(f"System: {self.system.format_output(try_diff_msg)}")
            
            restart_msg = "You can say 'yes' to try again, specify new preferences (e.g., 'chinese food'), or say 'exit' to leave"
            if self.system.allow_restarts:
                restart_msg += " or 'restart' to start over."
            else:
                restart_msg += "."
            print(f"System: {self.system.format_output(restart_msg)}")
    
    def _handle_current_restaurant_requests(self, user_input, user_intent):
        """
        Input: user_input (str), user_intent (str)
        Output: str (next state)
        """
        if user_intent == 'request':
            next_state = self.system.provide_restaurant_info(user_input.lower())
            return self.system.states['INFORM'] if next_state == 'await_next_request' else next_state
        
        if user_intent == 'reqalts':
            next_state = self.system.try_alternative()
            return next_state if next_state == self.system.states['SUGGEST_RESTAURANT'] else self.system.states['APOLOGIZE']
        
        utterance_lower = user_input.lower()
        
        if self._has_info_keywords(utterance_lower):
            next_state = self.system.provide_restaurant_info(user_input.lower())
            return self.system.states['INFORM'] if next_state == 'await_next_request' else next_state
        
        if self._has_alternative_keywords(utterance_lower):
            next_state = self.system.try_alternative()
            return next_state if next_state == self.system.states['SUGGEST_RESTAURANT'] else self.system.states['APOLOGIZE']
        
        return self._handle_no_restaurant_requests(user_intent, user_input)
    
    def _has_info_keywords(self, utterance):
        """
        Input: utterance (str)
        Output: bool
        """
        info_keywords = ['food', 'cuisine', 'serve', 'area', 'location', 'price', 'phone', 'address', 'postcode']
        return any(keyword in utterance for keyword in info_keywords)
    
    def _has_alternative_keywords(self, utterance):
        """
        Input: utterance (str)
        Output: bool
        """
        alt_keywords = ['alternative', 'different', 'other', 'another', 'else', 'more options']
        return any(keyword in utterance for keyword in alt_keywords)
    
    def _handle_no_restaurant_requests(self, user_intent, user_input):
        """
        Input: user_intent (str), user_input (str)
        Output: str (next state)
        """
        if user_intent in ['affirm', 'inform']:
            InputValidator.reset_system_state(self.system)
            return self.system.states['ASK_AREA']
        elif user_intent in ['negate', 'deny']:
            return self.system.states['GOODBYE']
        
        prefs = PreferenceExtractor.extract_all(user_input.lower())
        if any(prefs.get(key) not in [None, 'dontcare'] for key in ['area', 'price', 'food']):
            for key, value in prefs.items():
                if value and value != 'dontcare':
                    self.system.user_requirements[key] = value
            return self.system.check_next_stage()
        
        clarify_msg = "I didn't understand. Please let me know if you'd like restaurant information, alternatives, or want to search for different restaurants."
        print(f"System: {self.system.format_output(clarify_msg)}")
        return self.system.states['INFORM']
    
    def goodbye(self):
        """
        Input: None
        Output: None
        """
        bye_msg = "Thank you for using the Cambridge restaurant system. Goodbye!"
        print(f"System: {self.system.format_output(bye_msg)}")
        self.system.conversation_ended = True
        return None
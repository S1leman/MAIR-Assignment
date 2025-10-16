from utils import format_restaurant_suggestion, detect_restart_command, detect_exit_command
from preference_extraction import PreferenceExtractor


class InputValidator:
    @staticmethod
    def validate_and_handle_special_commands(user_input, system):
        """
        Check for exit and restart commands in user input.
        
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
        Reset all user preferences and system state to initial values.
        
        Input: system (RestaurantSystem)
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
    
    @staticmethod
    def get_romantic_questions(restaurant_name):
        """Generate questions to resolve romantic conflict for a restaurant."""
        return [
            f"I found '{restaurant_name}', but I need your help to determine if it's romantic.",
            "This restaurant is both busy (less romantic) and allows long stays (more romantic).",
            "Do you prefer a quiet atmosphere or the ability to stay for a long time?",
            "Say 'quiet' for peaceful setting, or 'long stay' for leisurely meals."
        ]
    
    @staticmethod
    def get_touristic_questions(restaurant_name):
        """Generate questions to resolve touristic conflict for a restaurant."""
        return [
            f"I found '{restaurant_name}', but I need your help to determine if it's touristic.",
            "This restaurant offers good food at cheap prices (attracts tourists) but serves Romanian cuisine (unfamiliar to tourists).",
            "What's more important - good value for money or familiar cuisine?",
            "Say 'value' for good prices, or 'familiar' for well-known cuisines."
        ]
    
    @classmethod
    def present_conflict(cls, restaurant_name, conflict_type):
        """
        Get appropriate conflict resolution questions based on type.
        
        Input: restaurant_name (str), conflict_type (str)
        Output: list of str (questions to ask user)
        """
        if conflict_type == 'romantic':
            return cls.get_romantic_questions(restaurant_name)
        elif conflict_type == 'touristic':
            return cls.get_touristic_questions(restaurant_name)
        else:
            return []
    
    @classmethod
    def resolve(cls, user_input, restaurant, conflict_type, user_requirement, allow_restarts=False):
        """
        Process user response to conflict and determine resolution action.
        
        Input: user_input (str), restaurant (dict), conflict_type (str), user_requirement (bool), allow_restarts (bool)
        Output: tuple (action: str, data: dict or str or None)
        """
        user_input_lower = user_input.lower()
        
        if any(kw in user_input_lower for kw in ['exit', 'quit', 'bye']):
            return ('exit', None)
        if allow_restarts and any(kw in user_input_lower for kw in ['restart', 'start over', 'reset']):
            return ('restart', None)
        if any(kw in user_input_lower for kw in ['alternative', 'different', 'skip', 'next']):
            return ('skip', None)
        
        resolved_value, explanation = cls._resolve_user_preference(user_input_lower, conflict_type, restaurant['restaurantname'])
        
        if resolved_value == user_requirement:
            cls._update_restaurant_with_resolution(restaurant, conflict_type, resolved_value, explanation)
            return ('recommend', restaurant)
        else:
            reject_msg = f"Based on your preference, {restaurant['restaurantname']} doesn't meet your {conflict_type} requirement."
            return ('reject', reject_msg)
    
    @classmethod
    def _resolve_user_preference(cls, user_input_lower, conflict_type, restaurant_name):
        """Determine user preference from input and generate explanation."""
        if conflict_type == 'romantic':
            if any(word in user_input_lower for word in ['quiet', 'peaceful', 'calm', 'not busy']):
                return False, f"Since you prefer a quiet atmosphere, {restaurant_name} is not romantic due to being busy"
            else:
                return True, f"Since you value long meals, {restaurant_name} is romantic because you can take your time"
        
        elif conflict_type == 'touristic':
            if any(word in user_input_lower for word in ['familiar', 'cuisine', 'food type', 'romanian', 'unfamiliar']):
                return False, f"Since you prefer familiar cuisine, {restaurant_name} is not touristic (Romanian food is unfamiliar)"
            else:
                return True, f"Since you prioritize value, {restaurant_name} is touristic due to its affordable quality"
        
        return True, f"{restaurant_name} meets your requirements"
    
    @classmethod
    def _update_restaurant_with_resolution(cls, restaurant, conflict_type, resolved_value, explanation):
        """
        Update restaurant data with conflict resolution results.
        
        Input: restaurant (dict), conflict_type (str), resolved_value (bool), explanation (str)
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
    def __init__(self, system):
        self.system = system
    
    def welcome(self):  # State 1
        """
        Handle initial welcome state and user's first input.
        
        Output: str (next state)
        """
        msg = (
            "Hello, welcome to the Cambridge restaurant system?\n"
            "You can ask for restaurants by area, price range or food type.\n"
            "How may I help you?"
        )
        print(self.system.format_output(msg))
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
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
    
    def ask_area(self):  # State 2
        """
        Ask user for area preference.
        
        Output: str (next state)
        """
        message = (
            'What part of town do you have in mind?\n'
            'Please specify: north, south, east, west, or centre (you can also say any area or do not care).'
        )
        return self._handle_preference_question([message], 'ASK_AREA')
    
    def ask_price(self):  # State 3
        """
        Ask user for price range preference.
        
        Output: str (next state)
        """
        message = (
            "Would you like something in the cheap, moderate, or expensive price range?\n"
            "Please specify: cheap, moderate, expensive (you can also say 'any price' or 'don't care')."
        )
        return self._handle_preference_question([message], 'ASK_PRICE')
    
    def ask_food_type(self):  # State 4
        """
        Ask user for food type preference.
        
        Output: str (next state)
        """
        message = (
            "What kind of food would you like?\n"
            "Please specify a cuisine type (e.g., italian, chinese, indian, british, french, etc.) or say 'any food' if you don't mind."
        )
        return self._handle_preference_question([message], 'ASK_FOOD_TYPE')
    
    def _handle_preference_question(self, messages, context):
        """
        Generic handler for preference collection questions.
        
        Input: messages (list of str), context (str)
        Output: str (next state)
        """
        for msg in messages:
            print(self.system.format_output(msg))
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        should_continue, next_state = InputValidator.validate_and_handle_special_commands(user_input, self.system)
        if not should_continue:
            return next_state
        
        if user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        if user_intent in ['inform']:
            self.system.parse_user_input(user_input.lower(), context)
        
        return self.system.check_next_stage()
    
    def ask_additional_requirements(self):  # State 5
        """
        Ask user for additional requirements like romantic or touristic.
        
        Output: str (next state)
        """
        if not self.system.alternatives and not self.system.current_restaurant:
            self.system.search_restaurants()
        
        add_req_msg = (
                "Do you have any additional requirements?\n"
                "For example, would you like the restaurant to be touristic, romantic, child-friendly, or have assigned seats?\n"
                "You can say 'yes' and specify requirements, or 'no' if you don't have any additional preferences."
            )
        print(self.system.format_output(add_req_msg))
        
        user_input = self.system.get_user_input("User: ")
        
        if not user_input or not user_input.strip():
            return self._prompt_for_additional_requirements()
        
        user_intent = self.system.classify_utterance(user_input)
        
        should_continue, next_state = InputValidator.validate_and_handle_special_commands(user_input, self.system)
        if not should_continue:
            return next_state

        if user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']

        # FIRST: Try to parse additional requirements from the input
        # This must come BEFORE checking for general negation
        additional_reqs = self.parse_additional_requirements(user_input.lower())
        
        if additional_reqs:
            self.system.additional_requirements.update(additional_reqs)
            return self.system.states['CONFIRM']

        # THEN: Check for general negation (only if no specific requirements found)
        if user_intent in ['negate', 'deny'] or user_input.lower() in ['no', 'nope', 'none']:
            return self.system.states['CONFIRM']

        if user_intent in ['affirm'] or user_input.lower() in ['yes', 'yeah', 'yep']:
            return self._handle_affirmative_additional_requirements()

        if self._has_requirement_keywords(user_input.lower()):
            # We already parsed above, but double-check
            if not additional_reqs:
                additional_reqs = self.parse_additional_requirements(user_input.lower())
                if additional_reqs:
                    self.system.additional_requirements.update(additional_reqs)
                    return self.system.states['CONFIRM']
        
        return self._prompt_for_additional_requirements()
    
    def _has_requirement_keywords(self, user_input):
        """
        Check if user input contains requirement-related keywords.
        
        Input: user_input (str)
        Output: bool
        """
        # Positive keywords
        positive_keywords = ['romantic', 'touristic', 'tourist', 'child', 'family', 'seat', 'seating', 'assigned', 'intimate', 'cozy']
        # Negative patterns
        negative_patterns = ['not romantic', 'not touristic', 'not tourist', 'non-romantic', 'non-touristic', 'no child', 'no assigned', 'no seat']
        
        found_positive = any(word in user_input for word in positive_keywords)
        found_negative = any(pattern in user_input for pattern in negative_patterns)
        
        return found_positive or found_negative
    
    def _prompt_for_additional_requirements(self):
        """
        Prompt user to clarify additional requirements.
        
        Output: str (next state)
        """
        clarify_msg = "I didn't understand. Please let me know if you have any specific requirements like romantic atmosphere, child-friendly environment, etc., or say 'no' if you don't have additional preferences."
        print(self.system.format_output(clarify_msg))
        return self.system.states['ASK_ADDITIONAL_REQUIREMENTS']
    
    def _handle_affirmative_additional_requirements(self):
        """
        Handle when user says yes to additional requirements.
        
        Output: str (next state)
        """
        clarify = "What specific requirements do you have in mind? For example, romantic, touristic, child-friendly?"
        print(self.system.format_output(clarify))
        
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
    
    def confirm(self):  # State 7
        """
        Confirm user preferences before restaurant search.
        
        Output: str (next state)
        """
        confirmation_msg = self._build_confirmation_message()
        print(self.system.format_output(confirmation_msg))
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
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
            print(self.system.format_output(change_msg))
            InputValidator.reset_system_state(self.system)
            return self.system.states['ASK_AREA']
        
        return self.system.states['CONFIRM']
    
    def _build_confirmation_message(self):
        """
        Build confirmation message summarizing user preferences.
        
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
            return (
                f"You are looking for a restaurant {' '.join(prefs)}\n"
                f"{' and '.join(additional_prefs)}, right?"
            )
        else:
            return f"You are looking for a restaurant {' '.join(prefs)}, right?"
    
    def _build_additional_preferences_text(self):
        """
        Build text description of additional preferences.
        
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
    
    def suggest_restaurant(self):  # State 8
        """
        Suggest a restaurant to the user and handle conflicts.
        
        Output: str (next state)
        """
        if not self.system.current_restaurant:
            no_rest_msg = "I'm sorry but there is no restaurant serving that type of food"
            print(self.system.format_output(no_rest_msg))
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
                    self.system.parse_user_input('restart')
                    return self.system.states['WELCOME']
                else:
                    # If restarts are disabled, treat as unrecognized input and continue with conflict resolution
                    print(self.system.format_output('I did not understand. Please choose one of the options provided.'))
                    return self._try_next_restaurant()
            elif resolution_action in ['skip', 'reject']:
                return self._try_next_restaurant()
            elif resolution_action == 'recommend':
                restaurant = self.system.current_restaurant
        
        if 'inference_result' not in restaurant:
            inference_result = self.system.inference_engine.apply_rules(restaurant)
            restaurant['inference_result'] = inference_result
        
        suggestion_msg = format_restaurant_suggestion(restaurant, self.system.additional_requirements)
        import time
        print(self.system.format_output(suggestion_msg))
        time.sleep(1.2)  # Small delay after recommendation
        return self.system.states['INFORM']
    
    def _should_apply_inference_filtering(self):
        """
        Check if inference filtering should be applied.
        
        Output: bool
        """
        has_additional_reqs = any(v is not None for v in self.system.additional_requirements.values())
        not_already_applied = not hasattr(self.system, '_inference_applied')
        return has_additional_reqs and not_already_applied
    
    def _get_active_conflict(self, restaurant):
        """
        Determine if restaurant has an active conflict that needs resolution.
        
        Input: restaurant (dict)
        Output: str or None ('romantic', 'touristic', or None)
        """
        if hasattr(self.system, '_handling_conflict_restaurant'):
            conflict_type = self.system._handling_conflict_restaurant
            if self.system.additional_requirements.get(conflict_type) is not None:
                return conflict_type
        
        restaurant_name = restaurant['restaurantname']
        
        # Check for romantic conflicts
        if self.system.additional_requirements.get('romantic') is not None:
            for conflict_restaurant in self.system.romantic_conflicts:
                if conflict_restaurant['restaurantname'] == restaurant_name:
                    return 'romantic'
        
        # Check for touristic conflicts
        if self.system.additional_requirements.get('touristic') is not None:
            for conflict_restaurant in self.system.touristic_conflicts:
                if conflict_restaurant['restaurantname'] == restaurant_name:
                    return 'touristic'
        
        return None
    
    def _handle_conflict(self, restaurant, conflict_type):
        """
        Handle conflict resolution dialog with user.
        
        Input: restaurant (dict), conflict_type (str)
        Output: str ('recommend', 'reject', 'skip', 'restart', or 'exit')
        """
        messages = ConflictResolver.present_conflict(restaurant['restaurantname'], conflict_type)
        for msg in messages:
            print(self.system.format_output(msg))
        
        user_input = self.system.get_user_input("User: ")
        user_requirement = self.system.additional_requirements[conflict_type]
        action, data = ConflictResolver.resolve(user_input, restaurant, conflict_type, user_requirement, self.system.allow_restarts)
        
        if action == 'recommend':
            self.system.current_restaurant = data
        elif action == 'reject':
            print(self.system.format_output(data))
            self._remove_restaurant_from_lists(restaurant['restaurantname'])
        elif action == 'exit':
            print(self.system.format_output('Thank you for using the restaurant system. Goodbye!'))
        elif action == 'skip':
            print(self.system.format_output('I understand you want to skip this restaurant. Let me find another option for you.'))
            self._remove_restaurant_from_lists(restaurant['restaurantname'])
        
        # Clear conflict handling flag after processing
        if hasattr(self.system, '_handling_conflict_restaurant'):
            delattr(self.system, '_handling_conflict_restaurant')
        
        return action
    
    def _remove_restaurant_from_lists(self, restaurant_name):
        """
        Remove restaurant from all alternative and conflict lists to prevent re-suggestion.
        
        Input: restaurant_name (str)
        """
        # Remove from regular alternatives
        self.system.alternatives = [
            alt for alt in self.system.alternatives 
            if (isinstance(alt, dict) and alt.get('restaurantname', '').lower() != restaurant_name.lower()) or
               (isinstance(alt, str) and alt.lower() != restaurant_name.lower())
        ]
        
        # Remove from romantic conflicts
        self.system.romantic_conflicts = [
            conflict for conflict in self.system.romantic_conflicts
            if conflict.get('restaurantname', '').lower() != restaurant_name.lower()
        ]
        
        # Remove from touristic conflicts
        self.system.touristic_conflicts = [
            conflict for conflict in self.system.touristic_conflicts
            if conflict.get('restaurantname', '').lower() != restaurant_name.lower()
        ]
    
    def _try_next_restaurant(self):
        """
        Try to find and suggest the next alternative restaurant.
        
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
        Get the next alternative restaurant from available options.
        
        Output: dict or None (next restaurant)
        """
        if hasattr(self.system, '_handling_conflict_restaurant'):
            return self._get_next_conflict_restaurant()
        
        if self.system.alternatives and self.system.suggestion_index < len(self.system.alternatives):
            return self._get_next_regular_alternative()
        
        return self._get_first_conflict_restaurant()
    
    def _get_next_conflict_restaurant(self):
        """
        Get next restaurant from conflict lists.
        
        Output: dict or None
        """
        conflict_type = self.system._handling_conflict_restaurant
        
        if conflict_type == 'romantic' and self.system.romantic_conflicts:
            return self.system.romantic_conflicts.pop(0)
        elif conflict_type == 'touristic' and self.system.touristic_conflicts:
            return self.system.touristic_conflicts.pop(0)
        
        # Try the other conflict type
        if conflict_type == 'romantic' and self.system.touristic_conflicts:
            self.system._handling_conflict_restaurant = 'touristic'
            return self.system.touristic_conflicts.pop(0)
        elif conflict_type == 'touristic' and self.system.romantic_conflicts:
            self.system._handling_conflict_restaurant = 'romantic'
            return self.system.romantic_conflicts.pop(0)
        
        return None
    
    def _get_next_regular_alternative(self):
        """
        Get next restaurant from regular alternatives list.
        
        Output: dict or None
        """
        alt_restaurant_dict = self.system.alternatives[self.system.suggestion_index]
        self.system.suggestion_index += 1
        
        if not isinstance(alt_restaurant_dict, dict):
            alt_restaurant_dict = self.system.restaurant_lookup.find_restaurant_by_name(alt_restaurant_dict)
        
        return alt_restaurant_dict
    
    def _get_first_conflict_restaurant(self):
        """
        Get first available restaurant from any conflict list.
        
        Output: dict or None
        """
        if self.system.romantic_conflicts:
            self.system._handling_conflict_restaurant = 'romantic'
            return self.system.romantic_conflicts.pop(0)
        
        if self.system.touristic_conflicts:
            self.system._handling_conflict_restaurant = 'touristic'
            return self.system.touristic_conflicts.pop(0)
        
        return None
    
    def parse_additional_requirements(self, user_input):
        """
        Extract additional requirements from user input.
        
        Input: user_input (str)
        Output: dict (extracted requirements)
        """
        requirements = {}
        
        # Check for touristic preferences (negative first to avoid conflicts)
        negative_touristic = ['not tourist', 'not touristic', 'non-tourist', 'non-touristic', 'not a tourist']
        positive_touristic = ['tourist', 'touristic', 'popular', 'famous']
        
        if any(phrase in user_input for phrase in negative_touristic) or any(word in user_input for word in ['local', 'hidden', 'authentic']):
            requirements['touristic'] = False
        elif any(word in user_input for word in positive_touristic):
            requirements['touristic'] = True
        
        # Check for romantic preferences (negative first to avoid conflicts)
        negative_romantic = ['not romantic', 'non-romantic', 'not a romantic']
        positive_romantic = ['romantic', 'romance', 'intimate', 'cozy', 'date']
        
        if any(phrase in user_input for phrase in negative_romantic) or any(word in user_input for word in ['casual', 'business']):
            requirements['romantic'] = False
        elif any(word in user_input for word in positive_romantic):
            requirements['romantic'] = True
        
        # Check for children preferences (negative first to avoid conflicts)
        negative_children = ['no child', 'no children', 'no kids', 'adults only', 'not child-friendly']
        positive_children = ['child', 'children', 'kid', 'kids', 'family', 'child-friendly']
        
        if any(phrase in user_input for phrase in negative_children) or 'quiet' in user_input:
            requirements['children'] = False
        elif any(word in user_input for word in positive_children):
            requirements['children'] = True
        
        # Check for seating preferences (negative first to avoid conflicts)
        negative_seating = ['no assigned seat', 'no assigned seats', 'no seat assignment', 'free seating', 'choose seat', 'pick seat']
        positive_seating = ['assigned seat', 'assigned seats', 'seat assignment', 'waiter choose']
        
        if any(phrase in user_input for phrase in negative_seating):
            requirements['assigned_seats'] = False
        elif any(phrase in user_input for phrase in positive_seating):
            requirements['assigned_seats'] = True
        
        return requirements
    
    def apologize(self):  # State 6
        """
        Apologize for no matching restaurants and restart search.
        
        Output: str (next state)
        """
        sorry_msg = (
            "I'm sorry, no restaurants were found matching your criteria.\n"
            "Let's try a new search with different preferences."
        )
        import time
        print(self.system.format_output(sorry_msg))
        time.sleep(1.2)  # Small delay before next print
        InputValidator.reset_system_state(self.system)
        return self.system.states['ASK_AREA']
    
    def inform(self):  # State 9
        """
        Handle user requests for restaurant information or alternatives.
        
        Output: str (next state)
        """
        self._print_inform_prompt()
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
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
        """Display appropriate prompt based on current restaurant status."""
        if self.system.current_restaurant:
            info_msg = (
                "Would you like more information about the restaurant (phone, address), an alternative restaurant, or would you like to try a different search?"
            )

            guidance_msg = info_msg + (
                "\nYou can ask for 'phone', 'address', say 'alternative' for other options,\n"
                "specify new preferences, or say 'exit' to leave"
            )
            if self.system.allow_restarts:
                guidance_msg += " or 'restart' to start over."
            else:
                guidance_msg += "."
            print(self.system.format_output(guidance_msg))
        else:
            try_diff_msg = "Would you like to try a different type of food or change your preferences?"
            print(self.system.format_output(try_diff_msg))
            
            restart_msg = "You can say 'yes' to try again, specify new preferences (e.g., 'chinese food'), or say 'exit' to leave"
            if self.system.allow_restarts:
                restart_msg += " or 'restart' to start over."
            else:
                restart_msg += "."
            print(self.system.format_output(restart_msg))
    
    def _handle_current_restaurant_requests(self, user_input, user_intent):
        """
        Handle user requests when a current restaurant is available.
        
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
        Check if utterance contains information request keywords.
        
        Input: utterance (str)
        Output: bool
        """
        info_keywords = ['food', 'cuisine', 'serve', 'area', 'location', 'price', 'phone', 'address', 'postcode']
        return any(keyword in utterance for keyword in info_keywords)
    
    def _has_alternative_keywords(self, utterance):
        """
        Check if utterance contains alternative request keywords.
        
        Input: utterance (str)
        Output: bool
        """
        alt_keywords = ['alternative', 'different', 'other', 'another', 'else', 'more options']
        return any(keyword in utterance for keyword in alt_keywords)
    
    def _handle_no_restaurant_requests(self, user_intent, user_input):
        """
        Handle user requests when no current restaurant is available.
        
        Input: user_intent (str), user_input (str)
        Output: str (next state)
        """
        if user_intent in ['affirm', 'inform']:
            InputValidator.reset_system_state(self.system)
            return self.system.states['ASK_AREA']
        elif user_intent in ['negate', 'deny']:
            return self.system.states['GOODBYE']
        
        prefs = PreferenceExtractor.extract_all(user_input.lower())
        has_new_prefs = False
        for key, value in prefs.items():
            if value and value != 'dontcare':
                self.system.user_requirements[key] = value
                has_new_prefs = True
        
        if has_new_prefs:
            return self.system.check_next_stage()
        
        clarify_msg = "I didn't understand. Please let me know if you'd like restaurant information, alternatives, or want to search for different restaurants."
        print(self.system.format_output(clarify_msg))
        return self.system.states['INFORM']
    
    def goodbye(self):  # State 10
        """End conversation with farewell message."""
        bye_msg = (
            "Thank you for using the Cambridge restaurant system.\n"
            "Goodbye!"
        )
        print(self.system.format_output(bye_msg))
        self.system.conversation_ended = True
        return None
from utils import format_restaurant_suggestion, detect_restart_command, detect_new_search_request, detect_exit_command
from preference_extraction import PreferenceExtractor

class ConversationStates: 
    
    def __init__(self, system):
        self.system = system 
    
    # Stage 1: WELCOME
    def welcome(self):
        print("System: Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?")
        print("System: Please tell me what kind of restaurant you're looking for (e.g., 'Italian food', 'cheap restaurant', 'restaurant in the south').")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        # Check for exit commands first
        if detect_exit_command(user_input):
            return self.system.states['GOODBYE']
        
        # Check for restart commands only if restarts are allowed
        if self.system.allow_restarts and detect_restart_command(user_input):
            return self.system.states['WELCOME']
        
        # Check for goodbye/thankyou intents
        if user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        # Handle inform intent and parse preferences
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower())
            if parse_result == 'restart' and self.system.allow_restarts: 
                return self.system.states['WELCOME']
        
        # Proceed to next stage based on current preferences
        next_state = self.system.check_next_stage() 
        
        return next_state
    
    # Stage 2: ASK_AREA
    def ask_area(self):
        print("System: What part of town do you have in mind?")
        print("System: Please specify: north, south, east, west, or centre (you can also say 'any area' or 'don't care').")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        # Check for exit commands
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        # Check for restart commands only if allowed
        if self.system.allow_restarts and detect_restart_command(user_input):
            return self.system.states['WELCOME']
        
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower(), 'ASK_AREA')
            if parse_result == 'restart' and self.system.allow_restarts:
                return self.system.states['WELCOME']
        
        next_state = self.system.check_next_stage()

        return next_state
    
    # Stage 3: ASK_PRICE
    def ask_price(self):
        print("System: Would you like something in the cheap, moderate, or expensive price range?")
        print("System: Please specify: cheap, moderate, expensive (you can also say 'any price' or 'don't care').")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        # Check for exit commands
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        # Check for restart commands only if allowed
        if self.system.allow_restarts and detect_restart_command(user_input):
            return self.system.states['WELCOME']
        
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower(), 'ASK_PRICE')
            if parse_result == 'restart' and self.system.allow_restarts:
                return self.system.states['WELCOME']
        
        next_state = self.system.check_next_stage()
        
        return next_state
    
    # Stage 4: ASK_FOOD_TYPE
    def ask_food_type(self):
        print("System: What kind of food would you like?")
        print("System: Please specify a cuisine type (e.g., italian, chinese, indian, british, french, etc.) or say 'any food' if you don't mind.")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        # Check for exit commands
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        # Check for restart commands only if allowed
        if self.system.allow_restarts and detect_restart_command(user_input):
            return self.system.states['WELCOME']
        
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower(), 'ASK_FOOD_TYPE')
            if parse_result == 'restart' and self.system.allow_restarts:
                return self.system.states['WELCOME']
        
        next_state = self.system.check_next_stage()

        return next_state
    
    # Stage 5: CONFIRM
    def confirm(self):
        prefs = []
        if self.system.user_requirements['area'] and self.system.user_requirements['area'] != 'dontcare':
            prefs.append(f"in the {self.system.user_requirements['area']} of town")
        if self.system.user_requirements['price']:
            if(self.system.user_requirements['price'] != 'dontcare'):
                prefs.append(f"in the {self.system.user_requirements['price']} price range")
            else:
                prefs.append("in any price range")
        if self.system.user_requirements['food'] and self.system.user_requirements['food'] != 'dontcare':
            prefs.append(f"serving {self.system.user_requirements['food']} food")
        
        confirmation_msg = f"You are looking for a restaurant {' '.join(prefs)}, right?"
        print(f"System: {confirmation_msg}")
        print("System: Please answer 'yes' to confirm or 'no' to change your preferences.")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        print(f"[Final preferences confirmed: {self.system.user_requirements}]")
        
        # Check for exit commands
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        # Check for restart commands only if allowed
        if self.system.allow_restarts and detect_restart_command(user_input):
            self.system.user_requirements = {'area': None, 'price': None, 'food': None}
            return self.system.states['WELCOME']
        
        if user_intent == 'affirm':
            # Search for restaurants and suggest
            print("[Searching restaurant database...]")
            self.system.search_restaurants()
            
            # Check if any restaurants were found
            if not self.system.current_restaurant:
                print("[No restaurants found matching criteria]")
                return self.system.states['APOLOGIZE']
            
            print(f"[Restaurant found: {self.system.current_restaurant['restaurantname']}]")
            print(f"[Alternatives available: {len(self.system.alternatives)}]")
            next_state = self.system.states['ASK_ADDITIONAL_REQUIREMENTS']
        elif user_intent == 'negate':
            # User said no, apologize and restart
            next_state = self.system.states['APOLOGIZE']
        else:
            # Repeat confirmation
            next_state = self.system.states['CONFIRM']
        
        return next_state
    
    # Stage 6: ASK_ADDITIONAL_REQUIREMENTS
    def ask_additional_requirements(self):
        if not self.system.alternatives and not self.system.current_restaurant:
            # No restaurants found, go to inform state
            return self.system.states['INFORM']
        
        print("System: Do you have any additional requirements?")
        print("System: For example, would you like the restaurant to be touristic, romantic, child-friendly, or have assigned seats?")
        print("System: You can say 'yes' and specify requirements, or 'no' if you don't have any additional preferences.")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        # Check for exit commands
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        # Check for restart commands only if allowed
        if self.system.allow_restarts and detect_restart_command(user_input):
            self.system.user_requirements = {'area': None, 'price': None, 'food': None}
            self.system.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
            return self.system.states['WELCOME']
        
        # Always try to parse additional requirements regardless of intent classification
        additional_reqs = self.parse_additional_requirements(user_input.lower())
        
        if user_intent in ['negate', 'deny'] or user_input.lower() in ['no', 'nope', 'none']:
            # No additional requirements, proceed to suggest restaurant
            print("[No additional requirements specified]")
            return self.system.states['SUGGEST_RESTAURANT']
        elif additional_reqs:
            # Found additional requirements
            print(f"[Additional requirements parsed: {additional_reqs}]")
            self.system.additional_requirements.update(additional_reqs)
            print(f"[Current additional requirements: {self.system.additional_requirements}]")
            
            # Apply inference rules to filter restaurants
            self.system.apply_inference_filtering()
            return self.system.states['SUGGEST_RESTAURANT']
        elif user_intent in ['affirm'] or user_input.lower() in ['yes', 'yeah', 'yep']:
            # User said yes but didn't specify requirements
            print("System: What specific requirements do you have in mind? For example, romantic, touristic, child-friendly?")
            return self.system.states['ASK_ADDITIONAL_REQUIREMENTS']
        else:
            # Try to detect if this might be a requirement even if not classified correctly
            if any(word in user_input.lower() for word in ['romantic', 'touristic', 'child', 'family', 'seat']):
                additional_reqs = self.parse_additional_requirements(user_input.lower())
                if additional_reqs:
                    print(f"[Additional requirements parsed: {additional_reqs}]")
                    self.system.additional_requirements.update(additional_reqs)
                    self.system.apply_inference_filtering()
                    return self.system.states['SUGGEST_RESTAURANT']
            
            # Clarify or repeat
            print("System: I didn't understand. Please let me know if you have any specific requirements like romantic atmosphere, child-friendly environment, etc., or say 'no' if you don't have additional preferences.")
            return self.system.states['ASK_ADDITIONAL_REQUIREMENTS']
    
    def parse_additional_requirements(self, user_input):
        """
        Parse additional requirements from user input.
        Returns a dictionary with requirement properties.
        """
        requirements = {}
        
        # Touristic
        if any(word in user_input for word in ['tourist', 'touristic', 'popular', 'famous']):
            requirements['touristic'] = True
        elif any(word in user_input for word in ['not tourist', 'local', 'hidden', 'authentic']):
            requirements['touristic'] = False
            
        # Romantic
        if any(word in user_input for word in ['romantic', 'romance', 'intimate', 'cozy', 'date']):
            requirements['romantic'] = True
        elif any(word in user_input for word in ['not romantic', 'casual', 'business']):
            requirements['romantic'] = False
            
        # Children
        if any(word in user_input for word in ['child', 'children', 'kid', 'family', 'child-friendly']):
            requirements['children'] = True
        elif any(word in user_input for word in ['no child', 'adults only', 'quiet']):
            requirements['children'] = False
            
        # Assigned seats
        if any(word in user_input for word in ['assigned seat', 'assigned seats', 'seat assignment', 'waiter choose']):
            requirements['assigned_seats'] = True
        elif any(word in user_input for word in ['choose seat', 'pick seat', 'free seating']):
            requirements['assigned_seats'] = False
            
        return requirements

    # Stage 7: APOLOGIZE
    def apologize(self):
        print("System: I'm sorry, no restaurants were found matching your criteria. Let me help you find what you're looking for.")
         
        self.system.user_requirements = {'area': None, 'price': None, 'food': None}
        self.system.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
        return self.system.states['ASK_AREA']
    
    # Stage 8: SUGGEST_RESTAURANT
    def suggest_restaurant(self):
        # Check if we have romantic conflicts to resolve first - but ONLY if user actually requested romantic
        if (hasattr(self.system, 'romantic_conflicts') and 
            self.system.romantic_conflicts and
            self.system.additional_requirements.get('romantic') is True):
            # User requested romantic restaurant and we have conflicts - let them choose
            return self._handle_romantic_requirement_conflicts()
        
        if not self.system.current_restaurant:
            print("System: I'm sorry but there is no restaurant serving that type of food")
            return self.system.states['INFORM']
        
        restaurant = self.system.current_restaurant
        
        # Check for conflicts in inference rules before suggesting
        inference_result = self.system.inference_engine.apply_rules(restaurant)
        
        # Handle romantic property conflicts - but ONLY if user specifically requested romantic
        if ('conflict' in inference_result and 
            inference_result['conflict']['requires_user_input'] and
            inference_result['conflict']['property'] == 'romantic' and
            self.system.additional_requirements.get('romantic') is True):
            return self._handle_romantic_conflict(restaurant, inference_result['conflict'])
        
        # No conflict, proceed with normal suggestion
        suggestion_msg = format_restaurant_suggestion(restaurant)
        print(suggestion_msg)
        
        print(f"[Restaurant details: {restaurant['restaurantname']} - {restaurant['food']}, {restaurant['pricerange']}, {restaurant['area']}]")
        
        # Add guidance message
        guidance_msg = "System: Would you like more information (phone, address), an alternative restaurant, or do you have any other requests?"
        if self.system.allow_restarts:
            guidance_msg += " You can also say 'restart' to start over or 'exit' to leave."
        else:
            guidance_msg += " You can say 'exit' to leave."
        print(guidance_msg)
        
        return self.await_user_response()
    
    def _handle_romantic_conflict(self, restaurant, conflict_info):
        """
        Handle romantic property conflicts by asking user for priority.
        """
        rule1 = conflict_info['rule1']
        rule2 = conflict_info['rule2']
        
        print(f"System: I found '{restaurant['restaurantname']}', but I need your help to determine if it's romantic.")
        print(f"System: This restaurant has both busy periods and allows for long stays, which gives mixed signals about romance.")
        print(f"System: Some people find busy restaurants less romantic because of the noise and crowds.")
        print(f"System: Others find restaurants where you can stay for a long time more romantic for intimate conversations.")
        print(f"System: What's more important to you - avoiding crowds or having time for a long, relaxed meal?")
        print(f"System: Please say 'avoid crowds' or 'long meal' to help me decide.")
        
        # Wait for user preference
        while True:
            user_input = self.system.get_user_input("User: ").lower()
            if user_input:
                break
        
        # Determine user's priority and apply it
        if 'avoid' in user_input or 'crowd' in user_input or 'busy' in user_input or 'quiet' in user_input:
            # User prioritizes avoiding crowds
            romantic_value = False
            explanation = f"Since you prefer to avoid crowds, I'd say {restaurant['restaurantname']} is not particularly romantic due to it being quite busy."
        elif 'long' in user_input or 'meal' in user_input or 'time' in user_input or 'stay' in user_input:
            # User prioritizes long stays
            romantic_value = True
            explanation = f"Since you value having time for a long meal, I'd say {restaurant['restaurantname']} is romantic because you can take your time and enjoy intimate conversations."
        else:
            # Default to prioritizing long stays if unclear
            romantic_value = True
            explanation = f"Since you didn't specify clearly, I'll assume having time for a relaxed meal is more important. {restaurant['restaurantname']} is romantic because you can stay as long as you like."
        
        # Update restaurant with resolved conflict
        restaurant['romantic'] = romantic_value
        restaurant['conflict_resolution'] = explanation
        
        # Now suggest the restaurant with resolved conflict
        name = restaurant['restaurantname']
        area = restaurant['area']
        pricerange = restaurant['pricerange'] 
        food = restaurant['food']
        
        area_desc = " in the city centre" if area == 'centre' else f" in the {area} of town"
        main_desc = f"System: I recommend '{name}', it is {pricerange} {food} restaurant{area_desc}."
        
        print(f"{main_desc} {explanation}")
        print(f"[Restaurant details: {restaurant['restaurantname']} - {restaurant['food']}, {restaurant['pricerange']}, {restaurant['area']}]")
        
        # Add guidance message
        guidance_msg = "System: Would you like more information (phone, address), an alternative restaurant, or do you have any other requests?"
        if self.system.allow_restarts:
            guidance_msg += " You can also say 'restart' to start over or 'exit' to leave."
        else:
            guidance_msg += " You can say 'exit' to leave."
        print(guidance_msg)
        
        return self.await_user_response()
        
    def _handle_romantic_requirement_conflicts(self):
        """
        Handle conflicts when user specifically requested romantic restaurants.
        Present restaurants with conflicts and ask user to resolve them.
        """
        conflicts = self.system.romantic_conflicts
        if not conflicts:
            return self.system.states['INFORM']
            
        # Take the first restaurant with romantic conflict
        restaurant = conflicts[0]
        conflict_info = restaurant['inference_result']['conflict']
        
        print(f"System: You asked for romantic restaurants. I found '{restaurant['restaurantname']}' which matches your other criteria,")
        print("System: but there's a conflict about whether it's romantic.")
        
        rule1 = conflict_info['rule1']
        rule2 = conflict_info['rule2']
        
        print(f"System: Rule {rule1['id']}: {rule1['description']} -> romantic = {rule1['value']}")
        print(f"System: Rule {rule2['id']}: {rule2['description']} -> romantic = {rule2['value']}")
        print(f"System: Which do you prioritize more - that it's not too busy (romantic = {rule1['value'] if rule1['id'] == 5 else rule2['value']}) or that you can stay for a long time (romantic = {rule1['value'] if rule1['id'] == 6 else rule2['value']})?")
        print(f"System: Please say 'not busy' or 'long stay' to help me decide.")
        
        # Wait for user preference
        while True:
            user_input = self.system.get_user_input("User: ").lower()
            if user_input:
                break
        
        # Determine user's priority and resolve conflict
        if 'not busy' in user_input or 'busy' in user_input:
            # User prioritizes "not busy" - use rule 5 (busy -> not romantic)
            chosen_rule = rule1 if rule1['id'] == 5 else rule2
            romantic_value = chosen_rule['value']
            explanation = f"Based on your priority of avoiding busy places, {restaurant['restaurantname']} is {'romantic' if romantic_value else 'not romantic'} because {chosen_rule['description']}"
        elif 'long stay' in user_input or 'long' in user_input:
            # User prioritizes "long stay" - use rule 6 (long stay -> romantic)
            chosen_rule = rule1 if rule1['id'] == 6 else rule2
            romantic_value = chosen_rule['value']
            explanation = f"Based on your priority of staying for a long time, {restaurant['restaurantname']} is {'romantic' if romantic_value else 'not romantic'} because {chosen_rule['description']}"
        else:
            # Default to rule 6 if unclear
            chosen_rule = rule1 if rule1['id'] == 6 else rule2
            romantic_value = chosen_rule['value']
            explanation = f"Since you didn't specify clearly, I'll assume the long stay aspect is more important. {restaurant['restaurantname']} is {'romantic' if romantic_value else 'not romantic'} because {chosen_rule['description']}"
        
        # Check if resolved restaurant meets user's romantic requirement
        if romantic_value == self.system.additional_requirements.get('romantic', True):
            # Restaurant meets requirement after conflict resolution
            restaurant['romantic'] = romantic_value
            restaurant['conflict_resolution'] = explanation
            
            # Set this as current restaurant
            self.system.current_restaurant = restaurant
            self.system.current_restaurant_name = restaurant['restaurantname']
            
            # Clear conflicts
            self.system.romantic_conflicts = []
            
            # Suggest the resolved restaurant
            name = restaurant['restaurantname']
            area = restaurant['area']
            pricerange = restaurant['pricerange'] 
            food = restaurant['food']
            
            area_desc = " in the city centre" if area == 'centre' else f" in the {area} of town"
            main_desc = f"System: I recommend '{name}', it is {pricerange} {food} restaurant{area_desc}."
            
            print(f"{main_desc} {explanation}")
            print(f"[Restaurant details: {restaurant['restaurantname']} - {restaurant['food']}, {restaurant['pricerange']}, {restaurant['area']}]")
            
            # Add guidance message
            guidance_msg = "System: Would you like more information (phone, address), an alternative restaurant, or do you have any other requests?"
            if self.system.allow_restarts:
                guidance_msg += " You can also say 'restart' to start over or 'exit' to leave."
            else:
                guidance_msg += " You can say 'exit' to leave."
            print(guidance_msg)
            
            return self.await_user_response()
        else:
            # Even after resolution, restaurant doesn't meet requirement
            # Try next restaurant or inform no match
            remaining_conflicts = self.system.romantic_conflicts[1:]
            if remaining_conflicts:
                self.system.romantic_conflicts = remaining_conflicts
                return self._handle_romantic_requirement_conflicts()  # Try next one
            else:
                self.system.romantic_conflicts = []
                print("System: I'm sorry, none of the available restaurants meet your romantic requirement after resolving the conflicts.")
                return self.system.states['INFORM']
        
    def await_user_response(self):
        # Use a loop instead of recursion to prevent stack overflow
        while True:
            user_input = self.system.get_user_input("User: ")
            if user_input:
                break
        
        user_intent = self.system.classify_utterance(user_input)
        print(f"\n[Classified as: {user_intent}]")
        
        if self.system.allow_restarts and detect_restart_command(user_input):
            print("[User requested restart]")
            self.system.user_requirements = {'area': None, 'price': None, 'food': None}
            return self.system.states['WELCOME']
         
        if detect_new_search_request(user_input): 
            print(f"[Detecting new restaurant search request...]")
            new_prefs = PreferenceExtractor.extract_all(user_input.lower())
            if any(new_prefs.get(key) not in [None, 'dontcare'] for key in ['area', 'price', 'food']):
                print(f"[New preferences detected: {new_prefs}]") 
                # Update system requirements with new preferences
                for key, value in new_prefs.items():
                    if value and value != 'dontcare':
                        if key == 'price':
                            self.system.user_requirements['price'] = value
                        elif key in ['area', 'food']:
                            self.system.user_requirements[key] = value
                print(f"[Updated requirements: {self.system.user_requirements}]") 
                return self.system.check_next_stage() 
         
        if user_intent == 'request':
            print(f"[Processing information request...]")
            next_state = self.system.provide_restaurant_info(user_input.lower())
            if next_state == 'await_next_request': 
                return self.await_user_response()
            else:
                return next_state
        elif user_intent == 'reqalts':
            print(f"[Processing alternative request...]")
            next_state = self.system.try_alternative()
            if next_state == self.system.states['SUGGEST_RESTAURANT']:
                print(f"[Alternative restaurant provided: {self.system.current_restaurant['restaurantname']}]")
            return next_state
        elif user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        else: 
            repeat_msg = f"{self.system.current_restaurant['restaurantname']} is a great restaurant"
            print(f"System: {repeat_msg}") 
            return self.await_user_response()
    
    # Stage 9: INFORM
    def inform(self):
        print("System: Would you like to try a different type of food or change your preferences?")
        restart_msg = "System: You can say 'yes' to try again, specify new preferences (e.g., 'chinese food'), or say 'exit' to leave"
        if self.system.allow_restarts:
            restart_msg += " or 'restart' to start over."
        else:
            restart_msg += "."
        print(restart_msg)
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        # Check for exit commands
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
         
        if self.system.allow_restarts and detect_restart_command(user_input):
            print("[User requested restart]") 
            self.system.user_requirements = {'area': None, 'price': None, 'food': None}
            self.system.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
            return self.system.states['WELCOME']
        
        # Check if user wants to try again or change preferences
        if user_intent in ['affirm', 'inform']: 
            print("[Restarting restaurant search...]")
            self.system.user_requirements = {'area': None, 'price': None, 'food': None}
            self.system.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
            return self.system.states['ASK_AREA']
        elif user_intent in ['negate', 'deny']:
            return self.system.states['GOODBYE']
        else: 
            # For unrecognized input, try to extract preferences and restart if found
            prefs = PreferenceExtractor.extract_all(user_input.lower())
            if any(prefs.get(key) not in [None, 'dontcare'] for key in ['area', 'price', 'food']):
                print(f"[New preferences detected: {prefs}]")
                print("[Starting new search with updated preferences...]")
                # Update preferences
                for key, value in prefs.items():
                    if value and value != 'dontcare':
                        self.system.user_requirements[key] = value
                return self.system.check_next_stage()
            else:
                # Ask for clarification instead of exiting
                print("System: I didn't understand. Please let me know if you'd like to search for different restaurants, or say 'exit' if you want to leave.")
                return self.system.states['INFORM']
    
    # Stage 10: GOODBYE
    def goodbye(self):
        if self.system.allow_restarts:
            print("System: Thank you for using the Cambridge restaurant system. Feel free to run the system again to search for more restaurants. Goodbye!")
        else:
            print("System: Thank you for using the Cambridge restaurant system. Goodbye!")
        self.system.conversation_ended = True
        return None
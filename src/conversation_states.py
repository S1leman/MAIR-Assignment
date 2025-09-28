from utils import format_restaurant_suggestion, detect_restart_command, detect_new_search_request, detect_exit_command
from preference_extraction import PreferenceExtractor

class ConversationStates: 
    
    def __init__(self, system):
        self.system = system 
    
    # Stage 1: WELCOME
    def welcome(self):
        msg = "Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?"
        print(f"System: {self.system.format_output(msg)}")
        
        guidance = "Please tell me what kind of restaurant you're looking for (e.g., 'Italian food', 'cheap restaurant', 'restaurant in the south')."
        print(f"System: {self.system.format_output(guidance)}")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        if detect_exit_command(user_input):
            return self.system.states['GOODBYE']
        
        if self.system.allow_restarts and detect_restart_command(user_input):
            return self.system.states['WELCOME']
        
        if user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower())
            if parse_result == 'restart' and self.system.allow_restarts: 
                return self.system.states['WELCOME']
        
        next_state = self.system.check_next_stage() 
        return next_state
    
    # Stage 2: ask area
    def ask_area(self):
        print(f"System: {self.system.format_output('What part of town do you have in mind?')}")
        print(f"System: {self.system.format_output('Please specify: north, south, east, west, or centre (you can also say any area or do not care).')}")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        if self.system.allow_restarts and detect_restart_command(user_input):
            return self.system.states['WELCOME']
        
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower(), 'ASK_AREA')
            if parse_result == 'restart' and self.system.allow_restarts:
                return self.system.states['WELCOME']
        
        next_state = self.system.check_next_stage()
        return next_state
    
    # Stage 3: ask price  
    def ask_price(self):
        price_msg = "Would you like something in the cheap, moderate, or expensive price range?"
        print(f"System: {self.system.format_output(price_msg)}")
        detail_msg = "Please specify: cheap, moderate, expensive (you can also say 'any price' or 'don't care')."
        print(f"System: {self.system.format_output(detail_msg)}")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        if self.system.allow_restarts and detect_restart_command(user_input):
            return self.system.states['WELCOME']
        
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower(), 'ASK_PRICE')
            if parse_result == 'restart' and self.system.allow_restarts:
                return self.system.states['WELCOME']
        
        next_state = self.system.check_next_stage()
        return next_state
    
    def ask_food_type(self):
        # Stage 4: ASK_FOOD_TYPE
        food_msg = "What kind of food would you like?"
        print(f"System: {self.system.format_output(food_msg)}")
        
        detail = "Please specify a cuisine type (e.g., italian, chinese, indian, british, french, etc.) or say 'any food' if you don't mind."
        print(f"System: {self.system.format_output(detail)}")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        if self.system.allow_restarts and detect_restart_command(user_input):
            return self.system.states['WELCOME']
        
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower(), 'ASK_FOOD_TYPE')
            if parse_result == 'restart' and self.system.allow_restarts:
                return self.system.states['WELCOME']
        
        next_state = self.system.check_next_stage()
        return next_state
    
    def confirm(self):
        # Stage 7: CONFIRM
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
        
        # Add additional requirements to confirmation if any were specified
        additional_prefs = []
        for req_key, req_value in self.system.additional_requirements.items():
            if req_value is not None:
                if req_value:
                    # User wants this property
                    if req_key == 'touristic':
                        additional_prefs.append("that is touristic")
                    elif req_key == 'romantic':
                        additional_prefs.append("that is romantic")
                    elif req_key == 'children':
                        additional_prefs.append("that is child-friendly")
                    elif req_key == 'assigned_seats':
                        additional_prefs.append("with assigned seating")
                else:
                    # User doesn't want this property
                    if req_key == 'touristic':
                        additional_prefs.append("that is not touristic")
                    elif req_key == 'romantic':
                        additional_prefs.append("that is not romantic")
                    elif req_key == 'children':
                        additional_prefs.append("that is not suitable for children")
                    elif req_key == 'assigned_seats':
                        additional_prefs.append("where you can choose your own seats")
        
        # Build confirmation message
        if additional_prefs:
            confirmation_msg = f"You are looking for a restaurant {' '.join(prefs)} {' and '.join(additional_prefs)}, right?"
        else:
            confirmation_msg = f"You are looking for a restaurant {' '.join(prefs)}, right?"
            
        print(f"System: {self.system.format_output(confirmation_msg)}")
        confirm_help = "Please answer 'yes' to confirm or 'no' to change your preferences."
        print(f"System: {self.system.format_output(confirm_help)}")
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        print(f"[Final basic preferences: {self.system.user_requirements}]")
        print(f"[Final additional preferences: {self.system.additional_requirements}]")
        
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        if self.system.allow_restarts and detect_restart_command(user_input):
            self.system.user_requirements = {'area': None, 'price': None, 'food': None}
            self.system.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
            return self.system.states['WELCOME']
        
        if user_intent == 'affirm':
            # restaurants already found in check_next_stage, go to suggest
            print("[Preferences confirmed - proceeding to restaurant suggestion]")
            
            # verify restaurant data
            if not self.system.current_restaurant:
                print("[No restaurant data available - searching again]")
                self.system.search_restaurants()
                if not self.system.current_restaurant:
                    return self.system.states['APOLOGIZE']
            
            print(f"[Restaurant selected: {self.system.current_restaurant['restaurantname']}]")
            print(f"[Alternatives available: {len(self.system.alternatives)}]")
            next_state = self.system.states['SUGGEST_RESTAURANT']
        elif user_intent == 'negate':
            change_msg = "No problem! Let me help you find a different restaurant."
            print(f"System: {self.system.format_output(change_msg)}")
            self.system.user_requirements = {'area': None, 'price': None, 'food': None}
            self.system.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
            next_state = self.system.states['ASK_AREA']
        else:
            # repeat confirm
            next_state = self.system.states['CONFIRM']
        
        return next_state
    
    def ask_additional_requirements(self):
        # Stage 5: ASK_ADDITIONAL_REQUIREMENTS
        # restaurants already loaded
        if not self.system.alternatives and not self.system.current_restaurant:
            # fallback: search if no restaurants
            print("[No restaurant data available - searching database]")
            self.system.search_restaurants()
            if not self.system.current_restaurant:
                return self.system.states['INFORM']
        
        add_req_msg = "Do you have any additional requirements? For example, would you like the restaurant to be touristic, romantic, child-friendly, or have assigned seats? You can say 'yes' and specify requirements, or 'no' if you don't have any additional preferences."
        print(f"System: {self.system.format_output(add_req_msg)}")
        
        user_input = self.system.get_user_input("User: ")
        
        # handle empty input
        if not user_input or not user_input.strip():
            empty_msg = "Please provide an answer. Do you have any additional requirements like romantic, touristic, child-friendly, or say 'no' if you don't have any preferences?"
            print(f"System: {self.system.format_output(empty_msg)}")
            return self.system.states['ASK_ADDITIONAL_REQUIREMENTS']
        
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
        
        if self.system.allow_restarts and detect_restart_command(user_input):
            self.system.user_requirements = {'area': None, 'price': None, 'food': None}
            self.system.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
            return self.system.states['WELCOME']
        
        additional_reqs = self.parse_additional_requirements(user_input.lower())
        
        if user_intent in ['negate', 'deny'] or user_input.lower() in ['no', 'nope', 'none']:
            print("[No additional requirements specified]")
            return self.system.states['CONFIRM']
        elif additional_reqs:
            print(f"[Additional requirements parsed: {additional_reqs}]")
            self.system.additional_requirements.update(additional_reqs)
            print(f"[Current additional requirements: {self.system.additional_requirements}]")
            
            # apply inference rules to filter restaurants
            self.system.apply_inference_filtering()
            
            # check if any restaurants remain after filtering
            if not self.system.current_restaurant:
                print("[No restaurants meet the additional requirements]")
                return self.system.states['APOLOGIZE']
                
            return self.system.states['CONFIRM']
        elif user_intent in ['affirm'] or user_input.lower() in ['yes', 'yeah', 'yep']:
            clarify = "What specific requirements do you have in mind? For example, romantic, touristic, child-friendly?"
            print(f"System: {self.system.format_output(clarify)}")
            
            # get the specific requirements immediately
            follow_up_input = self.system.get_user_input("User: ")
            
            if not follow_up_input or not follow_up_input.strip():
                empty_msg = "Please specify your requirements or say 'no' if you don't have any."
                print(f"System: {self.system.format_output(empty_msg)}")
                return self.system.states['ASK_ADDITIONAL_REQUIREMENTS']
            
            # check for exit/restart commands
            if detect_exit_command(follow_up_input):
                return self.system.states['GOODBYE']
            
            if self.system.allow_restarts and detect_restart_command(follow_up_input):
                self.system.user_requirements = {'area': None, 'price': None, 'food': None}
                self.system.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
                return self.system.states['WELCOME']
            
            # parse the follow-up input for additional requirements
            follow_up_reqs = self.parse_additional_requirements(follow_up_input.lower())
            
            if follow_up_reqs:
                print(f"[Additional requirements parsed: {follow_up_reqs}]")
                self.system.additional_requirements.update(follow_up_reqs)
                print(f"[Current additional requirements: {self.system.additional_requirements}]")
                
                # apply inference rules to filter restaurants
                self.system.apply_inference_filtering()
                
                if not self.system.current_restaurant:
                    print("[No restaurants meet the additional requirements]")
                    return self.system.states['APOLOGIZE']
                    
                return self.system.states['CONFIRM']
            else:
                # if no requirements found, assume user changed mind
                if follow_up_input.lower() in ['no', 'none', 'nothing', 'nope']:
                    print("[No additional requirements specified]")
                    return self.system.states['CONFIRM']
                else:
                    clarify_more = "I didn't understand your requirements. Please specify something like 'romantic', 'touristic', 'child-friendly', or say 'no' for no additional requirements."
                    print(f"System: {self.system.format_output(clarify_more)}")
                    return self.system.states['ASK_ADDITIONAL_REQUIREMENTS']
        else:
            if any(word in user_input.lower() for word in ['romantic', 'touristic', 'child', 'family', 'seat']):
                additional_reqs = self.parse_additional_requirements(user_input.lower())
                if additional_reqs:
                    print(f"[Additional requirements parsed: {additional_reqs}]")
                    self.system.additional_requirements.update(additional_reqs)
                    self.system.apply_inference_filtering()
                    
                    if not self.system.current_restaurant:
                        print("[No restaurants meet the additional requirements]")
                        return self.system.states['APOLOGIZE']
                        
                    return self.system.states['CONFIRM']
            
            clarify_msg = "I didn't understand. Please let me know if you have any specific requirements like romantic atmosphere, child-friendly environment, etc., or say 'no' if you don't have additional preferences."
            print(f"System: {self.system.format_output(clarify_msg)}")
            return self.system.states['ASK_ADDITIONAL_REQUIREMENTS']
    
    def parse_additional_requirements(self, user_input):
        # parse additional reqs from user input
        requirements = {}
        
        if any(word in user_input for word in ['tourist', 'touristic', 'popular', 'famous']):
            requirements['touristic'] = True
        elif any(word in user_input for word in ['not tourist', 'local', 'hidden', 'authentic']):
            requirements['touristic'] = False
            
        if any(word in user_input for word in ['romantic', 'romance', 'intimate', 'cozy', 'date']):
            requirements['romantic'] = True
        elif any(word in user_input for word in ['not romantic', 'casual', 'business']):
            requirements['romantic'] = False
            
        if any(word in user_input for word in ['child', 'children', 'kid', 'family', 'child-friendly']):
            requirements['children'] = True
        elif any(word in user_input for word in ['no child', 'adults only', 'quiet']):
            requirements['children'] = False
            
        if any(word in user_input for word in ['assigned seat', 'assigned seats', 'seat assignment', 'waiter choose']):
            requirements['assigned_seats'] = True
        elif any(word in user_input for word in ['choose seat', 'pick seat', 'free seating']):
            requirements['assigned_seats'] = False
            
        return requirements

    def apologize(self):
        # Stage 6: APOLOGIZE
        sorry_msg = "I'm sorry, no restaurants were found matching your criteria. Let me help you find what you're looking for."
        print(f"System: {self.system.format_output(sorry_msg)}")
         
        self.system.user_requirements = {'area': None, 'price': None, 'food': None}
        self.system.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
        return self.system.states['ASK_AREA']
    
    def suggest_restaurant(self):
        # Stage 8: SUGGEST_RESTAURANT
        # check romantic conflicts if requested
        if (hasattr(self.system, 'romantic_conflicts') and 
            self.system.romantic_conflicts and
            self.system.additional_requirements.get('romantic') is not None):
            # user requested romantic restaurant and we have conflicts - let them choose
            return self._handle_romantic_requirement_conflicts()
        
        # check touristic conflicts if requested
        if (hasattr(self.system, 'touristic_conflicts') and 
            self.system.touristic_conflicts and
            self.system.additional_requirements.get('touristic') is not None):
            # user requested touristic restaurant and we have conflicts - let them choose
            return self._handle_touristic_requirement_conflicts()
        
        if not self.system.current_restaurant:
            no_rest_msg = "I'm sorry but there is no restaurant serving that type of food"
            print(f"System: {self.system.format_output(no_rest_msg)}")
            return self.system.states['APOLOGIZE']
        
        restaurant = self.system.current_restaurant
        
        # check conflicts in inference rules before suggesting
        inference_result = self.system.inference_engine.apply_rules(restaurant)
        
        # handle romantic property conflicts if requested
        if ('conflict' in inference_result and 
            inference_result['conflict']['requires_user_input'] and
            inference_result['conflict']['property'] == 'romantic' and
            self.system.additional_requirements.get('romantic') is True):
            return self._handle_romantic_conflict(restaurant, inference_result['conflict'])
        
        suggestion_msg = format_restaurant_suggestion(restaurant)
        print(self.system.format_output(suggestion_msg))
        
        print(f"[Restaurant details: {restaurant['restaurantname']} - {restaurant['food']}, {restaurant['pricerange']}, {restaurant['area']}]")
        
        return self.system.states['INFORM']
    
    def _handle_romantic_conflict(self, restaurant, conflict_info):
        rule1 = conflict_info['rule1']
        rule2 = conflict_info['rule2']
        
        found_msg = f"I found '{restaurant['restaurantname']}', but I need your help to determine if it's romantic."
        print(f"System: {self.system.format_output(found_msg)}")
        
        mixed_msg = "This restaurant has both busy periods and allows for long stays, which gives mixed signals about romance."
        print(f"System: {self.system.format_output(mixed_msg)}")
        
        busy_msg = "Some people find busy restaurants less romantic because of the noise and crowds."
        print(f"System: {self.system.format_output(busy_msg)}")
        
        long_msg = "Others find restaurants where you can stay for a long time more romantic for intimate conversations."
        print(f"System: {self.system.format_output(long_msg)}")
        
        question_msg = "What's more important to you - avoiding crowds or having time for a long, relaxed meal?"
        print(f"System: {self.system.format_output(question_msg)}")
        
        choice_msg = "Please say 'avoid crowds' or 'long meal' to help me decide."
        print(f"System: {self.system.format_output(choice_msg)}")
        
        # wait for user preference
        while True:
            user_input = self.system.get_user_input("User: ").lower()
            if user_input:
                break
        
        # determine user's priority and apply it
        if 'avoid' in user_input or 'crowd' in user_input or 'busy' in user_input or 'quiet' in user_input:
            # user prioritizes avoiding crowds
            romantic_value = False
            explanation = f"Since you prefer to avoid crowds, I'd say {restaurant['restaurantname']} is not particularly romantic due to it being quite busy."
        elif 'long' in user_input or 'meal' in user_input or 'time' in user_input or 'stay' in user_input:
            # user prioritizes long stays
            romantic_value = True
            explanation = f"Since you value having time for a long meal, I'd say {restaurant['restaurantname']} is romantic because you can take your time and enjoy intimate conversations."
        else:
            # default to prioritizing long stays if unclear
            romantic_value = True
            explanation = f"Since you didn't specify clearly, I'll assume having time for a relaxed meal is more important. {restaurant['restaurantname']} is romantic because you can stay as long as you like."
        
        # update restaurant with resolved conflict
        restaurant['romantic'] = romantic_value
        restaurant['conflict_resolution'] = explanation
        
        # now suggest the restaurant with resolved conflict
        name = restaurant['restaurantname']
        area = restaurant['area']
        pricerange = restaurant['pricerange'] 
        food = restaurant['food']
        
        area_desc = " in the city centre" if area == 'centre' else f" in the {area} of town"
        main_desc = f"I recommend '{name}', it is {pricerange} {food} restaurant{area_desc}."
        
        full_msg = f"{main_desc} {explanation}"
        print(f"System: {self.system.format_output(full_msg)}")
        print(f"[Restaurant details: {restaurant['restaurantname']} - {restaurant['food']}, {restaurant['pricerange']}, {restaurant['area']}]")
        
        return self.system.states['INFORM']
        
    def _handle_romantic_requirement_conflicts(self):
        # handle conflicts when user specifically requested romantic restaurants
        # present restaurants with conflicts and ask user to resolve them
        conflicts = self.system.romantic_conflicts
        if not conflicts:
            return self.system.states['APOLOGIZE']
            
        # take the first restaurant with romantic conflict
        restaurant = conflicts[0]
        conflict_info = restaurant['inference_result']['conflict']
        
        found_msg = f"You asked for romantic restaurants. I found '{restaurant['restaurantname']}' which matches your other criteria,"
        print(f"System: {self.system.format_output(found_msg)}")
        conflict_msg = "but there's a conflict about whether it's romantic."
        print(f"System: {self.system.format_output(conflict_msg)}")
        
        rule1 = conflict_info['rule1']
        rule2 = conflict_info['rule2']
        
        # Make the messages more natural and understandable
        natural_msg = "I have mixed information about whether this restaurant is romantic."
        print(f"System: {self.system.format_output(natural_msg)}")
        
        # Explain the conflict in natural language
        if rule1['id'] == 5 and rule2['id'] == 6:
            # Busy vs Long stay conflict
            busy_msg = "On one hand, busy restaurants tend to be less romantic due to the noise and crowds."
            print(f"System: {self.system.format_output(busy_msg)}")
            long_msg = "On the other hand, restaurants where you can stay for a long time are generally more romantic for intimate conversations."
            print(f"System: {self.system.format_output(long_msg)}")
        elif rule1['id'] == 6 and rule2['id'] == 5:
            # Same conflict, different order
            long_msg = "On one hand, restaurants where you can stay for a long time are generally more romantic for intimate conversations."
            print(f"System: {self.system.format_output(long_msg)}")
            busy_msg = "On the other hand, busy restaurants tend to be less romantic due to the noise and crowds."
            print(f"System: {self.system.format_output(busy_msg)}")
        
        priority_msg = "What's more important to you for a romantic atmosphere - a quieter setting or the ability to stay for a long time?"
        print(f"System: {self.system.format_output(priority_msg)}")
        
        choice_help = "Please say 'quieter' if you prefer less crowded places, or 'long time' if you value being able to stay longer."
        print(f"System: {self.system.format_output(choice_help)}")
        
        # wait for user preference
        while True:
            user_input = self.system.get_user_input("User: ").lower()
            if user_input:
                break
        
        # determine user's priority and resolve conflict
        if 'quieter' in user_input or 'quiet' in user_input or 'not busy' in user_input or 'less crowd' in user_input:
            # user prioritizes quieter atmosphere - use rule 5 (busy -> not romantic)
            chosen_rule = rule1 if rule1['id'] == 5 else rule2
            romantic_value = chosen_rule['value']
            explanation = f"Based on your preference for a quieter atmosphere, {restaurant['restaurantname']} is {'romantic' if romantic_value else 'not romantic'} because it tends to be busy."
        elif 'long time' in user_input or 'long' in user_input or 'stay' in user_input:
            # user prioritizes long stay - use rule 6 (long stay -> romantic)
            chosen_rule = rule1 if rule1['id'] == 6 else rule2
            romantic_value = chosen_rule['value']
            explanation = f"Based on your preference for being able to stay for a long time, {restaurant['restaurantname']} is {'romantic' if romantic_value else 'not romantic'} because you can spend quality time there."
        else:
            # default to rule 6 if unclear
            chosen_rule = rule1 if rule1['id'] == 6 else rule2
            romantic_value = chosen_rule['value']
            explanation = f"Since you didn't specify clearly, I'll consider the ability to spend time together as more important. {restaurant['restaurantname']} is {'romantic' if romantic_value else 'not romantic'} for intimate dining."
        
        # check if resolved restaurant meets user's romantic requirement
        if romantic_value == self.system.additional_requirements.get('romantic', True):
            # restaurant meets requirement after conflict resolution
            restaurant['romantic'] = romantic_value
            restaurant['conflict_resolution'] = explanation
            
            # set this as current restaurant
            self.system.current_restaurant = restaurant
            self.system.current_restaurant_name = restaurant['restaurantname']
            
            # clear conflicts
            self.system.romantic_conflicts = []
            
            # suggest the resolved restaurant
            name = restaurant['restaurantname']
            area = restaurant['area']
            pricerange = restaurant['pricerange'] 
            food = restaurant['food']
            
            area_desc = " in the city centre" if area == 'centre' else f" in the {area} of town"
            main_desc = f"I recommend '{name}', it is {pricerange} {food} restaurant{area_desc}."
            
            full_suggestion = f"{main_desc} {explanation}"
            print(f"System: {self.system.format_output(full_suggestion)}")
            print(f"[Restaurant details: {restaurant['restaurantname']} - {restaurant['food']}, {restaurant['pricerange']}, {restaurant['area']}]")
            
            return self.system.states['INFORM']
        else:
            # even after resolution, restaurant doesn't meet requirement
            # try next restaurant or inform no match
            remaining_conflicts = self.system.romantic_conflicts[1:]
            if remaining_conflicts:
                self.system.romantic_conflicts = remaining_conflicts
                return self._handle_romantic_requirement_conflicts()  # try next one
            else:
                self.system.romantic_conflicts = []
                no_match_msg = "I'm sorry, none of the available restaurants meet your romantic requirement after resolving the conflicts."
                print(f"System: {self.system.format_output(no_match_msg)}")
                return self.system.states['APOLOGIZE']
    
    def _handle_touristic_requirement_conflicts(self):
        # handle conflicts when user specifically requested touristic restaurants
        # present restaurants with conflicts and ask user to resolve them
        conflicts = self.system.touristic_conflicts
        if not conflicts:
            return self.system.states['APOLOGIZE']
            
        # take the first restaurant with touristic conflict
        restaurant = conflicts[0]
        conflict_info = restaurant['inference_result']['conflict']
        
        found_msg = f"You asked for touristic restaurants. I found '{restaurant['restaurantname']}' which matches your other criteria,"
        print(f"System: {self.system.format_output(found_msg)}")
        conflict_msg = "but there's a conflict about whether it's touristic."
        print(f"System: {self.system.format_output(conflict_msg)}")
        
        rule1 = conflict_info['rule1']
        rule2 = conflict_info['rule2']
        
        # Make the messages more natural and understandable
        natural_msg = "I have mixed information about whether this restaurant is popular with tourists."
        print(f"System: {self.system.format_output(natural_msg)}")
        
        # Explain the conflict in natural language (rules 1 vs 2)
        if (rule1['id'] == 1 and rule2['id'] == 2) or (rule1['id'] == 2 and rule2['id'] == 1):
            cheap_good_msg = "On one hand, it's a cheap restaurant with good food, which usually attracts many tourists."
            print(f"System: {self.system.format_output(cheap_good_msg)}")
            romanian_msg = "On the other hand, it serves Romanian cuisine, which is less familiar to most tourists who prefer well-known food types."
            print(f"System: {self.system.format_output(romanian_msg)}")
        
        priority_msg = "What's more important to you - good value for money or familiar cuisine?"
        print(f"System: {self.system.format_output(priority_msg)}")
        
        choice_help = "Please say 'value' if you care more about good food at a good price, or 'familiar' if you prefer well-known cuisine types."
        print(f"System: {self.system.format_output(choice_help)}")
        
        # wait for user preference
        while True:
            user_input = self.system.get_user_input("User: ").lower()
            if user_input:
                break
        
        # determine user's priority and resolve conflict
        if 'value' in user_input or 'price' in user_input or 'cheap' in user_input or 'good food' in user_input:
            # user prioritizes value/price - use rule 1 (cheap+good -> touristic)
            chosen_rule = rule1 if rule1['id'] == 1 else rule2
            touristic_value = chosen_rule['value']
            explanation = f"Based on your preference for good value, {restaurant['restaurantname']} is {'popular with tourists' if touristic_value else 'not popular with tourists'} because it offers good food at reasonable prices."
        elif 'familiar' in user_input or 'cuisine' in user_input or 'food type' in user_input or 'romanian' in user_input:
            # user prioritizes familiar cuisine - use rule 2 (romanian -> not touristic)
            chosen_rule = rule1 if rule1['id'] == 2 else rule2
            touristic_value = chosen_rule['value']
            explanation = f"Based on your preference for familiar cuisine, {restaurant['restaurantname']} is {'popular with tourists' if touristic_value else 'not popular with tourists'} because Romanian food is less familiar to most visitors."
        else:
            # default to rule 1 if unclear (prioritize value)
            chosen_rule = rule1 if rule1['id'] == 1 else rule2
            touristic_value = chosen_rule['value']
            explanation = f"Since you didn't specify clearly, I'll consider the good value aspect as more important. {restaurant['restaurantname']} is {'popular with tourists' if touristic_value else 'not popular with tourists'} for its affordable quality."
        
        # check if resolved restaurant meets user's touristic requirement
        if touristic_value == self.system.additional_requirements.get('touristic', True):
            # restaurant meets requirement after conflict resolution
            restaurant['touristic'] = touristic_value
            restaurant['conflict_resolution'] = explanation
            
            # set this as current restaurant
            self.system.current_restaurant = restaurant
            self.system.current_restaurant_name = restaurant['restaurantname']
            
            # suggest the resolved restaurant
            name = restaurant['restaurantname']
            area = restaurant['area']
            pricerange = restaurant['pricerange'] 
            food = restaurant['food']
            
            area_desc = " in the city centre" if area == 'centre' else f" in the {area} of town"
            main_desc = f"I recommend '{name}', it is {pricerange} {food} restaurant{area_desc}."
            
            full_suggestion = f"{main_desc} {explanation}"
            print(f"System: {self.system.format_output(full_suggestion)}")
            print(f"[Restaurant details: {restaurant['restaurantname']} - {restaurant['food']}, {restaurant['pricerange']}, {restaurant['area']}]")
            
            return self.system.states['INFORM']
        else:
            # even after resolution, restaurant doesn't meet requirement
            # try next restaurant or inform no match
            remaining_conflicts = self.system.touristic_conflicts[1:]
            if remaining_conflicts:
                self.system.touristic_conflicts = remaining_conflicts
                return self._handle_touristic_requirement_conflicts()  # try next one
            else:
                self.system.touristic_conflicts = []
                no_match_msg = "I'm sorry, none of the available restaurants meet your touristic requirement after resolving the conflicts."
                print(f"System: {self.system.format_output(no_match_msg)}")
                return self.system.states['APOLOGIZE']
        
    def inform(self):
        # Stage 9: INFORM
        # check if we have a current restaurant to provide info about
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
        
        user_input = self.system.get_user_input("User: ")
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        # exit checks
        if detect_exit_command(user_input) or user_intent in ['bye', 'thankyou']:
            return self.system.states['GOODBYE']
         
        if self.system.allow_restarts and detect_restart_command(user_input):
            print("[User requested restart]") 
            self.system.user_requirements = {'area': None, 'price': None, 'food': None}
            self.system.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
            return self.system.states['WELCOME']
        
        # handle info requests if we have current restaurant
        if self.system.current_restaurant and user_intent == 'request':
            print(f"[Processing information request...]")
            next_state = self.system.provide_restaurant_info(user_input.lower())
            if next_state == 'await_next_request': 
                return self.system.states['INFORM']  # stay in INFORM for more requests
            else:
                return next_state
        
        # handle alternative requests if we have alternatives
        elif self.system.current_restaurant and user_intent == 'reqalts':
            print(f"[Processing alternative request...]")
            next_state = self.system.try_alternative()
            if next_state == self.system.states['SUGGEST_RESTAURANT']:
                print(f"[Alternative restaurant provided: {self.system.current_restaurant['restaurantname']}]")
                return next_state
            else:
                # no more alternatives, go to apologize
                return self.system.states['APOLOGIZE']
        
        # check for info requests even if not classified as such
        elif self.system.current_restaurant:
            utterance_lower = user_input.lower()
            info_keywords = ['food', 'cuisine', 'serve', 'food type', 'type of food', 'area', 'part of town', 
                           'location', 'where is it', 'which area', 'price', 'cost', 'price range',
                           'phone', 'number', 'address', 'postcode', 'post code']
            
            if any(keyword in utterance_lower for keyword in info_keywords):
                print(f"[Detected potential information request - processing...]")
                next_state = self.system.provide_restaurant_info(user_input.lower())
                if next_state == 'await_next_request': 
                    return self.system.states['INFORM']
                else:
                    return next_state
            
            # check for alternative request keywords
            alt_keywords = ['alternative', 'different', 'other', 'another', 'else', 'more options']
            if any(keyword in utterance_lower for keyword in alt_keywords):
                print(f"[Processing alternative request...]")
                next_state = self.system.try_alternative()
                if next_state == self.system.states['SUGGEST_RESTAURANT']:
                    print(f"[Alternative restaurant provided: {self.system.current_restaurant['restaurantname']}]")
                    return next_state
                else:
                    return self.system.states['APOLOGIZE']
        
        # handle preference changes or new searches
        if user_intent in ['affirm', 'inform']: 
            print("[Starting new restaurant search...]")
            self.system.user_requirements = {'area': None, 'price': None, 'food': None}
            self.system.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
            return self.system.states['ASK_AREA']
        elif user_intent in ['negate', 'deny']:
            return self.system.states['GOODBYE']
        else: 
            # try to extract preferences and restart if found
            prefs = PreferenceExtractor.extract_all(user_input.lower())
            if any(prefs.get(key) not in [None, 'dontcare'] for key in ['area', 'price', 'food']):
                print(f"[New preferences detected: {prefs}]")
                print("[Starting new search with updated preferences...]")
                # update preferences
                for key, value in prefs.items():
                    if value and value != 'dontcare':
                        self.system.user_requirements[key] = value
                return self.system.check_next_stage()
            else:
                # ask for clarification
                clarify_msg = "I didn't understand. Please let me know if you'd like restaurant information, alternatives, or want to search for different restaurants."
                print(f"System: {self.system.format_output(clarify_msg)}")
                return self.system.states['INFORM']
    
    def goodbye(self):
        # Stage 10: GOODBYE
        if self.system.allow_restarts:
            bye_msg = "Thank you for using the Cambridge restaurant system. Feel free to run the system again to search for more restaurants. Goodbye!"
        else:
            bye_msg = "Thank you for using the Cambridge restaurant system. Goodbye!"
        print(f"System: {self.system.format_output(bye_msg)}")
        self.system.conversation_ended = True
        return None
from ml_models import mlp_classifier
from lookup import RestaurantLookup
from utils import load_data
from preference_extraction import PreferenceExtractor

class RestaurantSystem:
    def __init__(self):
        self.restaurant_lookup = RestaurantLookup("data/restaurant_info.csv")
        self.preference_extractor = PreferenceExtractor()
        
        self.states = {
            'WELCOME': 'welcome',                    # Stage 1
            'ASK_AREA': 'ask_area',                 # Stage 2
            'ASK_PRICE': 'ask_price',               # Stage 3
            'ASK_FOOD_TYPE': 'ask_food_type',       # Stage 4
            'CONFIRM': 'confirm',                   # Stage 5
            'APOLOGIZE': 'apologize',               # Stage 6
            'SUGGEST_RESTAURANT': 'suggest_restaurant', # Stage 7
            'INFORM': 'inform',                     # Stage 8
            'GOODBYE': 'goodbye'                    # Stage 9
        }
        
        self.current_state = self.states['WELCOME']
        
        self.user_requirements = {
            'area': None,      
            'pricerange': None,  
            'food': None         
        }
        
        self.mlp_model = None
        self.mlp_vectorizer = None
        self.mlp_label_encoder = None
        self.is_trained = False
        
        self.current_restaurant = None
        self.current_restaurant_name = None
        self.alternatives = []
        self.suggestion_index = 0
        self.conversation_ended = False
        
        self.conversation_turn = 0
    
    def train_classifier(self):
        data = load_data()
        train_acts, test_acts, train_utterances, test_utterances = data['orig']

        self.mlp_model, self.mlp_vectorizer, self.mlp_label_encoder = mlp_classifier(
                train_acts, test_acts, train_utterances, test_utterances, return_model=True
            )
            
        self.is_trained = True
        print("MLP classifier trained successfully")
        return True
           
    def classify_utterance(self, user_utterance):
        user_utterance = user_utterance.lower()
    
        X_input = self.mlp_vectorizer.transform([user_utterance])
        prediction_int = self.mlp_model.predict(X_input)[0]
        predicted_act = self.mlp_label_encoder.inverse_transform([prediction_int])[0]
        return predicted_act

    def parse_user_input(self, user_input: str, context_stage=None):
        old_prefs = self.user_requirements.copy()
        
        # Check for restart command first
        if user_input.lower().strip() in ['start over', 'start again', 'reset']:
            print("[User requested restart - preferences reset]")
            self.user_requirements = {'area': None, 'pricerange': None, 'food': None}
            return 'restart'
        
        # Handle simple "any" responses based on context
        if user_input.strip().lower() in ['any', 'anything', "doesn't matter", "dont care", "any will do", "i dont care", "any type", "any food"]:
            if context_stage == 'ASK_AREA':
                self.user_requirements['area'] = 'dontcare'
                print(f"[Extracted: {{'area': 'dontcare'}}]")
                if self.user_requirements != old_prefs:
                    print(f"[Preferences updated: {self.user_requirements}]")
                return
            elif context_stage == 'ASK_PRICE':
                self.user_requirements['pricerange'] = 'dontcare'
                print(f"[Extracted: {{'pricerange': 'dontcare'}}]")
                if self.user_requirements != old_prefs:
                    print(f"[Preferences updated: {self.user_requirements}]")
                return
            elif context_stage == 'ASK_FOOD_TYPE':
                self.user_requirements['food'] = 'dontcare'
                print(f"[Extracted: {{'food': 'dontcare'}}]")
                if self.user_requirements != old_prefs:
                    print(f"[Preferences updated: {self.user_requirements}]")
                return
        
        # Extract preferences using the sophisticated extractor
        extracted_prefs = self.preference_extractor.extract_preferences(user_input)
        
        # Validate the extracted preferences
        validated_prefs, errors = self.preference_extractor.validate_preferences(extracted_prefs)
        
        if errors:
            print(f"[Validation warnings: {', '.join(errors)}]")
        
        # Context-aware preference updating
        if context_stage:
            # In specific asking stages, be more selective about what to update
            if context_stage == 'ASK_AREA' and 'area' in validated_prefs:
                self.user_requirements['area'] = validated_prefs['area']
            elif context_stage == 'ASK_PRICE' and 'pricerange' in validated_prefs:
                self.user_requirements['pricerange'] = validated_prefs['pricerange']
            elif context_stage == 'ASK_FOOD_TYPE' and 'food' in validated_prefs:
                self.user_requirements['food'] = validated_prefs['food']
            else:
                # Update any preferences found, but prioritize context
                for pref_type, value in validated_prefs.items():
                    if self.user_requirements[pref_type] is None:
                        self.user_requirements[pref_type] = value
        else:
            # General case: update all found preferences
            for pref_type, value in validated_prefs.items():
                if self.user_requirements[pref_type] is None:
                    self.user_requirements[pref_type] = value
        
        # Show what was extracted vs what was updated
        if validated_prefs:
            print(f"[Extracted: {validated_prefs}]")
        
        if self.user_requirements != old_prefs:
            print(f"[Preferences updated: {self.user_requirements}]")

    # Stage 1: WELCOME
    def welcome(self):
        print("System: Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?")
        
        user_input = input("User: ").strip()
        user_intent = self.classify_utterance(user_input)
        
        print(f"[Classified as: {user_intent}]")
        
        if user_intent in ['inform']:
            parse_result = self.parse_user_input(user_input.lower())
            if parse_result == 'restart':
                print(f"[State transition: WELCOME → WELCOME (restart)]")
                return self.states['WELCOME']
        
        next_state = self.check_next_stage()
        print(f"[State transition: WELCOME → {list(self.states.keys())[list(self.states.values()).index(next_state)]}]")
        
        return next_state
    
    # Stage 2: ASK_AREA
    def ask_area(self):
        print("System: What part of town do you have in mind?")
        
        user_input = input("User: ").strip()
        user_intent = self.classify_utterance(user_input)
        
        print(f"[Classified as: {user_intent}]")
                
        if user_intent in ['inform']:
            parse_result = self.parse_user_input(user_input.lower(), 'ASK_AREA')
            if parse_result == 'restart':
                print(f"[State transition: ASK_AREA → WELCOME (restart)]")
                return self.states['WELCOME']
        
        next_state = self.check_next_stage()
        print(f"[State transition: ASK_AREA → {list(self.states.keys())[list(self.states.values()).index(next_state)]}]")
        
        return next_state
    
    # Stage 3: ASK_PRICE
    def ask_price(self):
        print("System: Would you like something in the cheap, moderate, or expensive price range?")
        
        user_input = input("User: ").strip()
        user_intent = self.classify_utterance(user_input)
        
        print(f"[Classified as: {user_intent}]")
        
        if user_intent in ['inform']:
            parse_result = self.parse_user_input(user_input.lower(), 'ASK_PRICE')
            if parse_result == 'restart':
                print(f"[State transition: ASK_PRICE → WELCOME (restart)]")
                return self.states['WELCOME']
        
        next_state = self.check_next_stage()
        print(f"[State transition: ASK_PRICE → {list(self.states.keys())[list(self.states.values()).index(next_state)]}]")
        
        return next_state
    
    # Stage 4: ASK_FOOD_TYPE
    def ask_food_type(self):
        print("System: What kind of food would you like?")
        
        user_input = input("User: ").strip()
        user_intent = self.classify_utterance(user_input)
        
        print(f"[Classified as: {user_intent}]")
        
        if user_intent in ['inform']:
            parse_result = self.parse_user_input(user_input.lower(), 'ASK_FOOD_TYPE')
            if parse_result == 'restart':
                print(f"[State transition: ASK_FOOD_TYPE → WELCOME (restart)]")
                return self.states['WELCOME']
        
        next_state = self.check_next_stage()
        print(f"[State transition: ASK_FOOD_TYPE → {list(self.states.keys())[list(self.states.values()).index(next_state)]}]")
        
        return next_state
    
    # Stage 5: CONFIRM
    def confirm(self):
        prefs = []
        if self.user_requirements['area'] and self.user_requirements['area'] != 'dontcare':
            prefs.append(f"in the {self.user_requirements['area']} of town")
        if self.user_requirements['pricerange']:
            prefs.append(f"in the {self.user_requirements['pricerange']} price range")
        if self.user_requirements['food'] and self.user_requirements['food'] != 'dontcare':
            prefs.append(f"serving {self.user_requirements['food']} food")
        
        confirmation_msg = f"You are looking for a restaurant {' '.join(prefs)}, right?"
        print(f"System: {confirmation_msg}")
        
        user_input = input("User: ").strip()
        user_intent = self.classify_utterance(user_input)
        
        print(f"[Classified as: {user_intent}]")
        print(f"[Final preferences confirmed: {self.user_requirements}]")
        
        if user_intent == 'affirm':
            # Search for restaurants and suggest
            print("[Searching restaurant database...]")
            self.search_restaurants()
            next_state = self.states['SUGGEST_RESTAURANT']
            print(f"[State transition: CONFIRM → SUGGEST_RESTAURANT]")
            if self.current_restaurant:
                print(f"[Restaurant found: {self.current_restaurant['restaurantname']}]")
                print(f"[Alternatives available: {len(self.alternatives)}]")
            else:
                print("[No restaurants found matching criteria]")
        elif user_intent == 'negate':
            # User said no, apologize and restart
            next_state = self.states['APOLOGIZE']
            print(f"[State transition: CONFIRM → APOLOGIZE]")
        else:
            # Repeat confirmation
            next_state = self.states['CONFIRM']
            print(f"[State transition: CONFIRM → CONFIRM (repeat)]")
        
        return next_state
    
    # Stage 6: APOLOGIZE
    def apologize(self):
        print("I'm sorry for the confusion. Let me help you find what you're looking for.")
        
        # Reset preferences and start over
        self.user_requirements = {'area': None, 'pricerange': None, 'food': None}
        return self.states['ASK_AREA']
    
    # Stage 7: SUGGEST_RESTAURANT
    def suggest_restaurant(self):
        if not self.current_restaurant:
            # No restaurant found - provide proper message
            print("System: I'm sorry but there is no restaurant serving that type of food")
            print(f"[State transition: SUGGEST_RESTAURANT → INFORM]")
            return self.states['INFORM']
        
        # Present restaurant (only if not just provided info)
        restaurant = self.current_restaurant
        area_desc = f" in the {restaurant['area']} of town" if restaurant['area'] != 'centre' else " in the city centre"
        price_desc = f" and the prices are {restaurant['pricerange']}" if restaurant['pricerange'] != 'dontcare' else ""
        
        suggestion_msg = f"{restaurant['restaurantname']} is a nice place{area_desc}{price_desc}"
        print(f"System: {suggestion_msg}")
        print(f"[Restaurant details: {restaurant['restaurantname']} - {restaurant['food']}, {restaurant['pricerange']}, {restaurant['area']}]")
        
        return self.await_user_response()
        
    def await_user_response(self):
        """Wait for user response and handle accordingly"""
        user_input = input("User: ").strip()
        
        # Don't process empty input
        if not user_input:
            return self.await_user_response()
            
        user_intent = self.classify_utterance(user_input)
        print(f"[Classified as: {user_intent}]")
        
        # Check for restart command first
        if user_input.lower().strip() in ['start over', 'start again', 'reset']:
            print("[User requested restart]")
            print(f"[State transition: SUGGEST_RESTAURANT → WELCOME (restart)]")
            self.user_requirements = {'area': None, 'pricerange': None, 'food': None}
            return self.states['WELCOME']
        
        # Check if user is asking for a different restaurant (new preferences)
        input_lower = user_input.lower()
        if any(keyword in input_lower for keyword in ['is there', 'do you have', 'find me', 'looking for', 'want', 'need']):
            # This looks like a new search request - extract new preferences
            print(f"[Detecting new restaurant search request...]")
            new_prefs = self.preference_extractor.extract_preferences(user_input)
            if any(new_prefs[key] not in [None, 'dontcare'] for key in new_prefs):
                print(f"[New preferences detected: {new_prefs}]")
                # Update requirements with new preferences
                for key, value in new_prefs.items():
                    if value and value != 'dontcare':
                        self.user_requirements[key] = value
                print(f"[Updated requirements: {self.user_requirements}]")
                # Search with new criteria
                print(f"[State transition: SUGGEST_RESTAURANT → ASK_AREA (new search)]")
                return self.states['ASK_AREA']  # Let system re-confirm and search
        
        # Handle user response based on classified intent
        if user_intent == 'request':
            print(f"[Processing information request...]")
            next_state = self.provide_restaurant_info(user_input.lower())
            if next_state == 'await_next_request':
                print(f"[State transition: SUGGEST_RESTAURANT → SUGGEST_RESTAURANT (info provided, awaiting next)]")
                # Directly await next request without repeating suggestion
                return self.await_user_response()
            else:
                print(f"[State transition: SUGGEST_RESTAURANT → SUGGEST_RESTAURANT (info provided)]")
                return next_state
        elif user_intent == 'reqalts':
            print(f"[Processing alternative request...]")
            next_state = self.try_alternative()
            if next_state == self.states['SUGGEST_RESTAURANT']:
                print(f"[Alternative restaurant provided: {self.current_restaurant['restaurantname']}]")
                print(f"[State transition: SUGGEST_RESTAURANT → SUGGEST_RESTAURANT (alternative)]")
            else:
                print(f"[State transition: SUGGEST_RESTAURANT → INFORM (no alternatives)]")
            return next_state
        elif user_intent in ['bye', 'thankyou']:
            print(f"[State transition: SUGGEST_RESTAURANT → GOODBYE]")
            return self.states['GOODBYE']
        else:
            # Repeat or continue
            repeat_msg = f"{self.current_restaurant['restaurantname']} is a great restaurant"
            print(f"System: {repeat_msg}")
            print(f"[State transition: SUGGEST_RESTAURANT → SUGGEST_RESTAURANT (repeat)]")
            return self.await_user_response()
    
    # Stage 8: INFORM
    def inform(self):
        print("System: Would you like to try a different type of food or change your preferences?")
        
        user_input = input("User: ").strip()
        user_intent = self.classify_utterance(user_input)
        
        print(f"[Classified as: {user_intent}]")
        
        # Check for restart command
        if user_input.lower().strip() in ['start over', 'start again', 'reset']:
            print("[User requested restart]")
            print(f"[State transition: INFORM → WELCOME (restart)]")
            self.user_requirements = {'area': None, 'pricerange': None, 'food': None}
            return self.states['WELCOME']
        
        if user_intent in ['affirm', 'inform']:
            # User wants to search for restaurants again
            print("[Restarting restaurant search...]")
            self.user_requirements = {'area': None, 'pricerange': None, 'food': None}
            print(f"[State transition: INFORM → ASK_AREA]")
            return self.states['ASK_AREA']
        elif user_intent in ['bye', 'thankyou']:
            print(f"[State transition: INFORM → GOODBYE]")
            return self.states['GOODBYE']
        else:
            # Continue to goodbye
            print(f"[State transition: INFORM → GOODBYE]")
            return self.states['GOODBYE']
    
    # Stage 9: GOODBYE
    def goodbye(self):
        print("Thank you for using the Cambridge restaurant system. Goodbye!")
        self.conversation_ended = True
        return None
    
    # Helper functions
    def check_next_stage(self):
        if not self.user_requirements['area']:
            return self.states['ASK_AREA']
        elif not self.user_requirements['pricerange']:
            return self.states['ASK_PRICE']
        elif not self.user_requirements['food']:
            return self.states['ASK_FOOD_TYPE']
        else:
            return self.states['CONFIRM']
    
    def search_restaurants(self):
        filters = {
            "food": self.user_requirements['food'] or "dontcare",
            "pricerange": self.user_requirements['pricerange'] or "dontcare", 
            "area": self.user_requirements['area'] or "dontcare"
        }
        
        # Use your lookup class - returns restaurant_name, alternatives
        restaurant_name, alternatives = self.restaurant_lookup.lookup(filters)
        
        if restaurant_name:
            # Find the full restaurant data
            self.current_restaurant_name = restaurant_name
            # Get full restaurant data from dataframe
            restaurant_row = self.restaurant_lookup.df[
                self.restaurant_lookup.df['restaurantname'].str.lower() == restaurant_name.lower()
            ]
            if not restaurant_row.empty:
                self.current_restaurant = restaurant_row.iloc[0].to_dict()
            else:
                self.current_restaurant = None
            
            self.alternatives = alternatives
            self.suggestion_index = 0
        else:
            self.current_restaurant = None
            self.alternatives = []
    
    def provide_restaurant_info(self, user_input: str):
        if not self.current_restaurant:
            print("System: I'm sorry, I don't have any restaurant information available to provide details.")
            return self.states['INFORM']
        
        restaurant = self.current_restaurant
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
        
        # If specific requests were made, provide only those
        if phone_requested or address_requested or postcode_requested:
            if len(info_parts) == 1:
                print(f"System: {info_parts[0]}.")
            elif len(info_parts) == 2:
                print(f"System: {info_parts[0]} and {info_parts[1].lower()}.")
            else:
                print(f"System: {info_parts[0]}, {info_parts[1].lower()}, and {info_parts[2].lower()}.")
        else:
            # Default: provide all available information if no specific request
            response_parts = []
            response_parts.append(f"The phone number of {restaurant['restaurantname']} is {phone}")
            
            if has_address:
                response_parts.append(f"it is on {addr}")
            
            if has_postcode:
                response_parts.append(f"the post code is {postcode}")
            
            if len(response_parts) == 1:
                print(f"System: {response_parts[0]}.")
            elif len(response_parts) == 2:
                print(f"System: {response_parts[0]} and {response_parts[1]}.")
            else:
                print(f"System: {response_parts[0]}, {response_parts[1]}, and {response_parts[2]}.")
        
        # After providing info, wait for next user input instead of repeating suggestion
        return 'await_next_request'
    
    def try_alternative(self):
        if self.alternatives and self.suggestion_index < len(self.alternatives):
            alt_restaurant_dict = self.alternatives[self.suggestion_index]
            self.current_restaurant = alt_restaurant_dict
            self.current_restaurant_name = alt_restaurant_dict['restaurantname']
            self.suggestion_index += 1
            return self.states['SUGGEST_RESTAURANT']
        else:
            print("I'm sorry, I don't have any more alternatives to suggest.")
            return self.states['INFORM']
    
    def run_conversation(self):
        """Main conversation loop - clean and simple 9-stage flow with ML classification"""
        print("=" * 60)
        print("CAMBRIDGE RESTAURANT SYSTEM DIALOG")
        print("=" * 60)
        
        while self.current_state and not self.conversation_ended:
            try:
                self.conversation_turn += 1
                print(f"\n--- Turn {self.conversation_turn} ---")
                current_state_name = list(self.states.keys())[list(self.states.values()).index(self.current_state)]
                print(f"[Current State: {current_state_name}]\n")
                
                # Execute current stage
                if self.current_state == self.states['WELCOME']:
                    self.current_state = self.welcome()
                    
                elif self.current_state == self.states['ASK_AREA']:
                    self.current_state = self.ask_area()
                    
                elif self.current_state == self.states['ASK_PRICE']:
                    self.current_state = self.ask_price()
                    
                elif self.current_state == self.states['ASK_FOOD_TYPE']:
                    self.current_state = self.ask_food_type()
                    
                elif self.current_state == self.states['CONFIRM']:
                    self.current_state = self.confirm()
                    
                elif self.current_state == self.states['APOLOGIZE']:
                    self.current_state = self.apologize()
                    
                elif self.current_state == self.states['SUGGEST_RESTAURANT']:
                    self.current_state = self.suggest_restaurant()
                    
                elif self.current_state == self.states['INFORM']:
                    self.current_state = self.inform()
                    
                elif self.current_state == self.states['GOODBYE']:
                    self.current_state = self.goodbye()
                    
                # Visual separator between stages
                if self.current_state and not self.conversation_ended:
                    print("-" * 30)
                    
            except KeyboardInterrupt:
                print("\nConversation interrupted by user.")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
        
        print("=" * 60)
        print(f"CONVERSATION COMPLETED - Total turns: {self.conversation_turn}")
        print("=" * 60)
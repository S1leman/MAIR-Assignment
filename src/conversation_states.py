from utils import format_restaurant_suggestion, detect_restart_command, detect_new_search_request
from preference_extraction import PreferenceExtractor

class ConversationStates: 
    
    def __init__(self, system):
        self.system = system 
    
    # Stage 1: WELCOME
    def welcome(self):
        print("System: Hello, welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?")
        
        user_input = input("User: ").strip()
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower())
            if parse_result == 'restart': 
                return self.system.states['WELCOME']
        
        next_state = self.system.check_next_stage() 
        
        return next_state
    
    # Stage 2: ASK_AREA
    def ask_area(self):
        print("System: What part of town do you have in mind?")
        
        user_input = input("User: ").strip()
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
                
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower(), 'ASK_AREA')
            if parse_result == 'restart':
                return self.system.states['WELCOME']
        
        next_state = self.system.check_next_stage()

        return next_state
    
    # Stage 3: ASK_PRICE
    def ask_price(self):
        print("System: Would you like something in the cheap, moderate, or expensive price range?")
        
        user_input = input("User: ").strip()
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower(), 'ASK_PRICE')
            if parse_result == 'restart':
                return self.system.states['WELCOME']
        
        next_state = self.system.check_next_stage()
        
        return next_state
    
    # Stage 4: ASK_FOOD_TYPE
    def ask_food_type(self):
        print("System: What kind of food would you like?")
        
        user_input = input("User: ").strip()
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        
        if user_intent in ['inform']:
            parse_result = self.system.parse_user_input(user_input.lower(), 'ASK_FOOD_TYPE')
            if parse_result == 'restart':
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
        
        user_input = input("User: ").strip()
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
        print(f"[Final preferences confirmed: {self.system.user_requirements}]")
        
        if user_intent == 'affirm':
            # Search for restaurants and suggest
            print("[Searching restaurant database...]")
            self.system.search_restaurants()
            next_state = self.system.states['SUGGEST_RESTAURANT']
            if self.system.current_restaurant:
                print(f"[Restaurant found: {self.system.current_restaurant['restaurantname']}]")
                print(f"[Alternatives available: {len(self.system.alternatives)}]")
            else:
                print("[No restaurants found matching criteria]")
        elif user_intent == 'negate':
            # User said no, apologize and restart
            next_state = self.system.states['APOLOGIZE']
        else:
            # Repeat confirmation
            next_state = self.system.states['CONFIRM']
        
        return next_state
    
    # Stage 6: APOLOGIZE
    def apologize(self):
        print("I'm sorry for the confusion. Let me help you find what you're looking for.")
         
        self.system.user_requirements = {'area': None, 'price': None, 'food': None}
        return self.system.states['ASK_AREA']
    
    # Stage 7: SUGGEST_RESTAURANT
    def suggest_restaurant(self):
        if not self.system.current_restaurant:
            print("System: I'm sorry but there is no restaurant serving that type of food")
            return self.system.states['INFORM']
        
        restaurant = self.system.current_restaurant
        suggestion_msg = format_restaurant_suggestion(restaurant)
        print(f"System: {suggestion_msg}")
        print(f"[Restaurant details: {restaurant['restaurantname']} - {restaurant['food']}, {restaurant['pricerange']}, {restaurant['area']}]")
        
        return self.await_user_response()
        
    def await_user_response(self):
        user_input = input("User: ").strip()
        
        # Don't process empty input
        if not user_input:
            return self.await_user_response()
            
        user_intent = self.system.classify_utterance(user_input)
        print(f"\n[Classified as: {user_intent}]")
        
        if detect_restart_command(user_input):
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
    
    # Stage 8: INFORM
    def inform(self):
        print("System: Would you like to try a different type of food or change your preferences?")
        
        user_input = input("User: ").strip()
        user_intent = self.system.classify_utterance(user_input)
        
        print(f"\n[Classified as: {user_intent}]")
         
        if detect_restart_command(user_input):
            print("[User requested restart]") 
            self.system.user_requirements = {'area': None, 'price': None, 'food': None}
            return self.system.states['WELCOME']
        
        if user_intent in ['affirm', 'inform']: 
            print("[Restarting restaurant search...]")
            self.system.user_requirements = {'area': None, 'price': None, 'food': None} 
            return self.system.states['ASK_AREA']
        elif user_intent in ['bye', 'thankyou']: 
            return self.system.states['GOODBYE']
        else: 
            return self.system.states['GOODBYE']
    
    # Stage 9: GOODBYE
    def goodbye(self):
        print("Thank you for using the Cambridge restaurant system. Goodbye!")
        self.system.conversation_ended = True
        return None
from ml_models import mlp_classifier
from lookup import RestaurantLookup
from utils import load_data
from preference_extraction import PreferenceExtractor
from conversation_states import ConversationStates

class RestaurantSystem:
    def __init__(self): 
        self.conversation_states = ConversationStates(self)
         
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
        
        # ML classifier
        self.mlp_model = None
        self.mlp_vectorizer = None
        self.mlp_label_encoder = None
        self.is_trained = False
        
        # Restaurant tracking
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
        """Extract and update user preferences from input"""
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
        validated_prefs, errors = self.preference_extractor.validate_preferences(extracted_prefs)
        
        if errors:
            print(f"[Validation warnings: {', '.join(errors)}]")
        
        # Context-aware preference updating
        if context_stage:
            if context_stage == 'ASK_AREA' and 'area' in validated_prefs:
                self.user_requirements['area'] = validated_prefs['area']
            elif context_stage == 'ASK_PRICE' and 'pricerange' in validated_prefs:
                self.user_requirements['pricerange'] = validated_prefs['pricerange']
            elif context_stage == 'ASK_FOOD_TYPE' and 'food' in validated_prefs:
                self.user_requirements['food'] = validated_prefs['food']
            else:
                for pref_type, value in validated_prefs.items():
                    if self.user_requirements[pref_type] is None:
                        self.user_requirements[pref_type] = value
        else:
            for pref_type, value in validated_prefs.items():
                if self.user_requirements[pref_type] is None:
                    self.user_requirements[pref_type] = value
        
        if validated_prefs:
            print(f"[Extracted: {validated_prefs}]")
        
        if self.user_requirements != old_prefs:
            print(f"[Preferences updated: {self.user_requirements}]")

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
        
        restaurant_name, alternatives = self.restaurant_lookup.lookup(filters)
        
        if restaurant_name:
            self.current_restaurant_name = restaurant_name
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
        
        # Generate response
        if phone_requested or address_requested or postcode_requested:
            if len(info_parts) == 1:
                print(f"System: {info_parts[0]}.")
            elif len(info_parts) == 2:
                print(f"System: {info_parts[0]} and {info_parts[1].lower()}.")
            else:
                print(f"System: {info_parts[0]}, {info_parts[1].lower()}, and {info_parts[2].lower()}.")
        else:
            # Default: provide all available information
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
        print("=" * 60)
        print("CAMBRIDGE RESTAURANT SYSTEM DIALOG")
        print("=" * 60)
        
        while self.current_state and not self.conversation_ended:
            try:
                self.conversation_turn += 1
                print(f"\n--- Turn {self.conversation_turn} ---")
                current_state_name = list(self.states.keys())[list(self.states.values()).index(self.current_state)]
                print(f"[Current State: {current_state_name}]\n")
                
                # Execute current stage using the conversation states handler
                if self.current_state == self.states['WELCOME']:
                    self.current_state = self.conversation_states.welcome()
                    
                elif self.current_state == self.states['ASK_AREA']:
                    self.current_state = self.conversation_states.ask_area()
                    
                elif self.current_state == self.states['ASK_PRICE']:
                    self.current_state = self.conversation_states.ask_price()
                    
                elif self.current_state == self.states['ASK_FOOD_TYPE']:
                    self.current_state = self.conversation_states.ask_food_type()
                    
                elif self.current_state == self.states['CONFIRM']:
                    self.current_state = self.conversation_states.confirm()
                    
                elif self.current_state == self.states['APOLOGIZE']:
                    self.current_state = self.conversation_states.apologize()
                    
                elif self.current_state == self.states['SUGGEST_RESTAURANT']:
                    self.current_state = self.conversation_states.suggest_restaurant()
                    
                elif self.current_state == self.states['INFORM']:
                    self.current_state = self.conversation_states.inform()
                    
                elif self.current_state == self.states['GOODBYE']:
                    self.current_state = self.conversation_states.goodbye()
                    
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
 
if __name__ == "__main__":
    system = RestaurantSystem()
    system.train_classifier()
    system.run_conversation()
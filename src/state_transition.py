from ml_models import mlp_classifier
from lookup import RestaurantLookup
from utils import (load_data, format_restaurant_info_response, 
                   detect_restart_command, get_state_name_from_value, update_preferences_with_context, log_preference_changes,
                   execute_conversation_state, train_classifier, load_trained_model)
from preference_extraction import PreferenceExtractor
from conversation_states import ConversationStates
from baseline_models import (rules_baseline_model,majority_baseline_model)

class RestaurantSystem:
    def __init__(self): 
        self.conversation_states = ConversationStates(self)
         
        self.restaurant_lookup = RestaurantLookup("data/restaurant_info.csv")
         
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
            'price': None, 
            'food': None         
        }
        
        # ML classifier
        self.mlp_model = None
        self.mlp_vectorizer = None
        self.mlp_label_encoder = None
        self.is_trained = False
        self.model_path = "models/"  # Directory of trained model
        self.model_files = {
            'model': 'mlp_model.pkl',
            'vectorizer': 'mlp_vectorizer.pkl', 
            'label_encoder': 'mlp_label_encoder.pkl'
        }
        #Baseline classifier
        self.classifier_type = None
        self.majority_label = "inform"

        # Restaurant tracking
        self.current_restaurant = None
        self.current_restaurant_name = None
        self.alternatives = []
        self.suggestion_index = 0
        self.conversation_ended = False
        
        self.conversation_turn = 0
    
    def ensure_model_ready(self):
        
        if self.is_trained:
            return True
        
        if load_trained_model(self):
            return True
            
        print("No pre-trained model available. Training new model...")
        return train_classifier(self)
           
    def classify_utterance(self, user_utterance):

        if self.classifier_type == "mlp":
            user_utterance = user_utterance.lower()
            X_input = self.mlp_vectorizer.transform([user_utterance])
            prediction_int = self.mlp_model.predict(X_input)[0]
            predicted_act = self.mlp_label_encoder.inverse_transform([prediction_int])[0]
            return predicted_act
        
        elif self.classifier_type == "majority":
            return majority_baseline_model([user_utterance], self.majority_label)[0]
        
        elif self.classifier_type == "rules":
            return rules_baseline_model([user_utterance])[0]
        
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")



    def parse_user_input(self, user_input: str, context_stage=None): 
        old_prefs = self.user_requirements.copy()
        
        # Check for restart command first
        if detect_restart_command(user_input):
            print("[User requested restart - preferences reset]")
            self.user_requirements = {'area': None, 'price': None, 'food': None}
            return 'restart'
        
        extracted_prefs = PreferenceExtractor.extract_all(user_input)

        print(f"[Extracted preferences: {extracted_prefs}]")    
        # Update user requirements based on context
        update_preferences_with_context(self.user_requirements, extracted_prefs, context_stage)
        
        # Log all changes and results
        log_preference_changes(extracted_prefs, self.user_requirements, old_prefs, [])

    def check_next_stage(self): 
        if not self.user_requirements['area']:
            return self.states['ASK_AREA']
        elif not self.user_requirements['price']:
            return self.states['ASK_PRICE']
        elif not self.user_requirements['food']:
            return self.states['ASK_FOOD_TYPE']
        else:
            return self.states['CONFIRM']
    
    def search_restaurants(self): 
        filters = {
            "food": self.user_requirements['food'] or "dontcare",
            "pricerange": self.user_requirements['price'] or "dontcare",
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
        
        response = format_restaurant_info_response(self.current_restaurant, user_input)
        print(f"System: {response}")
        
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
        self._print_conversation_header()
        
        while self.current_state and not self.conversation_ended:
            try:
                self._handle_conversation_turn()
            except KeyboardInterrupt:
                print("\nConversation interrupted by user.")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
        
        self._print_conversation_footer()
    
    def _print_conversation_header(self): 
        print("=" * 60)
        print("CAMBRIDGE RESTAURANT SYSTEM DIALOG")
        print("=" * 60)
    
    def _print_conversation_footer(self): 
        print("=" * 60)
        print(f"CONVERSATION COMPLETED - Total turns: {self.conversation_turn}")
        print("=" * 60)
    
    def _handle_conversation_turn(self): 
        self.conversation_turn += 1
        current_state_name = get_state_name_from_value(self.states, self.current_state)
        
        print(f"\n--- Turn {self.conversation_turn} ---")
        print(f"[Current State: {current_state_name}]\n")
        
        # Execute current state and get next state
        self.current_state = execute_conversation_state(self, self.current_state, self.states)
        
        if self.current_state and not self.conversation_ended:
            print("-" * 30)
 
if __name__ == "__main__":
    system = RestaurantSystem()
    system.ensure_model_ready()  # This will load existing model or train new one
    system.run_conversation()
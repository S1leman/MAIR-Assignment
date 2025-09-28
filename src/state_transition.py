from lookup import RestaurantLookup
from utils import (format_restaurant_info_response, 
                   detect_restart_command, get_state_name_from_value, update_preferences_with_context, log_preference_changes,
                   execute_conversation_state, train_classifier, load_trained_model)
from preference_extraction import PreferenceExtractor
from conversation_states import ConversationStates
from baseline_models import (rules_baseline_model,majority_baseline_model)
from inference_engine import InferenceEngine

class RestaurantSystem:
    def __init__(self): 
        self.conversation_states = ConversationStates(self)
         
        self.restaurant_lookup = RestaurantLookup("data/restaurant_info_updated.csv")
        
        self.inference_engine = InferenceEngine()
         
        self.states = {
            'WELCOME': 'welcome',                    
            'ASK_AREA': 'ask_area',                 
            'ASK_PRICE': 'ask_price',               
            'ASK_FOOD_TYPE': 'ask_food_type',       
            'ASK_ADDITIONAL_REQUIREMENTS': 'ask_additional_requirements', 
            'APOLOGIZE': 'apologize',               
            'CONFIRM': 'confirm',                   
            'SUGGEST_RESTAURANT': 'suggest_restaurant', 
            'INFORM': 'inform',                     
            'GOODBYE': 'goodbye'                    
        }
        
        self.current_state = self.states['WELCOME']
         
        self.user_requirements = {
            'area': None,      
            'price': None, 
            'food': None        
        }
        
        # additional reqs for inference rules
        self.additional_requirements = {
            'touristic': None,
            'assigned_seats': None,
            'children': None,
            'romantic': None
        }
        
        # config settings
        self.allow_restarts = True
        self.output_caps = False
        self.use_tts = False
        self.tts_engine = None
        
        # for handling conflicts
        self.romantic_conflicts = []
        self.touristic_conflicts = []
        
        # ML classifier components
        self.mlp_model = None
        self.mlp_vectorizer = None
        self.mlp_label_encoder = None
        self.is_trained = False
        self.model_path = "models/"
        self.model_files = {
            'model': 'mlp_model.pkl',
            'vectorizer': 'mlp_vectorizer.pkl', 
            'label_encoder': 'mlp_label_encoder.pkl'
        }
        # baseline classifier
        self.classifier_type = None
        self.majority_label = "inform"

        # restaurant tracking
        self.current_restaurant = None
        self.current_restaurant_name = None
        self.alternatives = []
        self.suggestion_index = 0
        self.conversation_ended = False
        
        # conflict resolution
        self.pending_conflict = None
        self.conflict_restaurant = None
        
        self.conversation_turn = 0
    
    def format_output(self, message):
        """format system output and speak if TTS enabled"""
        formatted_message = message.upper() if self.output_caps else message
        
        # speak the message if TTS is enabled
        if self.use_tts and self.tts_engine:
            try:
                self.tts_engine.say(formatted_message)
                self.tts_engine.runAndWait()
            except:
                # if TTS fails, try reinitializing once
                try:
                    self.initialize_tts()
                    self.tts_engine.say(formatted_message)
                    self.tts_engine.runAndWait()
                except:
                    pass  # ignore TTS errors completely
        
        return formatted_message
    
    def initialize_tts(self):
        """simple TTS initialization"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)
            self.tts_engine.setProperty('volume', 0.8)
            return True
        except:
            raise Exception("TTS initialization failed")
    
    def _normalize_slot_values(self):
        """normalize slot values to match CSV data format"""
        a = self.user_requirements.get('area')
        if a == 'center': 
            self.user_requirements['area'] = 'centre'
        
        p = self.user_requirements.get('price')
        if p == 'moderately priced': 
            self.user_requirements['price'] = 'moderate'
    
    def ensure_model_ready(self):
        if self.is_trained:
            return True
        
        if load_trained_model(self):
            return True
            
        print("No pre-trained model available. Training new model...")
        return train_classifier(self)
    
    def get_user_input(self, prompt: str = "User: ") -> str:
        """get user input from text input"""
        return input(prompt)
           
    def classify_utterance(self, user_utterance):
        # handle empty or whitespace-only input
        if not user_utterance or not user_utterance.strip():
            return "null"
            
        try:
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
        except Exception as e:
            print(f"[Classification error: {e}]")
            return "null"

    def parse_user_input(self, user_input: str, context_stage=None): 
        old_prefs = self.user_requirements.copy()
        
        # check restart command first
        if detect_restart_command(user_input):
            if self.allow_restarts:
                print("[User requested restart - preferences reset]")
                self.user_requirements = {'area': None, 'price': None, 'food': None}
                return 'restart'
            else:
                print("[Restart requested but disabled]")
                return None
        

        extracted_prefs = PreferenceExtractor.extract_all(user_input, context=context_stage)

        print(f"[Extracted preferences: {extracted_prefs}]")    

        update_preferences_with_context(self.user_requirements, extracted_prefs, context_stage)
        
        # normalize slot values to match CSV format
        self._normalize_slot_values()
        

        log_preference_changes(extracted_prefs, self.user_requirements, old_prefs, [])

    def check_next_stage(self): 

        self._normalize_slot_values()
        
        if not self.user_requirements['area']:
            return self.states['ASK_AREA']
        elif not self.user_requirements['price']:
            return self.states['ASK_PRICE']
        elif not self.user_requirements['food']:
            return self.states['ASK_FOOD_TYPE']
        else:
            # all basic reqs collected, search for restaurants
            print("[All basic preferences collected. Searching restaurant database...]")
            self.search_restaurants()
            
            # check if any restaurants found
            if not self.current_restaurant:
                print("[No restaurants found matching basic criteria]")
                return self.states['APOLOGIZE']
            
            # calculate total matches (current + alternatives)
            total_matches = 1 + len(self.alternatives)
            print(f"[Found {total_matches} restaurant(s) matching criteria]")
            
            # always ask for additional requirements to allow user to narrow down selection
            print("[Proceeding to collect additional requirements]")
            return self.states['ASK_ADDITIONAL_REQUIREMENTS']
    
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
    
    def apply_inference_filtering(self):
        """apply inference rules to filter restaurants based on additional reqs"""
        if not self.additional_requirements:
            return
        
        # get active additional reqs (non-None values)
        active_requirements = {k: v for k, v in self.additional_requirements.items() if v is not None}
        
        if not active_requirements:
            return
        
        print(f"[Applying inference rules with requirements: {active_requirements}]")
        
        # get all candidate restaurants (current + alternatives)
        all_candidates = []
        if self.current_restaurant:
            all_candidates.append(self.current_restaurant)
        
        # convert alternatives to full restaurant objects
        for alt in self.alternatives:
            if isinstance(alt, dict):
                all_candidates.append(alt)
            else:
                found = self.restaurant_lookup.find_restaurant_by_name(alt)
                if found:
                    all_candidates.append(found)
        
        # filter restaurants using inference engine
        filter_result = self.inference_engine.filter_restaurants_by_requirements(
            all_candidates, active_requirements
        )
        
        # handle romantic conflicts
        if filter_result['has_romantic_conflicts']:
            # store conflict restaurants for user resolution
            self.romantic_conflicts = filter_result['romantic_conflict_restaurants']
            print(f"[Found {len(self.romantic_conflicts)} restaurants with romantic conflicts]")
            return  # will be handled in conversation state
        
        # handle touristic conflicts
        if filter_result['has_touristic_conflicts']:
            # store conflict restaurants for user resolution
            self.touristic_conflicts = filter_result['touristic_conflict_restaurants']
            print(f"[Found {len(self.touristic_conflicts)} restaurants with touristic conflicts]")
            return  # will be handled in conversation state
        
        filtered_restaurants = filter_result['restaurants']
        print(f"[Filtered from {len(all_candidates)} to {len(filtered_restaurants)} restaurants]")
        
        if filtered_restaurants:
            # update current restaurant to first filtered result
            self.current_restaurant = filtered_restaurants[0]
            self.current_restaurant_name = filtered_restaurants[0]['restaurantname']
            
            # update alternatives with remaining filtered restaurants
            remaining_restaurants = filtered_restaurants[1:]
            self.alternatives = remaining_restaurants  # store full restaurant dicts
            self.suggestion_index = 0
            
            print(f"[Updated current restaurant: {self.current_restaurant_name}]")
            alt_names = [r.get('restaurantname', 'Unknown') for r in self.alternatives if isinstance(r, dict)]
            print(f"[Updated alternatives: {alt_names}]")
        else:
            # no restaurants meet the additional reqs
            self.current_restaurant = None
            self.current_restaurant_name = None
            self.alternatives = []
            print("[No restaurants match the additional requirements]")
    
    def provide_restaurant_info(self, user_input: str): 
        if not self.current_restaurant:
            no_info_msg = "I'm sorry, I don't have any restaurant information available to provide details."
            print(f"System: {self.format_output(no_info_msg)}")
            return self.states['APOLOGIZE']  # changed from INFORM
        
        try:
            response = format_restaurant_info_response(self.current_restaurant, user_input)
            print(f"System: {self.format_output(response)}")
        except Exception as e:
            error_msg = f"I'm sorry, I'm having trouble accessing the restaurant information right now. Error: {e}"
            print(f"System: {self.format_output(error_msg)}")
            return self.states['APOLOGIZE']  # changed from INFORM
        
        return 'await_next_request'
    
    def try_alternative(self): 
        if self.alternatives and self.suggestion_index < len(self.alternatives):
            alt_restaurant_dict = self.alternatives[self.suggestion_index]
            
            # ensure we have a proper dict
            if not isinstance(alt_restaurant_dict, dict):
                # if it's a string (restaurant name), look it up in the database
                restaurant_name = alt_restaurant_dict
                restaurant_row = self.restaurant_lookup.df[
                    self.restaurant_lookup.df['restaurantname'].str.lower() == restaurant_name.lower()
                ]
                if not restaurant_row.empty:
                    alt_restaurant_dict = restaurant_row.iloc[0].to_dict()
                else:
                    error_msg = "I'm sorry, there was an error processing the alternative restaurant."
                    print(f"System: {self.format_output(error_msg)}")
                    return self.states['APOLOGIZE']  # changed from INFORM
                
            self.current_restaurant = alt_restaurant_dict
            self.current_restaurant_name = alt_restaurant_dict.get('restaurantname', 'Unknown Restaurant')
            self.suggestion_index += 1
            return self.states['SUGGEST_RESTAURANT']
        else:
            no_more_msg = "I'm sorry, I don't have any more alternatives to suggest."
            print(f"System: {self.format_output(no_more_msg)}")
            return self.states['APOLOGIZE']  # changed from INFORM
    
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
        
        # execute current state and get next state
        self.current_state = execute_conversation_state(self, self.current_state, self.states)
        
        if self.current_state and not self.conversation_ended:
            print("-" * 30)
 
if __name__ == "__main__":
    system = RestaurantSystem()
    system.ensure_model_ready()  # load existing model or train new one
    system.run_conversation()
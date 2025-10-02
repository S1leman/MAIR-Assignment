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
        """format system output"""
        formatted_message = message.upper() if self.output_caps else message
        return formatted_message
    
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
                self.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
                
                # Clear inference flags to ensure filtering works after restart
                if hasattr(self, '_inference_applied'):
                    delattr(self, '_inference_applied')
                if hasattr(self, '_handling_conflict_restaurant'):
                    delattr(self, '_handling_conflict_restaurant')
                
                # Clear conflict lists
                self.romantic_conflicts = []
                self.touristic_conflicts = []
                
                return 'restart'
            else:
                print("[Restart requested but disabled]")
                return None
        

        extracted_prefs = PreferenceExtractor.extract_all(user_input, context=context_stage)


        update_preferences_with_context(self.user_requirements, extracted_prefs, context_stage)
        
        log_preference_changes(extracted_prefs, self.user_requirements, old_prefs, [])

    def check_next_stage(self): 

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
            
            # Debug print: Show initial restaurants found
            total_found = 1 + len(alternatives)
            print(f"[DEBUG] Initial search found {total_found} restaurants:")
            print(f"[DEBUG]   Current: {restaurant_name}")
            if alternatives:
                alt_names = [alt if isinstance(alt, str) else alt.get('restaurantname', 'Unknown') for alt in alternatives]
                print(f"[DEBUG]   Alternatives: {alt_names}")
        else:
            self.current_restaurant = None
            self.alternatives = []
            print(f"[DEBUG] No restaurants found matching criteria: {filters}")
    
    def apply_inference_filtering(self):
        """apply inference rules to filter restaurants based on additional reqs"""
        if not self.additional_requirements:
            return
        
        # Clear any previous conflict lists to avoid carrying over conflicts from previous searches
        self.romantic_conflicts = []
        self.touristic_conflicts = []
        
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
        
        # Debug print: Show restaurants before filtering
        candidate_names = [r.get('restaurantname', 'Unknown') for r in all_candidates]
        print(f"[DEBUG] Before filtering: {len(all_candidates)} restaurants - {candidate_names}")
        
        # filter restaurants using inference engine
        filter_result = self.inference_engine.filter_restaurants_by_requirements(
            all_candidates, active_requirements
        )
        
        filtered_restaurants = filter_result['restaurants']
        
        # Debug print: Show filtered restaurants
        filtered_names = [r.get('restaurantname', 'Unknown') for r in filtered_restaurants]
        print(f"[DEBUG] After filtering: {len(filtered_restaurants)} clear matches - {filtered_names}")
        
        # Debug print: Show conflicts if any
        if filter_result['has_romantic_conflicts']:
            romantic_conflict_names = [r.get('restaurantname', 'Unknown') for r in filter_result['romantic_conflict_restaurants']]
            print(f"[DEBUG] Romantic conflicts: {len(romantic_conflict_names)} restaurants - {romantic_conflict_names}")
        if filter_result['has_touristic_conflicts']:
            touristic_conflict_names = [r.get('restaurantname', 'Unknown') for r in filter_result['touristic_conflict_restaurants']]
            print(f"[DEBUG] Touristic conflicts: {len(touristic_conflict_names)} restaurants - {touristic_conflict_names}")
        
        print(f"[Filtered from {len(all_candidates)} to {len(filtered_restaurants)} restaurants]")
        
        # PRIORITIZE clear matches over conflicts
        if filtered_restaurants:
            # We have clear matches - use them first
            self.current_restaurant = filtered_restaurants[0]
            self.current_restaurant_name = filtered_restaurants[0]['restaurantname']
            
            # update alternatives with remaining filtered restaurants
            remaining_restaurants = filtered_restaurants[1:]
            self.alternatives = remaining_restaurants  # store full restaurant dicts
            self.suggestion_index = 0
            
            print(f"[Updated current restaurant: {self.current_restaurant_name}]")
            alt_names = [r.get('restaurantname', 'Unknown') for r in self.alternatives if isinstance(r, dict)]
            print(f"[Updated alternatives: {alt_names}]")
            
            # Store conflicts for later use if user asks for alternatives and we run out of clear matches
            if filter_result['has_romantic_conflicts']:
                self.romantic_conflicts = filter_result['romantic_conflict_restaurants']
                print(f"[Also found {len(self.romantic_conflicts)} romantic conflicts for potential later use]")
            if filter_result['has_touristic_conflicts']:
                self.touristic_conflicts = filter_result['touristic_conflict_restaurants']
                print(f"[Also found {len(self.touristic_conflicts)} touristic conflicts for potential later use]")
            
            return
        
        # Only handle conflicts if NO clear matches are available
        # handle romantic conflicts
        if filter_result['has_romantic_conflicts']:
            # store conflict restaurants for user resolution
            self.romantic_conflicts = filter_result['romantic_conflict_restaurants']
            romantic_conflict_names = [r.get('restaurantname', 'Unknown') for r in self.romantic_conflicts]
            print(f"[No clear matches found. Found {len(self.romantic_conflicts)} restaurants with romantic conflicts: {romantic_conflict_names}]")
            
            # update current restaurant to first conflict restaurant for consistency
            if self.romantic_conflicts:
                self.current_restaurant = self.romantic_conflicts[0]  # Fixed: remove ['restaurant'] key
                self.current_restaurant_name = self.current_restaurant['restaurantname']
                print(f"[Updated current restaurant to conflict restaurant: {self.current_restaurant_name}]")
            
            return  # will be handled in conversation state
        
        # handle touristic conflicts
        if filter_result['has_touristic_conflicts']:
            # store conflict restaurants for user resolution
            self.touristic_conflicts = filter_result['touristic_conflict_restaurants']
            touristic_conflict_names = [r.get('restaurantname', 'Unknown') for r in self.touristic_conflicts]
            print(f"[No clear matches found. Found {len(self.touristic_conflicts)} restaurants with touristic conflicts: {touristic_conflict_names}]")
            
            # update current restaurant to first conflict restaurant for consistency
            if self.touristic_conflicts:
                self.current_restaurant = self.touristic_conflicts[0]  # Fixed: remove ['restaurant'] key
                self.current_restaurant_name = self.current_restaurant['restaurantname']
                print(f"[Updated current restaurant to conflict restaurant: {self.current_restaurant_name}]")
            
            return  # will be handled in conversation state
        
        # no restaurants meet the additional reqs and no conflicts
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
        print(f"[DEBUG] Current suggestion_index: {self.suggestion_index}")
        print(f"[DEBUG] Number of alternatives: {len(self.alternatives) if self.alternatives else 0}")
        print(f"[DEBUG] Current restaurant: {self.current_restaurant_name if self.current_restaurant_name else 'None'}")
        
        # Check if we're currently handling a conflict restaurant
        if hasattr(self, '_handling_conflict_restaurant'):
            conflict_type = self._handling_conflict_restaurant
            print(f"[DEBUG] Currently handling {conflict_type} conflict, looking for more {conflict_type} conflicts...")
            
            # Continue with remaining conflicts of the same type
            if conflict_type == 'romantic' and hasattr(self, 'romantic_conflicts') and self.romantic_conflicts:
                remaining_romantic_names = [r.get('restaurantname', 'Unknown') for r in self.romantic_conflicts]
                print(f"[DEBUG] Available romantic conflicts: {remaining_romantic_names}")
                
                conflict_restaurant = self.romantic_conflicts.pop(0)
                self.current_restaurant = conflict_restaurant
                self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
                print(f"[DEBUG] Switching to next romantic conflict restaurant: {self.current_restaurant_name}")
                print(f"[DEBUG] Remaining romantic conflicts: {len(self.romantic_conflicts)}")
                return self.states['SUGGEST_RESTAURANT']
            
            elif conflict_type == 'touristic' and hasattr(self, 'touristic_conflicts') and self.touristic_conflicts:
                remaining_touristic_names = [r.get('restaurantname', 'Unknown') for r in self.touristic_conflicts]
                print(f"[DEBUG] Available touristic conflicts: {remaining_touristic_names}")
                
                conflict_restaurant = self.touristic_conflicts.pop(0)
                self.current_restaurant = conflict_restaurant
                self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
                print(f"[DEBUG] Switching to next touristic conflict restaurant: {self.current_restaurant_name}")
                print(f"[DEBUG] Remaining touristic conflicts: {len(self.touristic_conflicts)}")
                return self.states['SUGGEST_RESTAURANT']
            
            else:
                # No more conflicts of current type, try other conflict types
                print(f"[DEBUG] No more {conflict_type} conflicts available")
                if conflict_type == 'romantic' and hasattr(self, 'touristic_conflicts') and self.touristic_conflicts:
                    print(f"[DEBUG] Trying touristic conflicts...")
                    available_touristic_names = [r.get('restaurantname', 'Unknown') for r in self.touristic_conflicts]
                    print(f"[DEBUG] Available touristic conflicts: {available_touristic_names}")
                    
                    conflict_restaurant = self.touristic_conflicts.pop(0)
                    self.current_restaurant = conflict_restaurant
                    self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
                    self._handling_conflict_restaurant = 'touristic'
                    print(f"[DEBUG] Switching to touristic conflict restaurant: {self.current_restaurant_name}")
                    return self.states['SUGGEST_RESTAURANT']
                elif conflict_type == 'touristic' and hasattr(self, 'romantic_conflicts') and self.romantic_conflicts:
                    available_romantic_names = [r.get('restaurantname', 'Unknown') for r in self.romantic_conflicts]
                    print(f"[DEBUG] Available romantic conflicts: {available_romantic_names}")
                    
                    conflict_restaurant = self.romantic_conflicts.pop(0)
                    self.current_restaurant = conflict_restaurant
                    self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
                    self._handling_conflict_restaurant = 'romantic'
                    print(f"[DEBUG] Switching to romantic conflict restaurant: {self.current_restaurant_name}")
                    return self.states['SUGGEST_RESTAURANT']
                else:
                    print(f"[DEBUG] No more conflicts of any type available")
                    return self.states['APOLOGIZE']
        
        # First try regular alternatives (when not handling conflicts)
        if self.alternatives and self.suggestion_index < len(self.alternatives):
            alt_restaurant_dict = self.alternatives[self.suggestion_index]
            
            print(f"[DEBUG] Trying alternative at index {self.suggestion_index}: {alt_restaurant_dict.get('restaurantname', 'Unknown') if isinstance(alt_restaurant_dict, dict) else alt_restaurant_dict}")
            
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
            print(f"[DEBUG] Updated to alternative: {self.current_restaurant_name}, new suggestion_index: {self.suggestion_index}")
            return self.states['SUGGEST_RESTAURANT']
        
        # If no more regular alternatives, try conflict restaurants
        elif hasattr(self, 'romantic_conflicts') and self.romantic_conflicts:
            print(f"[DEBUG] No more regular alternatives. Trying romantic conflicts...")
            available_romantic_names = [r.get('restaurantname', 'Unknown') for r in self.romantic_conflicts]
            print(f"[DEBUG] Available romantic conflicts: {available_romantic_names}")
            
            conflict_restaurant = self.romantic_conflicts.pop(0)  # Take first conflict
            
            self.current_restaurant = conflict_restaurant
            self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
            
            # Set flag to indicate we're handling a conflict restaurant
            self._handling_conflict_restaurant = 'romantic'
            
            print(f"[DEBUG] Switching to conflict restaurant: {self.current_restaurant_name}")
            print(f"[DEBUG] Remaining romantic conflicts: {len(self.romantic_conflicts)}")
            return self.states['SUGGEST_RESTAURANT']  # This will trigger conflict resolution
        
        elif hasattr(self, 'touristic_conflicts') and self.touristic_conflicts:
            print(f"[DEBUG] No more regular alternatives. Trying touristic conflicts...")
            available_touristic_names = [r.get('restaurantname', 'Unknown') for r in self.touristic_conflicts]
            print(f"[DEBUG] Available touristic conflicts: {available_touristic_names}")
            
            conflict_restaurant = self.touristic_conflicts.pop(0)  # Take first conflict
            
            self.current_restaurant = conflict_restaurant
            self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
            
            # Set flag to indicate we're handling a conflict restaurant
            self._handling_conflict_restaurant = 'touristic'
            
            print(f"[DEBUG] Switching to conflict restaurant: {self.current_restaurant_name}")
            print(f"[DEBUG] Remaining touristic conflicts: {len(self.touristic_conflicts)}")
            return self.states['SUGGEST_RESTAURANT']  # This will trigger conflict resolution
        
        else:
            print(f"[DEBUG] No more alternatives or conflicts available")
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
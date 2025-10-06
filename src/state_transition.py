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
            'WELCOME': 'welcome',  #State 1             
            'ASK_AREA': 'ask_area',  #State 2
            'ASK_PRICE': 'ask_price',  #State 3
            'ASK_FOOD_TYPE': 'ask_food_type',  #State 4
            'ASK_ADDITIONAL_REQUIREMENTS': 'ask_additional_requirements',  #State 5
            'APOLOGIZE': 'apologize',  #State 6    
            'CONFIRM': 'confirm',  #State 7
            'SUGGEST_RESTAURANT': 'suggest_restaurant',  #State 8
            'INFORM': 'inform',  #State 9
            'GOODBYE': 'goodbye' #State 10                 
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
        """
        Format system output based on user preferences.
        
        Input: message (str) - Raw system message
        Output: str - Formatted message (uppercase if caps enabled)
        """
        formatted_message = message.upper() if self.output_caps else message
        return formatted_message
    
    def ensure_model_ready(self):
        """
        Ensure ML model is loaded and ready for classification.
        
        Input: None
        Output: bool - True if model ready, False if training failed
        """
        if self.is_trained:
            return True
        
        if load_trained_model(self):
            return True
            
        print("No pre-trained model available. Training new model...")
        return train_classifier(self)
    
    def get_user_input(self, prompt: str = "User: ") -> str:
        """
        Get user input from terminal.
        
        Input: prompt (str) - Input prompt to display
        Output: str - User's text input
        """
        return input(prompt)
           
    def classify_utterance(self, user_utterance):
        """
        Classify user utterance into dialog act using selected classifier.
        
        Input: user_utterance (str) - User's input text
        Output: str - Predicted dialog act or 'null' if error
        """
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
                return "null"
        except Exception:
            return "null"

    def parse_user_input(self, user_input: str, context_stage=None):
        """
        Parse user input to extract preferences and handle restart commands.
        
        Input: user_input (str), context_stage (str, optional) - Current dialog context
        Output: str or None - 'restart' if restart requested, None otherwise
        """
        old_prefs = self.user_requirements.copy()
        
        if detect_restart_command(user_input):
            if self.allow_restarts:
                self.user_requirements = {'area': None, 'price': None, 'food': None}
                self.additional_requirements = {'touristic': None, 'assigned_seats': None, 'children': None, 'romantic': None}
                
                if hasattr(self, '_inference_applied'):
                    delattr(self, '_inference_applied')
                if hasattr(self, '_handling_conflict_restaurant'):
                    delattr(self, '_handling_conflict_restaurant')
                
                self.romantic_conflicts = []
                self.touristic_conflicts = []
                
                return 'restart'
            else:
                return None
        
        extracted_prefs = PreferenceExtractor.extract_all(user_input, context=context_stage)
        update_preferences_with_context(self.user_requirements, extracted_prefs, context_stage)
        log_preference_changes(extracted_prefs, self.user_requirements, old_prefs, [])

    def check_next_stage(self):
        """
        Determine next conversation stage based on collected preferences.
        
        Input: None
        Output: str - Next dialog state identifier
        """
        if not self.user_requirements['area']:
            return self.states['ASK_AREA']
        elif not self.user_requirements['price']:
            return self.states['ASK_PRICE']
        elif not self.user_requirements['food']:
            return self.states['ASK_FOOD_TYPE']
        else:
            self.search_restaurants()
            
            if not self.current_restaurant:
                return self.states['APOLOGIZE']
            
            return self.states['ASK_ADDITIONAL_REQUIREMENTS']
    
    def search_restaurants(self):
        """
        Search restaurant database using current user preferences.
        
        Input: None
        Output: None - Updates current_restaurant and alternatives
        """
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
        """
        Apply inference rules to filter restaurants based on additional requirements.
        
        Input: None
        Output: None - Updates current_restaurant and alternatives based on inference rules
        """
        if not self.additional_requirements:
            return
        
        self.romantic_conflicts = []
        self.touristic_conflicts = []
        
        active_requirements = {k: v for k, v in self.additional_requirements.items() if v is not None}
        
        if not active_requirements:
            return
        
        print(f"[Applying inference rules with requirements: {active_requirements}]")
        
        all_candidates = []
        if self.current_restaurant:
            all_candidates.append(self.current_restaurant)
        
        for alt in self.alternatives:
            if isinstance(alt, dict):
                all_candidates.append(alt)
            else:
                found = self.restaurant_lookup.find_restaurant_by_name(alt)
                if found:
                    all_candidates.append(found)
        
        filter_result = self.inference_engine.filter_restaurants_by_requirements(
            all_candidates, active_requirements
        )
        
        filtered_restaurants = filter_result['restaurants']
        
        if filtered_restaurants:
            self.current_restaurant = filtered_restaurants[0]
            self.current_restaurant_name = filtered_restaurants[0]['restaurantname']
            
            remaining_restaurants = filtered_restaurants[1:]
            self.alternatives = remaining_restaurants
            self.suggestion_index = 0
            
            if filter_result['has_romantic_conflicts']:
                self.romantic_conflicts = filter_result['romantic_conflict_restaurants']
            if filter_result['has_touristic_conflicts']:
                self.touristic_conflicts = filter_result['touristic_conflict_restaurants']
            
            return
        
        # Handle conflicts if no clear matches
        if filter_result['has_romantic_conflicts']:
            self.romantic_conflicts = filter_result['romantic_conflict_restaurants']
            
            if self.romantic_conflicts:
                self.current_restaurant = self.romantic_conflicts[0]
                self.current_restaurant_name = self.current_restaurant['restaurantname']
            
            return
        
        if filter_result['has_touristic_conflicts']:
            self.touristic_conflicts = filter_result['touristic_conflict_restaurants']
            
            if self.touristic_conflicts:
                self.current_restaurant = self.touristic_conflicts[0]
                self.current_restaurant_name = self.current_restaurant['restaurantname']
            
            return
        
        # No restaurants match the requirements
        self.current_restaurant = None
        self.current_restaurant_name = None
        self.alternatives = []
    
    def provide_restaurant_info(self, user_input: str):
        """
        Provide detailed information about current restaurant based on user request.
        
        Input: user_input (str) - User's information request
        Output: str - Next dialog state
        """
        if not self.current_restaurant:
            no_info_msg = "I'm sorry, I don't have any restaurant information available to provide details."
            print(f"System: {self.format_output(no_info_msg)}")
            return self.states['APOLOGIZE']
        
        try:
            response = format_restaurant_info_response(self.current_restaurant, user_input)
            print(f"System: {self.format_output(response)}")
        except Exception as e:
            error_msg = f"I'm sorry, I'm having trouble accessing the restaurant information right now."
            print(f"System: {self.format_output(error_msg)}")
            return self.states['APOLOGIZE']
        
        return 'await_next_request'
    
    def try_alternative(self):
        """
        Try next alternative restaurant or conflict restaurant if available.
        
        Input: None
        Output: str - Next dialog state
        """
        # Handle ongoing conflicts first
        if hasattr(self, '_handling_conflict_restaurant'):
            conflict_type = self._handling_conflict_restaurant
            
            if conflict_type == 'romantic' and self.romantic_conflicts:
                conflict_restaurant = self.romantic_conflicts.pop(0)
                self.current_restaurant = conflict_restaurant
                self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
                return self.states['SUGGEST_RESTAURANT']
            
            elif conflict_type == 'touristic' and self.touristic_conflicts:
                conflict_restaurant = self.touristic_conflicts.pop(0)
                self.current_restaurant = conflict_restaurant
                self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
                return self.states['SUGGEST_RESTAURANT']
            
            else:
                # Switch conflict types if available
                if conflict_type == 'romantic' and self.touristic_conflicts:
                    conflict_restaurant = self.touristic_conflicts.pop(0)
                    self.current_restaurant = conflict_restaurant
                    self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
                    self._handling_conflict_restaurant = 'touristic'
                    return self.states['SUGGEST_RESTAURANT']
                elif conflict_type == 'touristic' and self.romantic_conflicts:
                    conflict_restaurant = self.romantic_conflicts.pop(0)
                    self.current_restaurant = conflict_restaurant
                    self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
                    self._handling_conflict_restaurant = 'romantic'
                    return self.states['SUGGEST_RESTAURANT']
                else:
                    return self.states['APOLOGIZE']
        
        # Try regular alternatives
        if self.alternatives and self.suggestion_index < len(self.alternatives):
            alt_restaurant_dict = self.alternatives[self.suggestion_index]
            
            if not isinstance(alt_restaurant_dict, dict):
                restaurant_name = alt_restaurant_dict
                restaurant_row = self.restaurant_lookup.df[
                    self.restaurant_lookup.df['restaurantname'].str.lower() == restaurant_name.lower()
                ]
                if not restaurant_row.empty:
                    alt_restaurant_dict = restaurant_row.iloc[0].to_dict()
                else:
                    return self.states['APOLOGIZE']
                
            self.current_restaurant = alt_restaurant_dict
            self.current_restaurant_name = alt_restaurant_dict.get('restaurantname', 'Unknown Restaurant')
            self.suggestion_index += 1
            return self.states['SUGGEST_RESTAURANT']
        
        # Try conflict restaurants if no regular alternatives
        elif self.romantic_conflicts:
            conflict_restaurant = self.romantic_conflicts.pop(0)
            self.current_restaurant = conflict_restaurant
            self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
            self._handling_conflict_restaurant = 'romantic'
            return self.states['SUGGEST_RESTAURANT']
        
        elif self.touristic_conflicts:
            conflict_restaurant = self.touristic_conflicts.pop(0)
            self.current_restaurant = conflict_restaurant
            self.current_restaurant_name = conflict_restaurant.get('restaurantname', 'Unknown Restaurant')
            self._handling_conflict_restaurant = 'touristic'
            return self.states['SUGGEST_RESTAURANT']
        
        else:
            return self.states['APOLOGIZE']
    
    def run_conversation(self):
        """
        Execute the main conversation loop until completion.
        
        Input: None
        Output: None - Runs interactive conversation with user
        """
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
        """
        Print conversation header banner.
        
        Input: None
        Output: None - Prints header to console
        """
        print("=" * 60)
        print("CAMBRIDGE RESTAURANT SYSTEM DIALOG")
        print("=" * 60)
    
    def _print_conversation_footer(self):
        """
        Print conversation completion footer.
        
        Input: None
        Output: None - Prints footer to console
        """
        print("=" * 60)
        print(f"CONVERSATION COMPLETED - Total turns: {self.conversation_turn}")
        print("=" * 60)
    
    def _handle_conversation_turn(self):
        """
        Handle a single conversation turn.
        
        Input: None
        Output: None - Processes current state and transitions to next
        """
        self.conversation_turn += 1
        current_state_name = get_state_name_from_value(self.states, self.current_state)
        
        print(f"\n--- Turn {self.conversation_turn} ---")
        
        self.current_state = execute_conversation_state(self, self.current_state, self.states)
        
        if self.current_state and not self.conversation_ended:
            print("-" * 30)
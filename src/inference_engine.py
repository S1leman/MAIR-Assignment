class InferenceEngine:
    def __init__(self):
        # CONDITION MAPPINGS: Map abstract conditions to actual restaurant properties
        # Format: 'condition_name': (property_field, expected_value)
        # Example: 'cheap' maps to checking if restaurant['pricerange'] == 'cheap'
        self.condition_mappings = {
            'cheap': ('pricerange', 'cheap'),      # Restaurant is cheap
            'good': ('food_quality', 'good'),      # Restaurant has good food quality
            'romanian': ('food', 'romanian'),      # Restaurant serves Romanian cuisine
            'busy': ('crowdedness', 'busy'),       # Restaurant is busy/crowded
            'long': ('length_stay', 'long')        # Customers stay for long time
        }
        
        self.logical_operators = {
            'SINGLE': self._check_single_operator,
            'AND': self._check_and_operator
        }
        
        # INFERENCE RULES: Logical rules for deriving new properties
        # Each rule has:
        # - antecedent: conditions that must be true (IF part)
        # - antecedent_type: 'SINGLE' for one condition, 'AND' for all conditions
        # - consequent: property being inferred (THEN part)
        # - consequent_value: True/False value for the inferred property
        # - description: human-readable explanation
        
        self.inference_rules = [
            # RULE 1: IF (cheap AND good food) THEN touristic = True
            # Example: Da Vinci Pizzeria (cheap=True, food_quality=good) → touristic=True
            # Logic: Tourists love good value - cheap restaurants with good food attract them
            {
                'id': 1,
                'antecedent': ['cheap', 'good'],           # Both conditions must be true
                'antecedent_type': 'AND',                  # Logical AND operation
                'consequent': 'touristic',                 # Property being inferred
                'consequent_value': True,                  # touristic = True
                'description': 'a cheap restaurant with good food attracts tourists'
            },
            
            # RULE 2: IF (Romanian cuisine) THEN touristic = False  
            # Example: Any Romanian restaurant → touristic=False
            # Logic: Romanian food is unfamiliar to most tourists
            # *** CONFLICTS WITH RULE 1 *** if restaurant is cheap + good + Romanian
            {
                'id': 2,
                'antecedent': ['romanian'],                # Single condition
                'antecedent_type': 'SINGLE',               # Only one condition needed
                'consequent': 'touristic',                 # Same property as Rule 1!
                'consequent_value': False,                 # touristic = False (opposite!)
                'description': 'Romanian cuisine is unknown for most tourists and they prefer familiar food'
            },
            # RULE 3: IF (busy) THEN assigned_seats = True
            # Example: Pizza Hut (crowdedness=busy) → assigned_seats=True
            # Logic: Busy restaurants need to manage seating efficiently
            {
                'id': 3,
                'antecedent': ['busy'],
                'antecedent_type': 'SINGLE',
                'consequent': 'assigned_seats',
                'consequent_value': True,
                'description': 'in a busy restaurant the waiter decides where you sit'
            },
            
            # RULE 4: IF (long stay) THEN children = False
            # Example: Fine dining restaurant (length_stay=long) → children=False
            # Logic: Long meals are not suitable for children
            {
                'id': 4,
                'antecedent': ['long'],
                'antecedent_type': 'SINGLE',
                'consequent': 'children',
                'consequent_value': False,
                'description': 'spending a long time in a restaurant is not advised when taking children'
            },
            
            # RULE 5: IF (busy) THEN romantic = False
            # Example: Fast food place (crowdedness=busy) → romantic=False
            # Logic: Busy, noisy places are not romantic
            # *** CONFLICTS WITH RULE 6 *** if restaurant is both busy AND allows long stays
            {
                'id': 5,
                'antecedent': ['busy'],
                'antecedent_type': 'SINGLE',
                'consequent': 'romantic',                  # Same property as Rule 6!
                'consequent_value': False,                 # romantic = False
                'description': 'a busy restaurant is not romantic'
            },
            
            # RULE 6: IF (long stay) THEN romantic = True
            # Example: Fine dining (length_stay=long) → romantic=True
            # Logic: Taking time for a meal is romantic
            # *** CONFLICTS WITH RULE 5 *** if restaurant is both busy AND allows long stays
            {
                'id': 6,
                'antecedent': ['long'],
                'antecedent_type': 'SINGLE',
                'consequent': 'romantic',                  # Same property as Rule 5!
                'consequent_value': True,                  # romantic = True (opposite!)
                'description': 'spending a long time in a restaurant is romantic'
            }
        ]
        
        # CONFLICT PAIRS: Rules that can contradict each other
        # When both rules in a pair apply, they produce opposite truth values
        # Format: 'property': (rule_id1, rule_id2)
        
        # CONFLICT EXAMPLE 1: Rules 5 & 6 conflict on 'romantic' property
        # Scenario: Restaurant with crowdedness=busy AND length_stay=long
        # Rule 5: busy → romantic=False ("busy restaurants are not romantic")
        # Rule 6: long → romantic=True ("long stays are romantic")
        # Result: Conflict! System asks user to choose preference
        
        # CONFLICT EXAMPLE 2: Rules 1 & 2 conflict on 'touristic' property  
        # Scenario: Romanian restaurant with pricerange=cheap AND food_quality=good
        # Rule 1: (cheap AND good) → touristic=True ("good value attracts tourists")
        # Rule 2: romanian → touristic=False ("unfamiliar cuisine deters tourists")
        # Result: Conflict! System asks user to prioritize value vs familiarity
        
        self.conflict_pairs = {
            'romantic': (5, 6),     # Rule 5 vs Rule 6
            'touristic': (1, 2)     # Rule 1 vs Rule 2
        }
          
        self._categorize_properties()
        
    # PRACTICAL EXAMPLES OF HOW THE INFERENCE ENGINE WORKS:
    
    def example_no_conflict(self):
        """
        Example 1: Simple rule application without conflicts.
        
        Input: Restaurant data
        Output: Inferred properties
        """
        # Restaurant: Da Vinci Pizzeria 
        restaurant = {
            'restaurantname': 'Da Vinci Pizzeria',
            'pricerange': 'cheap',      # Triggers 'cheap' condition
            'food_quality': 'good',     # Triggers 'good' condition  
            'food': 'italian',          # Does NOT trigger 'romanian'
            'crowdedness': 'not busy',  # Does NOT trigger 'busy'
            'length_stay': 'short'      # Does NOT trigger 'long'
        }
        
        # Rule 1 applies: cheap=True AND good=True → touristic=True
        # Rule 2 does NOT apply: romanian=False
        # Result: {'touristic': True}
        return self.apply_rules(restaurant)
    
    def example_romantic_conflict(self):
        """
        Example 2: Conflict between Rules 5 and 6 on 'romantic' property.
        
        Input: Restaurant with busy=True AND long=True
        Output: Conflict requiring user resolution
        """
        # Restaurant: Upscale bistro that gets busy but allows long meals
        restaurant = {
            'restaurantname': 'Le Bistro',
            'pricerange': 'expensive',
            'food_quality': 'good',
            'food': 'french',
            'crowdedness': 'busy',      # Triggers Rule 5: romantic=False
            'length_stay': 'long'       # Triggers Rule 6: romantic=True
        }
        
        # CONFLICT: 
        # Rule 5: busy=True → romantic=False
        # Rule 6: long=True → romantic=True  
        # System detects conflict and requires user input
        return self.apply_rules(restaurant)
    
    def example_touristic_conflict(self):
        """
        Example 3: Conflict between Rules 1 and 2 on 'touristic' property.
        
        Input: Cheap Romanian restaurant with good food
        Output: Conflict between value and familiarity
        """
        # Restaurant: Good value Romanian place
        restaurant = {
            'restaurantname': 'Bucharest Grill',
            'pricerange': 'cheap',      # Triggers Rule 1 (with good food)
            'food_quality': 'good',     # Triggers Rule 1 (with cheap price)
            'food': 'romanian',         # Triggers Rule 2: touristic=False
            'crowdedness': 'not busy',
            'length_stay': 'short'
        }
        
        # CONFLICT:
        # Rule 1: (cheap=True AND good=True) → touristic=True
        # Rule 2: romanian=True → touristic=False
        # User must choose: value vs cuisine familiarity
        return self.apply_rules(restaurant)
    
    def example_multiple_rules(self):
        """
        Example 4: Multiple rules applying without conflicts.
        
        Input: Busy restaurant with short stays
        Output: Multiple inferred properties
        """
        # Restaurant: Fast-casual place
        restaurant = {
            'restaurantname': 'Quick Eats',
            'pricerange': 'moderate',
            'food_quality': 'average',
            'food': 'american',
            'crowdedness': 'busy',      # Triggers Rules 3 & 5
            'length_stay': 'short'      # Does NOT trigger Rules 4 & 6
        }
        
        # Multiple rules apply:
        # Rule 3: busy=True → assigned_seats=True
        # Rule 5: busy=True → romantic=False
        # Result: {'assigned_seats': True, 'romantic': False}
        return self.apply_rules(restaurant)
    
    def demonstrate_all_examples(self):
        """
        Run all examples to show how the inference engine works.
        
        Input: None
        Output: None (prints results to console)
        """
        print("=" * 60)
        print("INFERENCE ENGINE DEMONSTRATION")
        print("=" * 60)
        
        print("\n1. NO CONFLICT EXAMPLE:")
        print("Restaurant: Da Vinci Pizzeria (cheap + good + italian)")
        result1 = self.example_no_conflict()
        print(f"Inferred properties: {result1['inferred_properties']}")
        print(f"Has conflict: {result1['has_conflict']}")
        if result1['reasoning']:
            print(f"Reasoning: {result1['reasoning']}")
        
        print("\n2. ROMANTIC CONFLICT EXAMPLE:")
        print("Restaurant: Le Bistro (busy + long stays)")
        result2 = self.example_romantic_conflict()
        print(f"Inferred properties: {result2['inferred_properties']}")
        print(f"Has conflict: {result2['has_conflict']}")
        if result2['has_conflict']:
            conflict = result2['conflicts'][0]
            print(f"Conflict on: {conflict['property']}")
            print(f"Rule {conflict['rule1']['id']}: {conflict['rule1']['description']} → {conflict['property']}={conflict['rule1']['value']}")
            print(f"Rule {conflict['rule2']['id']}: {conflict['rule2']['description']} → {conflict['property']}={conflict['rule2']['value']}")
        
        print("\n3. TOURISTIC CONFLICT EXAMPLE:")
        print("Restaurant: Bucharest Grill (cheap + good + romanian)")
        result3 = self.example_touristic_conflict()
        print(f"Inferred properties: {result3['inferred_properties']}")
        print(f"Has conflict: {result3['has_conflict']}")
        if result3['has_conflict']:
            conflict = result3['conflicts'][0]
            print(f"Conflict on: {conflict['property']}")
            print(f"Rule {conflict['rule1']['id']}: {conflict['rule1']['description']} → {conflict['property']}={conflict['rule1']['value']}")
            print(f"Rule {conflict['rule2']['id']}: {conflict['rule2']['description']} → {conflict['property']}={conflict['rule2']['value']}")
        
        print("\n4. MULTIPLE RULES EXAMPLE:")
        print("Restaurant: Quick Eats (busy + short stays)")
        result4 = self.example_multiple_rules()
        print(f"Inferred properties: {result4['inferred_properties']}")
        print(f"Has conflict: {result4['has_conflict']}")
        if result4['reasoning']:
            print(f"Reasoning: {result4['reasoning']}")
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)

    def _categorize_properties(self):
        """
        Categorize properties by inference type.
        
        Input: None (uses self.inference_rules)
        Output: None (sets self.properties_* attributes)
        """
        property_inferences = {}
        
        for rule in self.inference_rules:
            consequent = rule['consequent']
            value = rule['consequent_value']
            
            if consequent not in property_inferences:
                property_inferences[consequent] = set()
            
            property_inferences[consequent].add(value)
        
        self.properties_positive_only = set()
        self.properties_negative_only = set()
        self.properties_both = set()
        
        for prop, values in property_inferences.items():
            if values == {True}:
                self.properties_positive_only.add(prop)
            elif values == {False}:
                self.properties_negative_only.add(prop)
            else:
                self.properties_both.add(prop)
    
    def _check_single_operator(self, conditions, restaurant):
        """Helper for SINGLE logical operator."""
        return self._check_single_condition(conditions[0], restaurant)
    
    def _check_and_operator(self, conditions, restaurant):
        """Helper for AND logical operator."""
        for condition in conditions:
            if not self._check_single_condition(condition, restaurant):
                return False
        return True
    
    def evaluate_antecedent(self, rule, restaurant):
        """
        Check if rule's antecedent conditions are satisfied.
        
        Input: rule (dict), restaurant (dict)
        Output: bool (True if conditions met)
        """
        antecedent = rule['antecedent']
        antecedent_type = rule['antecedent_type']
        
        if antecedent_type in self.logical_operators:
            return self.logical_operators[antecedent_type](antecedent, restaurant)
        
        return False
    
    def _check_single_condition(self, condition, restaurant):
        """
        Verify single condition against restaurant properties.
        
        Input: condition (str), restaurant (dict)
        Output: bool (True if condition satisfied)
        """
        if condition in self.condition_mappings:
            property_name, expected_value = self.condition_mappings[condition]
            actual_value = restaurant.get(property_name, '') or ''
            return actual_value.lower() == expected_value.lower()
        
        return False
    
    def apply_rules(self, restaurant):
        """
        Apply all inference rules to restaurant.
        
        Input: restaurant (dict with properties)
        Output: dict with keys: inferred_properties, applied_rules, reasoning, 
                has_conflict, conflict_type, conflicts (list of all conflicts)
        """
        inferred_properties = {}
        applied_rules = []
        reasoning = []
        conflicts = []
        
        for rule in self.inference_rules:
            if not self.evaluate_antecedent(rule, restaurant):
                continue
                
            rule_id = rule['id']
            consequent = rule['consequent']
            consequent_value = rule['consequent_value']
            description = rule['description']
            
            applied_rules.append(rule_id)
            
            if consequent in inferred_properties:
                old_rule_id = inferred_properties[consequent]['rule_id']
                old_value = inferred_properties[consequent]['value']
                
                if old_value != consequent_value:
                    old_description = ""
                    for r in self.inference_rules:
                        if r['id'] == old_rule_id:
                            old_description = r['description']
                            break
                    
                    # Check if this is a known conflict
                    rule1 = old_rule_id
                    rule2 = rule_id
                    if consequent in self.conflict_pairs:
                        conflict_rules = self.conflict_pairs[consequent]
                        if (rule1 in conflict_rules and rule2 in conflict_rules):
                            
                            del inferred_properties[consequent]
                            
                            conflict_info = {
                                'property': consequent,
                                'rule1': {
                                    'id': old_rule_id,
                                    'value': old_value,
                                    'description': old_description
                                },
                                'rule2': {
                                    'id': rule_id,
                                    'value': consequent_value,
                                    'description': description
                                },
                                'requires_user_input': True
                            }
                            
                            conflicts.append(conflict_info)
                            continue
            
            inferred_properties[consequent] = {
                'value': consequent_value,
                'rule_id': rule_id,
                'description': description
            }
            
            reasoning.append(f"Rule {rule_id}: {description}")
        
        final_properties = {}
        for prop, data in inferred_properties.items():
            final_properties[prop] = data['value']
         
        if len(conflicts) > 0:
            has_conflict = True
            conflict_type = conflicts[0]['property']
        else:
            has_conflict = False
            conflict_type = None
        
        result = {
            'inferred_properties': final_properties,
            'applied_rules': applied_rules,
            'reasoning': reasoning,
            'has_conflict': has_conflict,
            'conflict_type': conflict_type,
            'conflicts': conflicts
        }
        
        if has_conflict:
            result['conflict'] = conflicts[0]
        
        return result
    
    def filter_restaurants_by_requirements(self, restaurants, user_requirements):
        """
        Filter restaurants based on user requirements.
        
        Input: restaurants (list of dicts), user_requirements (dict)
        Output: dict with keys: restaurants, has_romantic_conflicts, has_touristic_conflicts,
                romantic_conflict_restaurants, touristic_conflict_restaurants
        """
        filtered_restaurants = []
        romantic_conflicts = []
        touristic_conflicts = []
        
        for restaurant in restaurants:
            inference_result = self.apply_rules(restaurant)
            inferred_properties = inference_result['inferred_properties']
            
            if inference_result['has_conflict']:
                restaurant_with_inference = restaurant.copy()
                restaurant_with_inference['inference_result'] = inference_result
                
                for conflict in inference_result['conflicts']:
                    conflict_type = conflict['property']
                    
                    if (conflict_type in user_requirements and 
                        user_requirements[conflict_type] is not None):
                        
                        if conflict_type == 'romantic':
                            romantic_conflicts.append(restaurant_with_inference)
                        elif conflict_type == 'touristic':
                            touristic_conflicts.append(restaurant_with_inference)
                
                continue
            
            meets_requirements = self._check_requirements(
                inferred_properties, 
                user_requirements
            )
            
            if meets_requirements:
                restaurant_with_inference = restaurant.copy()
                restaurant_with_inference['inference_result'] = inference_result
                filtered_restaurants.append(restaurant_with_inference)
        
        return {
            'restaurants': filtered_restaurants,
            'has_romantic_conflicts': len(romantic_conflicts) > 0,
            'has_touristic_conflicts': len(touristic_conflicts) > 0,
            'romantic_conflict_restaurants': romantic_conflicts,
            'touristic_conflict_restaurants': touristic_conflicts
        }

    def _check_requirements(self, inferred_properties, user_requirements, exclude_property=None):
        """
        Check if inferred properties meet user requirements.
        
        Input: inferred_properties (dict), user_requirements (dict), exclude_property (str or None)
        Output: bool (True if all requirements met)
        """
        for req_property, req_value in user_requirements.items():
            if req_property == exclude_property:
                continue
            
            if req_value is None:
                continue
            
            inferred_value = inferred_properties.get(req_property)
            
            if req_property in self.properties_positive_only:
                if req_value and inferred_value is not True:
                    return False
                if not req_value and inferred_value is True:
                    return False
            
            elif req_property in self.properties_negative_only:
                if req_value and inferred_value is False:
                    return False
                if not req_value and inferred_value is not False:
                    return False
            
            else:
                if req_value and inferred_value is not True:
                    return False
                if not req_value and inferred_value is True:
                    return False
        
        return True
    
    def explain_recommendation(self, restaurant):
        """
        Generate natural language explanation.
        
        Input: restaurant (dict with inference_result)
        Output: str (explanation text)
        """
        if 'inference_result' not in restaurant:
            return ""
        
        inference_result = restaurant['inference_result']
        reasoning = inference_result.get('reasoning', [])
        
        if not reasoning:
            return ""
        
        explanations = []
        for reason in reasoning:
            if 'Rule 1:' in reason:
                explanations.append("This restaurant is popular with tourists because it offers good food at affordable prices")
            elif 'Rule 2:' in reason:
                explanations.append("This restaurant is not typically visited by tourists because Romanian cuisine is less familiar")
            elif 'Rule 3:' in reason:
                explanations.append("The restaurant has assigned seating because it gets busy")
            elif 'Rule 4:' in reason:
                explanations.append("The restaurant is not ideal for children because guests typically stay for a long time")
            elif 'Rule 5:' in reason:
                explanations.append("The restaurant is not romantic because it tends to be busy and noisy")
            elif 'Rule 6:' in reason:
                explanations.append("The restaurant is romantic because you can take your time and enjoy a leisurely meal")
        
        return ". ".join(explanations) + "." if explanations else ""


def main():
    """
    Main function to demonstrate the inference engine capabilities.

    """
    print("CAMBRIDGE RESTAURANT INFERENCE ENGINE")
    print("Demonstrating rule-based reasoning and conflict detection\n")
    
    engine = InferenceEngine()
    
    engine.demonstrate_all_examples()
    

if __name__ == "__main__":
    main()
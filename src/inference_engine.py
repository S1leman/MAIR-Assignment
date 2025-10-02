"""
Inference Engine for Restaurant Recommendation Rules

Input: Restaurant properties (dict)
Output: Inferred properties with conflict detection
"""


class InferenceEngine:
    """
    Applies inference rules to derive additional restaurant properties.
    
    Input: Restaurant data dictionaries
    Output: Inference results with conflict information
    """
    
    def __init__(self):
        self.condition_mappings = {
            'cheap': ('pricerange', 'cheap'),
            'good': ('food_quality', 'good'),
            'romanian': ('food', 'romanian'),
            'busy': ('crowdedness', 'busy'),
            'long': ('length_stay', 'long')
        }
        
        self.logical_operators = {
            'SINGLE': lambda conditions, restaurant: self._check_single_condition(conditions[0], restaurant),
            'AND': lambda conditions, restaurant: all(self._check_single_condition(c, restaurant) for c in conditions)
        }
        
        self.inference_rules = [
            {
                'id': 1,
                'antecedent': ['cheap', 'good'],
                'antecedent_type': 'AND',
                'consequent': 'touristic',
                'consequent_value': True,
                'description': 'a cheap restaurant with good food attracts tourists'
            },
            {
                'id': 2,
                'antecedent': ['romanian'],
                'antecedent_type': 'SINGLE',
                'consequent': 'touristic',
                'consequent_value': False,
                'description': 'Romanian cuisine is unknown for most tourists and they prefer familiar food'
            },
            {
                'id': 3,
                'antecedent': ['busy'],
                'antecedent_type': 'SINGLE',
                'consequent': 'assigned_seats',
                'consequent_value': True,
                'description': 'in a busy restaurant the waiter decides where you sit'
            },
            {
                'id': 4,
                'antecedent': ['long'],
                'antecedent_type': 'SINGLE',
                'consequent': 'children',
                'consequent_value': False,
                'description': 'spending a long time in a restaurant is not advised when taking children'
            },
            {
                'id': 5,
                'antecedent': ['busy'],
                'antecedent_type': 'SINGLE',
                'consequent': 'romantic',
                'consequent_value': False,
                'description': 'a busy restaurant is not romantic'
            },
            {
                'id': 6,
                'antecedent': ['long'],
                'antecedent_type': 'SINGLE',
                'consequent': 'romantic',
                'consequent_value': True,
                'description': 'spending a long time in a restaurant is romantic'
            }
        ]
        
        self.conflict_pairs = {
            'romantic': frozenset([5, 6]),
            'touristic': frozenset([1, 2])
        }
          
        self._categorize_properties()

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
                    rule_pair = frozenset([rule_id, old_rule_id])
                    
                    if (consequent in self.conflict_pairs and 
                        rule_pair == self.conflict_pairs[consequent]):
                        
                        del inferred_properties[consequent]
                        
                        conflict_info = {
                            'property': consequent,
                            'rule1': {
                                'id': old_rule_id,
                                'value': old_value,
                                'description': [r['description'] for r in self.inference_rules if r['id'] == old_rule_id][0]
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
        
        final_properties = {prop: data['value'] for prop, data in inferred_properties.items()}
        
        has_conflict = len(conflicts) > 0
        conflict_type = conflicts[0]['property'] if has_conflict else None
        
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


def demonstrate_both_conflicts():
    """
    Demonstrate conflict detection.
    
    Input: None
    Output: None (prints demonstration)
    """
    engine = InferenceEngine()
    
    print("="*70)
    print("INFERENCE ENGINE - CONFLICT DEMONSTRATION")
    print("="*70)
    
    romantic_conflict_restaurant = {
        'restaurantname': 'The Busy Romantic',
        'pricerange': 'moderate',
        'food_quality': 'good',
        'food': 'french',
        'crowdedness': 'busy',
        'length_stay': 'long'
    }
    
    print("\n--- Example 1: Romantic Conflict ---")
    print(f"Restaurant: {romantic_conflict_restaurant['restaurantname']}")
    
    result1 = engine.apply_rules(romantic_conflict_restaurant)
    
    if result1['has_conflict']:
        for conflict in result1['conflicts']:
            print(f"\nConflict on: {conflict['property']}")
            print(f"  Rule {conflict['rule1']['id']}: → {conflict['property']} = {conflict['rule1']['value']}")
            print(f"  Rule {conflict['rule2']['id']}: → {conflict['property']} = {conflict['rule2']['value']}")
    
    touristic_conflict_restaurant = {
        'restaurantname': 'Casa Romaneasca',
        'pricerange': 'cheap',
        'food_quality': 'good',
        'food': 'romanian',
        'crowdedness': 'not busy',
        'length_stay': 'short'
    }
    
    print("\n--- Example 2: Touristic Conflict ---")
    print(f"Restaurant: {touristic_conflict_restaurant['restaurantname']}")
    
    result2 = engine.apply_rules(touristic_conflict_restaurant)
    
    if result2['has_conflict']:
        for conflict in result2['conflicts']:
            print(f"\nConflict on: {conflict['property']}")
            print(f"  Rule {conflict['rule1']['id']}: → {conflict['property']} = {conflict['rule1']['value']}")
            print(f"  Rule {conflict['rule2']['id']}: → {conflict['property']} = {conflict['rule2']['value']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    demonstrate_both_conflicts()
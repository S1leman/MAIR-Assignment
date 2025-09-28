class InferenceEngine:
    """
    Applies inference rules to determine additional preferences
    """
    
    def __init__(self):
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
    
    def evaluate_antecedent(self, rule, restaurant):
        """
        Evaluate if the antecedent conditions of a rule are satisfied by the restaurant
        """
        antecedent = rule['antecedent']
        antecedent_type = rule['antecedent_type']
        
        if antecedent_type == 'SINGLE':
            condition = antecedent[0]
            return self._check_single_condition(condition, restaurant)
        
        elif antecedent_type == 'AND': 
            return all(self._check_single_condition(condition, restaurant) for condition in antecedent)
        
        return False
    
    def _check_single_condition(self, condition, restaurant):
        """
        Check if a single condition is satisfied by the restaurant
        """ 
        if condition == 'cheap':
            value = restaurant.get('pricerange', '') or ''
            return value.lower() == 'cheap'
        elif condition == 'good':
            value = restaurant.get('food_quality', '') or ''
            return value.lower() == 'good'
        elif condition == 'romanian':
            value = restaurant.get('food', '') or ''
            return value.lower() == 'romanian'
        elif condition == 'busy':
            value = restaurant.get('crowdedness', '') or ''
            return value.lower() == 'busy'
        elif condition == 'long':
            value = restaurant.get('length_stay', '') or ''
            return value.lower() == 'long'
        
        return False
    
    def apply_rules(self, restaurant):
        """
        Apply all applicable inference rules to a restaurant and return inferred properties
        """
        inferred_properties = {}
        applied_rules = []
        reasoning = []
        
        # Initialize all possible consequent properties
        possible_properties = {
            'touristic': None,
            'assigned_seats': None,
            'children': None,
            'romantic': None
        }
        
        # Apply each rule
        for rule in self.inference_rules:
            if self.evaluate_antecedent(rule, restaurant):
                consequent = rule['consequent']
                consequent_value = rule['consequent_value']
                description = rule['description']
                
                applied_rules.append(rule['id'])
                
                if consequent in inferred_properties:
                    if inferred_properties[consequent]['value'] != consequent_value:
                        # Contradiction detected - don't auto-resolve, return conflict info
                        old_rule_id = inferred_properties[consequent]['rule_id']
                        old_value = inferred_properties[consequent]['value']
                        old_description = inferred_properties[consequent]['description']
                        
                        # Special handling for romantic property conflicts
                        if consequent == 'romantic' and {rule['id'], old_rule_id} == {5, 6}:
                            conflict_info = {
                                'property': consequent,
                                'rule1': {'id': old_rule_id, 'value': old_value, 'description': old_description},
                                'rule2': {'id': rule['id'], 'value': consequent_value, 'description': description},
                                'requires_user_input': True
                            }
                            
                            return {
                                'inferred_properties': possible_properties,
                                'applied_rules': applied_rules,
                                'reasoning': reasoning,
                                'has_contradictions': True,
                                'conflict': conflict_info
                            }
                        # Special handling for touristic property conflicts  
                        elif consequent == 'touristic' and {rule['id'], old_rule_id} == {1, 2}:
                            conflict_info = {
                                'property': consequent,
                                'rule1': {'id': old_rule_id, 'value': old_value, 'description': old_description},
                                'rule2': {'id': rule['id'], 'value': consequent_value, 'description': description},
                                'requires_user_input': True
                            }
                            
                            return {
                                'inferred_properties': possible_properties,
                                'applied_rules': applied_rules,
                                'reasoning': reasoning,
                                'has_contradictions': True,
                                'conflict': conflict_info
                            }
                        else:
                            reasoning.append(f"Contradiction detected for {consequent}: "
                                           f"Rule {old_rule_id} ({old_description}) suggests {old_value}, "
                                           f"Rule {rule['id']} ({description}) suggests {consequent_value}. "
                                           f"Keeping {consequent} = {old_value}")
                            continue
                                
                inferred_properties[consequent] = {
                    'value': consequent_value,
                    'rule_id': rule['id'],
                    'description': description
                }
                
                reasoning.append(f"Rule {rule['id']}: {description} -> {consequent} = {consequent_value}")
        
        for prop, data in inferred_properties.items():
            possible_properties[prop] = data['value']
        
        return {
            'inferred_properties': possible_properties,
            'applied_rules': applied_rules,
            'reasoning': reasoning,
            'has_contradictions': any('Contradiction detected' in reason for reason in reasoning)
        }
    
    def filter_restaurants_by_requirements(self, restaurants, user_requirements):
        """
        Filter restaurants based on additional user requirements and inference results
        """
        filtered_restaurants = []
        romantic_conflicts = []
        touristic_conflicts = []
        
        for restaurant in restaurants:
            inference_result = self.apply_rules(restaurant)
            
            # Check for romantic conflicts when user specifically asked for romantic restaurant
            if ('conflict' in inference_result and 
                inference_result['conflict']['requires_user_input'] and
                inference_result['conflict']['property'] == 'romantic' and
                'romantic' in user_requirements and 
                user_requirements['romantic'] is not None):
                restaurant_with_inference = restaurant.copy()
                restaurant_with_inference['inference_result'] = inference_result
                romantic_conflicts.append(restaurant_with_inference)
                continue
            
            # Check for touristic conflicts when user specifically asked for touristic restaurant
            if ('conflict' in inference_result and 
                inference_result['conflict']['requires_user_input'] and
                inference_result['conflict']['property'] == 'touristic' and
                'touristic' in user_requirements and 
                user_requirements['touristic'] is not None):
                restaurant_with_inference = restaurant.copy()
                restaurant_with_inference['inference_result'] = inference_result
                touristic_conflicts.append(restaurant_with_inference)
                continue
                
            inferred_properties = inference_result['inferred_properties']
            
            # Check if restaurant meets all user requirements
            meets_requirements = True
            for req_property, req_value in user_requirements.items():
                if req_value is not None:  
                    inferred_value = inferred_properties.get(req_property)
                    if inferred_value is not None and inferred_value != req_value:
                        meets_requirements = False
                        break
            
            if meets_requirements:
                restaurant_with_inference = restaurant.copy()
                restaurant_with_inference['inference_result'] = inference_result
                filtered_restaurants.append(restaurant_with_inference)
        
        # If we have romantic conflicts and user wants romantic restaurant, return conflict info
        if romantic_conflicts and 'romantic' in user_requirements and user_requirements['romantic'] is not None:
            return {
                'restaurants': filtered_restaurants,
                'has_romantic_conflicts': True,
                'has_touristic_conflicts': False,
                'romantic_conflict_restaurants': romantic_conflicts,
                'touristic_conflict_restaurants': []
            }
        
        # If we have touristic conflicts and user wants touristic restaurant, return conflict info
        if touristic_conflicts and 'touristic' in user_requirements and user_requirements['touristic'] is not None:
            return {
                'restaurants': filtered_restaurants,
                'has_romantic_conflicts': False,
                'has_touristic_conflicts': True,
                'romantic_conflict_restaurants': [],
                'touristic_conflict_restaurants': touristic_conflicts
            }
        
        return {
            'restaurants': filtered_restaurants,
            'has_romantic_conflicts': False,
            'has_touristic_conflicts': False,
            'romantic_conflict_restaurants': [],
            'touristic_conflict_restaurants': []
        }
    
    def explain_recommendation(self, restaurant):
        """
        Generate explanation text for why a restaurant was recommended based on inference rules. 
        """
        if 'inference_result' not in restaurant:
            return ""
        
        inference_result = restaurant['inference_result']
        reasoning = inference_result['reasoning']
        inferred_properties = inference_result['inferred_properties']
        
        if not reasoning:
            return ""
        
        explanations = []
        
        # Add romantic explanation if property was inferred
        if inferred_properties.get('romantic') is not None:
            romantic_value = inferred_properties['romantic']
            if romantic_value: 
                for reason in reasoning:
                    if 'romantic = True' in reason or 'romantic = true' in reason:
                        if 'long time' in reason.lower() or 'long stay' in reason.lower():
                            explanations.append("The restaurant is romantic because it allows you to stay for a long time")
                        break
                if not explanations and romantic_value:
                    explanations.append("The restaurant is romantic")
            else:  
                for reason in reasoning:
                    if 'romantic = False' in reason or 'romantic = false' in reason:
                        if 'busy' in reason.lower():
                            explanations.append("The restaurant is not romantic because it is busy")
                        break
         
        if inferred_properties.get('touristic') is not None:
            touristic_value = inferred_properties['touristic']
            if touristic_value:
                explanations.append("The restaurant is touristic because it offers good food at cheap prices")
            else:
                if 'romanian' in str(reasoning).lower():
                    explanations.append("The restaurant is not touristic because Romanian cuisine is unfamiliar to most tourists")
        
        if inferred_properties.get('assigned_seats'):
            explanations.append("The restaurant has assigned seating because it is busy")
            
        if inferred_properties.get('children') is False:
            explanations.append("The restaurant is not suitable for children because guests typically stay for a long time")
         
        contradiction_explanations = []
        for reason in reasoning:
            if 'Contradiction' in reason:
                contradiction_explanations.append(reason)
        
        result = ". ".join(explanations)
        if result:
            result += "."
         
        if contradiction_explanations:
            if result:
                result += " "
            result += " ".join(contradiction_explanations)
        
        return result
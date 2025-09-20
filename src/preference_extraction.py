import re
from typing import Dict, List, Optional, Tuple
import Levenshtein

class PreferenceExtractor: 
    def __init__(self): 
        self.food_types = [
            'world', 'swedish', 'tuscan', 'international', 'cuban', 'catalan', 
            'chinese', 'persian', 'italian', 'french', 'british', 'indian',
            'thai', 'vietnamese', 'european', 'mediterranean', 'seafood',
            'bistro', 'asian oriental', 'gastropub', 'portuguese', 'american',
            'japanese', 'mexican', 'spanish', 'danish', 'polish', 'korean',
            'greek', 'turkish', 'lebanese', 'moroccan', 'ethiopian', 'russian'
        ]
         
        self.areas = [
            'center', 'centre', 'north', 'south', 'east', 'west',
            'central', 'northern', 'southern', 'eastern', 'western'
        ]
         
        self.price_ranges = {
            'cheap': ['cheap', 'inexpensive', 'budget', 'affordable'],
            'moderate': ['moderate', 'moderately priced', 'medium', 'mid-range'],
            'expensive': ['expensive', 'pricey', 'upscale', 'high-end', 'costly']
        }
         
        self.food_patterns = [
            r"(?:serves?|serving|serve)\s+(\w+)\s+food",
            r"(\w+)\s+food",
            r"(\w+)\s+restaurant",
            r"looking for\s+(\w+)\s+food",
            r"want\s+(\w+)\s+food",
            r"find\s+(?:a\s+)?(\w+)\s+restaurant"
        ]
        
        self.area_patterns = [
            r"in\s+the\s+(\w+)(?:\s+part)?(?:\s+of\s+town)?",
            r"in\s+(\w+)(?:\s+part)?(?:\s+of\s+town)?",
            r"(\w+)\s+part\s+of\s+town",
            r"restaurant\s+in\s+(?:the\s+)?(\w+)"
        ]
        
        self.price_patterns = [
            r"(cheap|expensive|moderate(?:ly\s+priced)?)\s+restaurant",
            r"restaurant.*?(cheap|expensive|moderate(?:ly\s+priced)?)",
            r"(?:want|looking\s+for).*?(cheap|expensive|moderate(?:ly\s+priced)?)",
            r"(cheap|expensive|moderate(?:ly\s+priced)?)"
        ]
         
        self.dontcare_expressions = [
            'any', 'anywhere', 'any area', 'any part', 'any food', 'any type',
            'dont care', "don't care", 'doesnt matter', "doesn't matter",
            'whatever', 'anything'
        ]
    
    def extract_preferences(self, utterance: str) -> Dict[str, str]:
        utterance = utterance.lower().strip()
        preferences = {}
         
        food_pref = self._extract_food_preference(utterance)
        if food_pref:
            preferences['food'] = food_pref
         
        area_pref = self._extract_area_preference(utterance)
        if area_pref:
            preferences['area'] = area_pref
         
        price_pref = self._extract_price_preference(utterance)
        if price_pref:
            preferences['pricerange'] = price_pref
        
        return preferences
    
    def _extract_food_preference(self, utterance: str) -> Optional[str]: 
        # Check for food-specific dontcare expressions
        food_dontcare = ['any food', 'any type of food', 'any cuisine', 'doesnt matter what food', 
                        "doesn't matter what food", 'whatever food', 'anything to eat']
        if any(expr in utterance for expr in food_dontcare):
            return 'dontcare'
            
        # Check for general dontcare only if no specific context is present
        general_dontcare = ['dont care', "don't care", 'doesnt matter', "doesn't matter", 'whatever', 'anything']
        if any(expr in utterance for expr in general_dontcare) and 'food' in utterance:
            return 'dontcare'
            
        for pattern in self.food_patterns:
            matches = re.findall(pattern, utterance, re.IGNORECASE)
            for match in matches:
                food_type = self._fuzzy_match_food(match)
                if food_type:
                    return food_type
         
        words = utterance.split()
        for word in words:
            food_type = self._fuzzy_match_food(word)
            if food_type:
                return food_type
        
        return None
    
    def _extract_area_preference(self, utterance: str) -> Optional[str]: 
        if any(expr in utterance for expr in ['any area', 'any part', 'anywhere']):
            return 'dontcare' 
        
        for pattern in self.area_patterns:
            matches = re.findall(pattern, utterance, re.IGNORECASE)
            for match in matches:
                area = self._fuzzy_match_area(match)
                if area:
                    return area
         
        words = utterance.split()
        for word in words:
            area = self._fuzzy_match_area(word)
            if area:
                return area
        
        return None
    
    def _extract_price_preference(self, utterance: str) -> Optional[str]:
        for pattern in self.price_patterns:
            matches = re.findall(pattern, utterance, re.IGNORECASE)
            for match in matches:
                price = self._normalize_price_term(match)
                if price:
                    return price
        
        return None
    
    def _fuzzy_match_food(self, word: str) -> Optional[str]:
        word = word.lower().strip()
         
        if word in self.food_types:
            return word
         
        # Don't fuzzy match very short words or common words that could be confused
        if len(word) <= 3 or word in ['that', 'this', 'with', 'what', 'want', 'have', 'like', 'good']:
            return None
         
        best_match = None
        min_distance = float('inf')
        
        for food_type in self.food_types:
            distance = Levenshtein.distance(word, food_type)
              
            if distance <= 2 and len(word) > 4 and distance < min_distance:
                min_distance = distance
                best_match = food_type
         
        if min_distance <= min(2, len(word) // 3):
            return best_match
        
        return None
    
    def _fuzzy_match_area(self, word: str) -> Optional[str]: 
        word = word.lower().strip()
         
        if word in self.areas: 
            if word in ['center', 'centre', 'central']:
                return 'centre'
            return word
         
        best_match = None
        min_distance = float('inf')
        
        for area in self.areas:
            distance = Levenshtein.distance(word, area)
            if distance <= 2 and distance < min_distance:
                min_distance = distance
                best_match = area
        
        if min_distance <= 2: 
            if best_match in ['center', 'centre', 'central']:
                return 'centre'
            return best_match
        
        return None
    
    def _normalize_price_term(self, term: str) -> Optional[str]: 
        term = term.lower().strip()
        
        for price_category, variations in self.price_ranges.items():
            if any(variation in term for variation in variations):
                return price_category
        
        return None
    
    def validate_preferences(self, preferences: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
        validated = {}
        errors = []
        
        for pref_type, value in preferences.items():
            if pref_type == 'food':
                if value == 'dontcare' or value in self.food_types:
                    validated[pref_type] = value
                else:
                    errors.append(f"Unrecognized food type: {value}")
            
            elif pref_type == 'area':
                if value == 'dontcare' or value in ['centre', 'north', 'south', 'east', 'west']:
                    validated[pref_type] = value
                else:
                    errors.append(f"Unrecognized area: {value}")
            
            elif pref_type == 'pricerange':
                if value in ['cheap', 'moderate', 'expensive']:
                    validated[pref_type] = value
                else:
                    errors.append(f"Unrecognized price range: {value}")
        
        return validated, errors

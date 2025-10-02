import re as re 
import Levenshtein 


class PreferenceExtractor:
    """
    Extract restaurant preferences from user utterances using multiple strategies.
    
    Features: exact matching, regex patterns, fuzzy matching, "don't care" detection
    """

    food_types = [
    'british', 'modern european', 'italian', 'romanian', 'seafood', 'chinese',
    'steakhouse', 'asian oriental', 'french', 'portuguese', 'indian', 'spanish',
    'european', 'vietnamese', 'korean', 'thai', 'moroccan', 'swiss', 'fusion',
    'gastropub', 'tuscan', 'international', 'traditional', 'mediterranean',
    'polynesian', 'african', 'turkish', 'bistro', 'north american', 'australasian',
    'persian', 'jamaican', 'lebanese', 'cuban', 'japanese', 'catalan'
    ]
    
    areas = ['west', 'north', 'south', 'centre', 'center', 'east']  
    
    price_ranges = ['cheap', 'moderate', 'expensive', 'moderately priced']
 
    @staticmethod
    def _get_context_handlers():
        """
        Retrieves mapping of dialog contexts to their specialized extraction functions.
        
        Output: dict - Mapping of context strings to extraction function references
               Keys: 'ASK_FOOD_TYPE', 'ASK_PRICE', 'ASK_AREA'
               Values: Corresponding static method references
        """
        return {
            'ASK_FOOD_TYPE': PreferenceExtractor.food_extraction,
            'ASK_PRICE': PreferenceExtractor.price_extraction,
            'ASK_AREA': PreferenceExtractor.area_extraction
        }
     
    @staticmethod
    def _get_normalization_rules():
        """
        Provides value normalization rules for standardizing extracted preferences.
        
        Output: dict - Nested mapping for preference value normalization
               Structure: {preference_type: {variant: standard_form}}
        
        Examples: 'center' -> 'centre', 'moderately priced' -> 'moderate'
        """
        return {
            'area': {'center': 'centre'},
            'price': {'moderately priced': 'moderate'}
        }

    @staticmethod
    def is_dontcare_response(utterance):
        """
        Detect if utterance indicates "don't care" response.
        
        Input: utterance (str) - User input text
        Output: bool - True if no preference expressed
        """
        dontcare_patterns = [
            'any', 'anything', 'dont care', 'i dont care', "i don't care",  "don't care", 'doesnt matter', 
            "doesn't matter", 'no preference', 'whatever', 'i dont mind',
            "i don't mind", 'all', 'either', 'both', 'anywhere', 'anyplace'
        ]
        utterance_lower = utterance.lower().strip()
        
        # Check for exact matches with primary patterns
        if utterance_lower in dontcare_patterns:
            return True
        
        # Check for contextual variations with common suffixes
        for pattern in dontcare_patterns:
            if utterance_lower == pattern or utterance_lower == f"{pattern} is fine" or \
               utterance_lower == f"{pattern} is good" or utterance_lower == f"{pattern} is ok":
                return True
        
        return False

    @staticmethod
    def closest_term(word, terms, max_distance=3):
        """
        Find closest matching term using Levenshtein distance.
        
        Input: word (str), terms (list), max_distance (int, default=3)
        Output: str or None - Best match within distance threshold
        """
        best_term = None
        best_distance = float("inf")
        for term in terms:
            dist = Levenshtein.distance(word, term)
            if dist < best_distance:
                best_distance = dist
                best_term = term
        if best_distance <= max_distance and best_term[:1] == word[:1]:
            return best_term
        return None
    
    @staticmethod
    def food_extraction(utterance, preference, context=None):
        """
        Extract food/cuisine type from user utterance.
        
        Input: utterance (str), preference (dict), context (str, optional)
        Output: dict - Updated preference with 'food' key if found
        """
        utt = utterance.lower()
        
        # Priority 1: Context-aware "don't care" detection for food type questions
        if context == 'ASK_FOOD_TYPE' and PreferenceExtractor.is_dontcare_response(utterance):
            preference["food"] = 'dontcare'
            return preference
        
        potential_food_type = ''

        # Priority 2: Regex pattern definitions for structured extraction
        patterns = [
            r"(\w+)\s+food",                # Pattern: "<cuisine> food"
            r"(\w+)\s+restaurant",          # Pattern: "<cuisine> restaurant"
            r"\b(\w+(?:\s+\w+)?)\b"        # Pattern: General word/phrase extraction
        ]

        # Priority 3: Exact token matching for precise food type identification
        for word in utt.split():
            if word in PreferenceExtractor.food_types:
                preference["food"] = word
                return preference

        # Priority 4: Regex-based candidate extraction and validation
        for patt in patterns:
            match = re.search(patt, utt)
            if match:
                potential_food_type = match.group(1).strip()
                if potential_food_type in PreferenceExtractor.food_types:
                    preference["food"] = potential_food_type
                    return preference

        # Priority 5: Fuzzy matching for typo tolerance (minimum 4 characters)
        if len(potential_food_type) >= 4:
            closest = PreferenceExtractor.closest_term(potential_food_type, PreferenceExtractor.food_types)
            if closest:
                preference["food"] = closest
                return preference

        # Priority 6: Specialized "don't care" phrase detection for food context
        food_dontcare = ['any food', 'any type of food', 'any cuisine', 'doesnt matter what food', 
                        "doesn't matter what food", 'whatever food', 'anything to eat', 'any type',
                        'any kind of food', 'any kind']
        if any(expr in utterance for expr in food_dontcare):
            preference["food"] = 'dontcare'
            return preference
        
        return preference

    @staticmethod
    def price_extraction(utterance, preference, context=None):
        """
        Extract price range preference from user utterance.
        
        Input: utterance (str), preference (dict), context (str, optional)
        Output: dict - Updated preference with 'pricerange' key if found
        """
        utt = utterance.lower()
        
        # Priority 1: Context-aware "don't care" detection for price questions
        if context == 'ASK_PRICE' and PreferenceExtractor.is_dontcare_response(utterance):
            preference["price"] = 'dontcare'
            return preference
        
        potential_price = ''

        # Regex patterns for price extraction
        patterns = [
            r"(\w+)\s+restaurant",          # Pattern: "<price> restaurant"
            r"(\w+)\s+price",              # Pattern: "<price> price"
            r"(\w+(?:\s+\w+)?)\s+restaurant",  # Pattern: "<price phrase> restaurant"
            r"\b(\w+(?:\s+\w+)?)\b"        # Pattern: General word/phrase extraction
        ]
        
        # Priority 2: Exact token matching for standard price terms
        for word in utt.split():
            if word in PreferenceExtractor.price_ranges:
                preference["price"] = word
                return preference
            # Special case: handle "moderately" variation
            if word == "moderately":
                preference["price"] = "moderate"
                return preference

        # Priority 3: Regex-based candidate extraction with fuzzy matching
        for patt in patterns:
            match = re.search(patt, utt)
            if match:
                potential_price = match.group(1).strip()

            # Apply fuzzy matching for longer candidates
            if len(potential_price) >= 4:
                closest = PreferenceExtractor.closest_term(potential_price, PreferenceExtractor.price_ranges)
                if closest:
                    preference["price"] = closest
                    return preference

        # Priority 4: Specialized "don't care" phrase detection for price context
        price_dontcare = ['any price', 'any price range', 'any budget', 'whatever price', 
                         'any cost', 'doesnt matter how much', "doesn't matter how much",
                         "doesn't matter what it costs", "doesnt matter what it costs",
                         'whatever it costs', 'any amount', 'price doesnt matter',
                         "price doesn't matter"]
        if any(expr in utterance for expr in price_dontcare):
            preference["price"] = 'dontcare'
            return preference
        
        return preference

    @staticmethod
    def area_extraction(utterance, preference, context=None):
        """
        Extract area preference from user utterance.
        
        Input: utterance (str), preference (dict), context (str, optional)
        Output: dict - Updated preference with 'area' key if found
        """
        utt = utterance.lower()
        
        # Priority 1: Context-aware "don't care" detection for area questions
        if context == 'ASK_AREA' and PreferenceExtractor.is_dontcare_response(utterance):
            preference["area"] = 'dontcare'
            return preference
        
        potential_area = ''

        # Regex patterns for area extraction (ordered by specificity)
        patterns_area = [
            r"in the (\w+(?:\s+\w+)?) (?:part|side) of town",     # "in the north part of town"
            r"restaurant in the (\w+(?:\s+\w+)?) (?:part|side) of town",  # "restaurant in the south side of town"
            r"(?:the\s+)?(\w+)\s+(?:part|side) of town",          # "north part of town"
            r"in the (\w+(?:\s+\w+)?) (?:part|side)",            # "in the east part"
            r"\b(\w+(?:\s+\w+)?)\b"                               # General word extraction
        ]

        # Priority 2: Exact token matching for standard area terms
        for word in utt.split():
            if word in PreferenceExtractor.areas:
                preference["area"] = word
                return preference

        # Priority 3: Regex-based pattern matching with fuzzy fallback
        for patt in patterns_area:
            match = re.search(patt, utt)
            if match:
                potential_area = match.group(1).strip()
                
                # Check for exact match first
                if potential_area in PreferenceExtractor.areas:
                    preference["area"] = potential_area
                    return preference
                
                # Apply fuzzy matching for potential typos
                closest = PreferenceExtractor.closest_term(potential_area, PreferenceExtractor.areas)
                if closest:
                    preference["area"] = closest
                    return preference
                
        # Priority 4: Specialized "don't care" phrase detection for area context
        area_dontcare = ['any area', 'any part', 'anywhere', 'any part of town',
                        'any location', 'doesnt matter where', "doesn't matter where",
                        'location doesnt matter', "location doesn't matter"]
        if any(expr in utterance for expr in area_dontcare):
            preference["area"] = 'dontcare'
            return preference
        
        return preference

    @staticmethod
    def extract_all(utterance, context=None):
        """
        Extracts food, price, and area preferences from utterance.
        Context can be 'ASK_FOOD_TYPE', 'ASK_PRICE', or 'ASK_AREA' to help with appropriate extraction.
        """
        preference = {}
        context_handlers = PreferenceExtractor._get_context_handlers()
         
        if context and context in context_handlers:
            # Only extract the type being asked for
            preference = context_handlers[context](utterance, preference, context)
        else:
            # No specific context - extract all (for general "I want italian food in the south")
            for handler in context_handlers.values():
                preference = handler(utterance, preference, context)
         
        PreferenceExtractor._normalize_extracted_values(preference)
        
        return preference
    
    @staticmethod
    def _normalize_extracted_values(preference):
        """Normalize extracted preference values using structure-based rules."""
        normalization_rules = PreferenceExtractor._get_normalization_rules()
        
        # Apply normalization rules structure-based approach
        for pref_type, value in preference.items():
            if pref_type in normalization_rules and value in normalization_rules[pref_type]:
                preference[pref_type] = normalization_rules[pref_type][value]
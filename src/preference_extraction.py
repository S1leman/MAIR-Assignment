import re as re
import pandas as pd
import Levenshtein
import numpy as np

class PreferenceExtractor:

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
    def closest_term(word, terms, max_distance=3):
        best_term = None
        best_distance = float("inf")
        #print(f"were in the closest term function now, the word is: {word}")
        for term in terms:
            dist = Levenshtein.distance(word, term)
            if dist < best_distance:
                best_distance = dist
                best_term = term
        #print(f"were in the closest term function now, closest term: {best_term}")
        if best_distance <= max_distance and best_term[:1] == word[:1]:
            return best_term
        return None
    
    @staticmethod
    def food_extraction(utterance, preference):
        utt = utterance.lower()
        potential_food_type = ''

        patterns = [
            r"(\w+)\s+food",
            r"(\w+)\s+restaurant",
            r"\b(\w+(?:\s+\w+)?)\b"
        ]

        # Check losse woorden
        for word in utt.split():
            if word in PreferenceExtractor.food_types:
                preference["food"] = word
                return preference

        # Check patronen
        for patt in patterns:
            match = re.search(patt, utt)
            if match:
                potential_food_type = match.group(1).strip()
                if potential_food_type in PreferenceExtractor.food_types:
                    preference["food"] = potential_food_type
                    return preference
                
        if len(potential_food_type) >= 4:
            closest = PreferenceExtractor.closest_term(potential_food_type, PreferenceExtractor.food_types)
            if closest:
                preference["food"] = closest
                return preference

    
        food_dontcare = ['any food', 'any type of food', 'any cuisine', 'doesnt matter what food', 
                        "doesn't matter what food", 'whatever food', 'anything to eat']
        if any(expr in utterance for expr in food_dontcare):
            preference["food"] = 'dontcare'
            return preference
        
        return preference

    @staticmethod
    def price_extraction(utterance, preference):
        utt = utterance.lower()
        potential_price = ''

        patterns = [
            r"(\w+)\s+restaurant",
            r"(\w+)\s+price",
            r"(\w+(?:\s+\w+)?)\s+restaurant",
            r"\b(\w+(?:\s+\w+)?)\b"
        ]

        # Check losse woorden
        for word in utt.split():
            if word in PreferenceExtractor.price_ranges:
                preference["price"] = word
                return preference
            if word == "moderately":
                preference["price"] = "moderate"
                return preference


        for patt in patterns:
            #print(f"pattern:{patt}")
            match = re.search(patt, utt)
            if match:
                potential_price = match.group(1).strip()
                #print(f"match found:{potential_price}")

            #print(f"potential food type voor de lengte check: {potential_price}")
            if len(potential_price) >= 4:
                #print("length is groter of gelijk aan 4")
                closest = PreferenceExtractor.closest_term(potential_price, PreferenceExtractor.price_ranges)
                if closest:
                    preference["price"] = closest
                    return preference
            
        price_dontcare = ['any price', 'any price range', 'any budget', 'whatever price', 
                         'any cost', 'doesnt matter how much', "doesn't matter how much",
                         "doesn't matter what it costs", "doesnt matter what it costs",
                         'whatever it costs', 'any amount']
        if any(expr in utterance for expr in price_dontcare):
            preference["price"] = 'dontcare'
            return preference
        
        return preference

    @staticmethod
    def area_extraction(utterance, preference):
        utt = utterance.lower()
        potential_area = ''

        
        patterns_area = [
            r"in the (\w+(?:\s+\w+)?) (?:part|side) of town",
            r"restaurant in the (\w+(?:\s+\w+)?) (?:part|side) of town",
            r"(?:the\s+)?(\w+)\s+(?:part|side) of town",
            r"in the (\w+(?:\s+\w+)?) (?:part|side)",
            r"\b(\w+(?:\s+\w+)?)\b"
        ]

        for word in utt.split():
            if word in PreferenceExtractor.areas:
                preference["area"] = word
                return preference

        for patt in patterns_area:
            #print(f"pattern: {patt}")
            match = re.search(patt, utt)
            if match:
                potential_area = match.group(1).strip()
                #print(f"Match found : {potential_area}")
                if potential_area in PreferenceExtractor.areas:
                    preference["area"] = potential_area
                    return preference
                
                closest = PreferenceExtractor.closest_term(potential_area, PreferenceExtractor.areas)
                #print(f"closest = {closest}")
                if closest:
                    #print(f"closest term: {closest}")
                    preference["area"] = closest
                    return preference
        if any(expr in utterance for expr in ['any area', 'any part', 'anywhere']):
            preference["area"] = 'dontcare'
            return preference
        
        return preference

    @staticmethod
    def extract_all(utterance):
        preference = {}
        preference = PreferenceExtractor.food_extraction(utterance, preference)
        preference = PreferenceExtractor.price_extraction(utterance, preference)
        preference = PreferenceExtractor.area_extraction(utterance, preference)
        return preference
    

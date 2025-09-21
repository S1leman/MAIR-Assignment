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
        for term in terms:
            dist = Levenshtein.distance(word, term)
            if dist < best_distance:
                best_distance = dist
                best_term = term
        if best_distance <= max_distance and best_term[:2] == word[:2]:
            return best_term
        return None
    
    @staticmethod
    def food_extraction(utterance, preference):
        utt = utterance.lower()
        potential_food_type = ''

        patterns = [
            r"(\w+)\s+food",
            r"(\w+)\s+restaurant"
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

    @staticmethod
    def price_extraction(utterance, preference):
        utt = utterance.lower()
        potential_price = ''

        # Check losse woorden
        for word in utt.split():
            if word in PreferenceExtractor.price_ranges:
                preference["price"] = word
                return preference
            if word == "moderately":
                preference["price"] = "moderate"
                return preference

        patterns = [
            r"(\w+)\s+restaurant",
            r"(\w+)\s+price",
            r"(\w+(?:\s+\w+)?)\s+restaurant",
        ]

        for patt in patterns:
            match = re.search(patt, utt)
            if match:
                potential_price = match.group(1).strip()

        if len(potential_price) >= 4:
            closest = PreferenceExtractor.closest_term(potential_price, PreferenceExtractor.price_ranges)
            if closest:
                preference["price"] = closest
                return preference
            
        if any(expr in utterance for expr in ['any area', 'any part', 'anywhere']):
            preference["area"] = 'dontcare'
            return preference

        return preference

    @staticmethod
    def area_extraction(utterance, preference):
        utt = utterance.lower()
        potential_area = ''


        for word in utt.split():
            if word in PreferenceExtractor.areas:
                preference["area"] = word
                return preference

        patterns_area = [
            r"in the (\w+(?:\s+\w+)?) (?:part|side) of town",
            r"restaurant in the (\w+(?:\s+\w+)?) (?:part|side) of town",
            r"(?:the\s+)?(\w+)\s+(?:part|side) of town",
            r"in the (\w+(?:\s+\w+)?) (?:part|side)",
            r"\b(\w+(?:\s+\w+)?)\b"
        ]

        for patt in patterns_area:
            match = re.search(patt, utt)
            if match:
                potential_area = match.group(1).strip()
                if potential_area in PreferenceExtractor.areas:
                    preference["area"] = potential_area
                    return preference
                closest = PreferenceExtractor.closest_term(potential_area, PreferenceExtractor.areas)
                if closest:
                    preference["area"] = closest
                    return preference

        return preference

    @staticmethod
    def extract_all(utterance):
        preference = {}
        preference = PreferenceExtractor.food_extraction(utterance, preference)
        preference = PreferenceExtractor.price_extraction(utterance, preference)
        preference = PreferenceExtractor.area_extraction(utterance, preference)
        return preference
    

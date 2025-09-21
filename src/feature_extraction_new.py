import re as re
import pandas as pd
import Levenshtein
import numpy as np

class PreferenceExtractor:

    @staticmethod
    def food_extraction(utterance, preference, food_types):
        utt = utterance.lower()
        potential_food_type = ''

        patterns = [
            r"(\w+)\s+food",
            r"(\w+)\s+restaurant"
        ]

        # Check losse woorden
        for word in utt.split():
            if word in food_types:
                preference["food"] = word
                return preference

        # Check patronen
        for patt in patterns:
            match = re.search(patt, utt)
            if match:
                potential_food_type = match.group(1).strip()
                if potential_food_type in food_types:
                    preference["food"] = potential_food_type
                    return preference

        if len(potential_food_type) >= 4:
            closest = PreferenceExtractor.closest_term(potential_food_type, food_types)
            if closest:
                preference["food"] = closest
                return preference

        preference["food"] = None
        return preference

    @staticmethod
    def price_extraction(utterance, preference, price_ranges):
        utt = utterance.lower()
        potential_price = ''

        # Check losse woorden
        for word in utt.split():
            if word in price_ranges:
                preference["price"] = word
                return preference
            if word == "moderately":
                preference["price"] = "moderately priced"
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
            closest = PreferenceExtractor.closest_term(potential_price, price_ranges)
            if closest:
                preference["price"] = closest
                return preference

        preference["price"] = None
        return preference

    @staticmethod
    def area_extraction(utterance, preference, areas):
        utt = utterance.lower()
        potential_area = ''


        for word in utt.split():
            if word in areas:
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
                if potential_area in areas:
                    preference["area"] = potential_area
                    return preference
                closest = PreferenceExtractor.closest_term(potential_area, areas)
                if closest:
                    preference["area"] = closest
                    return preference

        preference["area"] = None
        return preference

    @staticmethod
    def extract_all(utterance, food_types, price_ranges, areas):
        preference = {"food": None, "price": None, "area": None}
        preference = PreferenceExtractor.food_extraction(utterance, preference, food_types)
        preference = PreferenceExtractor.price_extraction(utterance, preference, price_ranges)
        preference = PreferenceExtractor.area_extraction(utterance, preference, areas)
        return preference
def majority_baseline_model(test_utterances, majority_label):
    # Predicts the majority label for all test utterances (which is "inform" by default).
    return [majority_label] * len(test_utterances)


def rules_baseline_model(test_utterances):
    """
    Predicts labels using predefined keyword-matching rules.
    If a keyword is found in the utterance, assigns the corresponding label.
    If no rule matches, defaults to the majority class "inform".
    """
    rules = {
            'affirm': ['yes', 'correct', 'right', 'yeah', 'ye'],
            'thankyou': ['thank you', 'thanks'],
            'bye': ['goodbye', 'good bye', 'bye'],
            'request': ['phone number', 'postcode', 'address', 'type of food', 'post code', 'area', 'addre', 'part of town', 'price range'],
            'confirm': ['is it','does it'],
            'deny': ['wrong'],
            'hello': ['hello', 'hi'],
            'negate': ['no'],
            'null': ['unintelligible','cough', 'noise', 'sil'],
            'repeat': ['repeat', 'go back'],
            'reqalts': ['how about', 'how bout', 'anything else', 'what about'],
            'reqmore': ['more'],
            'restart': ['start'],
            'ack': ['okay', 'kay', 'okay um'] 

    }
    majority_class = "inform"

    predictions = []
    for utterance in test_utterances:
        u = utterance.lower()
        words = u.split()
        prediction = majority_class
        found = False

        for label, keywords_list in rules.items():
            for keyword in keywords_list:
                if " " in keyword:  
                    if keyword in u:
                        prediction = label
                        found = True
                        break
                else:
                    if keyword in words:
                        prediction = label
                        found = True
                        break
            if found:
                break

        predictions.append(prediction)
    return predictions
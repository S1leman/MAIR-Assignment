def majority_baseline_model(test_utterances, majority_label):
    """
    Simple baseline that predicts the same label for all utterances.
    
    Input: test_utterances (list of strings), majority_label (string)
    Output: list of predictions (all identical to majority_label)
    """
    return [majority_label] * len(test_utterances)


def rules_baseline_model(test_utterances):
    """
    Rule-based classifier using keyword matching for dialog act prediction.
    
    Input: test_utterances (list of strings)
    Output: list of predicted dialog acts based on keyword rules
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
    default_class = "inform"

    predictions = []
    for utterance in test_utterances:
        u = utterance.lower()
        words = u.split()
        prediction = default_class
        found = False

        for label, keywords_list in rules.items():
            for keyword in keywords_list:
                # Handle phrases and single words
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
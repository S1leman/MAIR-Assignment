from sklearn.model_selection import train_test_split

def read_data(path):
    dialogue_act = []
    utterance = []
    act_by_utterance_map = {}

    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().lower().split(' ', 1)

            if data[1] not in utterance: 
                act_by_utterance_map[data[1]] = data[0]
                dialogue_act.append(data[0])
                utterance.append(data[1])
            else:
                dialogue_act.append(act_by_utterance_map[data[1]])
                utterance.append(data[1])
                
    return dialogue_act, utterance



def baseline_model1(test_data, majority_label):
    return [majority_label] * len(test_data)

def rulebased_system(utterances):
    """ Function that defines the rule-based system"""
    rules = {
    "thankyou": ["thank you", "thanks", "thankyou", "thx"],
    "bye": ["goodbye", "good bye", "see you", "thank you good bye"],
    "hello": ["hello", "hi", "hey"],
    "affirm": ["yes", "yeah", "yep", "right", "correct", "exactly"],
    "ack": ["okay", "ok", "alright", "uh", "um", "mm"],
    "negate": ["no", "nope", "nah"],
    "deny": ["dont want", "don't want", "wrong", "not "],
    "confirm": ["is it", "is that", "does it", "do they", "are they"],
    "repeat": ["repeat", "say that again", "could you repeat", "please repeat"],
    "reqalts": ["anything else", "how about", "another option", "something else"],
    "reqmore": ["more"],
    "request": [
        "what is", "whats", "what's", "phone number", "address", "postcode",
        "post code", "price range", "and the price", "can i get", "give me", "tell me"
    ],
    "restart": ["start over", "restart"],
    "null": ["noise", "sil", "um", "uh"],
    # fallback: user preferences (slots/constraints)
    "inform": [
        "looking for", "i want", "i need", "restaurant", "food", "cuisine",
        "cheap", "expensive", "moderate", "north", "south", "east", "west",
        "centre", "center", "area", "any"
    ]
}


    majority_class = "inform"

    predictions = []
    for utternace in utterances:
        prediction = majority_class
        found = False
        for label, keywords_list in rules.items():
            for keyword in keywords_list:
                if keyword in utternace:
                    prediction = label
                    found = True
                    break
            if found == True:
                break

                
        predictions.append(prediction)
    return predictions

def split_and_save_dataset(dialogue_act, utterance, test_size):
    train_acts, test_acts, train_utterances, test_utterances = train_test_split(
        dialogue_act, utterance, test_size=test_size, random_state=42
    )

    with open('data/train_dataset.dat', 'w') as train_file:
        for act, utter in zip(train_acts, train_utterances):
            train_file.write(f"{act} {utter}\n")

    with open('data/test_dataset.dat', 'w') as test_file:
        for act, utter in zip(test_acts, test_utterances):
            test_file.write(f"{act} {utter}\n")

    return train_acts, test_acts, train_utterances, test_utterances

def accuracy(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def main(): 
    path = "data/dialog_acts.dat"
    dialogue_act, utterance = read_data(path)

    train_acts, test_acts, train_utterances, test_utterances = split_and_save_dataset(
        dialogue_act, utterance, 0.15
    )

    predictions = baseline_model1(test_acts, "inform") # given: "inform" -> the majority label
 

    y_pred_rule = rulebased_system(test_utterances) 
    print("Rule-based baseline accuracy:", accuracy(test_acts, y_pred_rule))

    rulebased_predictions = rulebased_system(utterance)
    
    while True:
        user_input = input("Enter an utterance (type 'exit' to quit): ")
        if user_input.strip().lower() == "exit":
            break
        prediction = rulebased_system([user_input])
        print(f"Predicted dialogue act: {prediction[0]}")

if __name__ == '__main__':
    main()



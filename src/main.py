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


def rulebased_system(test_data):
    """ Function that defines the rule-based system"""
    rules = {
            'thankyou': ['thank you', 'thanks'],
            'ack': ['okay'] ,
            'affrm': ['yes'],
            'bye': ['goodbye'],
            'confirm': ['is there'],
            'deny': ['dont want'],
            'hello': ['hello', 'hi', 'hey'],
            'inform': ['i am looking for', 'i want', 'i need'],
            'negate': ['no'],
            'null': ['cough'],
            'repeat': ['say that again'],
            'reqalts': ['how about'],
            'reqmore': ['more'],
            'request': ['what is'],
            'restart': ['start over']

    }
    majority_class = "inform"

    predictions = []
    for utternace in test_data:
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

def split_and_save_dataset(dialogue_act, utterance, train_path, test_path, test_size=0.15, random_state=42):
    train_acts, test_acts, train_utterances, test_utterances = train_test_split(
        dialogue_act, utterance, test_size=test_size, random_state=random_state
    )

    with open(train_path, 'w') as train_file:
        for act, utter in zip(train_acts, train_utterances):
            train_file.write(f"{act} {utter}\n")

    with open(test_path, 'w') as test_file:
        for act, utter in zip(test_acts, test_utterances):
            test_file.write(f"{act} {utter}\n")

    return train_acts, test_acts, train_utterances, test_utterances

def main(): 
    path = "data/dialog_acts_test.dat"
    dialogue_act, utterance = read_data(path)
    
    train_acts, test_acts, train_utterances, test_utterances = split_and_save_dataset(
        dialogue_act, utterance, 'data/train_dataset.dat', 'data/test_dataset.dat'
    )

    predictions = baseline_model1(test_acts, "inform") # given: "inform" -> the majority label

    rulebased_predictions = rulebased_system(utterance)
    print("Rule-based system predictions:", rulebased_predictions)
    
    while True:
        user_input = input("Enter an utterance (type 'exit' to quit): ")
        if user_input.strip().lower() == "exit":
            break
        prediction = rulebased_system([user_input])
        print(f"Predicted dialogue act: {prediction[0]}")

if __name__ == '__main__':
    main()



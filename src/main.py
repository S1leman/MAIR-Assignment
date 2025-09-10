from sklearn.model_selection import train_test_split

def read_data(path):
    duplicates = set()
    dialogue_act = []
    utterance = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().lower().split(' ', 1)
            act_utterance_pair = (data[0], data[1])
            list.append(act_utterance_pair)

            if act_utterance_pair not in duplicates: # Only add unique pairs
                duplicates.add(act_utterance_pair)
                dialogue_act.append(data[0])
                utterance.append(data[1])

    return dialogue_act, utterance



def baseline_model1(test_data, majority_label):
    return [majority_label] * len(test_data)


def rulebased_system(utterances):
    """ Function that defines the rule-based system"""
    rules = {'hi': 'hello'}
    majority_class = "inform"

    predictions = []
    for utter in utterances:
        prediction = majority_class
        for pattern, label in rules.items():
            if pattern in utter:
                prediction = label
                break
        predictions.append(prediction)
    return predictions

def main(): 
    path = "data/dialog_acts.dat"
    dialogue_act, utterance = read_data(path)
    print(dialogue_act)
    print(utterance)

    train_acts, test_acts, train_utterances, test_utterances = train_test_split(
        dialogue_act, utterance, test_size=0.15
    )

    predictions = baseline_model1(test_acts, "inform") # given: "inform" -> the majority label



if __name__ == '__main__':
    main()



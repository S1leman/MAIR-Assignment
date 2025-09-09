from sklearn.model_selection import train_test_split
def read_data(path):
    regels = [] # can be removed, used for testing
    dialogue_act = []
    utterance = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().lower().split(' ', 1)
            regels.append(data) # can be removed, used for testing
            dialogue_act.append(data[0])
            utterance.append(data[1])

    return dialogue_act, utterance

def baseline_model1(test_data, majority_label):
    return [majority_label] * len(test_data)

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
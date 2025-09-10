def read_data():
    duplicates = set()
    dialogue_act = []
    utterance = []
    list = []

    with open('data/dialog_acts.dat', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().lower().split(' ', 1)
            act_utterance_pair = (data[0], data[1])
            list.append(act_utterance_pair)

            if act_utterance_pair not in duplicates: # Only add unique pairs
                duplicates.add(act_utterance_pair)
                dialogue_act.append(data[0])
                utterance.append(data[1])

    return utterance, dialogue_act, duplicates, list

    #print(regels)
    #print(dialogue_act)
    #print(utterance)

#def majority_system(): 

def rulebased_system(utterances):
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


x ,y ,set, lijst = read_data()
print(len(x))
print(len(y))
print(len(set))

if set.issubset(lijst):
    print("Alle elementen van de set zitten in de lijst")
else:
    print("Niet alles zit in de lijst")


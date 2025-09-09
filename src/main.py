def read_data():
    regels = [] # can be removed, used for testing
    dialogue_act = []
    utterance = []
    with open('data/dialog_acts.dat', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().lower().split(' ', 1)
            regels.append(data) # can be removed, used for testing
            dialogue_act.append(data[0])
            utterance.append(data[1])

    #print(regels)
    #print(dialogue_act)
    #print(utterance)
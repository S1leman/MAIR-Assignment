# file = open('data/dialog_acts.dat', 'r')
# content = file.read()
# print(content)
regels = []
dialogue_act = []
utterance = []
with open('data/dialog_acts.dat', 'r') as f:
    lines = f.readlines()
    for line in lines[:3]:
        data = line.strip().lower().split(' ', 1)
        regels.append(data)
        dialogue_act.append(data[0])
        utterance.append(data[1])

print(regels)
print(dialogue_act)
print(utterance)
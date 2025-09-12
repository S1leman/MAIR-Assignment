from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

def read_data(path,dedup: bool = False):
    """
    If dedup = True, keep only the first occurrence of each utterance
             =  False, keep duplicates (but enforce consistent label)
    
    """
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
                if not dedup:
                    # keep duplicates, but always use the first-seen label
                    # if dedup=True then skip this utterance
                     dialogue_act.append(act_by_utterance_map[data[1]])
                     utterance.append(data[1])
                
    return dialogue_act, utterance






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
    path = "data/dialog_acts.dat"
    # original data
    dialogue_act_dup, utterance_dup = read_data(path, dedup=False)
    split_and_save_dataset(
        dialogue_act_dup, utterance_dup,
        train_path="data/train_dataset_dup.dat",
        test_path="data/test_dataset_dup_.dat",
        test_size=0.15, random_state=42
    )

    # deduplicated data
    dialogue_act_dedup, dialogue_act_dedup = read_data(path, dedup=True)
    split_and_save_dataset(
        dialogue_act_dedup, dialogue_act_dedup,
        train_path="data/train_dataset_dedup.dat",
        test_path="data/test_dataset_dedup.dat",
        test_size=0.15, random_state=42
    )

if __name__ == '__main__':
    main()


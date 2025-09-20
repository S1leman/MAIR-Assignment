from sklearn.model_selection import train_test_split
def read_data(path, deduplicate: bool = False):
    dialogue_act = []
    utterance = []
    utterance_to_first_act = {} 

    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().lower().split(' ', 1)
            
            current_act = data[0]
            current_utterance = data[1]
            
            if current_utterance in utterance_to_first_act:
                act_to_use = utterance_to_first_act[current_utterance]
                if not deduplicate:
                    dialogue_act.append(act_to_use)
                    utterance.append(current_utterance)
            else:
                utterance_to_first_act[current_utterance] = current_act
                dialogue_act.append(current_act)
                utterance.append(current_utterance)
                
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

def load_data():
    print("Loading data...")

    # Original data
    acts_orig, utterances_orig = read_data("data/dialog_acts.dat", deduplicate=False)
    train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig = split_and_save_dataset(
        acts_orig, utterances_orig, "data/train_orig.txt", "data/test_orig.txt"
    )
    
    # Deduplicated data
    acts_dedup, utterances_dedup = read_data("data/dialog_acts.dat", deduplicate=True)
    train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup = split_and_save_dataset(
        acts_dedup, utterances_dedup, "data/train_dedup.txt", "data/test_dedup.txt"
    )
    
    return {
        'orig': (train_acts_orig, test_acts_orig, train_utts_orig, test_utts_orig),
        'dedup': (train_acts_dedup, test_acts_dedup, train_utts_dedup, test_utts_dedup)
    }

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



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
def decision_tree(train_acts, test_acts, train_utterances, test_utterances): 
    # convert text to Bag of Words
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_utterances)
    X_test = vectorizer.transform(test_utterances)  # <- transform only, don’t fit!

    clf = DecisionTreeClassifier(random_state=42).fit(X_train, train_acts)
    y_pred = clf.predict(X_test)
    print(accuracy_score(test_acts, y_pred))
    return y_pred


def evaluate(y_true, y_pred,  name1="Model", name2="Dataset"):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n-- {name1} on {name2} --")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))





def main(): 
    path = "data/dialog_acts.dat"
    # original data
    dialogue_act_dup, utterance_dup = read_data(path, dedup=False)
    train_acts_dup, test_acts_dup, train_utterances_dup, test_utterances_dup=split_and_save_dataset(
        dialogue_act_dup, utterance_dup,
        train_path="data/train_dataset_dup.dat",
        test_path="data/test_dataset_dup.dat"
    )

    # deduplicated data
    dialogue_act_dedup, utterance_dedup = read_data(path, dedup=True)
    train_acts_dedup, test_acts_dedup, train_utterances_dedup, test_utterances_dedup=split_and_save_dataset(
        dialogue_act_dedup, utterance_dedup,
        train_path="data/train_dataset_dedup.dat",
        test_path="data/test_dataset_dedup.dat"
    )

    decisiontree_predictions = decision_tree(train_acts_dedup, test_acts_dedup, train_utterances_dedup, test_utterances_dedup)
    # Evaluate
    evaluate(test_acts_dedup, decisiontree_predictions, name1=f"Decision Tree", name2="Original data")

    
if __name__ == '__main__':
    main()


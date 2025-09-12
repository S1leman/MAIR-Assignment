from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
            'affirm': ['yes', 'correct', 'right', 'yeah', 'ye'],
            'thankyou': ['thank you', 'thanks'],
            'bye': ['goodbye', 'good bye', 'bye'],
            'request': ['phone number', 'postcode', 'address', 'type of food', 'post code', 'area', 'addre', 'part of town', 'price range'],
            'confirm': ['is it','does it'],
            'deny': ['wrong'],
            'hello': ['hello', 'hi'],
            'negate': ['no'],
            'null': ['unintelligible','cough', 'noise', 'sil'],
            'repeat': ['repeat', 'go back'],
            'reqalts': ['how about', 'how bout', 'anything else', 'what about'],
            'reqmore': ['more'],
            'restart': ['start'],
            'ack': ['okay', 'kay', 'okay um'] 

    }
    majority_class = "inform"

    predictions = []
    for utterance in test_data:
        u = utterance.lower()
        words = u.split()  # split into tokens
        prediction = majority_class
        found = False

        for label, keywords_list in rules.items():
            for keyword in keywords_list:
                if " " in keyword:  
                    # multi-word keyword → substring search
                    if keyword in u:
                        prediction = label
                        found = True
                        break
                else:
                    # single-word keyword → whole word search
                    if keyword in words:
                        prediction = label
                        found = True
                        break
            if found:
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


def evaluate(y_true, y_pred,  name="Model"):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n-- {name} --")
    print(f"Accuracy: {acc:.4f}")

def main(): 
    path = "data/dialog_acts.dat"
    dialogue_act, utterance = read_data(path)
    
    train_acts, test_acts, train_utterances, test_utterances = split_and_save_dataset(
        dialogue_act, utterance, 'data/train_dataset.dat', 'data/test_dataset.dat'
    )

    baseline_predictions = baseline_model1(test_acts, "inform") # given: "inform" -> the majority label
    rulebased_predictions = rulebased_system(test_utterances)
    #print("Rule-based system predictions:", rulebased_predictions)


    # Evaluate
    evaluate(test_acts, baseline_predictions, name=f"Baseline")
    evaluate(test_acts, rulebased_predictions,name="Rule-based system")

    
    while True:
        user_input = input("Enter an utterance (type 'exit' to quit): ")
        if user_input.strip().lower() == "exit":
            break
        prediction = rulebased_system([user_input])
        print(f"Predicted dialogue act: {prediction[0]}")

if __name__ == '__main__':
    main()


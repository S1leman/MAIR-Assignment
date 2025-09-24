from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import GradientBoostingClassifier

def gradient_boosting_classifier(train_acts, test_acts, train_utterances, test_utterances, return_model=False): 
    """
    Train a Gradient Boosting classifier on bag-of-words features and predict on the test set.
    Takes as input the training labels, test labels, train utterances, test utterances and a flag. 
    If flag is True then the function returns (fitted_model, vectorizer) instead of predictions.
    Returns the Predicted labels for the test set or (model, vectorizer) if flag is True.
    """
    #Vectorize text via bag-of-words (fit on train, apply to test)
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_utterances)    
    X_test = vectorizer.transform(test_utterances) 

    clf = GradientBoostingClassifier(n_estimators=250, ccp_alpha=1e-10,random_state=42)

    clf.fit(X_train, train_acts)
    y_pred = clf.predict(X_test)

    if return_model:
        return clf, vectorizer
    return y_pred

def decision_tree_classifier(train_acts, test_acts, train_utterances, test_utterances, return_model=False):
    """
    Train a Decision Tree classifier on n-gram bag-of-words features and predict on the test set.
    """
    #Vectorize text into up-to-5-gram BoW features
    vectorizer = CountVectorizer(ngram_range=(1,5)) 
    X_train = vectorizer.fit_transform(train_utterances)
    X_test = vectorizer.transform(test_utterances) 

    clf = DecisionTreeClassifier(ccp_alpha=1e-5,class_weight="balanced",random_state=42).fit(X_train, train_acts) 

    clf.fit(X_train, train_acts)  
    y_pred = clf.predict(X_test)
    if return_model:
        return clf, vectorizer
    return y_pred

def logistic_regression_classifier(train_acts, test_acts, train_utterances, test_utterances, return_model=False): 
    """
    Train a Logistic Regression  classifier on n-gram bag-of-words features and predict on the test set.
    """
    #Bag-of-words with unigrams + bigrams
    vectorizer = CountVectorizer(ngram_range=(1,2)) 
    X_train = vectorizer.fit_transform(train_utterances)
    X_test = vectorizer.transform(test_utterances) 

    clf = LogisticRegression(random_state=42).fit(X_train, train_acts)

    y_pred = clf.predict(X_test)
    if return_model:
        return clf, vectorizer
    return y_pred

def mlp_classifier(train_acts, test_acts, train_utterances, test_utterances, return_model=False): 
    """
    Train a feed-forward neural network MLP classifier on n-gram bag-of-words features and predict on the test set.
    """
    #Bag-of-words with unigrams + bigrams
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_utterances)
    X_test = vectorizer.transform(test_utterances)

    le = LabelEncoder()
    y_train = le.fit_transform(train_acts)
    y_test = le.transform(test_acts)
    sample_weights = compute_sample_weight('balanced', y_train)

    clf = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", max_iter=300, random_state=42)
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred_int = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_int)
    if return_model:
        return clf, vectorizer, le
    return y_pred
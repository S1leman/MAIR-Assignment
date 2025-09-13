from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

def decision_tree_classifier(train_acts, test_acts, train_utterances, test_utterances): 
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_utterances)
    X_test = vectorizer.transform(test_utterances) 
    clf = DecisionTreeClassifier(random_state=42).fit(X_train, train_acts)
    y_pred = clf.predict(X_test)
    return y_pred

def logistic_regression_classifier(train_acts, test_acts, train_utterances, test_utterances): 
    vectorizer = CountVectorizer(ngram_range=(1,2)) 
    X_train = vectorizer.fit_transform(train_utterances)
    X_test = vectorizer.transform(test_utterances) 
    clf = LogisticRegression(random_state=42).fit(X_train, train_acts)
    y_pred = clf.predict(X_test)
    return y_pred


def mlp_classifier(train_acts, test_acts, train_utterances, test_utterances): 
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_utterances).toarray()
    X_test = vectorizer.transform(test_utterances).toarray()
    le = LabelEncoder()
    y_train_int = le.fit_transform(train_acts)
    y_test_int = le.transform(test_acts)

    num_classes = len(le.classes_)
    y_train = to_categorical(y_train_int, num_classes=num_classes)
    y_test = to_categorical(y_test_int, num_classes=num_classes)

    model = Sequential([
        Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=8, batch_size=64, validation_split=0.1, verbose=0)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_int = y_pred_probs.argmax(axis=1)
    y_pred = le.inverse_transform(y_pred_int)
    return y_pred

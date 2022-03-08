# In this file, I have used Logistic Regression based solution
# Here, I have only used the text features for modelling
# It performed better than the FNN based solution

import numpy as np
import pandas as pd
from json import loads
from spacy import load
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def get_text(bp):
    text_dict = loads(bp)

    text = ""
    for value in text_dict.values():
        if value != None:
            text = text + " " + value

    return text


def preprocess(doc):
    doc = doc.apply(get_text)

    nlp = load("en_core_web_sm")
    count = 0

    for text in nlp.pipe(doc, n_process=4, batch_size=250, disable=["ner", "parser"]):
        doc[count] = " ".join([token.lemma_ for token in text if token.is_alpha and not token.is_stop])
        count += 1

    return doc


train_df = pd.read_csv("../input/stumbleupon/train.tsv", sep="\t")
test_df = pd.read_csv("../input/stumbleupon/test.tsv", sep="\t")
sub_df = pd.read_csv("../input/stumbleupon/sampleSubmission.csv")

train_text = train_df["boilerplate"]
target = train_df["label"]
del train_df

test_text = test_df["boilerplate"]
del test_df

train_text = preprocess(train_text)
test_text = preprocess(test_text)

X_train, X_val, y_train, y_val = train_test_split(
    train_text,
    target,
    stratify=target,
    shuffle=True,
    test_size=0.2,
    random_state=42
)

del train_text

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)
test_text = vectorizer.transform(test_text)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


clf = LogisticRegression()
clf.fit(X_train, y_train)

acc = accuracy_score(y_val, clf.predict(X_val))
print("Accuracy score:", acc)

auc = roc_auc_score(y_val.ravel(), clf.predict_proba(X_val)[:, 1])
print("ROC AUC score:", auc)

sub_df["label"] = clf.predict_proba(test_text)[:, 1]
sub_df.to_csv("submission.csv", index=False)
"""
Improved Tool: TF-IDF (bigrams) + Linear SVM for Bug Report Classification

Linear SVM is the standard choice for high-dimensional sparse text features (Joachims, 1998).
Bigrams capture two-word phrases (e.g. "memory leak", "slow training") that unigrams miss.
class_weight='balanced' compensates for the ~16% positive class rate.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def run_improved(X, y, n_repeats=30, test_size=0.3, random_state_base=0):
    precisions, recalls, f1s = [], [], []

    for i in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state_base + i, stratify=y
        )

        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000)
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    return np.array(precisions), np.array(recalls), np.array(f1s)

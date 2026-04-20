"""
Comparison model: TF-IDF (bigrams) + Logistic Regression

Logistic regression is a strong linear baseline for text classification.
Using the same bigrams TF-IDF as the SVM allows a direct classifier comparison.
class_weight='balanced' handles class imbalance identically to the SVM approach.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def run_logistic(X, y, n_repeats=30, test_size=0.3, random_state_base=0):
    precisions, recalls, f1s = [], [], []

    for i in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state_base + i, stratify=y
        )

        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs')
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    return np.array(precisions), np.array(recalls), np.array(f1s)

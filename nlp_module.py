import pandas as pd
import numpy as np
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score



nltk.download('stopwords')
from nltk.corpus import stopwords

class NLPModule:

    @staticmethod
    def preprocess_text(text):

        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))

        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]

        return " ".join(words)

    @staticmethod
    def train_spam_model(data, text_column, target_column):

        print("\nStarting NLP Spam Detection AutoML Training...")

        data[text_column] = data[text_column].apply(NLPModule.preprocess_text)

        X = data[text_column]
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Linear SVM": LinearSVC(),
            "Random Forest": RandomForestClassifier()
        }

        best_model = None
        best_score = 0
        best_name = ""

        for name, model in models.items():

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, pos_label="spam")

            print(f"\n{name} Accuracy: {accuracy}")
            print(f"{name} F1 Score (Spam): {f1}")

            if f1 > best_score:
                best_score = f1
                best_model = pipeline
                best_name = name

        print(f"\nBest Model Selected: {best_name}")
        print("Best F1 Score:", best_score)

        joblib.dump(best_model, "best_spam_model.pkl")
        print("\nBest Model Saved as best_spam_model.pkl ✅")

        return best_model
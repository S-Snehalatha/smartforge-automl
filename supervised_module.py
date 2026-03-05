import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class SupervisedModule:

    @staticmethod
    def clean_data(X):
        drop_cols = []
        for col in X.columns:
            if X[col].nunique() > 100:
                drop_cols.append(col)
        return X.drop(columns=drop_cols)

    @staticmethod
    def train_model(data, target_column, problem_type):

        print("\nStarting Advanced Supervised AutoML Training...")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        X = SupervisedModule.clean_data(X)

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if problem_type == "classification":

            models = {
                "Logistic Regression": (
                    LogisticRegression(max_iter=1000),
                    {"model__C": [0.1, 1, 10]}
                ),
                "Random Forest": (
                    RandomForestClassifier(),
                    {"model__n_estimators": [100, 200],
                     "model__max_depth": [None, 10, 20]}
                ),
                "SVM": (
                    SVC(),
                    {"model__C": [0.1, 1, 10],
                     "model__kernel": ["linear", "rbf"]}
                )
            }

            scoring_metric = "f1_weighted"

        else:

            models = {
                "Linear Regression": (
                    LinearRegression(),
                    {}
                ),
                "Random Forest Regressor": (
                    RandomForestRegressor(),
                    {"model__n_estimators": [100, 200],
                     "model__max_depth": [None, 10]}
                ),
                "SVR": (
                    SVR(),
                    {"model__C": [0.1, 1, 10],
                     "model__kernel": ["linear", "rbf"]}
                )
            }

            scoring_metric = "neg_mean_squared_error"

        best_score = -np.inf
        best_model = None
        best_model_name = ""
        model_results = []
        for name, (model, params) in models.items():

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            grid = GridSearchCV(
                pipeline,
                param_grid=params,
                cv=5,
                scoring=scoring_metric,
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            print(f"{name} Best CV Score: {grid.best_score_}")
            print(f"{name} Best Params: {grid.best_params_}\n")
            model_results.append({
                "Model": name,
                "Best CV Score": grid.best_score_
            })

            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_
                best_model_name = name

        print("=====================================")
        print(f"Best Model Selected: {best_model_name}")
        print(f"Best Cross-Validated Score: {best_score}")
        print("=====================================")

        # Final Evaluation on Test Set
        print("\nFinal Evaluation on Test Data")

        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)

        if problem_type == "classification":

            accuracy = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)
            report = classification_report(y_test, predictions)

            print("\nAccuracy:", accuracy)
            print("\nConfusion Matrix:\n", cm)
            print("\nClassification Report:\n", report)

            joblib.dump(best_model, "best_supervised_model.pkl")
            print("\nModel Saved as best_supervised_model.pkl ✅")

            return {
    "Best Model": best_model_name,
    "Best CV Score": best_score,
    "Test Accuracy": accuracy,
    "Confusion Matrix": cm.tolist(),
    "Leaderboard": model_results,
    
}
        
        else:
            mse = mean_squared_error(y_test, predictions)
            print("\nMean Squared Error:", mse)

            joblib.dump(best_model, "best_supervised_model.pkl")
            print("\nModel Saved as best_supervised_model.pkl ✅")

            return {
        "Best Model": best_model_name,
        "Best CV Score": best_score,
        "Test MSE": mse
    }
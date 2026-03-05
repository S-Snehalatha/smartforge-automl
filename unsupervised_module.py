import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score


class UnsupervisedModule:

    @staticmethod
    def cluster_data(data):

        print("\nStarting Unsupervised AutoML (Clustering)...")

        # Use only numeric columns
        X = data.select_dtypes(include=['int64', 'float64'])

        if X.shape[1] == 0:
            print("No numeric features found for clustering.")
            return None

        # Handle missing values
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        X_processed = pipeline.fit_transform(X)

        best_score = -1
        best_k = 2
        best_model = None

        # Try different cluster sizes
        for k in range(2, 8):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_processed)
            score = silhouette_score(X_processed, labels)

            print(f"K={k}, Silhouette Score={score}")

            if score > best_score:
                best_score = score
                best_k = k
                best_model = kmeans

        print("=====================================")
        print(f"Best Number of Clusters: {best_k}")
        print(f"Best Silhouette Score: {best_score}")
        print("=====================================")

        # Save model
        joblib.dump(best_model, "best_clustering_model.pkl")
        print("Clustering Model Saved as best_clustering_model.pkl ✅")

        return best_model
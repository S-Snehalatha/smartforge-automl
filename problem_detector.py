import pandas as pd

class ProblemDetector:

    @staticmethod
    def detect_problem(data, target_column=None):

        if target_column is None:
            print("No target column provided → Unsupervised Learning")
            return "unsupervised"

        if target_column not in data.columns:
            raise ValueError("Target column not found in dataset!")

        target_dtype = data[target_column].dtype

        if target_dtype == "object":
            print("Detected: Classification Problem")
            return "classification"

        elif target_dtype in ["int64", "float64"]:
            unique_values = data[target_column].nunique()

            if unique_values < 15:
                print("Detected: Classification Problem")
                return "classification"
            else:
                print("Detected: Regression Problem")
                return "regression"

        else:
            print("Unknown type. Defaulting to classification.")
            return "classification"
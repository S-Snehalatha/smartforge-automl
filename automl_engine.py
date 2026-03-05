from problem_detector import ProblemDetector
from nlp_module import NLPModule
from supervised_module import SupervisedModule
from unsupervised_module import UnsupervisedModule
import pandas as pd

class AutoMLEngine:

    @staticmethod
    def detect_text_column(data, threshold=0.6):
        """
        Detects if a column contains mostly text data.
        """
        for col in data.columns:
            if data[col].dtype == "object":
                sample = data[col].dropna().astype(str)
                avg_words = sample.apply(lambda x: len(x.split())).mean()
                if avg_words > 3:
                    return col
        return None

    @staticmethod
    def run(data, target_column=None):

        problem_type = ProblemDetector.detect_problem(data, target_column)

        # If supervised
        if problem_type in ["classification", "regression"]:

            text_column = AutoMLEngine.detect_text_column(data)

            if len(data.columns) == 2 and data[target_column].nunique() == 2:
                print("\nDetected Simple Text Classification Dataset → Routing to NLP Module")
                text_column = [col for col in data.columns if col != target_column][0]
                return NLPModule.train_spam_model(data, text_column, target_column)

            else:
                print("\nDetected Tabular Dataset → Routing to Supervised Module")
                return SupervisedModule.train_model(data, target_column, problem_type)
        # If unsupervised
        else:
            print("\nRouting to Unsupervised Module")
            return UnsupervisedModule.cluster_data(data)
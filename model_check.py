import pandas as pd
from model_impl.decision_tree import DecisionTree
from model_impl.balance_random_forest import BalanceRandomForest
from model_impl.easy_ensemble import EasyEnsemble
from model_impl.gradient_boosting import GradientBoosting
from model_impl.logistic import LogisticModel
from model_impl.random_forest import RandomForestModel
from model_impl.svm import SVMModel

if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train.to_csv("result/model_check/train.csv", index=False)
    test.to_csv("result/model_check/test.csv", index=False)
            
    model = LogisticModel(train, test, f"model_check")
    model.train_model()
    model.hit_rate()
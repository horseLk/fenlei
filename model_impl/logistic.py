## logistic回归分析

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from model import IModel

class LogisticModel(IModel):
    def __init__(self, train, test, feature_filter_type):
        super().__init__(train, test, feature_filter_type)

    def train_model(self):
        train = self._train
        y = train['是否凝血']
        X_raw = train.iloc[:, 1:]
        X = pd.get_dummies(X_raw, drop_first=True) # drop_first=True可以避免多重共线性
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
            'solver': ['liblinear', 'saga'],  # 需支持L1
            'class_weight': [None, 'balanced']
        }
        model = LogisticRegression(max_iter=1000)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X, y)

        print(f"最优参数: {grid_search.best_params_}")  
        print(f"最佳分数: {grid_search.best_score_:.4f}")
        self._model = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def hit_rate(self):
        test = self._test
        model = self._model
        y = test['是否凝血']
        X_raw = test.iloc[:, 1:]
        X = pd.get_dummies(X_raw, drop_first=True) # drop_first=True可以避免多重共线性
        y_pred = model.predict(X)
        print("\n--- 模型评估报告 ---")
        print(classification_report(y, y_pred))

        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        print(f"ROC AUC Score: {auc:.4f}")
        file = f"result/{self._feature_filter_type}/logistic.txt"
        result = f"--- 模型评估报告 ---\n{classification_report(y, y_pred)}\n\n\nROC AUC Score: {auc:.4f}\n"
        self.write_result(file, result)
    
    def feature_influence_sort(self):
        print('need to implements....')

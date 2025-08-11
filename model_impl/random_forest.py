from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap 
from collections import Counter
from scipy.stats import randint, uniform
from model import IModel

class RandomForestModel(IModel):
    def __init__(self, train, test, feature_filter_type):
        super().__init__(train, test, feature_filter_type)

    def train_model(self):
        train = self._train
        y = train['是否凝血']
        X_raw = train.iloc[:, 1:]
        # 3. 将分类变量进行独热编码 (One-Hot Encoding)
        X = pd.get_dummies(X_raw, drop_first=True) # drop_first=True可以避免多重共线性
        print("原始数据集分布:", Counter(y))
        print("数据集结果为:", X)
        rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        # 1. 关键参数粗调
        param_grid_basic = {
            'n_estimators': [50, 100, 200],
            'max_depth': [1, 2, 5, 8, 10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8, 10, 16],
            'max_features': ['sqrt', 'log2', 0.5, 0.7],
            'class_weight': [None, 'balanced', {0:1, 1:5}]  # 针对不平衡数据
        }
        grid_basic = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid_basic,
            cv=5,
            scoring='roc_auc',  # 推荐用于不平衡数据
            n_jobs=-1,
            verbose=1
        )
        grid_basic.fit(X, y)

        print("最佳基本参数:")
        print(grid_basic.best_params_)
        print(f"验证集AUC: {grid_basic.best_score_:.4f}")
        param_dist = {
            'max_depth': randint(5, 50),
            'min_samples_split': randint(10, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': uniform(0.1, 0.9),  # 连续值特征比例
            'max_samples': uniform(0.6, 0.4),   # 每棵树使用的样本比例
            'bootstrap': [True, False],
            'ccp_alpha': uniform(0, 0.1)        # 用于剪枝的复杂度参数
        }
        best_params = grid_basic.best_params_
        rf_optimized = RandomForestClassifier(
            n_estimators=500,  # 从上步学习曲线确定的最佳值
            random_state=42,
            n_jobs=-1,
            **{k: v for k, v in best_params.items() if k in [
                'class_weight', 'max_features']}
        )
        random_search = RandomizedSearchCV(
            estimator=rf_optimized,
            param_distributions=param_dist,
            n_iter=50,  # 迭代次数
            cv=5,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        random_search.fit(X, y)
        print("优化后最佳参数:")
        print(random_search.best_params_)
        print(f"最终AUC: {random_search.best_score_:.4f}")
        self._model = random_search.best_estimator_
        return random_search.best_estimator_

    def hit_rate(self):
        model = self._model
        test = self._test
        y = test['是否凝血']
        X_raw = test.iloc[:, 1:]
        # 3. 将分类变量进行独热编码 (One-Hot Encoding)
        X = pd.get_dummies(X_raw, drop_first=True)
        y_pred = model.predict(X)
        print("\n--- 模型评估报告 ---")
        print(classification_report(y, y_pred))

        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        print(f"ROC AUC Score: {auc:.4f}")
        file = f"result/{self._feature_filter_type}/randomForest.txt"
        result = f"--- 模型评估报告 ---\n{classification_report(y, y_pred)}\n\n\nROC AUC Score: {auc:.4f}\n"
        self.write_result(file, result)

    def feature_influence_sort(self):
        rf = self._model
        train = self._train
        ## todo 使用shap完成
        importances = rf.feature_importances_
        X_raw = train.iloc[:, 1:]
        # 3. 将分类变量进行独热编码 (One-Hot Encoding)
        X = pd.get_dummies(X_raw, drop_first=True) # drop_first=True可以避免多重共线性
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        print(feature_importance_df)

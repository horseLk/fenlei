from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_recall_curve
import numpy as np
from model import IModel
import pandas as pd
from collections import Counter


class EasyEnsemble(IModel):
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
        param_grid = {
            'n_estimators': [5, 10, 20],           # 基分类器数量（子集数量）
            'estimator': [                    # 基分类器选择
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(n_estimators=50)
            ],
            'sampling_strategy': [0.5, 0.7, 1.0], # 少数类采样比例（1.0表示完全平衡）
            'replacement': [True, False],          # 欠采样是否允许重复样本
            'n_jobs': [-1]                         # 并行计算（使用所有CPU核心）
        }
        easy_ensemble = EasyEnsembleClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=easy_ensemble,
            param_grid=param_grid,
            cv=5,                          # 5折交叉验证
            scoring='roc_auc',                  # 关注少数类的F1值
            n_jobs=-1,                     # 并行加速
            verbose=3                      # 输出调优过程
        )
        grid_search.fit(X, y)
        self.write_result(f"result/{self._feature_filter_type}/EasyEnsemble_param.txt", f"最佳参数组合: {grid_search.best_params_}")
        self._model =  grid_search.best_estimator_

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

        y_proba = model.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        y_pred_optimized = (y_proba > best_threshold).astype(int)

        print("\n=== 阈值优化后 (Threshold = %.3f) ===" % best_threshold)
        print("分类报告:\n", classification_report(y, y_pred_optimized))

        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        print(f"ROC AUC Score: {auc:.4f}")
        file = f"result/{self._feature_filter_type}/EasyEnsemble.txt"
        result = f"--- 模型评估报告 ---\n{classification_report(y, y_pred)}\n\n\nROC AUC Score: {auc:.4f}\n\n\n=== 阈值优化后 (Threshold = %.3f) ==={best_threshold}\n\n\n最终分类报告\n\n{classification_report(y, y_pred_optimized)}"
        self.write_result(file, result)
    
    def feature_influence_sort(self):
        print('need to implements....')
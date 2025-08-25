from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from model import IModel
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import GridSearchCV

class SVMModel(IModel):
    def __init__(self, train, test, feature_filter_type):
        super().__init__(train, test, feature_filter_type)

    def train_model(self):
        train = self._train
        self.write_result(f"result/{self._feature_filter_type}/svm_features.txt", f"数据保留特征: {train.columns.tolist()}")
        y = train['是否凝血']
        X_raw = train.iloc[:, 1:]
        # 3. 将分类变量进行独热编码 (One-Hot Encoding)
        X = pd.get_dummies(X_raw, drop_first=True) # drop_first=True可以避免多重共线性
        print("原始数据集分布:", Counter(y))
        print("数据集结果为:", X)

        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],      # 正则化强度
            'gamma': [0.001, 0.01, 0.1, 1],    # RBF核影响范围
            'kernel': ['linear', 'rbf', 'poly'] # 核函数类型
        }
        svm_model = SVC(probability=True)
        grid_search = GridSearchCV(
            estimator=svm_model,
            param_grid=param_grid,
            cv=5,                    # 五折交叉验证
            scoring='accuracy',       # 评估指标（可改为F1/AUC）
            n_jobs=-1,               # 并行加速
            verbose=1                # 输出训练日志
        )

        grid_search.fit(X, y)  # 训练模型

        self.write_result(f"result/{self._feature_filter_type}/svm_param.txt", f"最佳参数组合: {grid_search.best_params_}")
        self._model = grid_search.best_estimator_

    def hit_rate(self):
        test = self._test
        model = self._model
        y = test['是否凝血']
        X_raw = test.iloc[:, 1:]
        X = pd.get_dummies(X_raw, drop_first=True) # drop_first=True可以避免多重共线性
        y_pred = model.predict(X)
        print("\n--- 模型评估报告 ---")
        print(classification_report(y, y_pred))
        

        y_proba = model.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        y_pred_optimized = (y_proba > best_threshold).astype(int)

        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        print(f"ROC AUC Score: {auc:.4f}")
        file = f"result/{self._feature_filter_type}/svm.txt"
        result = f"--- 模型评估报告 ---\n{classification_report(y, y_pred)}\n\n\nROC AUC Score: {auc:.4f}\n\n\n=== 阈值优化后 (Threshold = %.3f) ==={best_threshold}\n\n\n最终分类报告\n\n{classification_report(y, y_pred_optimized)}"
        self.write_result(file, result)
    
    def feature_influence_sort(self):
        print('need to implements....')
    

from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
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

        ee = EasyEnsembleClassifier(
            n_estimators=10,  # 子模型数量
            random_state=42,
            # base_estimator=None  # 默认使用决策树桩（depth=1）
        )
        ee.fit(X, y)
        self._model = ee

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
        file = f"result/{self._feature_filter_type}/EasyEnsemble.txt"
        result = f"--- 模型评估报告 ---\n{classification_report(y, y_pred)}\n\n\nROC AUC Score: {auc:.4f}\n"
        self.write_result(file, result)
    
    def feature_influence_sort(self):
        print('need to implements....')
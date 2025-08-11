from imblearn.ensemble import BalancedRandomForestClassifier  
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from model import IModel
from collections import Counter
import pandas as pd

class BalanceRandomForest(IModel):
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
        brf = BalancedRandomForestClassifier(
            n_estimators=100,            # 树的数量（建议100~500）
            sampling_strategy='all',      # 过采样少数类使其与多数类数量一致
            replacement=True,             # 允许重复采样少数类样本
            bootstrap=False,              # 不使用自助采样（直接平衡抽样）
            max_features='sqrt',          # 每棵树分裂时随机选择sqrt(总特征数)
            random_state=42,
            class_weight='balanced'       # 可选：进一步调整类别权重
        )
        brf.fit(X, y)  # 训练模型
        self._model = brf

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
        file = f"result/{self._feature_filter_type}/balanceRandomForest.txt"
        result = f"--- 模型评估报告 ---\n{classification_report(y, y_pred)}\n\n\nROC AUC Score: {auc:.4f}\n"
        self.write_result(file, result)
    
    def feature_influence_sort(self):
        print('need to implements....')
    

from feature_processor import IFeatureProcessor
from pandas import DataFrame
from sklearn.feature_selection import RFE
from imblearn.ensemble import BalancedRandomForestClassifier  
import datetime

class RFEBalanceRandomForest(IFeatureProcessor):
    def __init__(self, feature_count):
        self._feature_count = feature_count
    def name(self) -> str:
        return "refWithBRF"
    def process_data(self, df) -> DataFrame:
        y = df["是否凝血"]
        X = df.iloc[:, 1:]
        brf = BalancedRandomForestClassifier(
            n_estimators=100,            # 树的数量（建议100~500）
            sampling_strategy='all',      # 过采样少数类使其与多数类数量一致
            replacement=True,             # 允许重复采样少数类样本
            bootstrap=False,              # 不使用自助采样（直接平衡抽样）
            max_features='sqrt',          # 每棵树分裂时随机选择sqrt(总特征数)
            random_state=42,
            class_weight='balanced'       # 可选：进一步调整类别权重
        )
        rfe = RFE(brf, n_features_to_select=self._feature_count)  # 保留10个特征
        rfe.fit(X, y)
        selected_features_rfe = set(X.columns[rfe.support_])
        print(f"特征数量: {len(selected_features_rfe)}")
        print(f"特征列表: {selected_features_rfe}")
        self.write_temp_data("result/temp/rfe_gb.txt", f"current_time: {datetime.datetime.now()},特征列表: {selected_features_rfe}")


        selected_features_rfe.add('是否凝血')
        columns_to_keep = [col for col in df.columns if col in selected_features_rfe]
        return df[columns_to_keep]
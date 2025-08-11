from feature_processor import IFeatureProcessor
from pandas import DataFrame
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import datetime

class RFERandomForest(IFeatureProcessor):
    def __init__(self, feature_count):
        self._feature_count = feature_count
    def name(self) -> str:
        return "rfeWithRF"
    def process_data(self, df) -> DataFrame:
        y = df["是否凝血"]
        X = df.iloc[:, 1:]
        model = RandomForestClassifier(random_state=42)
        rfe = RFE(model, n_features_to_select=self._feature_count)  # 保留10个特征
        rfe.fit(X, y)
        selected_features_rfe = set(X.columns[rfe.support_])
        print(f"特征数量: {len(selected_features_rfe)}")
        print(f"特征列表: {selected_features_rfe}")
        self.write_temp_data("result/temp/rfe_gb.txt", f"current_time: {datetime.datetime.now()},特征列表: {selected_features_rfe}")

        selected_features_rfe.add('是否凝血')
        columns_to_keep = [col for col in df.columns if col in selected_features_rfe]
        return df[columns_to_keep]
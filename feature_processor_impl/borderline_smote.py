from feature_processor import IFeatureProcessor
import numpy as np
from pandas import DataFrame
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter

class SmoteNCResample(IFeatureProcessor):
    def __init__(self, count = -1):
        self._count = count
    
    def process_data(self, df) -> DataFrame:
        train = df
        y = train['是否凝血']
        X_raw = train.iloc[:, 1:]
        ## resample with smote
        n_positive_original = np.sum(y == 1)
        n_negatives_original = np.sum(y == 0)
        print(f"原样本中正样本个数：{n_positive_original}, 负样本个数: {n_negatives_original}")
        n_positive_target = n_negatives_original
        if self._count != -1:
            n_positive_target = self._count
        b_smote = BorderlineSMOTE(kind='borderline-1', k_neighbors=5, random_state=42)
        X_resampled, y_resampled = b_smote.fit_resample(X_raw, y)
        print("SMOTE后训练集分布:", Counter(y_resampled))
        if isinstance(X_raw, pd.DataFrame):
            df_features = pd.DataFrame(X_resampled, columns=X_raw.columns)
        else:
            df_features = pd.DataFrame(X_resampled, columns=[f"feature_{i}" for i in range(X_resampled.shape[1])])
        df_labels = pd.DataFrame(y_resampled, columns=['是否凝血'])
        df_combined = pd.concat([df_labels, df_features], axis=1)
        print(df_combined.columns)
        return df_combined

    def name(self) -> str:
        return "borderlineSmote"

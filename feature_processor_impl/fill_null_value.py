from feature_processor import IFeatureProcessor
import numpy as np
from pandas import DataFrame

class FillNullValue(IFeatureProcessor):
    def __init__(self):
        super().__init__()
    def name(self) -> str:
        return "fillNullValue"
    def process_data(self, df) -> DataFrame:
        """缺失值填充"""
        print(df.head())
        ## 查询缺失值的数量
        mv = df.isnull().sum()
        print(mv)
        for column in df.columns:
            print(column)
            print(df[column])
            print(df[column].mode())
            mode_value = df[column].mode()[0]  # 获取该列的众数
            df[column].fillna(mode_value, inplace=True)
        random_state = 42
        shuffled_df = df.sample(frac=1, random_state=random_state)
        return shuffled_df
    
from model_impl.logistic import LogisticModel
from model_impl.random_forest import RandomForestModel
from model_impl.easy_ensemble import EasyEnsemble
from model_impl.balance_random_forest import BalanceRandomForest
from model_impl.gradient_boosting import GradientBoosting
from model_impl.decision_tree import DecisionTree
from model_impl.svm import SVMModel
import pandas as pd
from feature_processor_impl.replace_value import ReplaceValue
from feature_processor_impl.fill_null_value import FillNullValue
from feature_processor_impl.expand_enum_columns import ExpandEnumColumns
from feature_processor_impl.smote_nc_resample import SmoteNCResample
from feature_processor_impl.rfe_with_balance_random_forest import RFEBalanceRandomForest
from feature_processor_impl.rfe_with_gradient_boosting import RFEGradientBoosting
from feature_processor_impl.rfe_with_random_forest import RFERandomForest
from feature_processor_impl.rfe_with_svm_choice import RFESVM

import numpy as np


def split_data(df):
    """拆分成测试集和训练集"""
    train_size = 0.7  # 70%训练集
    split_point = int(len(df) * train_size)

    train_set = df[:split_point]
    test_set = df[split_point:]

    train_set.to_csv("data/train.csv",index=False)
    test_set.to_csv("data/test.csv",index=False)

if __name__ == "__main__":
    input_file = 'original_data/origin_data.xlsx'
    df = pd.read_excel(input_file)
    feature_processors = [
        ReplaceValue('#NULL!', np.nan), 
        FillNullValue(), 
        ExpandEnumColumns(exclude_column=['摇晃次数', '产次', '采血一次性成功', '来血顺畅', '民族', '糖尿病', '受孕方式', '分娩方式', '红细胞增多症', '@1血小板计数', '呼吸支持'])
    ]
    for fp in feature_processors:
        df = fp.process_data(df)
    feature_count = [4, 6, 8, 10, 12]
    for count in feature_count:
        feature_filters = [
            RFEBalanceRandomForest(count),
            RFEGradientBoosting(count),
            RFERandomForest(count),
            RFESVM(count)
        ]
        for filter_handler in feature_filters:
            use_df = filter_handler.process_data(df)
            print(f"{filter_handler.name()} data process success....")
            split_data(use_df)
            train_set_file = "data/train.csv"
            train = pd.read_csv(train_set_file)
            resample_train = SmoteNCResample(90).process_data(train)
            resample_train.to_csv(train_set_file, index=False)
            
            train = pd.read_csv("data/train.csv")
            test = pd.read_csv("data/test.csv")
            models = [
                BalanceRandomForest(train, test, f"{filter_handler.name()}_feature_count_{count}"),
                DecisionTree(train, test, f"{filter_handler.name()}_feature_count_{count}"),
                EasyEnsemble(train, test, f"{filter_handler.name()}_feature_count_{count}"),
                GradientBoosting(train, test, f"{filter_handler.name()}_feature_count_{count}"),
                LogisticModel(train, test, f"{filter_handler.name()}_feature_count_{count}"),
                RandomForestModel(train, test, f"{filter_handler.name()}_feature_count_{count}"),
                SVMModel(train, test, f"{filter_handler.name()}_feature_count_{count}"),
            ]
            for cur_model in models:
                cur_model.train_model()
                cur_model.hit_rate()
                cur_model.feature_influence_sort()
            



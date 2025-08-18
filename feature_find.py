import pandas as pd
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from feature_processor_impl.replace_value import ReplaceValue
from feature_processor_impl.fill_null_value import FillNullValue
from feature_processor_impl.expand_enum_columns import ExpandEnumColumns
import numpy as np
from feature_processor_impl.rfe_with_balance_random_forest import RFEBalanceRandomForest
from feature_processor_impl.smote_nc_resample import SmoteNCResample
from model_impl.easy_ensemble import EasyEnsemble
from main import split_data
import shap

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
    filter_handler = RFEBalanceRandomForest(11)
    use_df = filter_handler.process_data(df)
    print(f"{filter_handler.name()} data process success....")
    split_data(use_df)
    train_set_file = "data/train.csv"
    train = pd.read_csv(train_set_file)
    resample_train = SmoteNCResample(90).process_data(train)
    resample_train.to_csv(train_set_file, index=False)
    train = pd.read_csv("data/train.csv")

    y = train['是否凝血']
    X_raw = train.iloc[:, 1:]
    X = pd.get_dummies(X_raw, drop_first=True) # drop_first=True可以避免多重共线性
    test = pd.read_csv("data/test.csv")
    test_raw = test.iloc[:, 1:]
    test_X = pd.get_dummies(test_raw, drop_first=True)
    test_y = test['是否凝血']
    easy_ensemble = EasyEnsembleClassifier(random_state=42, estimator=RandomForestClassifier(max_depth=50), n_estimators=20, n_jobs=-1, replacement=False, sampling_strategy=0.5)
    easy_ensemble.fit(X, y)
    ## easy_ensemble的特征重要性排序
    
    # 计算并显示特征重要性排序
    print("\n=== EasyEnsemble 特征重要性排序 ===")
    
    # 获取所有基分类器的特征重要性
    all_importances = []
    feature_names = X.columns.tolist()
    
    for estimator in easy_ensemble.estimators_:
        if hasattr(estimator, 'feature_importances_'):
            all_importances.append(estimator.feature_importances_)
        elif hasattr(estimator, 'named_steps') and 'classifier' in estimator.named_steps:
            # 如果是Pipeline，获取其中的分类器
            classifier = estimator.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                all_importances.append(classifier.feature_importances_)
    
    if all_importances:
        # 计算平均特征重要性
        avg_importance = np.mean(all_importances, axis=0)
        
        # 创建特征重要性字典
        feature_importance_dict = dict(zip(feature_names, avg_importance))
        
        # 按重要性降序排序
        sorted_features = sorted(feature_importance_dict.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # 打印特征重要性排名
        for i, (feature, importance) in enumerate(sorted_features):
            print(f"{i+1:2d}. {feature}: {importance:.6f}")
        
        print(f"\n总共分析了 {len(sorted_features)} 个特征")
    else:
        print("无法获取特征重要性信息") 

    # first_pipeline = easy_ensemble.estimators_[0]
    # base_classifier = first_pipeline.named_steps['classifier']  # 关键步骤：通过键名 'classifier' 访问
    # print(base_classifier)
    # explainer = shap.TreeExplainer(base_classifier)  # 以第一个基分类器为例
    # # 计算测试集的SHAP值（二分类返回两个类别的SHAP数组）
    # shap_values = explainer.shap_values(test_X)  
    # # 提取正类（索引1）的SHAP值，形状为(n_samples, n_features)
    # shap_values_positive = shap_values[1]  
    # feature_importance = np.abs(shap_values_positive).mean(axis=0)
    # # 按重要性降序排序并获取特征名
    # sorted_idx = feature_importance.argsort()[::-1]
    # top_features = test_X.columns[sorted_idx]  # 假设X_test为DataFrame
    # print(top_features)
    # print("特征重要性排名：")
    # for i, feat in enumerate(top_features):
    #     print(f"{i+1}. {feat}: {feature_importance[sorted_idx[i]]:.4f}")
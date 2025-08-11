from feature_processor import IFeatureProcessor
from pandas import DataFrame
from collections import defaultdict
import pandas as pd
import re

class ExpandEnumColumns(IFeatureProcessor):
    def __init__(self, exclude_column:list):
        super().__init__()
        self._exclude_column = exclude_column

    def name(self) -> str:
        return "expandColumn"

    def process_data(self, df) -> DataFrame:
        exclude_column = self._exclude_column
        """自动识别并拆分多值枚举列"""
        columns_to_expand = []
        enum_mappings = defaultdict(dict)
        # 1. 识别需要处理的列
        for col in df.columns:
            if col in exclude_column:
                continue
            if self.should_process_column(df[col]):
                columns_to_expand.append(col)

                # 收集该列所有可能的枚举值
                all_vals = set()
                for val in df[col]:
                    processed = self.preprocess_value(val)
                    all_vals.update(processed)
                if len(all_vals) <= 2:
                    continue
                # 生成枚举值到新列名的映射
                enum_mappings[col] = {v: f"{col}_{v}" for v in all_vals if v}
        
        # 2. 创建新的DataFrame存放结果
        new_df = df.copy()


        # 3. 对每个需要处理的列执行拆分
        for col in columns_to_expand:
            mapping = enum_mappings[col]
            
            # 创建临时存储新列的DataFrame
            temp_df = pd.DataFrame(index=df.index)
            
            # 初始化所有新列为0
            for new_col in mapping.values():
                temp_df[new_col] = 0
                
            # 填充新列的值
            for idx, orig_val in enumerate(df[col]):
                vals = self.preprocess_value(orig_val)
                for v in vals:
                    if v and v in mapping:
                        new_col_name = mapping[v]
                        temp_df.loc[idx, new_col_name] = 1
            
            # 从原始DF中删除原列并添加新列
            new_df = new_df.drop(columns=[col])
            new_df = pd.concat([new_df, temp_df], axis=1)
            
            # 记录处理日志
            print(f"✅ 已拆分列 '{col}' => {len(mapping)}个新列")

            # 4. 返回处理结果
        if not columns_to_expand:
            print("⚠️ 未检测到需要拆分的多值枚举列")
        return new_df
    

    def preprocess_value(self, value):
        """统一处理各种格式的枚举值（数字、字符串、混合）"""
        if pd.isna(value):
            return ""
        # 处理用逗号/分号分隔的枚举值
        if isinstance(value, str):
            # 移除方括号和引号（如["1","2"]格式）
            cleaned = re.sub(r'[\[\]"\' ]', '', value)
            # 分割多种分隔符（支持逗号、分号、斜杠等）
            parts = re.split(r'[,;/|]', cleaned)
            # 过滤空值并标准化
            return [p.strip() for p in parts if p.strip()]
        # 处理单个数值型枚举
        return [str(value).strip()]

    def should_process_column(self, series, threshold=2):
        """判断列是否需要处理（唯一枚举值数量 > threshold）"""
        # 展开所有值并统计唯一枚举项
        all_values = []
        for val in series:
            processed = self.preprocess_value(val)
            all_values.extend(processed)
        
        # 统计唯一值数量（忽略空值）
        unique_count = len(set([v for v in all_values if v]))
        return unique_count > threshold
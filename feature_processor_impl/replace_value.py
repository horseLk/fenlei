from feature_processor import IFeatureProcessor
import numpy as np
from pandas import DataFrame

class ReplaceValue(IFeatureProcessor):
    def __init__(self, source, target):
        super().__init__()
        self._source = source
        self._target = target
    def name(self) -> str:
        return "replaceValue"
    def process_data(self, df) -> DataFrame:
        df.replace(self._source, self._target, inplace=True)
        return df
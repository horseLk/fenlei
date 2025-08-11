from abc import ABCMeta, abstractmethod
from pandas import DataFrame
import os

class IFeatureProcessor(metaclass=ABCMeta):
    @abstractmethod
    def process_data(self, df) -> DataFrame:
        pass
    def name(self) -> str:
        pass
    def write_temp_data(self, file_path, data):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 自动创建父目录
        with open(file_path, "a") as f:
            f.write(f"{data}\n\n")
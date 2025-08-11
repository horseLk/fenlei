from abc import ABCMeta, abstractmethod
import os

class IModel(metaclass=ABCMeta):
    def __init__(self, train, test, feature_filter_type):
        self._train = train
        self._test = test
        self._feature_filter_type = feature_filter_type
    @abstractmethod
    def train_model(self):
        pass
    @abstractmethod
    def hit_rate(self):
        pass
    @abstractmethod
    def feature_influence_sort(self):
        pass
    def write_result(self, file_path, result):
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print(f"文件{file_path}不存在")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 自动创建父目录
        with open(file_path, "w") as f:
            f.write(result)


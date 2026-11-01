"""Accumulate only scalar cells and small metadata, never all source DataFrames."""
from hypereeg_converter.core import Converted, ConversionError

class Accumulator:
    def __init__(self):
        self.value = Converted([], {}, {}, [], ())
        self.samples = {}
        self.raw_tables = []

    def add(self, item, raw):
        merged = self.value
        if merged.sample_columns and merged.sample_columns != item.sample_columns:
            raise ConversionError('样本键不同，禁止合并。')
        merged.sample_columns = item.sample_columns
        for sample in item.samples:
            key = sample['_key']
            if key in self.samples:
                for name, value in sample.items():
                    if name in self.samples[key] and self.samples[key][name] != value:
                        raise ConversionError(f'样本元数据冲突：{name}')
                self.samples[key].update(sample)
            else:
                self.samples[key] = sample.copy()
        for key, value in item.cells.items():
            if key in merged.cells:
                raise ConversionError('重复样本×特征，禁止静默覆盖。')
            merged.cells[key] = value
        for column, dictionary in item.features.items():
            if column in merged.features:
                merged.features[column]['origins'] = list(dict.fromkeys(merged.features[column]['origins'] + dictionary['origins']))
            else:
                merged.features[column] = dict(dictionary, origins=list(dictionary['origins']))
        self.raw_tables.append(raw)

    def finalize(self):
        self.value.samples = list(self.samples.values())
        return self.value

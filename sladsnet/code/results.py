import dataclasses as dt
import joblib

from .base import Sample


@dt.dataclass
class ResultOutputs:
    poly_features: list
    ratio_measured: list
    erds: list
    masks: list
    recons: list


class Result:
    def __init__(self, c_value: float = None):
        self.c_value = c_value
        self.outputs = ResultOutputs([], [], [], [], [])
        self.size = 0

    def add(self, sample: Sample):
        self.outputs.poly_features.append(sample.poly_features.copy().astype('float32'))
        self.outputs.erds.append(sample.ERD.copy().astype('float32'))
        self.outputs.ratio_measured.append(sample.ratio_measured.astype('float32'))
        self.outputs.masks.append(sample.mask.copy().astype('bool'))
        self.outputs.recons.append(sample.recon_image.copy().astype('float32'))
        self.size += 1

    def get_by_index(self, index):
        out_dict = {f.name: getattr(self.outputs, f.name)[index] for f in dt.fields(self.outputs)}
        return out_dict

    def get(self):
        return dt.asdict(self.outputs)

    def save(self, path: str):
        joblib.dump(self.get(), path)

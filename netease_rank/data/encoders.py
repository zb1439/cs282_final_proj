from netease_rank.utils import Registry
from sklearn.preprocessing import LabelEncoder as _LabelEncoder


ENCODERS = Registry("ENCODERS")


class Encoder:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, df, col, label_df=None):
        raise NotImplementedError


@ENCODERS.register()
class LabelEncoder(Encoder):
    def __init__(self, cfg):
        super(LabelEncoder, self).__init__(cfg)
        self.lbe = _LabelEncoder()

    def __call__(self, df, col, label_df=None):
        return self.lbe.fit_transform(df[col].values[:, None])

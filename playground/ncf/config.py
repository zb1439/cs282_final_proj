from netease_rank.config import BaseConfig

from model import SelfAttentionNCF2


_config_dict = dict(
    MODEL=dict(
        NAME="SelfAttentionNCF2",
    )
)


class BCEConfig(BaseConfig):
    def __init__(self):
        super(BCEConfig, self).__init__()
        self._register_configuration(_config_dict)


config = BCEConfig()

from netease_rank.config import BaseConfig


_config_dict = dict(
    TRAINING=dict(
        BATCH_SIZE=256,
    ),
    MODEL=dict(
        NAME="NCF",
    )
)


class BCEConfig(BaseConfig):
    def __init__(self):
        super(BCEConfig, self).__init__()
        self._register_configuration(_config_dict)


config = BCEConfig()

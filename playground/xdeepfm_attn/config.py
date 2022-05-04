from netease_rank.config import BaseConfig
from netease_rank.config.preprocess_func import *
from model import SelfAttentionXDeepFM


_config_dict = dict(
    FEATURE_ENG=dict(
        PREPROCESS=dict(
            USER=[
                # 'col_name', func, discrete, full_df_input (default false), new_name (default none)
                ('age', age_categorize, True),
                ('registeredMonthCnt', registered_month_categorize, True),
                ('followCnt', smallcount_to_id, True),
                ('level', int, True),
                # ('userIdx', lambda x: x, True),
                ('d_province', identity, True),
                ('d_gender', identity, True)
            ],
            ITEM=[
                # ('mlogindex', lambda x: x, True),
                ('mlog_userImprssionCount', bigcount_to_id, True),
                ('mlog_userClickCount', medcount_to_id, True),
                ('mlog_userLikeCount', smallcount_to_id, True),
                ('mlog_userCommentCount', smallcount_to_id, True),
                ('mlog_userShareCount', smallcount_to_id, True),
                ('mlog_userViewCommentCount', medcount_to_id, True),
                ('mlog_userIntoPersonalHomepageCount', smallcount_to_id, True),
                ('mlog_userFollowCreatorCount', smallcount_to_id, True),
                ('mlog_publishTime', publish_time_categorize, True),
                ('d_mlog_type', mlog_type_categorize, True),
                ('d_creator_gender', gender_categorize, True),
                ('creator_registeredMonthCnt', registered_month_categorize, True),
                ('creator_follows', smallcount_to_id, True),
                ('creator_followeds', bigcount_to_id, True),
                ('d_creatorType', int, True),
                ('d_creator_level', int, True)
            ],
        ),
        ENCODE=dict(
            USER=[],
            ITEM=[
                ('d_creatorType', "LabelEncoder"),
            ],
        ),
        DISCRETE_COLS=dict(
            USER=['d_age', 'd_registeredMonthCnt', 'd_level', 'userIdx', 'd_province', 'd_gender', 'd_followCnt'],
            ITEM=['mlogindex', 'd_mlog_publishTime', 'd_mlog_type', 'd_creator_gender', 'd_creator_registeredMonthCnt',
                  'd_creatorType', 'd_creator_level', 'd_mlog_userImprssionCount', 'd_mlog_userClickCount',
                  'd_mlog_userLikeCount', 'd_mlog_userCommentCount',
                  'd_mlog_userShareCount', 'd_mlog_userViewCommentCount', 'd_mlog_userIntoPersonalHomepageCount',
                  'd_mlog_userFollowCreatorCount', 'd_creator_follows', 'd_creator_followeds'],
        ),
        CONTINUOUS_COLS=dict(
            USER = [],
            ITEM=[],
        ),
        TARGET="label",
    ),
    MODEL=dict(
        NAME="SelfAttentionXDeepFM",
        SMALL_EMB_DIM=64,
        LARGE_EMB_DIM=64,
        FIELDWISE_LINEAR=False,
    ),
    TRAINING=dict(
        LR=5e-3,
        EVAL_EPOCH=1,
    )
)


class BCEConfig(BaseConfig):
    def __init__(self):
        super(BCEConfig, self).__init__()
        self._register_configuration(_config_dict)


config = BCEConfig()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

import collections
import numpy as np
import os
import os.path as osp
import pprint
import re
import six
from colorama import Back, Fore
from easydict import EasyDict
from tabulate import tabulate

from .config_helper import (
    _assert_with_logging,
    _cast_cfg_value_type,
    _decode_cfg_value,
    diff_dict,
    find_key,
    highlight,
)
from .preprocess_func import *

# python 3.8+ compatibility
try:
    collectionsAbc = collections.abc
except ImportError:
    collectionsAbc = collections


root = osp.dirname(osp.dirname(osp.dirname(__file__)))


_config_dict = dict(
    FILES=dict(
        USER=osp.join(root, 'csv_data/user_final.csv'),
        ITEM=osp.join(root, 'csv_data/mlog_final.csv'),
        # LABEL='csv_data/im_score_final.csv',
        LABEL=osp.join(root, 'csv_data/im_score_final_small.json'),
        SPLIT=osp.join(root, 'csv_data/test_pairs_small.txt'),
        TEST=osp.join(root, 'csv_data/user2negative_test_samples.json'),
        LOAD_FROM_JSON=False,
        DUMP_TO_JSON=True,
    ),
    FEATURE_ENG=dict(
        PREPROCESS=dict(
            USER=[
                # 'col_name', func, discrete, full_df_input (default false), new_name (default none)
                ('age', age_categorize, True),
                ('registeredMonthCnt', registered_month_categorize, True),
                ('followCnt', log, False),
                ('level', int, True),
                # ('userIdx', lambda x: x, True),
                ('d_province', identity, True),
                ('d_gender', identity, True)
            ],
            ITEM=[
                # ('mlogindex', lambda x: x, True),
                ('mlog_userImprssionCount', log, False),
                ('mlog_userClickCount', log, False),
                ('mlog_userLikeCount', log, False),
                ('mlog_userCommentCount', log, False),
                ('mlog_userShareCount', log, False),
                ('mlog_userViewCommentCount', log, False),
                ('mlog_ctr', compute_ctr, False, True),
                ('mlog_like_ratio', compute_like_ratio, False, True),
                ('mlog_comment_ratio', compute_comment_ratio, False, True),
                ('mlog_userIntoPersonalHomepageCount', log, False),
                ('mlog_userFollowCreatorCount', log, False),
                ('mlog_publishTime', publish_time_categorize, True),
                ('d_mlog_type', mlog_type_categorize, True),
                ('d_creator_gender', gender_categorize, True),
                ('creator_registeredMonthCnt', registered_month_categorize, True),
                ('creator_follows', log, False),
                ('creator_followeds', log, False),
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
            USER=['d_age', 'd_registeredMonthCnt', 'd_level', 'userIdx', 'd_province', 'd_gender'],
            ITEM=['mlogindex', 'd_mlog_publishTime', 'd_mlog_type', 'd_creator_gender', 'd_creator_registeredMonthCnt',
                  'd_creatorType', 'd_creator_level'],
        ),
        CONTINUOUS_COLS=dict(
            USER=['followCnt'],
            ITEM=['mlog_userImprssionCount', 'mlog_userClickCount', 'mlog_userLikeCount', 'mlog_userCommentCount',
                  'mlog_userShareCount', 'mlog_userViewCommentCount', 'mlog_userIntoPersonalHomepageCount',
                  'mlog_userFollowCreatorCount', 'creator_follows', 'creator_followeds'],
        ),
        TARGET="label",
    ),
    MODEL=dict(
        SMALL_EMB_DIM="AUTO",
        LARGE_EMB_DIM=128,
        NAME="MLP",
        FIELDWISE_LINEAR=False,
        DEEP=dict(
            HIDDEN_DIMS=[512, 256]
        ),
        NCF=dict(
            GMF_DIM=128,
        )
    ),
    EVALUATION=dict(
        EVALUATORS=[
            ["HitRate", dict(top_k=10)],
            ["NDCG", dict(k=10)],
        ],
    ),
    TRAINING=dict(
        ITEM_SIZE=16,
        NEGATIVE_RATIO=0.5,
        BATCH_SIZE=128,
        EPOCHS=20,
        LR=1e-4,
        OPTIM="SGD",
        LR_SCHEDULER=dict(
            NAME="Dummy",
            STEPS=[],
        ),
        CRITERIA=dict(
            NAME="RankNet",
        ),
        EVAL_EPOCH=2,
        EVAL_BATCHES=100,
        INFO_ITER=100,
    ),
    GLOBAL=dict(
        RANDOM_SEED=24,
        RESUME=False,
        TRAIN=True,
        TEST_EPOCH=-1,
        NUM_WORKERS=1,
    )
)


class ConfigDict(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if (
                not (k.startswith("__") and k.endswith("__"))
                and k not in self.funcname_not_in_attr()
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [EasyDict(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict):
            value = EasyDict(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def funcname_not_in_attr(self):
        return [
            "update", "pop", "merge",
            "merge_from_list", "find", "diff",
            "inner_dict", "funcname_not_in_attr"
        ]

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super().pop(k, d)

    def merge(self, config=None, **kwargs):
        """
        merge all key and values of config as BaseConfig's attributes.
        Note that kwargs will override values in config if they have the same keys
        Args:
            config (dict): custom config dict
        """
        def update_helper(d, u):
            for k, v in six.iteritems(u):
                dv = d.get(k, EasyDict())
                if not isinstance(dv, collectionsAbc.Mapping):
                    d[k] = v
                elif isinstance(v, collectionsAbc.Mapping):
                    d[k] = update_helper(dv, v)
                else:
                    d[k] = v
            return d

        if config is not None:
            update_helper(self, config)
        if kwargs:
            update_helper(self, kwargs)

    def merge_from_list(self, cfg_list):
        """
        Merge config (keys, values) in a list (e.g., from command line) into
        this config dict.
        Args:
            cfg_list (list): cfg_list must be divided exactly.
            For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".format(
                cfg_list
            ),
        )
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                _assert_with_logging(subkey in d, "Non-existent key: {}".format(full_key))
                d = d[subkey]
            subkey = key_list[-1]
            _assert_with_logging(subkey in d, "Non-existent key: {}".format(full_key))
            value = _decode_cfg_value(v)
            value = _cast_cfg_value_type(value, d[subkey], full_key)
            d[subkey] = value

    def diff(self, cfg) -> dict:
        """
        diff given config with current config object
        Args:
            cfg(ConfigDict): given config, could be any subclass of ConfigDict
        Returns:
            ConfigDict: contains all diff pair
        """
        assert isinstance(cfg, ConfigDict), "config is not a subclass of ConfigDict"
        diff_result = {}
        conf_keys = cfg.keys()
        for param in self.keys():
            if param not in conf_keys:
                diff_result[param] = getattr(self, param)
            else:
                self_val, conf_val = getattr(self, param), getattr(cfg, param)
                if self_val != conf_val:
                    if isinstance(self_val, dict) and isinstance(conf_val, dict):
                        diff_result[param] = diff_dict(self_val, conf_val)
                    else:
                        diff_result[param] = self_val
        return ConfigDict(diff_result)

    def find(self, key: str, show=True, color=Fore.BLACK + Back.YELLOW) -> dict:
        """
        find a given key and its value in config
        Args:
            key (str): the string you want to find
            show (bool): if show is True, print find result; or return the find result
            color (str): color of `key`, default color is black(foreground) yellow(background)
        Returns:
            dict: if  show is False, return dict that contains all find result
        """
        key = key.upper()
        find_result = {}
        for param, param_value in self.items():
            param_value = getattr(self, param)
            if re.search(key, param):
                find_result[param] = param_value
            elif isinstance(param_value, dict):
                find_res = find_key(param_value, key)
                if find_res:
                    find_result[param] = find_res
        if not show:
            return find_result
        else:
            pformat_str = repr(ConfigDict(find_result))
            print(highlight(key, pformat_str, color))

    def __repr__(self):
        param_list = [(k, pprint.pformat(v)) for k, v in self.items()]
        table_header = ["config params", "values"]
        return tabulate(param_list, headers=table_header, tablefmt="fancy_grid")


class BaseConfig(ConfigDict):

    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)

    def _register_configuration(self, config):
        self.merge(config)

    def link_log(self, link_name="log"):
        """
        create a softlink to output dir.
        Args:
            link_name(str): name of softlink
        """
        if os.path.islink(link_name) and os.readlink(link_name) != self.OUTPUT_DIR:
            os.system("rm " + link_name)
        if not os.path.exists(link_name):
            cmd = "ln -s {} {}".format(self.OUTPUT_DIR, link_name)
            os.system(cmd)

    def funcname_not_in_attr(self):
        namelist = super().funcname_not_in_attr()
        namelist.extend(["link_log", "_register_configuration"])
        return namelist


config = BaseConfig()

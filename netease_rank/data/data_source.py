import random
import json
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from .encoders import ENCODERS


class DataSource:
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = cfg.FEATURE_ENG.PREPROCESS
        self.encoders = cfg.FEATURE_ENG.ENCODE
        self.disc_cols = cfg.FEATURE_ENG.DISCRETE_COLS
        self.cont_cols = cfg.FEATURE_ENG.CONTINUOUS_COLS
        # self.label_df = pd.read_csv(cfg.FILES.LABEL)
        self.user2item2label, self.user2test_label, self.train_users, self.test_users, self.test_pairs = self.load_label_json()

        self.item_size = cfg.TRAINING.ITEM_SIZE
        self.negative_size = min(
            max(1, int(self.item_size * cfg.TRAINING.NEGATIVE_RATIO)),
            self.item_size - 1
        )
        self.positive_size = self.item_size - self.negative_size

        self.test_mode = False

        self.process_and_build_indices()

    @staticmethod
    def _process(processes, df):
        print("Preprocessing user and mlog dataframes ...")
        for proc in tqdm(processes):
            col, func, discrete = proc[:3]
            full_df_input = False if len(proc) < 4 else proc[3]
            new_col_name = col if len(proc) < 5 else proc[4]
            if full_df_input:
                df[new_col_name] = df.apply(func, axis=1)
            else:
                assert col in df, f"{col} not found!"
                df[new_col_name] = df[col].apply(func)
            if new_col_name != col:
                df.drop(col, axis=1)

            if discrete and not new_col_name.startswith("d_"):
                df.rename(columns={new_col_name: "d_" + new_col_name}, inplace=True)
        return df

    def _encode(self, processes, df, label):
        for col, enc_name in processes:
            assert col in df, f"{col} not found!"
            df[col] = ENCODERS.get(enc_name)(self.cfg)(df, col, label)
        return df

    def load_label_json(self):
        user2item2label = json.load(open(self.cfg.FILES.LABEL))
        test_pairs = pickle.load(open(self.cfg.FILES.SPLIT, 'rb'))
        formatted_user2item2label = {}
        for usr, item2label in user2item2label.items():
            assert len(item2label) > 0
            formatted_user2item2label[int(usr)] = {}
            for item, label in item2label.items():
                formatted_user2item2label[int(usr)][int(item)] = label['score']
        del user2item2label

        pairs_to_del = []
        user2test_label = {}
        for usr, mlog in test_pairs:
            score = formatted_user2item2label[usr][mlog]
            del formatted_user2item2label[usr][mlog]
            if len(formatted_user2item2label[usr]) == 0:
                del formatted_user2item2label[usr]
                pairs_to_del.append( (usr, mlog) )
            else:
                if usr not in user2test_label:
                    user2test_label[usr] = {mlog: score}
                else:
                    user2test_label[usr][mlog] = score
        for p in pairs_to_del:
            test_pairs.remove(p)

        train_users = list(formatted_user2item2label.keys())
        test_users = list(user2test_label.keys())
        return formatted_user2item2label, user2test_label, train_users, test_users, test_pairs


    def process_and_build_indices(self):
        print("Loading user and item dataframe...")
        self.user_df = pd.read_csv(self.cfg.FILES.USER)
        self.item_df = pd.read_csv(self.cfg.FILES.ITEM)
        self.user_df = DataSource._process(self.processes.USER, self.user_df)
        self.item_df = DataSource._process(self.processes.ITEM, self.item_df)
        self.user_df = self._encode(self.encoders.USER, self.user_df, self.user2item2label)
        self.item_df = self._encode(self.encoders.ITEM, self.item_df, self.user2item2label)

        self.user_df = self.user_df.set_index("userIdx")
        self.item_df = self.item_df.set_index("mlogindex")
        self.user_df["userIdx"] = self.user_df.index.values
        self.item_df["mlogindex"] = self.item_df.index.values
        self.user_df = self.user_df[self.disc_cols.USER + self.cont_cols.USER]
        self.item_df = self.item_df[self.disc_cols.ITEM + self.cont_cols.ITEM]

        self._cardinality = self.user_df.apply("nunique").to_dict()
        self._cardinality.update(self.item_df.apply("nunique").to_dict())

        self.user2positive_test_item = {}
        for user, mlog in self.test_pairs:
            if user in self.user2positive_test_item:
                self.user2positive_test_item[user].append(mlog)
            else:
                self.user2positive_test_item[user] = [mlog]
        user2negative_test_items = json.load(open(self.cfg.FILES.TEST))
        self.user2negative_test_items = {}
        for user, items in user2negative_test_items.items():
            self.user2negative_test_items[int(user)] = items

    @property
    def cardinality(self):
        assert hasattr(self, "_cardinality")
        return self._cardinality

    def __len__(self):
        if self.test_mode:
            return len(self.test_users)
        else: 
            return len(self.train_users)

    def __getitem__(self, idx):
        
        
        if self.test_mode:
            user = self.test_users[idx]
            user_feat = self.user_df.loc[user].values
            positive_mlogs, negative_mlogs = self.get_test_mlogs(user)
        else:
            user = self.train_users[idx]
            user_feat = self.user_df.loc[user].values
            positive_mlogs, negative_mlogs = self.get_train_mlogs(user)
        mlogs = positive_mlogs + negative_mlogs
        scores = np.zeros(len(mlogs))
        if not self.test_mode:
            scores[:len(positive_mlogs)] = np.array([self.user2item2label[user][mlog] for mlog in positive_mlogs])
        else:
            scores[:len(positive_mlogs)] = np.array([self.user2test_label[user][mlog] for mlog in positive_mlogs])

        user_feat = user_feat.astype(np.float32)
        item_feat = self.item_df.loc[mlogs].values.astype(np.float32)
        scores = scores.astype(np.float32)
        return user_feat, item_feat, scores

    def get_test_mlogs(self, user):
        positive_mlogs = self.user2positive_test_item[user]
        negative_mlogs = self.user2negative_test_items[user]
        return positive_mlogs, negative_mlogs

    def get_train_mlogs(self, user):
        positive_size = min(self.positive_size, len(self.user2item2label[user]))
        positive_mlogs = random.sample(self.user2item2label[user].keys(), positive_size)
        positive_mlogs = sorted(positive_mlogs, key=lambda x: self.user2item2label[user][x], reverse=True)

        negative_size = self.item_size - positive_size
        negative_mlogs = []
        while len(negative_mlogs) < negative_size:
            mlog = random.randint(0, len(self.item_df) - 1)
            if mlog not in self.user2item2label[user] and (user, mlog) not in self.test_pairs:
                negative_mlogs.append(mlog)
        return positive_mlogs, negative_mlogs


if __name__ == "__main__":
    from netease_rank.config import BaseConfig
    from torch.utils.data import DataLoader
    import time
    cfg = BaseConfig()
    cfg.GLOBAL.NUM_WORKERS = 0
    ds = DataSource(cfg)
    ds.test_mode = True
    loader = iter(DataLoader(ds, batch_size=100, num_workers=0))
    # start = time.time()
    # for i in range(100):
    #     next(loader)
    # print(time.time() - start)
    print(next(loader))

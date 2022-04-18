from collections import defaultdict
import json
import os
import os.path as osp
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from netease_rank.data.encoders import ENCODERS
# from .encoders import ENCODERS

class DataSource:
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = cfg.FEATURE_ENG.PREPROCESS
        self.encoders = cfg.FEATURE_ENG.ENCODE
        self.disc_cols = cfg.FEATURE_ENG.DISCRETE_COLS
        self.cont_cols = cfg.FEATURE_ENG.CONTINUOUS_COLS
        # self.label_df = pd.read_csv(cfg.FILES.LABEL)
        self.user2item2label = self.load_label_json()
        self.test_pairs = pickle.load(open(cfg.FILES.SPLIT, 'rb'))
        for usr, mlog in self.test_pairs:
            del self.user2item2label[usr][mlog]

        self.load_from_json = cfg.FILES.LOAD_FROM_JSON
        self.dump_to_json = cfg.FILES.DUMP_TO_JSON

        self.process_and_build_indices()

    @staticmethod
    def _process(processes, df):
        print("Preprocessing user and mlog dataframes ...")
        for proc in tqdm(processes):
            col, func, discrete = proc[:3]
            assert col in df, f"{col} not found!"
            full_df_input = False if len(proc) < 4 else proc[3]
            new_col_name = col if len(proc) < 5 else proc[4]
            if full_df_input:
                df[new_col_name] = df.apply(func)
            else:
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
        formatted_user2item2label = {}
        for usr, item2label in user2item2label.items():
            formatted_user2item2label[int(usr)] = {}
            for item, label in item2label.items():
                formatted_user2item2label[int(usr)][int(item)] = label['score']
        del user2item2label
        return formatted_user2item2label


    def process_and_build_indices(self):
        file_path = osp.join(os.getcwd(), "json")
        if not self.load_from_json or not osp.exists(self.processes.USER.replace(".csv", "_disc.json")):
            print("Loading user and item dataframe...")
            self.user_df = pd.read_csv(self.cfg.FILES.USER)
            self.item_df = pd.read_csv(self.cfg.FILES.ITEM)
            self.user_df = DataSource._process(self.processes.USER, self.user_df)
            self.item_df = DataSource._process(self.processes.ITEM, self.item_df)
            self.user_df = self._encode(self.encoders.USER, self.user_df, self.user2item2label)
            self.item_df = self._encode(self.encoders.ITEM, self.item_df, self.user2item2label)

            print("Building indices for user dataframe...")
            self.user2disc, self.user2cont = {}, {}
            for uid in tqdm(self.user_df.userIdx):
                self.user2disc[uid] = self.user_df.loc[uid, self.disc_cols.USER]
                self.user2cont[uid] = self.user_df.loc[uid, self.cont_cols.USER]

            print("Building indices for mlog dataframe...")
            self.item2disc, self.item2cont = {}, {}
            for mid in tqdm(self.item_df.mlogindex):
                self.item2disc[mid] = self.item_df.loc[mid, self.disc_cols.ITEM]
                self.item2cont[mid] = self.item_df.loc[mid, self.cont_cols.ITEM]

            self.cardinality = {}
            for col in self.disc_cols.USER:
                assert col in self.user_df, f"{col} not found in user dataframe"
                assert not np.any(self.user_df[col].isna().values)
                self.cardinality[col] = self.user_df[col].nunique()
            for col in self.disc_cols.ITEM:
                assert col in self.item_df, f"{col} not found in item dataframe"
                assert not np.any(self.item_df[col].isna().values)
                self.cardinality[col] = self.item_df[col].nunique()

            if self.dump_to_json:
                if osp.exists(file_path):
                    print(f"Warning: {file_path} exists, overwrite? Y/N")
                    resp = input()
                    if resp.lower() != 'yes' or resp.lower() != 'y':
                        raise Exception("Terminated")
                os.makedirs(file_path)
                json.dump(self.user2disc, open(
                    osp.join(file_path, osp.basename(self.processes.USER.replace(".csv", "_disc.json"))), "wb"))
                json.dump(self.user2cont, open(
                    osp.join(file_path, osp.basename(self.processes.USER.replace(".csv", "_cont.json"))), "wb"))
                json.dump(self.item2disc, open(
                    osp.join(file_path, osp.basename(self.processes.ITEM.replace(".csv", "_disc.json"))), "wb"))
                json.dump(self.item2cont, open(
                    osp.join(file_path, osp.basename(self.processes.ITEM.replace(".csv", "_cont.json"))), "wb"))
                json.dump(self.cardinality, open(osp.join(file_path, "cardinality.json"), "wb"))
        else:
            print("Loading indices for user and mlog dataframe...")
            self.user2disc = json.load(open(
                osp.join(file_path, osp.basename(self.processes.USER.replace(".csv", "_disc.json"))), 'rb'))
            self.user2cont = json.load(
                osp.join(file_path, osp.basename(open(self.processes.USER.replace(".csv", "_cont.json"))), 'rb'))
            self.item2disc = json.load(open(
                osp.join(file_path, osp.basename(self.processes.ITEM.replace(".csv", "_disc.json"))), 'rb'))
            self.item2cont = json.load(open(
                osp.join(file_path, osp.basename(self.processes.ITEM.replace(".csv", "_cont.json"))), 'rb'))
            self.cardinality = json.load(open(osp.join(file_path, "cardinality.json"), 'rb'))
            for col in self.disc_cols.USER:
                assert col in self.user_df, f"{col} not found in user dataframe"
                assert col in self.cardinality, f"{col} not found in cardinality"
            for col in self.disc_cols.ITEM:
                assert col in self.item_df, f"{col} not found in item dataframe"
                assert col in self.cardinality, f"{col} not found in cardinality"


if __name__ == "__main__":
    from netease_rank.config import BaseConfig
    cfg = BaseConfig()
    ds = DataSource(cfg)








from collections import defaultdict
import json
import os.path as osp
import pandas as pd
import pickle
from tqdm import tqdm


class DataSource:
    def __init__(self, cfg):
        self.processes = cfg.FEATURE_ENG.PREPROCESS
        self.disc_cols = cfg.FEATURE_ENG.DISCRETE_COLS
        self.cont_cols = cfg.FEATURE_ENG.CONTINUOUS_COLS
        self.user_df = pd.read_csv(cfg.FILES.USER)
        self.item_df = pd.read_csv(cfg.FILES.ITEM)
        self.label_df = pd.read_csv(cfg.FILES.LABEL)
        self.test_pairs = pickle.load(cfg.FILES.SPLIT)

        self.load_from_json = cfg.FILES.LOAD_FROM_JSON
        self.dump_to_json = cfg.FILES.DUMP_TO_JSON

        self.process_and_build_indices()

    @staticmethod
    def _process(processes, df):
        print("Preprocessing user and mlog dataframes ...")
        for proc in tqdm(processes):
            col, func, discrete = proc[:3]
            full_df_input = False if len(proc) < 4 else proc[3]
            inplace = True if len(proc) < 5 else proc[4]
            new_col_name = col if inplace else col + "_proc"
            if full_df_input:
                df[new_col_name] = df.apply(func)
            else:
                df[new_col_name] = df[col].apply(func)
            if discrete and not new_col_name.startswith("d_"):
                df.rename(columns={new_col_name: "d_" + new_col_name}, inplace=True)
        return df

    def process_and_build_indices(self):
        if not self.load_from_json or not osp.exists(self.processes.USER.replace(".csv", "_disc.json")):
            self.user_df = DataSource._process(self.processes.USER, self.user_df)
            self.item_df = DataSource._process(self.processes.ITEM, self.item_df)

            print("Building indices for user dataframe...")
            self.user2disc, self.user2cont = {}, {}
            for uid in tqdm(self.user_df.userIdx):
                self.user2disc[uid] = self.user_df[uid, self.disc_cols.USER]
                self.user2cont[uid] = self.user_df[uid, self.cont_cols.USER]

            print("Building indices for mlog dataframe...")
            self.item2disc, self.item2cont = {}, {}
            for mid in tqdm(self.item_df.mlogindex):
                self.item2disc[mid] = self.item_df[mid, self.disc_cols.ITEM]
                self.item2cont[mid] = self.item_df[mid, self.cont_cols.ITEM]

            if self.dump_to_json:
                json.dump(self.user2disc, open(self.processes.USER.replace(".csv", "_disc.json"), "wb"))
                json.dump(self.user2cont, open(self.processes.USER.replace(".csv", "_cont.json"), "wb"))
                json.dump(self.item2disc, open(self.processes.ITEM.replace(".csv", "_disc.json"), "wb"))
                json.dump(self.item2cont, open(self.processes.ITEM.replace(".csv", "_cont.json"), "wb"))
        else:
            print("Loading indices for user and mlog dataframe...")
            self.user2disc = json.load(open(self.processes.USER.replace(".csv", "_disc.json"), 'rb'))
            self.user2cont = json.load(open(self.processes.USER.replace(".csv", "_cont.json"), 'rb'))
            self.item2disc = json.load(open(self.processes.ITEM.replace(".csv", "_disc.json"), 'rb'))
            self.item2cont = json.load(open(self.processes.ITEM.replace(".csv", "_cont.json"), 'rb'))

        if not osp.exists("")
        print("Building indices for label dataframe...")
        self.user2item2label = defaultdict(defaultdict) # uid -> mlog -> (score, isTrain)
        for uid, group in self.label_df.groupby("userIdx"):
            for row in group.to_dict("index").values():
                self.user2item2label[uid][row["mlogindex"]] = (
                    row["score"], (uid, row["mlogindex"]) in self.test_pairs
                )






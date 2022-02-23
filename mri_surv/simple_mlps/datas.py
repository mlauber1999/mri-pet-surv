from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import random
import glob
import pandas as pd
import numpy as np
import csv
import re
import torch
import nibabel as nib
import json
import logging
import datetime
from icecream import ic

fname = 'logs/datas.log'
with open(fname, 'w') as fi:
    fi.write(str(datetime.datetime.today()))

logging.basicConfig(filename=fname, level=logging.DEBUG)

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def _read_csv_cox(filename, skip_ids: list=None):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        fileIDs, time_obs, hit, age, mmse = [], [], [], [], []
        for r in reader:
            if skip_ids is not None:
                if r['RID'] in skip_ids:
                    continue
            fileIDs += [str(r['RID'])]
            time_obs += [float(r['TIMES'])]  # changed to TIMES_ROUNDED. consider switching so observations for progressors are all < 1 year
            hit += [int(float(r['PROGRESSES']))]
            age += [float(r['AGE'])]
            if 'MMSCORE_mmse' in r.keys():
                mmse += [float(r['MMSCORE_mmse'])]
            else:
                mmse += [np.nan if r['MMSE'] == '' else float(r['MMSE'])]
    return fileIDs, np.asarray(time_obs), np.asarray(hit), age, mmse

def _read_csv_csf(filename):
    parcellation_tbl = pd.read_csv(filename)
    valid_columns = ["abeta", "tau","ptau"]
    parcellation_tbl = parcellation_tbl[valid_columns + ['RID']].copy()
    return parcellation_tbl

def _retrieve_kfold_partition(idxs, stage, folds=5, exp_idx=1, shuffle=True,
                              random_state=120):
    idxs = np.asarray(idxs).copy()
    if shuffle:
        np.random.seed(random_state)
        idxs = np.random.permutation(idxs)
    if 'all' in stage:
        return idxs
    if len(idxs.shape) > 1: raise ValueError
    fold_len = len(idxs) // folds
    folds_stitched = []
    for f in range(folds):
        folds_stitched.append(idxs[f*fold_len:(f+1)*fold_len])
    test_idx = exp_idx
    valid_idx = (exp_idx+1) % folds
    train_idx = np.setdiff1d(np.arange(0,folds,1),[test_idx, valid_idx])
    if 'test' in stage:
        return folds_stitched[test_idx]
    elif 'valid' in stage:
        return folds_stitched[valid_idx]
    elif 'train' in stage:
        return np.concatenate([folds_stitched[x] for x in train_idx], axis=0)
    else:
        raise ValueError

def deabbreviate_parcellation_columns(df):
    df_dict = pd.read_csv(
            './metadata/data_raw/neuromorphometrics/neuromorphometrics.csv',
                          usecols=['ROIabbr','ROIname'],sep=';')
    df_dict = df_dict.loc[[x[0] == 'l' for x in df_dict['ROIabbr']],:]
    df_dict['ROIabbr'] = df_dict['ROIabbr'].apply(
            lambda x: x[1:]
    )
    df_dict['ROIname'] = df_dict['ROIname'].apply(
            lambda x: x.replace('Left ', '')
    )
    df_dict = df_dict.set_index('ROIabbr').to_dict()['ROIname']
    df.rename(columns=df_dict, inplace=True)

def drop_ventricles(df, ventricle_list):
    df.drop(columns=ventricle_list, inplace=True)

def add_ventricle_info(parcellation_df, ventricle_df, ventricles):
    return parcellation_df.merge(ventricle_df[['RID'] + ventricles], on='RID',
                          validate="one_to_one")

class ParcellationDataMeta(Dataset):
    def __init__(self, seed, **kwargs):
        random.seed(1000)
        self.exp_idx = seed
        json_props = read_json('./simple_mlps/mlp_config'
                             '.json')
        self._json_props = json_props
        self.csv_directory = json_props['datadir']
        self.ventricles = json_props['ventricles']
        self.csvname = self.csv_directory + json_props['metadata_fi']
        self.parcellation_file = self.csv_directory + json_props['parcellation_fi']
        self.parcellation_file_csf = self.csv_directory + json_props[
            'parcellation_csf_fi']
        self.rids, self.time_obs, self.hit, self.age, self.mmse = \
            _read_csv_cox(self.csvname)

    def _prep_data(self, feature_df, stage):
        idxs = list(range(len(self.rids)))
        self.index_list = _retrieve_kfold_partition(idxs, stage, 5, self.exp_idx)
        self.rid = np.array(self.rids)
        logging.warning(f'selecting indices\n{self.rid[self.index_list]}\n\t '
                        f'for stage'
                        f'{stage} and random seed 1000')
        self.labels = feature_df.columns
        self.data_l = feature_df.to_numpy()
        self.data_l = torch.FloatTensor(self.data_l)
        self.data = self.data_l

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx_transformed = self.index_list[idx]
        x = self.data[idx_transformed]
        obs = self.time_obs[idx_transformed]
        hit = self.hit[idx_transformed]
        rid = self.rid[idx_transformed]
        return x, obs, hit, rid

    def get_features(self):
        return self.labels

    def get_data(self):
        return self.data

class ParcellationDataCSF(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2), add_age=False,
                 add_mmse=False):
        super().__init__(seed, stage=stage, ratio=ratio)
        parcellation_df = _read_csv_csf(self.csvname)
        parcellation_df['RID'] = parcellation_df['RID'].apply(
                lambda x: str(int(x)).zfill(4)
        )
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        self._prep_data(parcellation_df, stage)

class ParcellationData(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2), add_age=False,
                 add_mmse=False):
        super().__init__(seed, stage=stage, ratio=ratio)
        parcellation_df = pd.read_csv(self.parcellation_file)
        parcellation_df['RID'] = parcellation_df['RID'].apply(
                lambda x: str(int(x)).zfill(4)
        )
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        deabbreviate_parcellation_columns(parcellation_df)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        self._prep_data(parcellation_df, stage)

class ParcellationDataVentricles(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2),
                 ventricle_info=False, add_age=False,
                 add_mmse=False):
        super().__init__(seed, stage=stage)
        parcellation_df = pd.read_csv(self.parcellation_file, dtype={'RID':
                                                                         str})
        ventricle_df = pd.read_csv(self.parcellation_file_csf, dtype={'RID':
                                                                          str})
        drop_ventricles(parcellation_df, self.ventricles)
        if ventricle_info:
           parcellation_df = add_ventricle_info(parcellation_df, ventricle_df,
                                  self.ventricles)
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        deabbreviate_parcellation_columns(parcellation_df)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        self._prep_data(parcellation_df, stage)

class ParcellationDataNacc(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2), add_age=False,
                 add_mmse=False):
        self.seed = seed
        random.seed(1000)
        json_props = read_json('./simple_mlps/mlp_config'
                             '.json')
        self._json_props = json_props
        self.csv_directory = json_props['datadir']
        self.csvname = self.csv_directory + json_props['metadata_fi_nacc']
        self.rids, self.time_obs, self.hit, self.age, self.mmse = \
            _read_csv_cox(
                self.csvname)
        csvname2 = self.csv_directory + json_props['parcellation_fi_nacc']
        parcellation_df = pd.read_csv(csvname2)
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        self.exp_idx = 1
        deabbreviate_parcellation_columns(parcellation_df)
        self._prep_data(parcellation_df, stage)

class ParcellationDataVentriclesNacc(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2),
                 ventricle_info=False, add_age=False,
                 add_mmse=False):
        self.seed = seed
        random.seed(1000)
        json_props = read_json('./simple_mlps/mlp_config'
                             '.json')
        self._json_props = json_props
        self.ventricles = json_props['ventricles']
        self.csv_directory = json_props['datadir']
        self.csvname = self.csv_directory + json_props['metadata_fi_nacc']
        self.rids, self.time_obs, self.hit, self.age, self.mmse = \
            _read_csv_cox(self.csvname)
        csvname2 = self.csv_directory + json_props['parcellation_fi_nacc']
        csvname3 = self.csv_directory + json_props['parcellation_csf_fi_nacc']
        parcellation_df = pd.read_csv(csvname2, dtype={'RID': str})
        ventricle_df = pd.read_csv(csvname3, dtype={'RID': str})
        drop_ventricles(parcellation_df, self.ventricles)
        if ventricle_info:
            parcellation_df = add_ventricle_info(parcellation_df, ventricle_df,
                                  self.ventricles)
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        deabbreviate_parcellation_columns(parcellation_df)
        self.exp_idx = 1
        self._prep_data(parcellation_df, stage)

class ParcellationData(Dataset):
    def __init__(self, exp_idx, seed=1000, stage='train', dataset='ADNI', ratio=(0.6, 0.2, 0.2), add_age=False,
                 add_mmse=False, partitioner=_retrieve_kfold_partition):
        random.seed(seed)
        self.exp_idx = exp_idx
        self.ratio = ratio
        self.stage = stage
        self.partitioner = partitioner
        json_props = read_json('./simple_mlps/config.json')
        self.csv_directory = json_props['datadir']
        self.csvname = self.csv_directory + json_props['metadata_fi'][dataset]
        self.parcellation_file = pd.read_csv(
            self.csv_directory + json_props['parcellation_fi'], dtype={'RID': str})
        self.parcellation_file = self.parcellation_file.query(
            'Dataset == @dataset').drop(columns=['Dataset', 'PROGRESSION_CATEGORY']).copy()
        self.rids, self.time_obs, self.hit, self.age, self.mmse = \
            _read_csv_cox(self.csvname)
        self.parcellation_file['RID'] = self.parcellation_file['RID'].apply(
                lambda x: x.zfill(4)
        )
        self.parcellation_file.set_index('RID', inplace=True)
        self.parcellation_file = self.parcellation_file.loc[self.rids,:].reset_index(
                drop=True)
        if add_age:
            self.parcellation_file['age'] = self.age
        if add_mmse:
            self.parcellation_file['mmse'] = self.mmse
        self._prep_data(self.parcellation_file)

    def _prep_data(self, feature_df):
        idxs = list(range(len(self.rids)))
        self.index_list = self.partitioner(idxs, stage=self.stage, exp_idx=self.exp_idx)
        self.rid = np.array(self.rids)
        feature_df.drop(columns=["CSF",
                        "3thVen",
                        "4thVen",
                        "InfLatVen",
                        "LatVen"], inplace=True)
        self.labels = feature_df.columns
        self.data_l = feature_df.to_numpy()
        self.data_l = torch.FloatTensor(self.data_l)
        self.data = self.data_l

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx_transformed = self.index_list[idx]
        x = self.data[idx_transformed]
        obs = self.time_obs[idx_transformed]
        hit = self.hit[idx_transformed]
        rid = self.rid[idx_transformed]
        return x, obs, hit, rid

    def get_features(self):
        return self.labels

    def get_data(self):
        return self.data

def test():
    for _exp in range(5):
        for stage in ('train', 'valid', 'test','all'):
            for kwargs in (
                {'add_age': True, 'add_mmse': True}, 
                {'add_age': True, 'add_mmse': False},
                {'add_age': False, 'add_mmse': True},
                {'add_age': False, 'add_mmse': False}
                ):
                pd = ParcellationData(_exp, stage=stage, **kwargs) 
                pd2 = ParcellationDataVentricles(_exp, stage=stage, **kwargs)
                for i in range(len(pd)):
                    for j in range(len(pd[i])):
                        if j == 0:
                            assert(all(pd[i][j].numpy()==pd2[i][j].numpy()))
                        else:
                            assert(pd[i][j]==pd2[i][j])
        pd = ParcellationData(_exp, stage='all', dataset='NACC')
        pd2 = ParcellationDataVentriclesNacc(_exp, stage='all')
        for i in range(len(pd)):
            for j in range(len(pd[i])):
                if j == 0:
                    assert(all(pd[i][j].numpy()==pd2[i][j].numpy()))
                else:
                    assert(pd[i][j]==pd2[i][j])

    print("pass") 


if __name__ == "__main__":
    test()
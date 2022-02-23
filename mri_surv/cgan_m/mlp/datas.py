from torch.utils.data import Dataset
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

logging.basicConfig(filename='datas.log', level=logging.DEBUG)

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def _read_csv_cox(filename, return_age=False):
    with open(filename, 'r') as f:
        # reader = csv.reader(f)
        # your_list = list(reader)
        reader = csv.DictReader(f)
        fileIDs, time_obs, time_hit, age = [], [], [], []
        for r in reader:
            fileIDs += [r['RID']]
            time_obs += [int(float(r['TIME_TO_FINAL_DX']))]
            temp = r['TIME_TO_PROGRESSION']
            age += [float(r['AGE'])]
            if len(temp) == 0:
                time_hit += [0]
            else:
                time_hit += [int(float(r['TIME_TO_PROGRESSION']))]
    fileIDs = ['0'*(4-len(f))+f for f in fileIDs]
    if return_age:
        return fileIDs, time_obs, time_hit, age
    else:
        return fileIDs, time_obs, time_hit

def _read_csv_parcellations(filename):
    parcellation_tbl = pd.read_csv(filename)
    col_regex = re.compile(r'corr_vol_.*')
    valid_columns = [col for col in parcellation_tbl.columns if
                     col_regex.match(col)]
    replacement_dict = {x: x.replace('corr_vol_','') for x in valid_columns}
    parcellation_tbl = parcellation_tbl[valid_columns + ['RID']].copy()
    parcellation_tbl = parcellation_tbl.rename(columns=replacement_dict)
    return parcellation_tbl

def _read_csv_surfaces(filename):
    parcellation_tbl = pd.read_csv(filename)
    col_regex = re.compile(r'corr_surf_.*')
    valid_columns = [col for col in parcellation_tbl.columns if
                     col_regex.match(col)]
    replacement_dict = {x: x.replace('corr_surf_','') for x in valid_columns}
    parcellation_tbl = parcellation_tbl[valid_columns + ['RID']].copy()
    parcellation_tbl = parcellation_tbl.rename(columns=replacement_dict)
    return parcellation_tbl

def _read_csv_csf(filename):
    parcellation_tbl = pd.read_csv(filename)
    valid_columns = ["abeta", "tau","ptau"]
    parcellation_tbl = parcellation_tbl[valid_columns + ['RID']].copy()
    return parcellation_tbl

def _retrieve_index_partition(idxs, stage, _l, ratio):
    split1 = int(_l * ratio[0])
    split2 = int(_l * (ratio[0] + ratio[1]))
    if 'train' in stage:
        return idxs[:split1]
    elif 'valid' in stage:
        return idxs[split1:split2]
    elif 'test' in stage:
        return idxs[split2:]
    elif 'all' in stage:
        return idxs
    else:
        raise Exception('Unexpected Stage for FCN_Cox_Data!')

class ParcellationDataMeta(Dataset):
    def __init__(self, seed, *args, **kwargs):
        random.seed(seed)
        self.seed = seed
        json_props = read_json('mlp_config'
                             '.json')
        self._json_props = json_props
        self.csv_directory = json_props['datadir']
        self.csvname = self.csv_directory + json_props['metadata_fi']
        self.parcellation_file = self.csv_directory + json_props['parcellation_fi']
        self.rids, self.time_obs, self.time_hit, self.age = _read_csv_cox(
                self.csvname, return_age=True)

    def _prep_data(self, parcellation_df, stage, ratio):
        time_obs, time_hit = self.time_obs, self.time_hit
        self.time_obs = []
        self.time_hit = []
        for obs, hit in zip(time_obs, time_hit):
            self.time_obs += [hit if hit else obs]
            self.time_hit += [1 if hit else 0]
        l = len(self.rids)
        idxs = list(range(len(self.rids)))
        random.shuffle(idxs)
        self.index_list = _retrieve_index_partition(idxs, stage, l, ratio)
        logging.warning(f'selecting indices\n{self.index_list}\n\t for stage'
                        f'{stage} and random seed {self.seed}')
        self.fileIDs = np.array(self.rids)[self.index_list]
        self.data_l = parcellation_df
        self.data_l = torch.FloatTensor(self.data_l)
        self.data = self.data_l

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        x = self.data[idx]
        obs = self.time_obs[idx]
        hit = self.time_hit[idx]
        return x, obs, hit

class ParcellationDataCSF(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2)):
        super().__init__(seed, stage=stage, ratio=ratio)
        parcellation_df = _read_csv_csf(self.csvname)
        parcellation_df['RID'] = parcellation_df['RID'].apply(
                lambda x: str(int(x)).zfill(4)
        )
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        parcellation_df = parcellation_df.to_numpy()
        self._prep_data(parcellation_df, stage, ratio)

class ParcellationData(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2)):
        super().__init__(seed, stage=stage, ratio=ratio)
        parcellation_df = pd.read_csv(self.parcellation_file)
        parcellation_df['RID'] = parcellation_df['RID'].apply(
                lambda x: str(int(x)).zfill(4)
        )
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        parcellation_df = parcellation_df.to_numpy()
        self._prep_data(parcellation_df, stage, ratio)

class ParcellationDataNacc(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2)):
        random.seed(seed)
        self.seed = seed
        json_props = read_json('mlp_config'
                             '.json')
        self._json_props = json_props
        self.csv_directory = json_props['datadir']
        self.csvname = self.csv_directory + json_props['metadata_fi_nacc']
        self.rids, self.time_obs, self.time_hit = _read_csv_cox(
                self.csvname, return_age=False)
        csvname2 = self.csv_directory + json_props['parcellation_fi_nacc']
        parcellation_df = pd.read_csv(csvname2)
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        parcellation_df = parcellation_df.to_numpy()
        self._prep_data(parcellation_df, stage, ratio)

class ParcellationDataSurf(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2)):
        super().__init__(seed, stage=stage, ratio=ratio)
        csvname2 = self.csv_directory + 'mri3_cat12_cox.csv'
        parcellation_df = _read_csv_surfaces(csvname2)
        parcellation_df['RID'] = parcellation_df['RID'].apply(
                lambda x: str(int(x)).zfill(4)
        )
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        parcellation_df.drop(columns=['corpuscallosum', 'unknown'],
                             inplace=True)
        parcellation_df = parcellation_df.to_numpy()
        self._prep_data(parcellation_df, stage, ratio)


class CoxData(ParcellationDataMeta):
    def __init__(self, Data_dir, stage, ratio=(0.6, 0.2, 0.2), seed=1000):
        super().__init__(seed, stage=stage, ratio=ratio)
        self.Data_list = glob.glob(Data_dir + '*nii')
        rids_data_list = [x.split('_')[-1][:-4] for x in
                                   self.Data_list]
        df = pd.DataFrame(data={'FileName': self.Data_list},
                          index=rids_data_list,
                          )
        df = df.loc[self.rids,:]
        self.data_l = np.expand_dims(np.asarray([nib.load(x).get_fdata().squeeze()
                                                 for x in df['FileName']]),
                                                axis=1)
        self._prep_data(self.data_l, stage, ratio)
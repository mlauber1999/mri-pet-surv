# dataloader for vision transformer
# Created: 6/16/2021
# Status: in progress

import random
import glob
import os, sys

import numpy as np
import nibabel as nib

from torch.utils.data import Dataset
from utils import read_csv_cox, rescale, read_csv_cox_ext

SCALE = 1 #rescale to 0~2.5


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

class ViT_Data(Dataset):
    def __init__(self, data_dir, exp_idx, stage, ratio=(0.6, 0.2, 0.2), seed=1000, name='', fold=[], external=False):
        random.seed(seed)

        self.stage = stage
        self.exp_idx = exp_idx
        self.data_dir = data_dir

        # self.data_list = glob.glob(data_dir + 'coregistered*nii*')
        self.data_list = glob.glob(data_dir + '*nii*')

        # csvname = '~/mri-pet/metadata/data_processed/merged_dataframe_cox_pruned_final.csv'
        csvname = '~/mri-pet/mri_surv/metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv'
        if external:
            csvname = '~/mri-pet/mri_surv/metadata/data_processed/merged_dataframe_cox_test_pruned_final.csv'
        csvname = os.path.expanduser(csvname)
        if external:
            fileIDs, time_obs, time_hit = read_csv_cox_ext(csvname) #training file
        else:
            fileIDs, time_obs, time_hit = read_csv_cox(csvname) #training file

        tmp_f = []
        tmp_o = []
        tmp_h = []
        tmp_d = []
        for d in self.data_list:
            for f, o, h in zip(fileIDs, time_obs, time_hit):
                fname = os.path.basename(d)
                if f in fname:
                    tmp_f.append(f)
                    tmp_o.append(o)
                    tmp_h.append(h)
                    tmp_d.append(d)
                    break
        self.data_list = tmp_d
        self.time_obs  = tmp_o
        self.time_hit  = tmp_h
        self.fileIDs = np.array(tmp_f) #Note: this only for csv generation not used for data retrival

        FOLDS = True
        if FOLDS:
            # 1 fold for testing, 1 fold for validation, the rest for training
            # Using custom for unity experiments (Vit, Mlp & CNN)
            # print('using k-fold')
            # sys.exit()
            idxs = list(range(len(fileIDs)))
            self.index_list = _retrieve_kfold_partition(idxs, self.stage, 5, self.exp_idx)
        else:
            # print(len(tmp_f))
            l = len(self.data_list)
            split1 = int(l*ratio[0])
            split2 = int(l*(ratio[0]+ratio[1]))
            idxs = list(range(len(fileIDs)))
            random.shuffle(idxs)
            if 'train' in stage:
                self.index_list = idxs[:split1]
            elif 'valid' in stage:
                self.index_list = idxs[split1:split2]
            elif 'test' in stage:
                self.index_list = idxs[split2:]
            elif 'all' in stage:
                self.index_list = idxs
            else:
                raise Exception('Unexpected Stage for Vit_Data!')
        # print(len(self.index_list))
        # print((self.fileIDs[:10]))
        # for i, idx in enumerate(self.index_list):
            # if tmp_f[idx] == '0173':
                # print(tmp_f[idx])
                # print(tmp_o[idx])
                # print(tmp_h[idx])
                # sys.exit()
        self.fileIDs = np.array(tmp_f)[self.index_list]
        # sys.exit()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        obs = self.time_obs[idx]
        hit = self.time_hit[idx]

        data = nib.load(self.data_list[idx]).get_fdata().astype(np.float32)
        data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
            if 0:
                data = rescale(data, (0, 99))
                data = data.astype(np.int)
        data = np.expand_dims(data, axis=0)
        return data, obs, hit

    def get_sample_weights(self):
        num_classes = len(set(self.time_hit))
        counts = [self.time_hit.count(i) for i in range(num_classes)]
        count = len(self.time_hit)
        weights = [count / counts[i] for i in self.time_hit]
        class_weights = [count/c for c in counts]
        return weights, class_weights

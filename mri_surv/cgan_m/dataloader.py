# dataloader.py: Prepare the dataloader needed for the neural networks

import numpy as np
from sklearn.model_selection import StratifiedKFold
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from utils import read_csv, read_csv2, read_csv_cox, read_csv_cox2, read_csv_surv, read_csv_pre, padding, get_AD_risk, read_txt, rescale, chunks
import random
import sys
import os
import nibabel as nib

#SKULL: True indicate SKULL removed, False indicate SKULL remained
SKULL = True
#True indicate scaling the data in dataloader
SCALE = False
# false: 477, 491, 598
# true: better on average
#True to get cross set where all models correctly identified
CROSS_VALID=False

def filter(fileIDs, Data_list):
    skip = fileIDs.copy()
    for d in Data_list:
        v = os.path.basename(d)[:4]
        if v in skip:
            skip.remove(v)
    return skip

class CNN_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, ratio=(0.6, 0.2, 0.2), seed=1000, loc=False, name='', fold=None, yr=2):
        random.seed(seed)
        self.loc = loc
        self.Data_list = glob.glob(Data_dir + '*.*')

        csvname = './csv/{}.csv'.format('fdg_mri_amy_mci_csf_cumulative_pruned_final')
        fileIDs, labels = read_csv2(csvname) #training file

        skip = filter(fileIDs, self.Data_list)
        if 'train' in stage and 'ADNIP' not in Data_dir and skip != []:
            print ('loading: ', csvname)
            print(skip, 'not found, skipped')

        for s in skip:
            idx = fileIDs.index(s)
            fileIDs.pop(idx)
            labels.pop(idx)
        #self.Data_list.sort()

        tmp = []
        if 'mri' in name:
            name = 'mri'
        elif 'amyloid' in name:
            name = 'amyloid'
        else:
            name = 'fdg'
        self.name = name
        for d in self.Data_list:
            for f in fileIDs:
                fname = os.path.basename(d)
                if f in fname[:4]:
                    if SKULL:
                        if ('brain' in fname) and (name in fname):
                            tmp.append(d)
                            break
                    else:
                        if ('brain' not in fname) and (name in fname):
                            tmp.append(d)
                            break
        self.Data_list = tmp

        # 1 fold for testing, 1 fold for validation, the rest for training
        num_fold = fold
        idxs = list(range(len(fileIDs)))

        skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=120)

        i = 0
        for train_index, test_index in skf.split(idxs, labels):
            if i == exp_idx:
                if 'train' in stage:
                    self.index_list = train_index[:-len(test_index)]
                elif 'valid' in stage:
                    self.index_list = train_index[-len(test_index):]
                elif 'test' in stage:
                    self.index_list = test_index
                elif 'all' in stage:
                    self.index_list = idxs
                else:
                    self.index_list = []
                break
            i += 1
        self.fileIDs = np.array(fileIDs)[self.index_list]

        labels = np.array(labels)
        labels[labels==-1] = 0
        self.Label_list = labels.tolist()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        label = self.Label_list[idx]
        # ma, mb, mc = 160, 240, 256
        try:
            data = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
        except:
            data = np.load(self.Data_list[idx]).astype(np.float32)
        # if len(data.shape) == 4:
        #     data = data[0:160,0:240,0:256,0]
        data = np.expand_dims(data, axis=0)
        data = np.array(data)
        data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
        if self.loc:
            return data, label, self.Data_list[idx]
        return data, label

    def get_sample_weights(self):
        num_classes = len(set(self.Label_list))
        counts = [self.Label_list.count(i) for i in range(num_classes)]
        count = len(self.Label_list)
        weights = [count / counts[i] for i in self.Label_list]
        class_weights = [count/c for c in counts]
        return weights, class_weights

class FCN_Data(CNN_Data):
    def __init__(self, Data_dir, exp_idx, stage, transform=None, whole_volume=False, ratio=(0.6, 0.2, 0.2), seed=1000, patch_size=47, dim=1, name='', fold=[], yr=2):
        if type(Data_dir) == type([]):
            self.Data_dir = Data_dir
            super().__init__(Data_dir[0], exp_idx, stage, ratio, seed, name=name, fold=fold, yr=yr)
        else:
            self.Data_dir = None
            super().__init__(Data_dir, exp_idx, stage, ratio, seed, name=name, fold=fold, yr=yr)
        self.stage = stage
        self.transform = transform
        self.whole = whole_volume
        self.patch_size = patch_size
        self.patch_sampler = PatchGenerator(patch_size=self.patch_size)
        self.cache = []
        self.dim = dim
        # self.name = name # already defined in CNN_Data

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        label = self.Label_list[idx]
        # print(idx, self.Data_list[idx])
        if type(self.Data_dir) == type([]):
            self.Data_list2 = [dl.replace(self.Data_dir[0], self.Data_dir[1]) for dl in self.Data_list]
            if ('mri' in self.name) and ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('mri' in self.name) and ('amyloid' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
            elif ('mri' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('amyloid', 'fdg') for d in self.Data_list2]
            else:
                print('error: case not supported. make sure the model name contains input scan types (mri, amyloid, or fdg)')
                print(self.name)
                sys.exit()
            if len(self.Data_dir) == 2:
                try:
                    data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                    data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                except:
                    data1 = np.load(self.Data_list[idx]).astype(np.float32)
                    data2 = np.load(self.Data_list2[idx]).astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data = np.array([data1, data2])
            else:
                data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                data3 = nib.load(self.Data_list3[idx]).get_fdata().astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data3[data3 != data3] = 0
                data = np.array([data1, data2, data3])
        else:
            try:
                data = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
            except:
                data = np.load(self.Data_list[idx]).astype(np.float32)
            data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
        if self.whole:
            # if len(data.shape) == 4:
            #     data = data[0:160,0:240,0:256,0]
            if type(self.Data_dir) == type([]):
                if len(self.Data_dir) == 2:
                    data1 = padding(data[0], win_size=self.patch_size // 2)
                    data2 = padding(data[1], win_size=self.patch_size // 2)
                    data = np.array((data1, data2))
                else:
                    data1 = padding(data[0], win_size=self.patch_size // 2)
                    data2 = padding(data[1], win_size=self.patch_size // 2)
                    data3 = padding(data[2], win_size=self.patch_size // 2)
                    data = np.array((data1, data2, data3))
            else:
                data = np.expand_dims(padding(data, win_size=self.patch_size // 2), axis=0)
            return data, label
        elif self.stage == 'valid_patch' and len(self.cache) == len(self.Label_list):
            return self.cache[idx]
        elif self.stage == 'valid_patch':
            # data = np.load(self.Data_list[idx], mmap_mode='r').astype(np.float32)
            # if len(data.shape) == 4:
            #     data = data[0:160,0:240,0:256,0]
            array_list = []
            if type(self.Data_dir) == type([]):
                loc = (np.array(data[0].shape)-47)//5
            else:
                loc = (np.array(data.shape)-47)//5
            patch_locs = [loc*i for i in range(1, 6)]
            for i, loc in enumerate(patch_locs):
                x, y, z = loc
                if type(self.Data_dir) == type([]):
                    patch = data[:, x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
                    array_list.append(patch)
                else:
                    patch = data[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
                    array_list.append(np.expand_dims(patch, axis = 0))
            data = Variable(torch.FloatTensor(np.stack(array_list, axis = 0)))
            label = Variable(torch.LongTensor([label]*5))
            self.cache.append((data, label))
            return data, label
        elif self.stage == 'train':
            # data = np.load(self.Data_list[idx], mmap_mode='r').astype(np.float32)
            # if len(data.shape) == 4:
            #     data = data[0:160,0:240,0:256,0]
            if type(self.Data_dir) == type([]):
                if len(self.Data_dir) == 2:
                    patches = self.patch_sampler.random_sample(data[0], data[1])
                else:
                    patches = self.patch_sampler.random_sample(data[0], data[1], data[2])
                patch = np.array(patches)
            else:
                patch = self.patch_sampler.random_sample(data)
                patch = np.expand_dims(patch, axis=0)
            # if self.transform:
            #     patch = self.transform.apply(patch).astype(np.float32)
            return patch, label

class PatchGenerator:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def randInt(self, ran):
        return [random.randint(0, v-self.patch_size) for v in ran]

    def prob(self, arr):
        return 1 - np.count_nonzero(arr)/arr.size

    def random_sample(self, data1, data2=None, data3=None):
        """sample random patch from numpy array data"""
        x, y, z = self.randInt(data1.shape)
        if data2 is None:
            p = self.prob(data1[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size])
            while random.random() < p:
                x, y, z = self.randInt(data1.shape)
                p = self.prob(data1[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size])
            return data1[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
        else:
            if data3 is None:
                p = self.prob(data1[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size])
                while random.random() < p:
                    x, y, z = self.randInt(data1.shape)
                    p = self.prob(data1[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size])
                return data1[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size], data2[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
            else:
                p = self.prob(data1[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size])
                while random.random() < p:
                    x, y, z = self.randInt(data1.shape)
                    p = self.prob(data1[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size])
                return data1[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size], data2[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size], data3[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]


    def fixed_sample(self, data):
        """sample patch from fixed locations"""
        patches = []
        patch_locs = [[25, 90, 30], [115, 90, 30], [67, 90, 90], [67, 45, 60], [67, 135, 60]]
        for i, loc in enumerate(patch_locs):
            x, y, z = loc
            patch = data[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
            patches.append(np.expand_dims(patch, axis = 0))
        return patches

class MLP_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, roi_threshold, roi_count, choice, ratio=(0.6, 0.2, 0.2), seed=1000, yr=2):
        random.seed(seed)
        self.Data_list = glob.glob(Data_dir + '*nii*')

        csvname = './csv/{}.csv'.format('fdg_mri_amy_mci_csf_cumulative_pruned_final')
        fileIDs, labels = read_csv2(csvname) #training file

        tmp = []
        for f in fileIDs:
            for d in self.Data_list:
                if f in d:
                    tmp += [f]
        skip = list(set(tmp) ^ set(fileIDs))
        # print(skip, 'not found, skipped')
        for f in skip:
            idx = fileIDs.index(f)
            fileIDs.pop(idx)
            labels.pop(idx)
        #self.Data_list.sort()

        labels = np.array(labels)
        labels[labels==-1] = 0
        self.Label_list = labels.tolist()

        FIXED = False
        if FIXED:
            # 1 fold for testing, 1 fold for validation, the rest for training
            # Note: in this case, MLP supports only 5 fold cross validation!
            # Not in use, need to test before use!
            num_fold = 5
            idxs = list(range(len(fileIDs)))

            skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=120)

            i = 0
            for train_index, test_index in skf.split(idxs, labels):
                if i == exp_idx:
                    if 'train' in stage:
                        self.index_list = train_index[:-len(test_index)]
                    elif 'valid' in stage:
                        self.index_list = train_index[-len(test_index):]
                    elif 'test' in stage:
                        self.index_list = test_index
                    elif 'all' in stage:
                        self.index_list = idxs
                    else:
                        self.index_list = []
                    break
                i += 1

        else:
            l = len(self.Data_list)
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
        self.fileIDs = np.array(fileIDs)[self.index_list]

        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.roi_threshold = roi_threshold
        self.roi_count = roi_count
        if choice == 'count':
            self.select_roi_count()
        else:
            self.select_roi_thres()
        self.risk_list = [get_AD_risk(np.load(file))[self.roi] for file in self.Data_list]
        self.in_size = self.risk_list[0].shape[0]
        if CROSS_VALID:
            with open('cross/rid_order_index.txt', 'w') as f:
                f.write(self.index_list)
            with open('cross/rid_order_data.txt', 'w') as f:
                f.write(self.Data_list)

    def select_roi_thres(self):
        self.roi = np.load(self.Data_dir + 'train_MCC.npy')
        self.roi = self.roi > self.roi_threshold
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0:
                        self.roi[i,j,k] = False

    def select_roi_count(self):
        self.roi = np.load(self.Data_dir + 'train_MCC.npy')
        tmp = []
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0: continue
                    tmp.append((self.roi[i,j,k], i, j, k))
        tmp.sort()
        tmp = tmp[-self.roi_count:]
        self.roi = self.roi != self.roi
        for _, i, j, k in tmp:
            self.roi[i,j,k] = True

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        label = self.Label_list[idx]
        risk = self.risk_list[idx]
        # demor = self.demor_list[idx]
        return risk, label#, np.asarray(demor).astype(np.float32)


    def get_sample_weights(self):
        num_classes = len(set(self.Label_list))
        counts = [self.Label_list.count(i) for i in range(num_classes)]
        count = len(self.Label_list)
        weights = [count / counts[i] for i in self.Label_list]
        class_weights = [count/c for c in counts]
        return weights, class_weights

class MLP_Data_f1(Dataset):
    def __init__(self, Data_dir, mode, exp_idx, stage, roi_threshold, roi_count, choice, ratio=(0.6, 0.2, 0.2), seed=1000, yr=2):
        random.seed(seed)
        self.Data_list = glob.glob(Data_dir[0] + '*nii*')
        self.mode = mode

        csvname = './csv/{}.csv'.format('fdg_mri_amy_mci_csf_cumulative_pruned_final')
        fileIDs, labels = read_csv2(csvname) #training file

        tmp = []
        for f in fileIDs:
            for d in self.Data_list:
                if f in d:
                    tmp += [f]
        skip = list(set(tmp) ^ set(fileIDs))
        # print(skip, 'not found, skipped')
        for f in skip:
            idx = fileIDs.index(f)
            fileIDs.pop(idx)
            labels.pop(idx)
        #self.Data_list.sort()

        labels = np.array(labels)
        labels[labels==-1] = 0
        self.Label_list = labels.tolist()

        FIXED = False
        if FIXED:
            # 1 fold for testing, 1 fold for validation, the rest for training
            # Note: in this case, MLP supports only 5 fold cross validation!
            # Not in use, need to test before use!
            num_fold = 5
            idxs = list(range(len(fileIDs)))

            skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=120)

            i = 0
            for train_index, test_index in skf.split(idxs, labels):
                if i == exp_idx:
                    if 'train' in stage:
                        self.index_list = train_index[:-len(test_index)]
                    elif 'valid' in stage:
                        self.index_list = train_index[-len(test_index):]
                    elif 'test' in stage:
                        self.index_list = test_index
                    elif 'all' in stage:
                        self.index_list = idxs
                    else:
                        self.index_list = []
                    break
                i += 1

        else:
            l = len(self.Data_list)
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
        self.fileIDs = np.array(fileIDs)[self.index_list]

        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.roi_threshold = roi_threshold
        self.roi_count = roi_count

        risk_lists = []
        for d, m in zip(self.Data_dir, self.mode):
            if choice == 'count':
                self.select_roi_count(d)
            else:
                self.select_roi_thres(d)
            risk_lists += [[get_AD_risk(np.load(file.replace(Data_dir[0], d).replace(self.mode[0], m)))[self.roi] for file in self.Data_list]]
        self.risk_list = risk_lists[0]
        for r in risk_lists[1:]:
            self.risk_list = np.append(self.risk_list, r, axis=1)
        self.in_size = self.risk_list[0].shape[0]

        if CROSS_VALID:
            with open('cross/rid_order_index.txt', 'w') as f:
                f.write(self.index_list)
            with open('cross/rid_order_data.txt', 'w') as f:
                f.write(self.Data_list)

    def select_roi_thres(self, data_dir):
        self.roi = np.load(data_dir + 'train_MCC.npy')
        self.roi = self.roi > self.roi_threshold
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0:
                        self.roi[i,j,k] = False

    def select_roi_count(self, data_dir):
        self.roi = np.load(data_dir + 'train_MCC.npy')
        tmp = []
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0: continue
                    tmp.append((self.roi[i,j,k], i, j, k))
        tmp.sort()
        tmp = tmp[-self.roi_count:]
        self.roi = self.roi != self.roi
        for _, i, j, k in tmp:
            self.roi[i,j,k] = True

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        label = self.Label_list[idx]
        risk = self.risk_list[idx]
        # demor = self.demor_list[idx]
        return risk, label#, np.asarray(demor).astype(np.float32)

    def get_sample_weights(self):
        num_classes = len(set(self.Label_list))
        counts = [self.Label_list.count(i) for i in range(num_classes)]
        count = len(self.Label_list)
        weights = [count / counts[i] for i in self.Label_list]
        class_weights = [count/c for c in counts]
        return weights, class_weights

class FCN_Cox_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, transform=None, whole_volume=False, ratio=(0.6, 0.2, 0.2), seed=1000, patch_size=47, dim=1, name='', fold=[], yr=2):
        random.seed(seed)

        self.stage = stage
        self.transform = transform
        self.whole = whole_volume
        self.patch_size = patch_size
        self.patch_sampler = PatchGenerator(patch_size=self.patch_size)
        self.cache = []
        self.dim = dim
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir

        self.Data_list = glob.glob(Data_dir + '*nii*')

        csvname = './csv/{}.csv'.format('fdg_mri_amy_mci_csf_cumulative_pruned_final')
        fileIDs, time_obs, time_hit = read_csv_cox(csvname) #training file
        # print(np.array([time_hit[:18],time_obs[:18]]))

        if 'mri' in name:
            name = 'mri'
        elif 'amyloid' in name:
            name = 'amyloid'
        else:
            name = 'fdg'
        self.name = name

        tmp_f = []
        tmp_d = []
        for d in self.Data_list:
            for f in fileIDs:
                fname = os.path.basename(d)
                if f in fname[:4]:
                    if SKULL:
                        if ('brain' in fname) and (name in fname):
                            tmp_f.append(f)
                            tmp_d.append(d)
                            break
                    else:
                        if ('brain' not in fname) and (name in fname):
                            tmp_f.append(f)
                            tmp_d.append(d)
                            break
        self.Data_list = tmp_d

        skip = list(set(tmp_f) ^ set(fileIDs))

        # print(skip, 'not found, skipped')
        for f in skip:
            idx = fileIDs.index(f)
            fileIDs.pop(idx)
            time_obs.pop(idx)
            time_hit.pop(idx)
        #self.Data_list.sort()
        self.time_obs = []
        self.time_hit = []
        for obs, hit in zip(time_obs, time_hit):
            self.time_obs += [hit if hit else obs]
            self.time_hit += [1 if hit else 0]

        FIXED = False
        if FIXED:
            # 1 fold for testing, 1 fold for validation, the rest for training
            # Note: in this case, MLP supports only 5 fold cross validation!
            # Not in use, need to test before use!
            num_fold = 5
            idxs = list(range(len(fileIDs)))

            skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=120)

            i = 0
            for train_index, test_index in skf.split(idxs, labels):
                if i == exp_idx:
                    if 'train' in stage:
                        self.index_list = train_index[:-len(test_index)]
                    elif 'valid' in stage:
                        self.index_list = train_index[-len(test_index):]
                    elif 'test' in stage:
                        self.index_list = test_index
                    elif 'all' in stage:
                        self.index_list = idxs
                    else:
                        self.index_list = []
                    break
                i += 1

        else:
            l = len(self.Data_list)
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
                raise Exception('Unexpected Stage for FCN_Cox_Data!')
        self.fileIDs = np.array(fileIDs)[self.index_list] #Note: this only for csv generation not used for data retrival

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        obs = self.time_obs[idx]
        hit = self.time_hit[idx]

        if type(self.Data_dir) == type([]):
            self.Data_list2 = [dl.replace(self.Data_dir[0], self.Data_dir[1]) for dl in self.Data_list]
            if ('mri' in self.name) and ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('mri' in self.name) and ('amyloid' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
            elif ('mri' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('amyloid', 'fdg') for d in self.Data_list2]
            else:
                print('error: case not supported. make sure the model name contains input scan types (mri, amyloid, or fdg)')
                print(self.name)
                sys.exit()
            if len(self.Data_dir) == 2:
                try:
                    data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                    data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                except:
                    data1 = np.load(self.Data_list[idx]).astype(np.float32)
                    data2 = np.load(self.Data_list2[idx]).astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data = np.array([data1, data2])
            else:
                data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                data3 = nib.load(self.Data_list3[idx]).get_fdata().astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data3[data3 != data3] = 0
                data = np.array([data1, data2, data3])
        else:
            try:
                data = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
            except:
                data = np.load(self.Data_list[idx]).astype(np.float32)
            data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
        if self.whole:
            # if len(data.shape) == 4:
            #     data = data[0:160,0:240,0:256,0]
            if type(self.Data_dir) == type([]):
                if len(self.Data_dir) == 2:
                    data1 = padding(data[0], win_size=self.patch_size // 2)
                    data2 = padding(data[1], win_size=self.patch_size // 2)
                    data = np.array((data1, data2))
                else:
                    data1 = padding(data[0], win_size=self.patch_size // 2)
                    data2 = padding(data[1], win_size=self.patch_size // 2)
                    data3 = padding(data[2], win_size=self.patch_size // 2)
                    data = np.array((data1, data2, data3))
            else:
                data = np.expand_dims(padding(data, win_size=self.patch_size // 2), axis=0)
            return data, obs, hit
        elif self.stage == 'valid_patch' and len(self.cache) == len(self.time_hit):
            return self.cache[idx]
        elif self.stage == 'valid_patch':
            # data = np.load(self.Data_list[idx], mmap_mode='r').astype(np.float32)
            # if len(data.shape) == 4:
            #     data = data[0:160,0:240,0:256,0]
            array_list = []
            if type(self.Data_dir) == type([]):
                loc = (np.array(data[0].shape)-47)//5
            else:
                loc = (np.array(data.shape)-47)//5
            patch_locs = [loc*i for i in range(1, 6)]
            for i, loc in enumerate(patch_locs):
                x, y, z = loc
                if type(self.Data_dir) == type([]):
                    patch = data[:, x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
                    array_list.append(patch)
                else:
                    patch = data[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
                    array_list.append(np.expand_dims(patch, axis = 0))
            data = Variable(torch.FloatTensor(np.stack(array_list, axis = 0)))
            obs = Variable(torch.LongTensor([obs]*5))
            hit = Variable(torch.LongTensor([hit]*5))
            self.cache.append((data, obs, hit))
            return data, obs, hit
        elif self.stage == 'train' or self.stage == 'all' :
            # data = np.load(self.Data_list[idx], mmap_mode='r').astype(np.float32)
            # if len(data.shape) == 4:
            #     data = data[0:160,0:240,0:256,0]
            if type(self.Data_dir) == type([]):
                if len(self.Data_dir) == 2:
                    patches = self.patch_sampler.random_sample(data[0], data[1])
                else:
                    patches = self.patch_sampler.random_sample(data[0], data[1], data[2])
                patch = np.array(patches)
            else:
                patch = self.patch_sampler.random_sample(data)
                patch = np.expand_dims(patch, axis=0)
            # if self.transform:
            #     patch = self.transform.apply(patch).astype(np.float32)
            return patch, obs, hit

    def get_sample_weights(self):
        num_classes = len(set(self.time_hit))
        counts = [self.time_hit.count(i) for i in range(num_classes)]
        count = len(self.time_hit)
        weights = [count / counts[i] for i in self.time_hit]
        class_weights = [count/c for c in counts]
        return weights, class_weights

class MLP_Cox_Data(Dataset):
    def __init__(self, Data_dir, mode, exp_idx, stage, ratio=(0.6, 0.2, 0.2), seed=1000, yr=2):
        random.seed(seed)
        self.Data_list = glob.glob(Data_dir[0] + '*npy*')
        self.mode = mode

        csvname = './csv/{}.csv'.format('fdg_mri_amy_mci_csf_cumulative_pruned_final')
        fileIDs, time_obs, time_hit = read_csv_cox(csvname) #training file
        # print(np.array([time_hit[:18],time_obs[:18]]))

        tmp = []
        for f in fileIDs:
            for d in self.Data_list:
                if f in d:
                    tmp += [f]
        skip = list(set(tmp) ^ set(fileIDs))
        # print(skip, 'not found, skipped')
        for f in skip:
            idx = fileIDs.index(f)
            fileIDs.pop(idx)
            time_obs.pop(idx)
            time_hit.pop(idx)
        #self.Data_list.sort()

        self.time_obs = []
        self.time_hit = []
        for obs, hit in zip(time_obs, time_hit):
            self.time_obs += [hit if hit else obs]
            self.time_hit += [1 if hit else 0]

        FIXED = False
        if FIXED:
            # 1 fold for testing, 1 fold for validation, the rest for training
            # Note: in this case, MLP supports only 5 fold cross validation!
            # Not in use, need to test before use!
            num_fold = 5
            idxs = list(range(len(fileIDs)))

            skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=120)

            i = 0
            for train_index, test_index in skf.split(idxs, labels):
                if i == exp_idx:
                    if 'train' in stage:
                        self.index_list = train_index[:-len(test_index)]
                    elif 'valid' in stage:
                        self.index_list = train_index[-len(test_index):]
                    elif 'test' in stage:
                        self.index_list = test_index
                    elif 'all' in stage:
                        self.index_list = idxs
                    else:
                        self.index_list = []
                    break
                i += 1

        else:
            l = len(self.Data_list)
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
        self.fileIDs = np.array(fileIDs)[self.index_list] #Note: this only for csv generation not used for data retrival

        self.exp_idx = exp_idx
        self.Data_dir = Data_dir

        risk_lists = []
        for d, m in zip(self.Data_dir, self.mode):
            risk_lists += [[np.load(file.replace(Data_dir[0], d).replace(self.mode[0], m)).flatten() for file in self.Data_list]]
        self.risk_list = risk_lists
        for r in risk_lists[1:]:
            self.risk_list = np.append(self.risk_list, r, axis=1)
        self.in_size = self.risk_list[0][0].shape[0]
        self.risk_list = self.risk_list[0]

        if CROSS_VALID:
            with open('cross/rid_order_index.txt', 'w') as f:
                f.write(self.index_list)
            with open('cross/rid_order_data.txt', 'w') as f:
                f.write(self.Data_list)

    def select_roi_thres(self, data_dir):
        self.roi = np.load(data_dir + 'train_MCC.npy')
        self.roi = self.roi > self.roi_threshold
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0:
                        self.roi[i,j,k] = False

    def select_roi_count(self, data_dir):
        self.roi = np.load(data_dir + 'train_MCC.npy')
        tmp = []
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0: continue
                    tmp.append((self.roi[i,j,k], i, j, k))
        tmp.sort()
        tmp = tmp[-self.roi_count:]
        self.roi = self.roi != self.roi
        for _, i, j, k in tmp:
            self.roi[i,j,k] = True

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        obs = self.time_obs[idx]
        hit = self.time_hit[idx]
        risk = self.risk_list[idx]
        return risk, obs, hit

    def get_sample_weights(self):
        num_classes = len(set(self.time_hit))
        counts = [self.time_hit.count(i) for i in range(num_classes)]
        count = len(self.time_hit)
        weights = [count / counts[i] for i in self.time_hit]
        class_weights = [count/c for c in counts]
        return weights, class_weights

class Cox_Data(Dataset):
    def __init__(self, Data_dir, mode, exp_idx, stage, roi_threshold, roi_count, choice, ratio=(0.6, 0.2, 0.2), seed=1000, yr=2):
        random.seed(seed)
        self.Data_list = glob.glob(Data_dir[0] + '*nii*')
        self.mode = mode

        csvname = './csv/{}.csv'.format('fdg_mri_amy_mci_csf_cumulative_pruned_final')
        fileIDs, time_obs, time_hit = read_csv_cox(csvname) #training file
        # print(np.array([time_hit[:18],time_obs[:18]]))

        tmp = []
        for f in fileIDs:
            for d in self.Data_list:
                if f in d:
                    tmp += [f]
        skip = list(set(tmp) ^ set(fileIDs))
        # print(skip, 'not found, skipped')
        for f in skip:
            idx = fileIDs.index(f)
            fileIDs.pop(idx)
            time_obs.pop(idx)
            time_hit.pop(idx)
        #self.Data_list.sort()

        self.time_obs = []
        self.time_hit = []
        for obs, hit in zip(time_obs, time_hit):
            self.time_obs += [hit if hit else obs]
            self.time_hit += [1 if hit else 0]

        FIXED = False
        if FIXED:
            # 1 fold for testing, 1 fold for validation, the rest for training
            # Note: in this case, MLP supports only 5 fold cross validation!
            # Not in use, need to test before use!
            num_fold = 5
            idxs = list(range(len(fileIDs)))

            skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=120)

            i = 0
            for train_index, test_index in skf.split(idxs, labels):
                if i == exp_idx:
                    if 'train' in stage:
                        self.index_list = train_index[:-len(test_index)]
                    elif 'valid' in stage:
                        self.index_list = train_index[-len(test_index):]
                    elif 'test' in stage:
                        self.index_list = test_index
                    elif 'all' in stage:
                        self.index_list = idxs
                    else:
                        self.index_list = []
                    break
                i += 1

        else:
            l = len(self.Data_list)
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
        self.fileIDs = np.array(fileIDs)[self.index_list] #Note: this only for csv generation not used for data retrival

        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.roi_threshold = roi_threshold
        self.roi_count = roi_count

        risk_lists = []
        for d, m in zip(self.Data_dir, self.mode):
            if choice == 'count':
                self.select_roi_count(d)
            else:
                self.select_roi_thres(d)
            risk_lists += [[get_AD_risk(np.load(file.replace(Data_dir[0], d).replace(self.mode[0], m)))[self.roi] for file in self.Data_list]]
        self.risk_list = risk_lists[0]
        for r in risk_lists[1:]:
            self.risk_list = np.append(self.risk_list, r, axis=1)
        self.in_size = self.risk_list[0].shape[0]

        if CROSS_VALID:
            with open('cross/rid_order_index.txt', 'w') as f:
                f.write(self.index_list)
            with open('cross/rid_order_data.txt', 'w') as f:
                f.write(self.Data_list)

    def select_roi_thres(self, data_dir):
        self.roi = np.load(data_dir + 'train_MCC.npy')
        self.roi = self.roi > self.roi_threshold
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0:
                        self.roi[i,j,k] = False

    def select_roi_count(self, data_dir):
        self.roi = np.load(data_dir + 'train_MCC.npy')
        tmp = []
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0: continue
                    tmp.append((self.roi[i,j,k], i, j, k))
        tmp.sort()
        tmp = tmp[-self.roi_count:]
        self.roi = self.roi != self.roi
        for _, i, j, k in tmp:
            self.roi[i,j,k] = True

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        obs = self.time_obs[idx]
        hit = self.time_hit[idx]
        risk = self.risk_list[idx]
        # print(self.time_hit)
        # print('info')
        return risk, obs, hit

    def get_sample_weights(self):
        num_classes = len(set(self.time_hit))
        counts = [self.time_hit.count(i) for i in range(num_classes)]
        count = len(self.time_hit)
        weights = [count / counts[i] for i in self.time_hit]
        class_weights = [count/c for c in counts]
        return weights, class_weights

class AE_Cox_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, transform=None, whole_volume=True, ratio=(0.6, 0.2, 0.2), seed=1000, patch_size=47, dim=1, name='', fold=[], yr=2):
        random.seed(seed)

        self.stage = stage
        self.transform = transform
        self.whole = whole_volume
        self.patch_size = patch_size
        self.patch_sampler = PatchGenerator(patch_size=self.patch_size)
        self.cache = []
        self.dim = dim
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir

        self.Data_list = glob.glob(Data_dir + '*nii*')

        # csvname = './csv/{}.csv'.format('fdg_mri_amy_mci_csf_cumulative_pruned_final')
        # csvname = '/data2/MRI_PET_DATA/processed_images_final_cox/'+'merged_dataframe_cox_pruned_final.csv'
        csvname = '../metadata/data_processed/'+'merged_dataframe_cox_pruned_final.csv'
        # csvname = '../metadata/data_processed/'+'merged_dataframe_cox_noqc_pruned.csv'
        fileIDs, time_obs, time_hit = read_csv_cox2(csvname) #training file

        # print(np.array([time_hit[:18],time_obs[:18]]))

        if 'mri' in name:
            name = 'mri'
        elif 'amyloid' in name:
            name = 'amyloid'
        else:
            name = 'fdg'
        self.name = name

        tmp_f = []
        tmp_d = []
        for d in self.Data_list:
            for f in fileIDs:
                fname = os.path.basename(d)
                if f in fname:
                    tmp_f.append(f)
                    tmp_d.append(d)
                    break
        self.Data_list = tmp_d
        # print(len(self.Data_list))
        # sys.exit()

        skip = list(set(tmp_f) ^ set(fileIDs))

        # print(skip, 'not found, skipped')
        for f in skip:
            idx = fileIDs.index(f)
            fileIDs.pop(idx)
            time_obs.pop(idx)
            time_hit.pop(idx)
        #self.Data_list.sort()
        self.time_obs = []
        self.time_hit = []
        for obs, hit in zip(time_obs, time_hit):
            # m = np.mean(time_obs)
            self.time_obs += [hit if hit else obs]
            self.time_hit += [1 if hit else 0]

        FIXED = False
        if FIXED:
            # 1 fold for testing, 1 fold for validation, the rest for training
            # Note: in this case, MLP supports only 5 fold cross validation!
            # Not in use, need to test before use!
            num_fold = 5
            idxs = list(range(len(fileIDs)))

            skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=120)

            i = 0
            for train_index, test_index in skf.split(idxs, labels):
                if i == exp_idx:
                    if 'train' in stage:
                        self.index_list = train_index[:-len(test_index)]
                    elif 'valid' in stage:
                        self.index_list = train_index[-len(test_index):]
                    elif 'test' in stage:
                        self.index_list = test_index
                    elif 'all' in stage:
                        self.index_list = idxs
                    else:
                        self.index_list = []
                    break
                i += 1

        else:
            l = len(self.Data_list)
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
                raise Exception('Unexpected Stage for FCN_Cox_Data!')
        self.fileIDs = np.array(fileIDs)[self.index_list] #Note: this only for csv generation not used for data retrival

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        obs = self.time_obs[idx]
        hit = self.time_hit[idx]

        if type(self.Data_dir) == type([]):
            self.Data_list2 = [dl.replace(self.Data_dir[0], self.Data_dir[1]) for dl in self.Data_list]
            if ('mri' in self.name) and ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('mri' in self.name) and ('amyloid' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
            elif ('mri' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('amyloid', 'fdg') for d in self.Data_list2]
            else:
                print('error: case not supported. make sure the model name contains input scan types (mri, amyloid, or fdg)')
                print(self.name)
                sys.exit()
            if len(self.Data_dir) == 2:
                try:
                    data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                    data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                except:
                    data1 = np.load(self.Data_list[idx]).astype(np.float32)
                    data2 = np.load(self.Data_list2[idx]).astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data = np.array([data1, data2])
            else:
                data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                data3 = nib.load(self.Data_list3[idx]).get_fdata().astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data3[data3 != data3] = 0
                data = np.array([data1, data2, data3])
        else:
            try:
                data = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
            except:
                data = np.load(self.Data_list[idx]).astype(np.float32)
            data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
        if self.whole:
            # if len(data.shape) == 4:
            #     data = data[0:160,0:240,0:256,0]
            if type(self.Data_dir) == type([]):
                if len(self.Data_dir) == 2:
                    data1 = padding(data[0], win_size=self.patch_size // 2)
                    data2 = padding(data[1], win_size=self.patch_size // 2)
                    data = np.array((data1, data2))
                else:
                    data1 = padding(data[0], win_size=self.patch_size // 2)
                    data2 = padding(data[1], win_size=self.patch_size // 2)
                    data3 = padding(data[2], win_size=self.patch_size // 2)
                    data = np.array((data1, data2, data3))
            else:
                # otherwise would be 121*145*121
                # now 167*191*167
                data = np.expand_dims(data, axis=0)
            return data, obs, hit
        elif self.stage == 'valid_patch' and len(self.cache) == len(self.time_hit):
            return self.cache[idx]
        elif self.stage == 'valid_patch':
            # data = np.load(self.Data_list[idx], mmap_mode='r').astype(np.float32)
            # if len(data.shape) == 4:
            #     data = data[0:160,0:240,0:256,0]
            array_list = []
            if type(self.Data_dir) == type([]):
                loc = (np.array(data[0].shape)-47)//5
            else:
                loc = (np.array(data.shape)-47)//5
            patch_locs = [loc*i for i in range(1, 6)]
            for i, loc in enumerate(patch_locs):
                x, y, z = loc
                if type(self.Data_dir) == type([]):
                    patch = data[:, x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
                    array_list.append(patch)
                else:
                    patch = data[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
                    array_list.append(np.expand_dims(patch, axis = 0))
            data = Variable(torch.FloatTensor(np.stack(array_list, axis = 0)))
            obs = Variable(torch.LongTensor([obs]*5))
            hit = Variable(torch.LongTensor([hit]*5))
            self.cache.append((data, obs, hit))
            return data, obs, hit
        elif self.stage == 'train' or self.stage == 'all' or self.stage == 'valid' or self.stage == 'test' :
            # data = np.load(self.Data_list[idx], mmap_mode='r').astype(np.float32)
            # if len(data.shape) == 4:
            #     data = data[0:160,0:240,0:256,0]
            if type(self.Data_dir) == type([]):
                if len(self.Data_dir) == 2:
                    patches = self.patch_sampler.random_sample(data[0], data[1])
                else:
                    patches = self.patch_sampler.random_sample(data[0], data[1], data[2])
                patch = np.array(patches)
            else:
                patch = self.patch_sampler.random_sample(data)
                patch = np.expand_dims(patch, axis=0)
            # if self.transform:
            #     patch = self.transform.apply(patch).astype(np.float32)
            return patch, obs, hit

    def get_sample_weights(self):
        num_classes = len(set(self.time_hit))
        counts = [self.time_hit.count(i) for i in range(num_classes)]
        count = len(self.time_hit)
        weights = [count / counts[i] for i in self.time_hit]
        class_weights = [count/c for c in counts]
        return weights, class_weights

class CNN_Surv_Data_Pre(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, transform=None, whole_volume=True, ratio=(0.6, 0.2, 0.2), seed=1000, patch_size=47, dim=1, name='', fold=[], yr=2):
        random.seed(seed)

        self.stage = stage
        self.transform = transform
        self.whole = whole_volume
        self.patch_size = patch_size
        self.patch_sampler = PatchGenerator(patch_size=self.patch_size)
        self.cache = []
        self.dim = dim
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir

        self.Data_list = glob.glob(Data_dir + '*nii*')

        # csvname = './csv/{}.csv'.format('fdg_mri_amy_mci_csf_cumulative_pruned_final')
        # csvname = '/data2/MRI_PET_DATA/processed_images_final_cox/'+'merged_dataframe_cox_pruned_final.csv'
        csvname = '../metadata/data_processed/'+'merged_dataframe_unused_cox_pruned.csv'
        fileIDs, labels = read_csv_pre(csvname) #training file
        # print(len(fileIDs))
        # print((labels))
        # sys.exit()


        # print(np.array([time_hit[:18],time_obs[:18]]))

        if 'mri' in name:
            name = 'mri'
        elif 'amyloid' in name:
            name = 'amyloid'
        else:
            name = 'fdg'
        self.name = name

        tmp_l = []
        tmp_d = []
        for d in self.Data_list:
            for f in fileIDs:
                dname = os.path.basename(d)
                if f in dname:
                    tmp_l.append(labels[fileIDs.index(f)])
                    tmp_d.append(d)
                    break
        self.Data_list = tmp_d
        self.labels = tmp_l

        l = len(self.Data_list)
        split1 = int(l*ratio[0])
        split2 = int(l*(ratio[0]+ratio[1]))
        idxs = list(range(l))
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
            raise Exception('Unexpected Stage for FCN_Cox_Data!')
        self.fileIDs = np.array(fileIDs)[self.index_list] #Note: this only for csv generation not used for data retrival

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        label = self.labels[idx]

        if type(self.Data_dir) == type([]):
            self.Data_list2 = [dl.replace(self.Data_dir[0], self.Data_dir[1]) for dl in self.Data_list]
            if ('mri' in self.name) and ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('mri' in self.name) and ('amyloid' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
            elif ('mri' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('amyloid', 'fdg') for d in self.Data_list2]
            else:
                print('error: case not supported. make sure the model name contains input scan types (mri, amyloid, or fdg)')
                print(self.name)
                sys.exit()
            if len(self.Data_dir) == 2:
                try:
                    data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                    data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                except:
                    data1 = np.load(self.Data_list[idx]).astype(np.float32)
                    data2 = np.load(self.Data_list2[idx]).astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data = np.array([data1, data2])
            else:
                data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                data3 = nib.load(self.Data_list3[idx]).get_fdata().astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data3[data3 != data3] = 0
                data = np.array([data1, data2, data3])
        else:
            try:
                data = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
            except:
                data = np.load(self.Data_list[idx]).astype(np.float32)
            data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
        if self.whole:
            # if len(data.shape) == 4:
            #     data = data[0:160,0:240,0:256,0]
            if type(self.Data_dir) == type([]):
                if len(self.Data_dir) == 2:
                    data1 = padding(data[0], win_size=self.patch_size // 2)
                    data2 = padding(data[1], win_size=self.patch_size // 2)
                    data = np.array((data1, data2))
                else:
                    data1 = padding(data[0], win_size=self.patch_size // 2)
                    data2 = padding(data[1], win_size=self.patch_size // 2)
                    data3 = padding(data[2], win_size=self.patch_size // 2)
                    data = np.array((data1, data2, data3))
            else:
                # otherwise would be 121*145*121
                # now 167*191*167
                data = np.expand_dims(data, axis=0)
            return data, label
        elif self.stage == 'train' or self.stage == 'all' or self.stage == 'valid' or self.stage == 'test' :
            # data = np.load(self.Data_list[idx], mmap_mode='r').astype(np.float32)
            # if len(data.shape) == 4:
            #     data = data[0:160,0:240,0:256,0]
            if type(self.Data_dir) == type([]):
                if len(self.Data_dir) == 2:
                    patches = self.patch_sampler.random_sample(data[0], data[1])
                else:
                    patches = self.patch_sampler.random_sample(data[0], data[1], data[2])
                patch = np.array(patches)
            else:
                patch = self.patch_sampler.random_sample(data)
                patch = np.expand_dims(patch, axis=0)
            # if self.transform:
            #     patch = self.transform.apply(patch).astype(np.float32)
            return patch, label

    def get_sample_weights(self):
        num_classes = 3
        counts = [self.labels.count(i) for i in range(3)]
        count = len(self.labels)
        weights = [count / counts[i] for i in self.labels]
        class_weights = [count/c for c in counts]
        return np.array(weights)[self.index_list], class_weights


if __name__ == "__main__":
    # dataset = CNN_Data(Data_dir='/home/mfromano/data/adni/processed_images_test/', exp_idx=0, stage='train')
    # dataset = FCN_Data(Data_dir='/data2/ADNIP/', exp_idx=0, stage='valid_patch')
    # dataset = FCN_Data(Data_dir='/home/mfromano/data/adni/processed_images_test/', exp_idx=0, stage='train', whole_volume=True)
    # train_dataloader = DataLoader(dataset, batch_size=1)
    # sample_weight, _ = dataset.get_sample_weights()
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
    # train_w_dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    # for scan1, label in train_w_dataloader:
    # t = []
    # for scan1, label in dataset:
    #     if scan1.shape in t:
    #         pass
    #     else:
    #         t.append(scan1.shape)
    # print(t)
    train_data = MLP_Data('Data_dir', 0, stage='train', roi_threshold=1, roi_count=1, choice=1, seed=1, yr=2)

    dataset = GAN_Data(Data_dir='/home/mfromano/data/adni/processed_images/', stage='train')
    # train_dataloader = DataLoader(dataset, batch_size=1)
    t = []
    for scan1, scan2, label in dataset:
        if scan1.shape in t:
            pass
        else:
            t.append(scan1.shape)
    print(t)

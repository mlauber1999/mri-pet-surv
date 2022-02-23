import os
import sys
import collections
import shutil
import nilearn
import time
import glob
import csv
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib

from models import Vanila_CNN_Lite, _FCN, _FCNt, _MLP, _Encoder, _Decoder, _CNN
from utils import get_accu, write_raw_score, DPM_statistics, read_json, bold_axs_stick, kl_divergence, rescale
from dataloader import CNN_Data, FCN_Data, FCN_Cox_Data, MLP_Data, MLP_Cox_Data, MLP_Data_f1, AE_Cox_Data, CNN_Surv_Data_Pre
# from mlp_cox_csf import MLP_Data as CSF_Data
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored
sys.path.insert(1, './plot/')
from plot import plot_mri_tau, plot_mri_tau_overlay
from sklearn.metrics import confusion_matrix, classification_report
# import cv2

CROSS_VALID = False

# import matlab.engine

def cus_loss_cus(preds, obss, hits, all_logs=None, all_obss=None, debug=False, ver=0):
    '''
    Don't use entire set for now.
    Problems: inf value may happen
    Potential reason:
        1. patch-based, random picked -> unstable result
        2. lr too high
    '''

    # if ver == 0: # custom loss
    #     # requires full dataset
    #     preds, y, e = preds, obss, hits
    #     mask = torch.ones(y.shape[0], y.shape[0])
    #     # mask[(y.T - y) > 0] = 0
    #     mask[(y.view(-1, 1) - y) < 0] = 0 #chaned from > to <!
    #     # mask = mask.cuda() # whole trainig set does not fit
    #     log_loss = torch.mm(mask, torch.exp(preds))
    #     log_loss = torch.log(log_loss)
    #     neg_log_loss = -torch.sum((preds-log_loss) * e) / torch.sum(e)
    #
    #     if torch.isnan(neg_log_loss):
    #         print('nan')
    #         print(log_loss)
    #         print(preds)
    #         sys.exit()
    #         return None
    #     return neg_log_loss

    if ver == 0: # custom loss
        # requires full dataset
        preds, y, e = preds, obss, hits
        mask = torch.ones(y.shape[0], y.shape[0])
        mask2 = torch.ones(y.shape[0], y.shape[0])
        # mask[(y.T - y) > 0] = 0
        mask[(y.view(-1, 1) - y) < 0] = 0 #chaned from > to <!
        mask2[(y.view(-1, 1) - y) > 0] = 0 #chaned from > to <! (patients who still 'at risk' at time of patient y's progression time, i.e. observed before T_y)
        # mask = mask.cuda() # whole trainig set does not fit
        # print(preds[:10])
        preds = torch.exp(torch.abs(preds))
        # print(preds[:10])
        # sys.exit()
        sum_loss = torch.mm(mask, preds) / len(preds)
        sum_loss2 = torch.mm(mask2, preds) / len(preds)
        neg_sum_loss = -torch.sum((preds-sum_loss) * e) - torch.sum((sum_loss2-preds) * e)

        if torch.isnan(neg_sum_loss):
            print('nan')
            print(sum_loss)
            print(preds)
            sys.exit()
            return None
        return neg_sum_loss

    elif ver == 1: # local cox loss
        #\frac{1}{N_D} \sum_{i \in D}[F(x_i,\theta) - log(\sum_{j \in R_i} e^F(x_j,\theta))] - \lambda P(\theta)
        '''
        where:
            D is the set of observed events
            N_D is the number of observed events
            R_i is the set of examples that are still alive at time of death t_j
            F(x,\theta) = log hazard rate
        '''

        # #consider l1
        idxs = torch.argsort(obss, dim=0, descending=True)
        h_x = preds[idxs]
        obss = obss[idxs]
        hits = hits[idxs]

        num_hits = torch.sum(hits)

        e_h_x = torch.exp(h_x)
        log_e = torch.log(torch.cumsum(e_h_x, dim=0))
        diff = h_x - log_e
        hits = torch.reshape(hits, diff.shape) #convert into same shape, prevent errors
        diff = torch.sum(diff*hits)
        loss = -diff / num_hits
        if debug:
            print(preds.shape)
            print(h_x.shape)
            sys.exit()

        if torch.isnan(loss):
            print('nan')
            print(h_x)
            print(e_h_x)
            sys.exit()
            return None
        return loss

    return 0

def cox_loss(risk_pred, y, e, ver=0):
    risk_pred = risk_pred.view(-1,1)
    y = y.view(-1,1)
    e = e.reshape(-1,1)
    mask = torch.ones(y.shape[0], y.shape[0])
    mask[(y.T - y) > 0] = 0
    log_loss = torch.exp(risk_pred) * mask
    log_loss = torch.sum(log_loss, dim=0) #/ torch.sum(mask, dim=0)
    log_loss = torch.log(log_loss).reshape(-1, 1)
    neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
    return neg_log_loss

def sur_loss(preds, obss, hits, bins=torch.Tensor([[0, 12, 24, 36, 108]])):
    bin_centers = (bins[0, 1:] + bins[0, :-1])/2
    survived_bins_censored = torch.ge(torch.mul(obss.view(-1, 1),1-hits.view(-1,1)), bin_centers)
    survived_bins_hits = torch.ge(torch.mul(obss.view(-1,1), hits.view(-1,1)),bins[0,1:])
    survived_bins = torch.logical_or(survived_bins_censored, survived_bins_hits)
    survived_bins = torch.where(survived_bins, 1, 0)

    event_bins = torch.logical_and(torch.ge(obss.view(-1, 1), bins[0, :-1]),
                 torch.lt(obss.view(-1, 1), bins[0, 1:]))
    event_bins = torch.where(event_bins, 1, 0)
    hit_bins = torch.mul(event_bins, hits.view(-1, 1))
    l_h_x = 1+survived_bins*(preds-1)
    n_l_h_x = 1-hit_bins*preds
    cat_tensor = torch.cat((l_h_x, n_l_h_x), axis=0)
    total = -torch.log(torch.clamp(cat_tensor, min=1e-07))
    pos_sum = torch.sum(total)
    neg_sum = torch.sum(pos_sum)
    return neg_sum

def sur_loss_v1(preds, obss, hits, bins=torch.Tensor([[0, 12, 24, 36, 108]])):
    l_h_x = torch.log(preds.squeeze())
    n_l_h_x = torch.log(1-preds.squeeze())
    survived_bins = torch.ge(obss.view(-1,1), bins[0,1:])
    event_bins = torch.logical_and(torch.ge(obss.view(-1,1), bins[0,:-1]),
                 torch.lt(obss.view(-1, 1), bins[0,1:]))
    hit_bins = torch.logical_and(event_bins, hits.view(-1,1).bool())
    survived_bins = torch.logical_and(survived_bins, torch.logical_not(hit_bins))
    # print(l_h_x.shape)
    # print(hit_bins.shape)
    pos_sum = torch.sum(l_h_x[hit_bins], axis=0) + torch.sum(n_l_h_x[survived_bins], axis=0)
    # print(preds)
    # print(l_h_x)
    # print(torch.sum(l_h_x[hit_bins], axis=0))
    # sys.exit()
    return torch.sum(-pos_sum)

class CNN_Wrapper:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, model_name, metric):
        self.seed = seed
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.model_name = model_name
        if metric == 'accuracy':
            self.eval_metric = get_accu #need update
        elif metric == 'f1':
            self.eval_metric = get_f1
        else:
            self.eval_metric = get_MCC
        torch.manual_seed(seed)
        self.model = Vanila_CNN_Lite(fil_num=fil_num, drop_rate=drop_rate).cuda()
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def train(self, lr, epochs, verbose=0):
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).cuda()
        # print('ratio', self.imbalanced_ratio)
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            valid_matrix = self.valid_model_epoch()
            if verbose >= 2:
                print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix, 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
            self.save_checkpoint(valid_matrix)
        if verbose >= 2:
            print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric, self.optimal_valid_matrix)
        return self.optimal_valid_metric

    def test(self):
        print('testing ... ')
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        with torch.no_grad():
            # for stage in ['train', 'valid', 'test', 'AIBL', 'NACC']:
            for stage in ['train', 'valid', 'test']:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = CNN_Data(Data_dir, self.exp_idx, stage=stage, seed=self.seed)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                f = open(self.checkpoint_dir + 'raw_score_{}.txt'.format(stage), 'w')
                matrix = np.array([[0, 0], [0, 0]])
                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    preds = self.model(inputs)
                    write_raw_score(f, preds, labels)
                    matrix += confusion_matrix(labels.cpu(), [np.argmax(p) for p in preds.cpu()], labels=[0,1])
                    # print('AD:', np.argmax(np.array(preds.cpu()),axis=1))
                # print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
                print(stage + ' accuracy ', self.eval_metric(matrix))
                f.close()

    def save_checkpoint(self, valid_matrix):
        if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_matrix = valid_matrix
            self.optimal_valid_metric = self.eval_metric(valid_matrix)
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            torch.set_deterministic(False)
            loss.backward()
            torch.set_deterministic(True)
            self.optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            valid_matrix = np.array([[0, 0], [0, 0]])
            for inputs, labels in self.valid_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                preds = self.model(inputs)
                valid_matrix += confusion_matrix(labels.cpu(), [np.argmax(p) for p in preds.cpu()], labels=[0,1])
        return valid_matrix

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        # print('cnn dataloader')
        train_data = CNN_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed)
        valid_data = CNN_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed)
        test_data  = CNN_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        # the following if else blocks represent two ways of handling class imbalance issue
        if balanced == 1:
            # print('sending balanced data...')
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
            # self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class FCN_Wrapper(CNN_Wrapper):
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, patch_size, lr, augment=False, dim=1, yr=2):
        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.patch_size = patch_size
        self.model_name = model_name
        self.augment = augment
        #'macro avg' or 'weighted avg'
        self.eval_metric = metric
        self.dim = dim
        torch.manual_seed(seed)
        if 'test' in self.model_name:
            self.model = _FCNt(num=fil_num, p=drop_rate, dim=self.dim).cuda()
        else:
            self.model = _FCN(num=fil_num, p=drop_rate, dim=self.dim).cuda()
        self.yr=yr
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        if balanced == 1:
            # balance from sampling part
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            # balance from weight part
            self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.DPMs_dir = './DPMs/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.DPMs_dir):
            os.mkdir(self.DPMs_dir)

    def train(self, epochs):
        self.optimal_valid_matrix = None
        # self.optimal_valid_matrix = np.array([[0, 0, 0, 0]]*4)
        self.optimal_valid_metric = 0
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            if self.epoch % 10 == 0:
                valid_matrix, report = self.valid_model_epoch()
                self.save_checkpoint(valid_matrix, report)
                v_score = report[self.eval_metric]['f1-score']
                print('{}th epoch validation f1 score:'.format(self.epoch), '%.4f' % v_score, '[weighted, average]:', '%.4f' % report['weighted avg']['f1-score'], '%.4f' % report['macro avg']['f1-score'])
                # if self.epoch % (epochs//10) == 0:
                #     print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix.tolist(), 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def save_checkpoint(self, valid_matrix, report):
        # if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:
        score = report[self.eval_metric]['f1-score']
        if score >= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_matrix = valid_matrix
            self.optimal_valid_metric = score
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            preds_all = []
            labels_all = []
            for patches, labels in self.valid_dataloader:
                patches, labels = patches.cuda(), labels.cuda()
                preds_all += [np.argmax(p) for p in self.model(patches).cpu()]
                labels_all += labels.cpu()
                target_names = ['class ' + str(i) for i in range(4)]
            preds_all = np.array(preds_all)
            labels_all = np.array(labels_all)
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1,2,3], target_names=target_names, zero_division=0, output_dict=True)
            valid_matrix = confusion_matrix(y_true=labels_all, y_pred=preds_all, labels=[0,1,2,3])
        return valid_matrix, report

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        if self.augment:
            train_data = FCN_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size, transform=Augment(), dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        else:
            train_data = FCN_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = FCN_Data(Data_dir, self.exp_idx, stage='valid_patch', seed=self.seed, patch_size=self.patch_size, name=self.model_name, fold=self.num_fold, yr=self.yr)

    def test_and_generate_DPMs(self, epoch=None, stages=['train', 'valid', 'test'], single_dim=True, root=None, upsample=True, CSV=True):
        if epoch:
            self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, epoch)))
        else:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        print('testing and generating DPMs ... ')
        if root:
            print('\tcustom root directory detected:', root)
            self.DPMs_dir = self.DPMs_dir.replace('./', root)
        if os.path.isdir(self.DPMs_dir):
            shutil.rmtree(self.DPMs_dir)
        os.mkdir(self.DPMs_dir)
        self.fcn = self.model.dense_to_conv()
        self.fcn.train(False)
        with torch.no_grad():
            if single_dim:
                if os.path.isdir(self.DPMs_dir+'1d/'):
                    shutil.rmtree(self.DPMs_dir+'1d/')
                os.mkdir(self.DPMs_dir+'1d/')
                if os.path.isdir(self.DPMs_dir+'upsample_vis/'):
                    shutil.rmtree(self.DPMs_dir+'upsample_vis/')
                os.mkdir(self.DPMs_dir+'upsample_vis/')
                if os.path.isdir(self.DPMs_dir+'upsample/'):
                    shutil.rmtree(self.DPMs_dir+'upsample/')
                os.mkdir(self.DPMs_dir+'upsample/')
                if os.path.isdir(self.DPMs_dir+'nii_format/'):
                    shutil.rmtree(self.DPMs_dir+'nii_format/')
                os.mkdir(self.DPMs_dir+'nii_format/')
            for stage in stages:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = FCN_Data(Data_dir, self.exp_idx, stage=stage, whole_volume=True, seed=self.seed, patch_size=self.patch_size, name=self.model_name, fold=self.num_fold, yr=self.yr)
                fids = data.index_list
                filenames = data.Data_list
                dataloader = DataLoader(data, batch_size=1, shuffle=False)
                DPMs, Labels = [], []
                labels_all = []
                for idx, (inputs, labels) in enumerate(dataloader):
                    labels_all += labels.tolist()
                    inputs, labels = inputs.cuda(), labels.cuda()
                    DPM_tensor = self.fcn(inputs, stage='inference')
                    DPM = DPM_tensor.cpu().numpy().squeeze()
                    if single_dim:
                        m = nn.Softmax(dim=1) # dim=1, as the output shape is [1, 2, cube]
                        n = nn.LeakyReLU()
                        DPM2 = m(DPM_tensor).cpu().numpy().squeeze()[1]
                        # print(np.argmax(DPM, axis=0))
                        v = nib.Nifti1Image(DPM2, np.eye(4))
                        nib.save(v, self.DPMs_dir + 'nii_format/' + os.path.basename(filenames[fids[idx]]))

                        DPM3 = n(DPM_tensor).cpu().numpy().squeeze()[1]
                        DPM3 = np.around(DPM3, decimals=2)
                        np.save(self.DPMs_dir + '1d/' + os.path.basename(filenames[fids[idx]]), DPM2)
                        if upsample:
                            DPM_ni = nib.Nifti1Image(DPM3, np.eye(4))
                            # shape = list(inputs.shape[2:])
                            shape = [121, 145, 121] # fixed value here, because the input is padded, thus cannot be used here
                            vals = np.append(np.array(DPM_ni.shape)/np.array(shape)-0.005,[1])
                            affine = np.diag(vals)
                            DPM_ni = nilearn.image.resample_img(img=DPM_ni, target_affine=affine, target_shape=shape)
                            nib.save(DPM_ni, self.DPMs_dir + 'upsample/' + os.path.basename(filenames[fids[idx]]))
                            DPM_ni = DPM_ni.get_data()

                            plt.set_cmap("jet")
                            plt.subplots_adjust(wspace=0.3, hspace=0.3)
                            # fig, axs = plt.subplots(3, 3, figsize=(20, 15))
                            # fig, axs = plt.subplots(3, 1, figsize=(20, 15))
                            fig, axs = plt.subplots(3, 3, figsize=(40, 30))

                            DPM3 = inputs.cpu().numpy().squeeze()

                            slice1, slice2, slice3 = DPM_ni.shape[0]//2, DPM_ni.shape[1]//2, DPM_ni.shape[2]//2
                            slice1b, slice2b, slice3b = DPM3.shape[0]//2, DPM3.shape[1]//2, DPM3.shape[2]//2

                            axs[0,0].imshow(DPM_ni[slice1, :, :].T)
                            # print(DPM_ni[slice1, :, :].T)
                            # axs[0,0].imshow(DPM_ni[slice1, :, :].T, vmin=0, vmax=1)
                            axs[0,0].set_title(str(labels.cpu().numpy().squeeze()), fontsize=25)
                            axs[0,0].axis('off')
                            im1 = axs[0,1].imshow(DPM_ni[:, slice2, :].T)
                            # im1 = axs[0,1].imshow(DPM_ni[:, slice2, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[0,1].axis('off')
                            axs[0,2].imshow(DPM_ni[:, :, slice3].T)
                            # axs[0,2].imshow(DPM_ni[:, :, slice3].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[0,2].axis('off')

                            axs[1,0].imshow(DPM3[slice1b, :, :].T)
                            # axs[1,0].imshow(DPM3[slice1b, :, :].T, vmin=0, vmax=1)
                            axs[1,0].set_title(str(labels.cpu().numpy().squeeze()), fontsize=25)
                            axs[1,0].axis('off')
                            axs[1,1].imshow(DPM3[:, slice2b, :].T)
                            # axs[1,1].imshow(DPM3[:, slice2b, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[1,1].axis('off')
                            axs[1,2].imshow(DPM3[:, :, slice3b].T)
                            # axs[1,2].imshow(DPM3[:, :, slice3b].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[1,2].axis('off')

                            # axs[2,0].imshow(DPM[1][slice1b, :, :].T, vmin=0, vmax=1)
                            # axs[2,0].set_title(str(labels.cpu().numpy().squeeze()), fontsize=25)
                            # axs[2,0].axis('off')
                            # axs[2,1].imshow(DPM[1][:, slice2b, :].T, vmin=0, vmax=1)
                            # # axs[1].set_title('v2', fontsize=25)
                            # axs[2,1].axis('off')
                            # axs[2,2].imshow(DPM[1][:, :, slice3b].T, vmin=0, vmax=1)
                            # # axs[2].set_title('v3', fontsize=25)
                            # axs[2,2].axis('off')

                            # cbar = fig.colorbar(im1, ax=axs)
                            # cbar.ax.tick_params(labelsize=20)
                            plt.savefig(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png', dpi=150)
                            # print(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png')
                            plt.close()
                            # print(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png')
                            # if idx == 10:
                            #     sys.exit()
                    np.save(self.DPMs_dir + os.path.basename(filenames[fids[idx]]), DPM)
                    DPMs.append(DPM)
                    Labels.append(labels)

                if CSV:
                    rids = list(data.fileIDs)
                    filename = '{}_{}_{}'.format(self.exp_idx, stage, self.model_name) #exp_stage_model-scan
                    with open('fcn_csvs/'+filename+'.csv', 'w') as f:
                        wr = csv.writer(f)
                        wr.writerows([['label']+labels_all]+[['RID']+rids])
                matrix, ACCU, F1, MCC = DPM_statistics(DPMs, Labels)
                np.save(self.DPMs_dir + '{}_MCC.npy'.format(stage), MCC)
                np.save(self.DPMs_dir + '{}_F1.npy'.format(stage),  F1)
                np.save(self.DPMs_dir + '{}_ACCU.npy'.format(stage), ACCU)
                # print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))

        print('DPM generation is done')

class FCN_Cox_Wrapper():
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, patch_size, lr, augment=False, dim=1, yr=2, loss_v=0):
        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.patch_size = patch_size
        self.model_name = model_name
        self.augment = augment
        self.cox_local = 1
        #'macro avg' or 'weighted avg'
        self.eval_metric = metric
        self.dim = dim
        torch.manual_seed(seed)
        if 'test' in self.model_name:
            self.model = _FCNt(num=fil_num, p=drop_rate, dim=self.dim, out=1).cuda() #network output is 1 for cox model
        else:
            self.model = _FCN(num=fil_num, p=drop_rate, dim=self.dim, out=1).cuda()
        self.yr=yr
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        # self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        self.criterion = cox_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.DPMs_dir = './DPMs/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.DPMs_dir):
            os.mkdir(self.DPMs_dir)

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        if self.augment:
            train_data = FCN_Cox_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size, transform=Augment(), dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        else:
            train_data = FCN_Cox_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = FCN_Cox_Data(Data_dir, self.exp_idx, stage='valid_patch', seed=self.seed, patch_size=self.patch_size, name=self.model_name, fold=self.num_fold, yr=self.yr)

        all_data = FCN_Cox_Data(Data_dir, self.exp_idx, stage='all', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=len(train_data))

    def get_info(self, stage, debug=False):
        all_x, all_obss, all_hits = [],[],[]
        if stage == 'train':
            dl = self.train_all_dataloader
        elif stage == 'all':
            dl = self.all_dataloader
        else:
            raise Exception('Error in fn get info: stage unavailable')
        for items in dl:
            all_x, all_obss, all_hits = items
        idxs = torch.argsort(all_obss, dim=0, descending=True)
        all_x = all_x[idxs]
        all_obss = all_obss[idxs]
        # all_hits = all_hits[idxs]
        with torch.no_grad():
            h_x = self.model(all_x.cuda())

        all_logs = torch.log(torch.cumsum(torch.exp(h_x), dim=0))
        if debug:
            print('h_x', h_x[:10])
            print('exp(h_x)', torch.exp(h_x))
            print('cumsum(torch.exp(h_x)', torch.cumsum(torch.exp(h_x), dim=0))
            print('all_logs', all_logs)
        return all_logs, all_obss

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, obss, hits in self.train_dataloader:
            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()

            idxs = torch.argsort(obss, dim=0, descending=True)
            inputs = inputs[idxs]
            obss = obss[idxs]
            hits = hits[idxs]

            if torch.sum(hits) == 0:
                continue # because 0 indicates nothing to learn in this batch, we skip it
            self.model.zero_grad()
            preds = self.model(inputs)
            if self.cox_local == 0:
                self.logs, self.obss = self.get_info('train')
                if np.inf in all_logs:
                    print(all_logs)
                    all_logs, all_obss = self.get_info('train')
                    print(all_logs)
                loss = self.criterion(preds, obss, hits, all_logs, all_obss)
                if loss > 8000:
                    print('get_info')
                    all_logs, all_obss = self.get_info('train',debug=0)
                    # print('all_logs', self.logs)
                    # print('all_logs2', all_logs)
                    loss2 = self.criterion(preds, obss, hits, all_logs, all_obss, ver = self.cox_local)
                    print('loss', loss)
                    print('loss2', loss2)
                    print('\ncriterion')
                    loss2 = self.criterion(preds, obss, hits, self.logs, self.obss, debug=True)
                    print(loss)
            else:
                loss = self.criterion(preds, obss, hits, ver = self.cox_local)
            torch.set_deterministic(False)
            loss.backward()
            torch.set_deterministic(True)
            self.optimizer.step()

    def train(self, epochs):
        self.optimal_valid_matrix = None
        # self.optimal_valid_matrix = np.array([[0, 0, 0, 0]]*4)
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                print('{}th epoch validation loss:'.format(self.epoch), '[Cox-based]:', '%.4f' % val_loss)
                # if self.epoch % (epochs//10) == 0:
                #     print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix.tolist(), 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def save_checkpoint(self, loss):
        # if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:
        # need to modify the metric
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            patches_all = []
            # preds_all = []
            obss_all = []
            hits_all = []
            # for patches, labels in self.valid_dataloader:
            for data, obss, hits in self.valid_dataloader:
                # if torch.sum(hits) == 0:
                    # continue # because 0 indicates nothing to learn in this batch, we skip it
                patches, obs, hit = data, obss, hits

                # print(self.model(patches, stage='test').cpu())
                # preds = self.model(patches).cpu()
                # here only use 1 patch
                patches_all += [patches.numpy()[2]]
                # preds_all += [np.mean(preds.numpy())]
                obss_all += [obss.numpy()[0]]
                hits_all += [hits.numpy()[0]]

            # preds_all, obss_all, hits_all = torch.tensor(preds_all).view(-1, 1).cuda(), torch.tensor(obss_all).view(-1).cuda(), torch.tensor(hits_all).view(-1).cuda()

            idxs = np.argsort(obss_all, axis=0)[::-1]
            patches_all = np.array(patches_all)[idxs]
            obss_all = np.array(obss_all)[idxs]
            hits_all = np.array(hits_all)[idxs]

            preds_all = self.model(torch.tensor(patches_all).cuda()).cpu()
            preds_all, obss_all, hits_all = preds_all.view(-1, 1).cuda(), torch.tensor(obss_all).view(-1).cuda(), torch.tensor(hits_all).view(-1).cuda()


            if self.cox_local == 0:
                all_logs, all_obss = self.get_info('train')
                loss = self.criterion(preds_all, obss_all, hits_all, all_logs, all_obss)
                # loss = self.criterion(preds_all, obss_all, hits_all, self.logs, self.obss)
            else:
                loss = self.criterion(preds_all, obss_all, hits_all, ver=self.cox_local)
        return loss

    def test_and_generate_DPMs(self, epoch=None, stages=['train', 'valid', 'test'], single_dim=True, root=None, upsample=True, CSV=True):
        if epoch:
            self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, epoch)))
        else:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        print('testing and generating DPMs ... ')
        if root:
            print('\tcustom root directory detected:', root)
            self.DPMs_dir = self.DPMs_dir.replace('./', root)
        if os.path.isdir(self.DPMs_dir):
            shutil.rmtree(self.DPMs_dir)
        os.mkdir(self.DPMs_dir)
        self.fcn = self.model.dense_to_conv()
        self.fcn.train(False)
        with torch.no_grad():
            if single_dim:
                if os.path.isdir(self.DPMs_dir+'1d/'):
                    shutil.rmtree(self.DPMs_dir+'1d/')
                os.mkdir(self.DPMs_dir+'1d/')
                if os.path.isdir(self.DPMs_dir+'upsample_vis/'):
                    shutil.rmtree(self.DPMs_dir+'upsample_vis/')
                os.mkdir(self.DPMs_dir+'upsample_vis/')
                if os.path.isdir(self.DPMs_dir+'upsample/'):
                    shutil.rmtree(self.DPMs_dir+'upsample/')
                os.mkdir(self.DPMs_dir+'upsample/')
                if os.path.isdir(self.DPMs_dir+'nii_format/'):
                    shutil.rmtree(self.DPMs_dir+'nii_format/')
                os.mkdir(self.DPMs_dir+'nii_format/')
            for stage in stages:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = FCN_Cox_Data(Data_dir, self.exp_idx, stage=stage, whole_volume=True, seed=self.seed, patch_size=self.patch_size, name=self.model_name, fold=self.num_fold, yr=self.yr)
                fids = data.index_list
                filenames = data.Data_list
                dataloader = DataLoader(data, batch_size=1, shuffle=False)
                DPMs, Labels = [], []
                labels_all = []
                for idx, (inputs, obss, hits) in enumerate(dataloader):
                    labels_all += hits.tolist()
                    inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
                    DPM_tensor = self.fcn(inputs, stage='inference')
                    DPM = DPM_tensor.cpu().numpy().squeeze()
                    if single_dim:
                        m = nn.Softmax(dim=1) # dim=1, as the output shape is [1, 2, cube]
                        n = nn.LeakyReLU()
                        DPM2 = m(DPM_tensor).cpu().numpy().squeeze()
                        # DPM2 = m(DPM_tensor).cpu().numpy().squeeze()[1]
                        # print(np.argmax(DPM, axis=0))
                        v = nib.Nifti1Image(DPM2, np.eye(4))
                        nib.save(v, self.DPMs_dir + 'nii_format/' + os.path.basename(filenames[fids[idx]]))

                        DPM3 = n(DPM_tensor).cpu().numpy().squeeze()
                        # DPM3 = DPM_tensor.cpu().numpy().squeeze() #might produce strange edges on the side, see later comments

                        DPM3 = np.around(DPM3, decimals=2)
                        np.save(self.DPMs_dir + '1d/' + os.path.basename(filenames[fids[idx]]), DPM2)
                        if upsample:
                            DPM_ni = nib.Nifti1Image(DPM3, np.eye(4))
                            # shape = list(inputs.shape[2:])
                            # [167, 191, 167]
                            shape = [121, 145, 121] # fixed value here, because the input is padded, thus cannot be used here

                            # if not using the activation fn, need to subtract a small value to offset the boarder of the resized image
                            # vals = np.append(np.array(DPM_ni.shape)/np.array(shape)-0.005,[1])
                            vals = np.append(np.array(DPM_ni.shape)/np.array(shape),[1])

                            affine = np.diag(vals)
                            DPM_ni = nilearn.image.resample_img(img=DPM_ni, target_affine=affine, target_shape=shape)
                            nib.save(DPM_ni, self.DPMs_dir + 'upsample/' + os.path.basename(filenames[fids[idx]]))
                            DPM_ni = DPM_ni.get_data()

                            plt.set_cmap("jet")
                            plt.subplots_adjust(wspace=0.3, hspace=0.3)
                            fig, axs = plt.subplots(3, 3, figsize=(20, 15))
                            # fig, axs = plt.subplots(3, 3, figsize=(20, 15))
                            # fig, axs = plt.subplots(2, 3, figsize=(40, 30))

                            INPUT = inputs.cpu().numpy().squeeze()

                            slice1, slice2, slice3 = DPM_ni.shape[0]//2, DPM_ni.shape[1]//2, DPM_ni.shape[2]//2
                            slice1b, slice2b, slice3b = INPUT.shape[0]//2, INPUT.shape[1]//2, INPUT.shape[2]//2
                            slice1c, slice2c, slice3c = DPM3.shape[0]//2, DPM3.shape[1]//2, DPM3.shape[2]//2

                            axs[0,0].imshow(DPM_ni[slice1, :, :].T)
                            # print(DPM_ni[slice1, :, :].T)
                            # axs[0,0].imshow(DPM_ni[slice1, :, :].T, vmin=0, vmax=1)
                            axs[0,0].set_title('output. status:'+str(hits.cpu().numpy().squeeze()), fontsize=25)
                            axs[0,0].axis('off')
                            im1 = axs[0,1].imshow(DPM_ni[:, slice2, :].T)
                            # im1 = axs[0,1].imshow(DPM_ni[:, slice2, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[0,1].axis('off')
                            im = axs[0,2].imshow(DPM_ni[:, :, slice3].T)
                            # axs[0,2].imshow(DPM_ni[:, :, slice3].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[0,2].axis('off')
                            cbar = fig.colorbar(im, ax=axs[0,2])
                            cbar.ax.tick_params(labelsize=20)

                            axs[1,0].imshow(INPUT[slice1b, :, :].T)
                            # axs[1,0].imshow(DPM3[slice1b, :, :].T, vmin=0, vmax=1)
                            axs[1,0].set_title('input. status:'+str(hits.cpu().numpy().squeeze()), fontsize=25)
                            axs[1,0].axis('off')
                            axs[1,1].imshow(INPUT[:, slice2b, :].T)
                            # axs[1,1].imshow(DPM3[:, slice2b, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[1,1].axis('off')
                            im = axs[1,2].imshow(INPUT[:, :, slice3b].T)
                            # axs[1,2].imshow(DPM3[:, :, slice3b].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[1,2].axis('off')
                            cbar = fig.colorbar(im, ax=axs[1,2])
                            cbar.ax.tick_params(labelsize=20)

                            axs[2,0].imshow(DPM3[slice1c, :, :].T)
                            # axs[2,0].imshow(DPM3[slice1c, :, :].T, vmin=0, vmax=1)
                            axs[2,0].set_title('output small size. status:'+str(hits.cpu().numpy().squeeze()), fontsize=25)
                            axs[2,0].axis('off')
                            axs[2,1].imshow(DPM3[:, slice2c, :].T)
                            # axs[2,1].imshow(DPM3[:, slice2c, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[2,1].axis('off')
                            im = axs[2,2].imshow(DPM3[:, :, slice3c].T)
                            # axs[2,2].imshow(DPM3[:, :, slice3c].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[2,2].axis('off')
                            cbar = fig.colorbar(im, ax=axs[2,2])
                            cbar.ax.tick_params(labelsize=20)

                            # cbar = fig.colorbar(im1, ax=axs)
                            # cbar.ax.tick_params(labelsize=20)
                            plt.savefig(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png', dpi=150)
                            # print(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png')
                            plt.close()
                            # print(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png')
                            # if idx == 10:
                            #     sys.exit()
                    np.save(self.DPMs_dir + os.path.basename(filenames[fids[idx]]), DPM)
                    DPMs.append(DPM)
                    Labels.append(hits)

                if CSV:
                    rids = list(data.fileIDs)
                    filename = '{}_{}_{}'.format(self.exp_idx, stage, self.model_name) #exp_stage_model-scan
                    with open('fcn_csvs/'+filename+'.csv', 'w') as f:
                        wr = csv.writer(f)
                        wr.writerows([['label']+labels_all]+[['RID']+rids])
                # matrix, ACCU, F1, MCC = DPM_statistics(DPMs, Labels)
                # np.save(self.DPMs_dir + '{}_MCC.npy'.format(stage), MCC)
                # np.save(self.DPMs_dir + '{}_F1.npy'.format(stage),  F1)
                # np.save(self.DPMs_dir + '{}_ACCU.npy'.format(stage), ACCU)
                # print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))

        print('DPM generation is done')

    def predict():
        # TODO: given testing data, produce survival plot, based on time
        # could be a single patient
        return

class MLP_Wrapper(FCN_Wrapper):
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, lr, metric, roi_threshold, roi_count=200, choice='count', mode=None, yr=2):
        self.seed = seed
        self.mode = mode   # for normal FCN, mode is None; for FCN_GAN, mode is "gan_"
        self.choice = choice
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.roi_count = roi_count
        self.roi_threshold = roi_threshold
        self.eval_metric = metric
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.Data_dir = './DPMs/fcn_{}_exp{}/'.format(self.mode, exp_idx)
        self.yr = yr
        self.prepare_dataloader(batch_size, balanced, self.Data_dir)
        if balanced == 1:
            # balance from sampling part
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            # balance from weight part
            self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio))
        torch.manual_seed(seed)
        self.model = _MLP(in_size=self.in_size, fil_num=fil_num, drop_rate=drop_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = MLP_Data(Data_dir, self.exp_idx, stage='train', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed, yr=self.yr)
        valid_data = MLP_Data(Data_dir, self.exp_idx, stage='valid', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed, yr=self.yr)
        test_data  = MLP_Data(Data_dir, self.exp_idx, stage='test', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed, yr=self.yr)
        # the following if else blocks represent two ways of handling class imbalance issue
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            # use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            # sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.in_size = train_data.in_size

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = 0
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            valid_matrix, report = self.valid_model_epoch()
            v_score = report[self.eval_metric]['f1-score']
            # if self.epoch % 30 == 0:
                # print('{}th epoch validation f1 score:'.format(self.epoch), '%.4f' % v_score, '\t[weighted, average]:', '%.4f' % report['weighted avg']['f1-score'], '%.4f' % report['macro avg']['f1-score'])
            self.save_checkpoint(valid_matrix, report)
        return self.optimal_valid_metric

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs, labels
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            preds_all = []
            labels_all = []
            for patches, labels in self.valid_dataloader:
                patches, labels = patches, labels
                preds_all += [np.argmax(p) for p in self.model(patches)]
                labels_all += labels
            preds_all = np.array(preds_all)
            labels_all = np.array(labels_all)
            target_names = ['class ' + str(i) for i in range(4)]
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1,2,3], target_names=target_names, zero_division=0, output_dict=True)
            valid_matrix = confusion_matrix(y_true=labels_all, y_pred=preds_all, labels=[0,1,2,3])
        return valid_matrix, report

    def test(self, repe_idx):
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        accu_list = []
        with torch.no_grad():
            for stage in ['train', 'valid', 'test']: #, 'AIBL', 'NACC'
                data = MLP_Data(self.Data_dir, self.exp_idx, stage=stage, roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                f = open(self.checkpoint_dir + 'raw_score_{}_{}.txt'.format(stage, repe_idx), 'w')
                preds_all = []
                labels_all = []
                # a = []
                # b = []
                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs, labels
                    preds = self.model(inputs)
                    write_raw_score(f, preds, labels)
                    preds = [np.argmax(p) for p in preds]
                    preds_all += preds
                    labels_all += labels
                f.close()
                target_names = ['class ' + str(i) for i in range(4)]
                report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1,2,3], target_names=target_names, zero_division=0, output_dict=True)

                accu_list.append(report[self.eval_metric]['f1-score'])
        return accu_list

class MLP_Cox_Wrapper(FCN_Wrapper):
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, lr, metric, roi_threshold, roi_count=200, choice='count', mode=None, yr=2):
        self.loss_imp = 0.0
        self.loss_tot = 0.0
        self.double = False
        self.csf = 0

        self.seed = seed
        self.mode = mode
        self.choice = choice
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.roi_count = roi_count
        self.roi_threshold = roi_threshold
        self.eval_metric = metric
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        # self.Data_dir = ['./DPMs/fcn_{}_exp{}/'.format(m, exp_idx) for m in self.mode]
        self.Data_dir = ['/data2/MRI_PET_DATA/processed_images_final_cumulative/Cox/DPMs/fcn_{}_exp{}/'.format(m, exp_idx) for m in self.mode]
        self.yr = yr
        self.prepare_dataloader(batch_size, balanced, self.Data_dir)
        # if balanced == 1:
        #     # balance from sampling part
        #     self.criterion = nn.CrossEntropyLoss().cuda()
        # else:
        #     # balance from weight part
        #     self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio))
        self.criterion = cox_loss
        torch.manual_seed(seed)
        print(self.in_size)
        self.model = _MLP(in_size=self.in_size, fil_num=fil_num, drop_rate=drop_rate, out=1)
        if self.double:
            self.model.double()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.cox_local = 1

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        if self.csf:
            train_data = CSF_Data(self.Data_dir,self.seed, self.exp_idx, stage = 'train')
            valid_data = CSF_Data(self.Data_dir,self.seed, self.exp_idx, stage = 'valid')
            test_data = CSF_Data(self.Data_dir,self.seed, self.exp_idx, stage = 'test')
            self.all_data = CSF_Data(self.Data_dir,self.seed, self.exp_idx, stage = 'all')
            self.in_size = 3

        else:
            train_data = MLP_Cox_Data(Data_dir, self.mode, self.exp_idx, stage='train', seed=self.seed, yr=self.yr)
            valid_data = MLP_Cox_Data(Data_dir, self.mode, self.exp_idx, stage='valid', seed=self.seed, yr=self.yr)
            test_data  = MLP_Cox_Data(Data_dir, self.mode, self.exp_idx, stage='test', seed=self.seed, yr=self.yr)
            self.all_data =  MLP_Cox_Data(Data_dir, self.mode, self.exp_idx, stage='all', seed=self.seed, yr=self.yr)
            sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
            self.train_data = train_data
            self.in_size = train_data.in_size

        # the following if else blocks represent two ways of handling class imbalance issue
        if balanced == 1:
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            # use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            # sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        self.test_data = test_data
        # self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            val_loss = self.valid_model_epoch()
            self.save_checkpoint(val_loss)
            if self.epoch % 30 == 0:
                print('{}th epoch validation score:'.format(self.epoch), '%.4f, loss improved: %.2f' % (val_loss, self.loss_imp/self.loss_tot))
        print('Best cox model saved at the {}th epoch; cox-based loss:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

        return self.optimal_valid_metric

    def save_checkpoint(self, loss):
        # if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:
        # need to modify the metric
        score = loss
        torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, obss, hits in self.train_dataloader:
            # inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.double:
                inputs, obss, hits = inputs.double(), obss.double(), hits.double()
            if torch.sum(hits) == 0:
                continue # because 0 indicates nothing to learn in this batch, we skip it
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = self.criterion(preds, obss, hits, ver=self.cox_local)
            loss.backward()
            self.optimizer.step()
            loss2 = self.criterion(self.model(inputs), obss, hits, ver=self.cox_local)
            if loss2 < loss:
                self.loss_imp += 1
            self.loss_tot += 1

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            preds_all = []
            obss_all = []
            hits_all = []

            for data, obss, hits in self.valid_dataloader:
                # if torch.sum(hits) == 0:
                    # continue # because 0 indicates nothing to learn in this batch, we skip it
                patches, obs, hit = data, obss, hits
                if self.double:
                    patches, obs, hit = patches.double(), obs.double(), hit.double()
                # print(self.model(patches, stage='test').cpu())
                preds = self.model(patches)
                preds_all += [preds.numpy()]
                obss_all += [obss.numpy()]
                hits_all += [hits.numpy()]
            loss = self.criterion(torch.tensor(preds_all).squeeze(), torch.tensor(obss_all).squeeze(), torch.tensor(hits_all).squeeze(), ver=self.cox_local)#,debug=True)
        return loss

    def predict_plot(self, repe_idx, id=[10,30], average=False):
        # id: element id in the dataset to plot

        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = DataLoader(self.all_data, batch_size=len(self.all_data))
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]
        if self.double:
            train_x = train_x.double()
            train_obss = train_obss.double()
            train_hits = train_hits.double()

        times = train_obss

        data = self.test_data

        fig, ax = plt.subplots()
        with torch.no_grad():
            self.model.zero_grad()
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            input, obs, hit = data[id[0]]
            if average:
                inputs = []
                for d in data:
                    item = d[0]
                    if self.csf:
                        item = item.numpy()
                    inputs += [item]
                input = np.mean(inputs,axis=0)
            if self.double:
                pred = torch.exp(self.model(torch.tensor(input).view(1, -1).double()))
            else:
                pred = torch.exp(self.model(torch.tensor(input).view(1, -1)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            if average:
                title = 'Average plot'
                ax.plot(times, surv_chances, label=title)
                ax.set(xlabel='time (m)', ylabel='Surv', title='')
                ax.grid()
                ax.legend()
                fig.savefig("likelihood.png")
                return
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred1 = pred

            input, obs, hit = data[id[1]]
            if self.double:
                pred = torch.exp(self.model(torch.tensor(input).view(1, -1).double()))
            else:
                pred = torch.exp(self.model(torch.tensor(input).view(1, -1)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred2 = pred
        # print('hit', hit, obs)
        # print(pred)
        # print(sums)
        title = 'ratio (h(x_1)/h(x_2)): %.3f' % (pred1/pred2)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        if self.csf:
            fig.savefig("likelihood_csf.png")
        else:
            fig.savefig("likelihood.png")
        # plt.show()
        # print(len(surv_chances))
        # print(len(times))
        # print(pred)
        # print(sums[:10])
        # print(surv_chances[:10])
        # print(times[:10])
        # sys.exit()

        return

    def predict_plot_general(self, repe_idx):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = DataLoader(self.all_data, batch_size=len(self.all_data))
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]
        if self.double:
            train_x = train_x.double()
            train_obss = train_obss.double()
            train_hits = train_hits.double()

        times = train_obss

        data = self.test_data

        inputs_0, inputs_1 = [], []
        for d in data:
            item = d[0]
            if self.csf:
                item = item.numpy()
            if d[-1] == 0: #hit
                inputs_0 += [item]
            else:
                inputs_1 += [item]
        input_0 = np.mean(inputs_0,axis=0)
        input_1 = np.mean(inputs_1,axis=0)

        with torch.no_grad():
            self.model.zero_grad()
            input_0 = torch.tensor(input_0).view(1, -1)
            input_1 = torch.tensor(input_1).view(1, -1)
            if self.double:
                input_0 = input_0.double()
                input_1 = input_1.double()
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            pred_0 = torch.exp(self.model(input_0))
            pred_1 = torch.exp(self.model(input_1))

        fig, ax = plt.subplots()
        event_chances = pred_0 / (sums+pred_0) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 0')

        event_chances = pred_1 / (sums+pred_1) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 1')

        title = 'ratio (AD_0/AD_1): %.3f' % (pred_0/pred_1)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        if self.csf:
            fig.savefig("likelihood_general_csf.png")
        else:
            fig.savefig("likelihood_general.png")

        return

class MLP_Wrapper_f1(FCN_Wrapper):
    #fusion version 1:
    #   concatenate all DPMs and predict
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, lr, metric, roi_threshold, roi_count=200, choice='count', mode=None, yr=2):
        self.seed = seed
        self.mode = mode   # for normal FCN, mode is None; for FCN_GAN, mode is "gan_"
        self.choice = choice
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.roi_count = roi_count
        self.roi_threshold = roi_threshold
        self.eval_metric = metric
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.Data_dir = ['./DPMs/fcn_{}_exp{}/'.format(m, exp_idx) for m in self.mode]
        self.yr = yr
        self.prepare_dataloader(batch_size, balanced)
        if balanced == 1:
            # balance from sampling part
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            # balance from weight part
            self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio))
        torch.manual_seed(seed)
        self.model = _MLP(in_size=self.in_size, fil_num=fil_num, drop_rate=drop_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))

    def prepare_dataloader(self, batch_size, balanced):
        train_data = MLP_Data_f1(self.Data_dir, self.mode, self.exp_idx, stage='train', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed, yr=self.yr)
        valid_data = MLP_Data_f1(self.Data_dir, self.mode, self.exp_idx, stage='valid', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed, yr=self.yr)
        test_data  = MLP_Data_f1(self.Data_dir, self.mode, self.exp_idx, stage='test', roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed, yr=self.yr)
        # the following if else blocks represent two ways of handling class imbalance issue
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            # use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            # sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.in_size = train_data.in_size

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = 0
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            valid_matrix, report = self.valid_model_epoch()
            v_score = report[self.eval_metric]['f1-score']
            # if self.epoch % 30 == 0:
                # print('{}th epoch validation f1 score:'.format(self.epoch), '%.4f' % v_score, '\t[weighted, average]:', '%.4f' % report['weighted avg']['f1-score'], '%.4f' % report['macro avg']['f1-score'])
            self.save_checkpoint(valid_matrix, report)
        return self.optimal_valid_metric

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs, labels
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            preds_all = []
            labels_all = []
            for patches, labels in self.valid_dataloader:
                patches, labels = patches, labels
                preds_all += [np.argmax(p) for p in self.model(patches)]
                labels_all += labels
            preds_all = np.array(preds_all)
            labels_all = np.array(labels_all)
            target_names = ['class ' + str(i) for i in range(4)]
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1,2,3], target_names=target_names, zero_division=0, output_dict=True)
            valid_matrix = confusion_matrix(y_true=labels_all, y_pred=preds_all, labels=[0,1,2,3])
        return valid_matrix, report

    def test(self, repe_idx):
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        accu_list = []
        with torch.no_grad():
            for stage in ['train', 'valid', 'test']: #, 'AIBL', 'NACC'
                data = MLP_Data_f1(self.Data_dir, self.mode, self.exp_idx, stage=stage, roi_threshold=self.roi_threshold, roi_count=self.roi_count, choice=self.choice, seed=self.seed)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                f = open(self.checkpoint_dir + 'raw_score_{}_{}.txt'.format(stage, repe_idx), 'w')
                preds_all = []
                labels_all = []
                # a = []
                # b = []
                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs, labels
                    preds = self.model(inputs)
                    write_raw_score(f, preds, labels)
                    preds = [np.argmax(p) for p in preds]
                    preds_all += preds
                    labels_all += labels
                f.close()
                target_names = ['class ' + str(i) for i in range(4)]
                report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1,2,3], target_names=target_names, zero_division=0, output_dict=True)

                accu_list.append(report[self.eval_metric]['f1-score'])
        return accu_list

class AE_Cox_Wrapper:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, patch_size, lr, augment=False, dim=1, yr=2, loss_v=0):
        self.loss_imp = 0.0
        self.loss_tot = 0.0
        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.patch_size = patch_size
        self.model_name = model_name
        self.augment = augment
        self.cox_local = 1
        #'macro avg' or 'weighted avg'
        self.eval_metric = metric
        self.dim = dim
        torch.manual_seed(seed)

        fil_num = 30 #either this or batch size
        # in_size = 167*191*167
        vector_len = 3
        # fil_num = 512
        # self.model = _FCN(num=fil_num, p=drop_rate, dim=self.dim, out=1).cuda()
        self.encoder = _Encoder(drop_rate=.5, fil_num=fil_num, out_channels=vector_len).cuda()
        self.decoder = _Decoder(drop_rate=.5, fil_num=fil_num, in_channels=vector_len).cuda()
        # self.decoder = _Decoder(in_size=out, drop_rate=.5, out=in_size, fil_num=fil_num).cuda()

        self.yr=yr
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        # self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        # self.criterion = cox_loss
        self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerE = optim.Adam(self.encoder.parameters(), lr=lr)
        self.optimizerD = optim.Adam(self.decoder.parameters(), lr=lr)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.DPMs_dir = './DPMs/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.DPMs_dir):
            os.mkdir(self.DPMs_dir)
        if os.path.isdir('ae_valid/'):
            shutil.rmtree('ae_valid/')
        os.mkdir('ae_valid/')

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        if self.augment:
            train_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size, transform=Augment(), dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        else:
            train_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = AE_Cox_Data(Data_dir, self.exp_idx, stage='valid_patch', seed=self.seed, patch_size=self.patch_size, name=self.model_name, fold=self.num_fold, yr=self.yr)

        all_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='all', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=len(train_data))

    def get_info(self, stage, debug=False):
        all_x, all_obss, all_hits = [],[],[]
        if stage == 'train':
            dl = self.train_all_dataloader
        elif stage == 'all':
            dl = self.all_dataloader
        else:
            raise Exception('Error in fn get info: stage unavailable')
        for items in dl:
            all_x, all_obss, all_hits = items
        idxs = torch.argsort(all_obss, dim=0, descending=True)
        all_x = all_x[idxs]
        all_obss = all_obss[idxs]
        # all_hits = all_hits[idxs]
        with torch.no_grad():
            h_x = self.model(all_x.cuda())

        all_logs = torch.log(torch.cumsum(torch.exp(h_x), dim=0))
        if debug:
            print('h_x', h_x[:10])
            print('exp(h_x)', torch.exp(h_x))
            print('cumsum(torch.exp(h_x)', torch.cumsum(torch.exp(h_x), dim=0))
            print('all_logs', all_logs)
        return all_logs, all_obss

    def train_model_epoch(self):
        self.encoder.train(True)
        self.decoder.train(True)
        for inputs, obss, hits in self.train_dataloader:
            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()

            # idxs = torch.argsort(obss, dim=0, descending=True)
            # inputs = inputs[idxs]
            # obss = obss[idxs]
            # hits = hits[idxs]

            # if torch.sum(hits) == 0:
            #     continue # because 0 indicates nothing to learn in this batch, we skip it
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            # inputs = inputs.view(inputs.shape[0], -1)

            vector = self.encoder(inputs)
            outputs = self.decoder(vector)
            loss = self.criterion(outputs, inputs)
            loss.backward()

            self.optimizerE.step()
            self.optimizerD.step()

            # vector = self.encoder(inputs)
            # outputs = self.decoder(vector)
            # loss2 = self.criterion(outputs, inputs)
            # if loss2 < loss:
                # self.loss_imp += 1
            # self.loss_tot += 1

    def train(self, epochs):
        self.optimal_valid_matrix = None
        # self.optimal_valid_matrix = np.array([[0, 0, 0, 0]]*4)
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            if self.epoch % 20 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                # print('{}th epoch validation loss:'.format(self.epoch), '[MSE-based]:', '%.4f, loss improved: %.2f' % (val_loss, self.loss_imp/self.loss_tot))
                print('{}th epoch validation loss [MSE-based]:'.format(self.epoch), '%.4f' % (val_loss))
                # if self.epoch % (epochs//10) == 0:
                #     print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix.tolist(), 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def save_checkpoint(self, loss):
        # if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:
        # need to modify the metric
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            # torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
            torch.save(self.encoder.state_dict(), '{}{}_{}_en.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
            torch.save(self.decoder.state_dict(), '{}{}_{}_de.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def valid_model_epoch(self):
        with torch.no_grad():
            self.encoder.train(False)
            self.decoder.train(False)
            # loss_all = 0
            patches_all = []
            obss_all = []
            hits_all = []
            # for patches, labels in self.valid_dataloader:
            for data, obss, hits in self.valid_dataloader:
                # if torch.sum(hits) == 0:
                    # continue # because 0 indicates nothing to learn in this batch, we skip it
                patches, obs, hit = data, obss, hits

                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                patch = patches
                patches_all += [patch]
                # patches_all += [patch.numpy()]
                # obss_all += [obss.numpy()[0]]
                # hits_all += [hits.numpy()[0]]
                obss_all += [obss]
                hits_all += [hits]

            # idxs = np.argsort(obss_all, axis=0)[::-1]
            patches_all = np.array(patches_all)
            obss_all = np.array(obss_all)
            hits_all = np.array(hits_all)

            patches_all = torch.tensor(patches_all)

            preds_all = self.decoder(self.encoder((patches_all.cuda()))).cpu()
            # preds_all, obss_all, hits_all = preds_all.view(-1, 1).cuda(), torch.tensor(obss_all).view(-1).cuda(), torch.tensor(hits_all).view(-1).cuda()

            loss = self.criterion(preds_all, patches_all)

        with torch.no_grad():
            number = 2
            plt.figure(figsize=(20, 4))
            for index in range(number):
                # display original
                ax = plt.subplot(2, number, index + 1)
                plt.imshow(patches_all[index].cpu().numpy()[0,60])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display reconstruction
                ax = plt.subplot(2, number, index + 1 + number)
                plt.imshow(preds_all[index].cpu().numpy()[0,60])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()
            plt.savefig('ae_valid/'+str(self.epoch)+"AE.png")
            plt.close()
        return loss

    def test_and_generate_DPMs(self, epoch=None, stages=['train', 'valid', 'test'], single_dim=True, root=None, upsample=True, CSV=True):
        if epoch:
            self.encoder.load_state_dict(torch.load('{}{}_{}_en.pth'.format(self.checkpoint_dir, self.model_name, epoch)))
            self.decoder.load_state_dict(torch.load('{}{}_{}_de.pth'.format(self.checkpoint_dir, self.model_name, epoch)))
        else:
            dir1 = glob.glob(self.checkpoint_dir + '*_en.pth')
            dir2 = glob.glob(self.checkpoint_dir + '*_de.pth')
            self.encoder.load_state_dict(torch.load(dir1[0]))
            self.decoder.load_state_dict(torch.load(dir2[0]))
        print('testing and generating DPMs ... ')
        if root:
            print('\tcustom root directory detected:', root)
            self.DPMs_dir = self.DPMs_dir.replace('./', root)
        if os.path.isdir(self.DPMs_dir):
            shutil.rmtree(self.DPMs_dir)
        os.mkdir(self.DPMs_dir)
        # self.fcn = self.model.dense_to_conv()
        self.fcn = self.encoder
        self.fcn.train(False)
        with torch.no_grad():
            if single_dim:
                if os.path.isdir(self.DPMs_dir+'1d/'):
                    shutil.rmtree(self.DPMs_dir+'1d/')
                os.mkdir(self.DPMs_dir+'1d/')
                if os.path.isdir(self.DPMs_dir+'upsample_vis/'):
                    shutil.rmtree(self.DPMs_dir+'upsample_vis/')
                os.mkdir(self.DPMs_dir+'upsample_vis/')
                if os.path.isdir(self.DPMs_dir+'upsample/'):
                    shutil.rmtree(self.DPMs_dir+'upsample/')
                os.mkdir(self.DPMs_dir+'upsample/')
                if os.path.isdir(self.DPMs_dir+'nii_format/'):
                    shutil.rmtree(self.DPMs_dir+'nii_format/')
                os.mkdir(self.DPMs_dir+'nii_format/')
            for stage in stages:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = AE_Cox_Data(Data_dir, self.exp_idx, stage=stage, whole_volume=True, seed=self.seed, patch_size=self.patch_size, name=self.model_name, fold=self.num_fold, yr=self.yr)
                fids = data.index_list
                filenames = data.Data_list
                dataloader = DataLoader(data, batch_size=1, shuffle=False)
                DPMs, Labels = [], []
                labels_all = []
                print('len(data)', len(data), stage)
                for idx, (inputs, obss, hits) in enumerate(dataloader):
                    labels_all += hits.tolist()
                    inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()

                    # inputs = inputs.view(inputs.shape[0], -1)
                    DPM_tensor = self.fcn(inputs)
                    DPM = DPM_tensor.cpu().numpy().squeeze()
                    if single_dim:
                        m = nn.Softmax(dim=1) # dim=1, as the output shape is [1, 2, cube]
                        n = nn.LeakyReLU()
                        DPM2 = m(DPM_tensor).cpu().numpy().squeeze()
                        # DPM2 = m(DPM_tensor).cpu().numpy().squeeze()[1]
                        # print(np.argmax(DPM, axis=0))
                        v = nib.Nifti1Image(DPM2, np.eye(4))
                        nib.save(v, self.DPMs_dir + 'nii_format/' + os.path.basename(filenames[fids[idx]]))

                        DPM3 = n(DPM_tensor).cpu().numpy().squeeze()
                        # DPM3 = DPM_tensor.cpu().numpy().squeeze() #might produce strange edges on the side, see later comments

                        DPM3 = np.around(DPM3, decimals=2)
                        np.save(self.DPMs_dir + '1d/' + os.path.basename(filenames[fids[idx]]), DPM2)
                        if upsample:
                            DPM_ni = nib.Nifti1Image(DPM3, np.eye(4))
                            # shape = list(inputs.shape[2:])
                            # [167, 191, 167]
                            shape = [121, 145, 121] # fixed value here, because the input is padded, thus cannot be used here

                            # if not using the activation fn, need to subtract a small value to offset the boarder of the resized image
                            # vals = np.append(np.array(DPM_ni.shape)/np.array(shape)-0.005,[1])
                            vals = np.append(np.array(DPM_ni.shape)/np.array(shape),[1])

                            affine = np.diag(vals)
                            DPM_ni = nilearn.image.resample_img(img=DPM_ni, target_affine=affine, target_shape=shape)
                            nib.save(DPM_ni, self.DPMs_dir + 'upsample/' + os.path.basename(filenames[fids[idx]]))
                            DPM_ni = DPM_ni.get_data()

                            plt.set_cmap("jet")
                            plt.subplots_adjust(wspace=0.3, hspace=0.3)
                            fig, axs = plt.subplots(3, 3, figsize=(20, 15))
                            # fig, axs = plt.subplots(3, 3, figsize=(20, 15))
                            # fig, axs = plt.subplots(2, 3, figsize=(40, 30))

                            INPUT = inputs.cpu().numpy().squeeze()

                            slice1, slice2, slice3 = DPM_ni.shape[0]//2, DPM_ni.shape[1]//2, DPM_ni.shape[2]//2
                            slice1b, slice2b, slice3b = INPUT.shape[0]//2, INPUT.shape[1]//2, INPUT.shape[2]//2
                            slice1c, slice2c, slice3c = DPM3.shape[0]//2, DPM3.shape[1]//2, DPM3.shape[2]//2

                            axs[0,0].imshow(DPM_ni[slice1, :, :].T)
                            # print(DPM_ni[slice1, :, :].T)
                            # axs[0,0].imshow(DPM_ni[slice1, :, :].T, vmin=0, vmax=1)
                            axs[0,0].set_title('output. status:'+str(hits.cpu().numpy().squeeze()), fontsize=25)
                            axs[0,0].axis('off')
                            im1 = axs[0,1].imshow(DPM_ni[:, slice2, :].T)
                            # im1 = axs[0,1].imshow(DPM_ni[:, slice2, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[0,1].axis('off')
                            im = axs[0,2].imshow(DPM_ni[:, :, slice3].T)
                            # axs[0,2].imshow(DPM_ni[:, :, slice3].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[0,2].axis('off')
                            cbar = fig.colorbar(im, ax=axs[0,2])
                            cbar.ax.tick_params(labelsize=20)

                            axs[1,0].imshow(INPUT[slice1b, :, :].T)
                            # axs[1,0].imshow(DPM3[slice1b, :, :].T, vmin=0, vmax=1)
                            axs[1,0].set_title('input. status:'+str(hits.cpu().numpy().squeeze()), fontsize=25)
                            axs[1,0].axis('off')
                            axs[1,1].imshow(INPUT[:, slice2b, :].T)
                            # axs[1,1].imshow(DPM3[:, slice2b, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[1,1].axis('off')
                            im = axs[1,2].imshow(INPUT[:, :, slice3b].T)
                            # axs[1,2].imshow(DPM3[:, :, slice3b].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[1,2].axis('off')
                            cbar = fig.colorbar(im, ax=axs[1,2])
                            cbar.ax.tick_params(labelsize=20)

                            axs[2,0].imshow(DPM3[slice1c, :, :].T)
                            # axs[2,0].imshow(DPM3[slice1c, :, :].T, vmin=0, vmax=1)
                            axs[2,0].set_title('output small size. status:'+str(hits.cpu().numpy().squeeze()), fontsize=25)
                            axs[2,0].axis('off')
                            axs[2,1].imshow(DPM3[:, slice2c, :].T)
                            # axs[2,1].imshow(DPM3[:, slice2c, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[2,1].axis('off')
                            im = axs[2,2].imshow(DPM3[:, :, slice3c].T)
                            # axs[2,2].imshow(DPM3[:, :, slice3c].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[2,2].axis('off')
                            cbar = fig.colorbar(im, ax=axs[2,2])
                            cbar.ax.tick_params(labelsize=20)

                            # cbar = fig.colorbar(im1, ax=axs)
                            # cbar.ax.tick_params(labelsize=20)
                            plt.savefig(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png', dpi=150)
                            # print(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png')
                            plt.close()
                            # print(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png')
                            # if idx == 10:
                            #     sys.exit()
                    np.save(self.DPMs_dir + os.path.basename(filenames[fids[idx]]), DPM)
                    DPMs.append(DPM)
                    Labels.append(hits)

                if CSV:
                    rids = list(data.fileIDs)
                    filename = '{}_{}_{}'.format(self.exp_idx, stage, self.model_name) #exp_stage_model-scan
                    with open('fcn_csvs/'+filename+'.csv', 'w') as f:
                        wr = csv.writer(f)
                        wr.writerows([['label']+labels_all]+[['RID']+rids])
                # matrix, ACCU, F1, MCC = DPM_statistics(DPMs, Labels)
                # np.save(self.DPMs_dir + '{}_MCC.npy'.format(stage), MCC)
                # np.save(self.DPMs_dir + '{}_F1.npy'.format(stage),  F1)
                # np.save(self.DPMs_dir + '{}_ACCU.npy'.format(stage), ACCU)
                # print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
        # print(DPM.shape)

        print('DPM generation is done')

    def predict():
        # TODO: given testing data, produce survival plot, based on time
        # could be a single patient
        return

class CNN_Cox_Wrapper:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, patch_size, lr, dim=1, yr=2, loss_v=0):
        self.loss_imp = 0.0
        self.loss_tot = 0.0
        self.double = 0
        self.cox_local = 0 #0: global, 1: local
        self.loss_type = 0 # 1 for categorical
        # self.categorical = 0

        self.seed = seed
        print('seed:', seed)
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.patch_size = patch_size
        self.model_name = model_name
        #'macro avg' or 'weighted avg'
        self.eval_metric = metric
        self.dim = dim
        self.yr=yr
        torch.manual_seed(seed)
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        # self.time_intervals = list(set(self.all_data.time_obs))
        # self.time_intervals.sort()
        # self.time_class = {}
        # self.class_time = {}
        # for i,t in enumerate(self.time_intervals):
        #     self.time_class[t] = i
        #     self.class_time[i] = t

        # in_size = 167*191*167
        # vector_len = 4
        vector_len = 1
        # vector_len = len(self.time_intervals)
        # fil_num = 512
        self.model = _CNN(drop_rate=.5, fil_num=fil_num, out_channels=vector_len).cuda()
        if self.cox_local != 1:
            self.model = self.model.cpu()
        if self.double:
            self.model.double()

        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        self.criterion = cox_loss
        # self.criterion = sur_loss
        # self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.DPMs_dir = './DPMs/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.DPMs_dir):
            os.mkdir(self.DPMs_dir)

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            if self.cox_local != 1:
                self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            if self.cox_local != 1:
                self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, drop_last=True)
        self.valid_dataloader = AE_Cox_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed, patch_size=self.patch_size, name=self.model_name, fold=self.num_fold, yr=self.yr)

        test_data  = AE_Cox_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='all', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=len(train_data))

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, obss, hits in self.train_dataloader:
            # if self.categorical:
            #     obss = torch.tensor([self.time_class[o.item()] for o in obss])

            if self.double:
                inputs, obss, hits = inputs.double(), obss.double(), hits.double()
            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.cox_local != 1:
                inputs, obss, hits = inputs.cpu(), obss.cpu(), hits.cpu()

            if torch.sum(hits) == 0:
                continue # because 0 indicates nothing to learn in this batch, we skip it

            self.model.zero_grad()
            preds = self.model(inputs)

            if self.loss_type:
                loss = self.criterion(preds, obss, hits)
            else:
                loss = self.criterion(preds, obss, hits, ver=self.cox_local)
            loss.backward()
            # clip = 3
            # nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
        return loss

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            train_loss = self.train_model_epoch()
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                # print('{}th epoch validation [cox] loss:'.format(self.epoch), '%.4f, loss improved: %.2f' % (val_loss, self.loss_imp/self.loss_tot))
                print('{}th epoch validation loss [cox]:'.format(self.epoch), '%.4f' % (val_loss), 'train_loss: %.4f' % (train_loss))
                # if self.epoch % (epochs//10) == 0:
                #     print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix.tolist(), 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            patches_all = []
            obss_all = []
            hits_all = []
            # for patches, labels in self.valid_dataloader:
            for data, obss, hits in self.valid_dataloader:
                # if torch.sum(hits) == 0:
                    # continue # because 0 indicates nothing to learn in this batch, we skip it
                patches, obs, hit = data, obss, hits

                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                patch = patches
                patches_all += [patch]
                # patches_all += [patch.numpy()]
                # obss_all += [obss.numpy()[0]]
                # hits_all += [hits.numpy()[0]]
                obss_all += [obss]
                hits_all += [hits]

            # idxs = np.argsort(obss_all, axis=0)[::-1]
            patches_all = np.array(patches_all)
            obss_all = np.array(obss_all)
            hits_all = np.array(hits_all)
            # if self.categorical:
            #     obss_all = torch.tensor([self.time_class[o] for o in obss_all])

            patches_all = torch.tensor(patches_all)

            if self.cox_local != 1:
                preds_all = self.model(patches_all)

                if self.loss_type:
                    # print(preds_all.shape, obss_all.shape, hits_all.shape)
                    loss = self.criterion(torch.tensor(preds_all), torch.tensor(obss_all), torch.tensor(hits_all))
                else:
                    preds_all, obss_all, hits_all = preds_all.view(-1, 1), torch.tensor(obss_all).view(-1), torch.tensor(hits_all).view(-1)
                    loss = self.criterion(preds_all, obss_all, hits_all, ver=self.cox_local)

            else:
                preds_all = self.model(patches_all.cuda()).cpu()
                # preds_all, obss_all, hits_all = preds_all.view(-1, 1).cuda(), torch.tensor(obss_all).view(-1).cuda(), torch.tensor(hits_all).view(-1).cuda()
                preds_all, obss_all, hits_all = preds_all.view(-1, 1).cuda(), torch.tensor(obss_all).view(-1).cuda(), torch.tensor(hits_all).view(-1).cuda()

            # loss = self.criterion(preds_all, torch.tensor(obss_all.reshape(preds_all.shape)).float())
            # loss = self.criterion(preds_all, torch.tensor(obss_all).squeeze(), torch.tensor(hits_all).squeeze(), ver=self.cox_local)#,debug=True)
            # loss = self.criterion(preds_all, torch.tensor(obss_all).squeeze(), torch.tensor(hits_all).squeeze(), ver=self.cox_local)#,debug=True)

            # loss = self.criterion(preds_all, obss_all, hits_all, ver=self.cox_local)
        return loss

    def save_checkpoint(self, loss):
        # if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:
        # need to modify the metric
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def predict_plot(self, id=[10,30], average=False):
        # id: element id in the dataset to plot

        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = self.all_dataloader
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]
        if self.double:
            train_x = train_x.double()
            train_obss = train_obss.double()
            train_hits = train_hits.double()

        times = train_obss

        data = self.test_data

        fig, ax = plt.subplots()
        self.model = self.model.cpu()
        with torch.no_grad():
            self.model.zero_grad()
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            input, obs, hit = data[id[0]]
            if average:
                inputs = []
                for d in data:
                    item = d[0]
                    if self.csf:
                        item = item.numpy()
                    inputs += [item]
                input = np.mean(inputs,axis=0)
            if self.double:
                pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0).double()))
            else:
                pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            if average:
                title = 'Average plot'
                ax.plot(times, surv_chances, label=title)
                ax.set(xlabel='time (m)', ylabel='Surv', title='')
                ax.grid()
                ax.legend()
                fig.savefig("likelihood.png")
                return
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred1 = pred

            input, obs, hit = data[id[1]]
            if self.double:
                pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0).double()))
            else:
                pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred2 = pred
        # print('hit', hit, obs)
        # print(pred)
        # print(sums)
        title = 'ratio (h(x_1)/h(x_2)): %.3f' % (pred1/pred2)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        fig.savefig("likelihood.png")
        # plt.show()
        # print(len(surv_chances))
        # print(len(times))
        # print(pred)
        # print(sums[:10])
        # print(surv_chances[:10])
        # print(times[:10])
        # sys.exit()

        return

    def predict_plot_general(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = self.all_dataloader
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]
        if self.double:
            train_x = train_x.double()
            train_obss = train_obss.double()
            train_hits = train_hits.double()

        times = train_obss

        data = self.test_data

        inputs_0, inputs_1 = [], []
        for d in data:
            item = d[0]
            if d[-1] == 0: #hit
                inputs_0 += [item]
            else:
                inputs_1 += [item]
        input_0 = np.mean(inputs_0,axis=0)
        input_1 = np.mean(inputs_1,axis=0)

        with torch.no_grad():
            self.model.zero_grad()
            input_0 = torch.tensor(input_0).unsqueeze(dim=0)
            input_1 = torch.tensor(input_1).unsqueeze(dim=0)
            if self.double:
                input_0 = input_0.double()
                input_1 = input_1.double()
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            pred_0 = torch.exp(self.model(input_0))
            pred_1 = torch.exp(self.model(input_1))

        fig, ax = plt.subplots()
        event_chances = pred_0 / (sums+pred_0) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 0')

        event_chances = pred_1 / (sums+pred_1) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 1')

        title = 'ratio (AD_0/AD_1): %.3f' % (pred_0/pred_1)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        fig.savefig("likelihood_general.png")

        return

    def predict_plot_scatter(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        preds = []
        times = []
        for inputs, obss, hits in self.test_dataloader:

            if self.double:
                inputs, obss, hits = inputs.double(), obss.double(), hits.double()
            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.cox_local != 1:
                inputs, obss, hits = inputs.cpu(), obss.cpu(), hits.cpu()

            preds += [self.model(inputs).item()]
            times += [obss.item()]
        fig, ax = plt.subplots()

        ax.scatter(times, preds)
        title = 'Scatter plot, h_x vs time'
        ax.set(xlabel='time (m)', ylabel='h_x', title=title)
        ax.grid()

        fig.savefig("scatter_plot.png")

        return

    def concord(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        preds_all = []
        obss_all = []
        hits_all = []
        with torch.no_grad():
            for data, obss, hits in self.test_dataloader:
                # print(data.shape)
                # sys.exit()

                preds = self.model(data)
                preds_all += list(np.array(preds)[0])
                obss_all += [np.array(obss)[0]]
                hits_all += [np.array(hits)[0] == 1]
            c_index = concordance_index_censored(hits_all, obss_all, preds_all)
        # self.c_index.append(c_index)
        print(c_index)
        return c_index

class CNN_Cox_Wrapper_Pre:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, patch_size, lr, dim=1, yr=2, loss_v=0):
        self.loss_imp = 0.0
        self.loss_tot = 0.0
        self.double = 0
        self.cox_local = 0 #0: global, 1: local

        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.patch_size = patch_size
        self.model_name = model_name
        #'macro avg' or 'weighted avg'
        self.eval_metric = metric
        self.dim = dim
        torch.manual_seed(seed)

        # in_size = 167*191*167
        vector_len = 1
        # fil_num = 512
        self.model = _CNN(drop_rate=.5, fil_num=fil_num, out_channels=vector_len).cuda()
        if self.cox_local != 1:
            self.model = self.model.cpu()
        if self.double:
            self.model.double()

        self.yr=yr
        self.prepare_dataloader(batch_size, balanced, Data_dir)

        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        self.criterion = cox_loss
        # self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.01)
        #check weights?
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.1)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.DPMs_dir = './DPMs/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.DPMs_dir):
            os.mkdir(self.DPMs_dir)

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = CNN_Surv_Data_Pre(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            # sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), sampler=sampler)
            # if self.cox_local != 1:
                # self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, drop_last=True)
            # if self.cox_local != 1:
                # self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, drop_last=True)
        self.valid_dataloader = CNN_Surv_Data_Pre(Data_dir, self.exp_idx, stage='valid', seed=self.seed, patch_size=self.patch_size, name=self.model_name, fold=self.num_fold, yr=self.yr)

        test_data  = CNN_Surv_Data_Pre(Data_dir, self.exp_idx, stage='test', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = CNN_Surv_Data_Pre(Data_dir, self.exp_idx, stage='all', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=len(train_data))

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, labels in self.train_dataloader:
            # if self.categorical:
            #     obss = torch.tensor([self.time_class[o.item()] for o in obss])

            if self.double:
                inputs, labels = inputs.double(), labels.double()
            inputs, labels = inputs.cuda(), labels.cuda()
            if self.cox_local != 1:
                inputs, labels = inputs.cpu(), labels.cpu()

            self.model.zero_grad()
            preds = self.model(inputs)

            loss = self.criterion(preds, labels, labels == 2)
            loss.backward()
            self.optimizer.step()

            # print('this round bin:', torch.bincount(labels))

            # print(preds[:10])
            # print(self.model(inputs)[:10])

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            if self.epoch % 1 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                # print('{}th epoch validation loss:'.format(self.epoch), '%.4f, loss improved: %.2f' % (val_loss, self.loss_imp/self.loss_tot))
                print('{}th epoch validation loss [cox]:'.format(self.epoch), '%.4f' % (val_loss))
                # if self.epoch % (epochs//10) == 0:
                #     print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix.tolist(), 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def test(self):
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            labels_all = []
            for data, label in self.test_dataloader:
                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                preds_all += [self.model(data)]
                labels_all += [label]
            target_names = ['class ' + str(i) for i in range(2)]
            preds_all = [torch.argmax(p) for p in preds_all]
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1], target_names=target_names, zero_division=0, output_dict=False)
            print(report)
            # loss = self.criterion(torch.tensor(preds_all), torch.tensor(labels_all))

        # return loss

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            patches_all = []
            labels_all = []
            for data, label in self.valid_dataloader:
                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                patches_all += [data]
                labels_all += [label]
            patches_all = np.array(patches_all)
            labels_all = torch.tensor(labels_all)

            patches_all = torch.tensor(patches_all)

            preds_all = self.model(patches_all)
            # preds_all = self.model(patches_all.cuda()).cpu()

            loss = self.criterion(preds_all, labels_all, labels_all == 2)
            # target_names = ['class ' + str(i) for i in range(2)]
            # preds_all = [torch.argmax(p) for p in preds_all]
            # report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1], target_names=target_names, zero_division=0, output_dict=True)
            # print(report['accuracy'])
            # loss = -report['accuracy']

        return loss

    def save_checkpoint(self, loss):
        # if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:
        # need to modify the metric
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def concord(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        preds_all = []
        obss_all = []
        hits_all = []
        with torch.no_grad():
            for data, label in self.test_dataloader:
                pred = self.model(data)
                # preds_all += list(np.array(preds)[0])
                # obss_all += [np.array(obss)[0]]
                # hits_all += [np.array(hits)[0] == 1]
                preds_all += [pred]
                obss_all += [label]
                hits_all += [label == 2]

                # print(np.array(preds_all).shape)
                # sys.exit()
            print(hits_all[:10])
            print(obss_all[:10])
            print(preds_all[:10])
            c_index = concordance_index_censored(hits_all, obss_all, preds_all)
        # self.c_index.append(c_index)
        print(c_index)
        return c_index

if __name__ == "__main__":
    print('networks.py')
    o = [[1.,2.,1.],[2.,1.,1.],[3.,4.,2.]]
    output = torch.tensor(o, requires_grad=True)
    target = torch.empty(3).random_(2)
    print(output)
    print(target)
    cox_loss = cox_loss(output, target)
    print(cox_loss)

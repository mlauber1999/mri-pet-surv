from models import Vanila_CNN_Lite, _FCN, _FCNt, _MLP, _netD, _netG
from utils import get_accu, write_raw_score, DPM_statistics, read_json, bold_axs_stick, kl_divergence, rescale
from dataloader import CNN_Data, FCN_Data, MLP_Data
from torch.utils.data import DataLoader
from plot import plot_mri_tau, plot_mri_tau_overlay
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import collections
import numpy as np
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import nilearn
import time
import glob
# import cv2

CROSS_VALID = False


# import matlab.engine

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
            if self.epoch % 20 == 0:
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

    # def test_and_generate_DPMs(self, epoch=None, stages=['train', 'valid', 'test', 'AIBL', 'NACC']):
    def test_and_generate_DPMs(self, epoch=None, stages=['train', 'valid', 'test'], single_dim=True, root=None, upsample=True):
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
            for stage in stages:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = FCN_Data(Data_dir, self.exp_idx, stage=stage, whole_volume=True, seed=self.seed, patch_size=self.patch_size, name=self.model_name, fold=self.num_fold, yr=self.yr)
                fids = data.index_list
                filenames = data.Data_list
                dataloader = DataLoader(data, batch_size=1, shuffle=False)
                DPMs, Labels = [], []
                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    DPM_tensor = self.fcn(inputs, stage='inference')
                    DPM = DPM_tensor.cpu().numpy().squeeze()
                    if single_dim:
                        m = nn.Softmax(dim=1) # dim=1, as the output shape is [1, 2, cube]
                        n = nn.LeakyReLU()
                        DPM2 = m(DPM_tensor).cpu().numpy().squeeze()[1]
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

                            slice1, slice2, slice3 = DPM_ni.shape[0]//2, DPM_ni.shape[1]//2, DPM_ni.shape[2]//2
                            slice1b, slice2b, slice3b = DPM3.shape[0]//2, DPM3.shape[1]//2, DPM3.shape[2]//2

                            axs[0,0].imshow(DPM_ni[slice1, :, :].T, vmin=0, vmax=1)
                            axs[0,0].set_title(str(labels.cpu().numpy().squeeze()), fontsize=25)
                            axs[0,0].axis('off')
                            im1 = axs[0,1].imshow(DPM_ni[:, slice2, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[0,1].axis('off')
                            axs[0,2].imshow(DPM_ni[:, :, slice3].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[0,2].axis('off')

                            axs[1,0].imshow(DPM3[slice1b, :, :].T, vmin=0, vmax=1)
                            axs[1,0].set_title(str(labels.cpu().numpy().squeeze()), fontsize=25)
                            axs[1,0].axis('off')
                            axs[1,1].imshow(DPM3[:, slice2b, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[1,1].axis('off')
                            axs[1,2].imshow(DPM3[:, :, slice3b].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[1,2].axis('off')

                            axs[2,0].imshow(DPM[1][slice1b, :, :].T, vmin=0, vmax=1)
                            axs[2,0].set_title(str(labels.cpu().numpy().squeeze()), fontsize=25)
                            axs[2,0].axis('off')
                            axs[2,1].imshow(DPM[1][:, slice2b, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[2,1].axis('off')
                            axs[2,2].imshow(DPM[1][:, :, slice3b].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[2,2].axis('off')

                            cbar = fig.colorbar(im1, ax=axs)
                            cbar.ax.tick_params(labelsize=20)
                            plt.savefig(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png', dpi=150)
                            plt.close()
                            # print(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png')
                            # if idx == 10:
                            #     sys.exit()
                    np.save(self.DPMs_dir + os.path.basename(filenames[fids[idx]]), DPM)
                    DPMs.append(DPM)
                    Labels.append(labels)
                matrix, ACCU, F1, MCC = DPM_statistics(DPMs, Labels)
                np.save(self.DPMs_dir + '{}_MCC.npy'.format(stage), MCC)
                np.save(self.DPMs_dir + '{}_F1.npy'.format(stage),  F1)
                np.save(self.DPMs_dir + '{}_ACCU.npy'.format(stage), ACCU)
                # print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))

        print('DPM generation is done')

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


if __name__ == "__main__":
    print('networks.py')

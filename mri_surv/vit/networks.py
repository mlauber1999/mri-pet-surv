# network wrappers for vision transformer
# Created: 6/16/2021
# Status: in progress

import torch
import os, sys
import glob

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored, integrated_brier_score

from scipy import interpolate
from dataloader import ViT_Data
from models import _ViT_Model, _Pre_Model
from sklearn.metrics import classification_report

def make_struc_array(hits, obss):
    return np.array([(x,y) for x,y in zip(np.asarray(hits) == 1, obss)], dtype=[('hit',bool),('time',float)])

def sur_loss(preds, obss, hits, bins=torch.Tensor([[0, 24, 48, 108]])):
    if torch.cuda.is_available():
        bins = bins.cuda()
    bin_centers = (bins[0, 1:] + bins[0, :-1])/2
    survived_bins_censored = torch.ge(torch.mul(obss.view(-1, 1),1-hits.view(-1,1)), bin_centers)
    survived_bins_hits = torch.ge(torch.mul(obss.view(-1,1), hits.view(-1,1)), bins[0,1:])
    survived_bins = torch.logical_or(survived_bins_censored, survived_bins_hits)
    survived_bins = torch.where(survived_bins, 1, 0)
    event_bins = torch.logical_and(torch.ge(obss.view(-1, 1), bins[0, :-1]), torch.lt(obss.view(-1, 1), bins[0, 1:]))
    event_bins = torch.where(event_bins, 1, 0)
    hit_bins = torch.mul(event_bins, hits.view(-1, 1))
    l_h_x = 1+survived_bins*(preds-1)
    n_l_h_x = 1-hit_bins*preds
    cat_tensor = torch.cat((l_h_x, n_l_h_x), axis=0)
    total = -torch.log(torch.clamp(cat_tensor, min=1e-12))
    pos_sum = torch.sum(total)
    neg_sum = torch.sum(pos_sum)
    return neg_sum

class ViT_Wrapper:
    def __init__(self, config, exp_idx, num_fold, seed, model_name):
        self.gpu = 1

        self.config = config
        self.data_dir = config['data_dir']
        self.lr = config['lr']
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.seed = seed
        self.model_name = model_name
        self.metric = config['metric']
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        torch.manual_seed(seed)
        self.prepare_dataloader(config['batch_size'], self.data_dir)

        # in_size = 121*145*121
        vector_len = config['out_dim']
        self.targets = list(range(vector_len))
        self.model = _ViT_Model(config).cuda()
        if self.gpu != 1:
            self.model = self.model.cpu()

        if self.metric == 'Standard':
            self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
            # self.criterion = nn.MSELoss()
        else:
            self.criterion = sur_loss
        self.optimizer = optim.SGD(self.model.parameters(), lr=config['lr'], weight_decay=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)

    def prepare_dataloader(self, batch_size, data_dir):
        train_data = ViT_Data(data_dir, self.exp_idx, stage='train', seed=self.seed, name=self.model_name, fold=self.num_fold)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        self.train_data = train_data
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
        if self.gpu != 1:
            self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)

        valid_data = ViT_Data(data_dir, self.exp_idx, stage='valid', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

        test_data  = ViT_Data(data_dir, self.exp_idx, stage='test', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = ViT_Data(data_dir, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))

        Data_dir_NACC = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test/"
        external_data = ViT_Data(Data_dir_NACC, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold, external=True)
        self.external_data = external_data

    def load(self, dir, fixed=False):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        # need to update
        del st['t_decoder.bias']
        del st['t_decoder.weight']
        self.model.load_state_dict(st, strict=False)
        # need to update
        if fixed:
            ps = []
            for n, p in self.model.named_parameters():
                if n == 'l2.weight' or n == 'l2.bias' or n == 'l1.weight' or n == 'l1.bias':
                    ps += [p]
                else:
                    pass
                    p.requires_grad = False
            self.optimizer = optim.SGD(ps, lr=self.lr, weight_decay=0.01)
        print('loaded.')

    def train(self, epochs):
        print('training...')
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for self.epoch in range(epochs):
            train_loss = self.train_model_epoch()
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
            if self.epoch % (epochs//10) == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                cis, bss = self.concord(load=False)
                ci_t, ci_v = cis

                end.record()
                torch.cuda.synchronize()

                print('{}th epoch validation loss [{}] ='.format(self.epoch, self.config['metric']), '%.3f' % (val_loss), '|| train_loss = %.3f' % (train_loss), '|| CI (test vs valid) = %.3f : %.3f' % (ci_t[0], ci_v[0]), '|| time(s) =', start.elapsed_time(end)//1000, '|| BS (test vs valid) = %.3f : %.3f' %(bss[0], bss[1]))

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric.item())
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def train_model_epoch(self):
        self.model.train(True)
        total_loss = []
        for inputs, obss, hits in self.train_dataloader:
            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.gpu != 1:
                inputs, obss, hits = inputs.cpu(), obss.cpu(), hits.cpu()
            self.model.zero_grad()
            preds = self.model(inputs)
            if self.metric == 'Standard':
                loss = self.criterion(preds, hits)
            else:
                loss = self.criterion(preds, obss, hits)
            total_loss += [loss.item()]
            loss.backward()
            self.optimizer.step()
        return np.mean(total_loss)

    def valid_model_epoch(self):
        if self.metric == 'Surv':
            cis, bss = self.concord(load=False)
            ci_t, ci_v = cis
            return -ci_t[0]
        with torch.no_grad():
            self.model.train(False)
            preds_all = []
            obss_all = []
            hits_all = []
            for inputs, obss, hits in self.valid_dataloader:
                # here only use 1 patch
                if self.gpu == 1:
                    inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
                preds_all += [self.model(inputs).cpu().numpy().squeeze()]
                obss_all += [obss]
                hits_all += [hits]
            if self.gpu == 1:
                preds_all, obss_all, hits_all = torch.tensor(preds_all).cuda(), torch.tensor(obss_all).cuda(), torch.tensor(hits_all).cuda()
            else:
                preds_all, obss_all, hits_all = torch.tensor(preds_all), torch.tensor(obss_all), torch.tensor(hits_all)

            if self.metric == 'Standard':
                loss = self.criterion(preds_all, hits_all)
            else:
                loss = self.criterion(preds_all, obss_all, hits_all)
        return loss

    def save_checkpoint(self, loss):
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

    def concord_old(self, load=True, all=False):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        cis = []
        bss = []
        bins = [0, 24, 48, 108][:self.config['out_dim']+1]
        with torch.no_grad():
            dls = [self.test_dataloader, self.valid_dataloader]
            if all:
                train_dl = DataLoader(self.train_data, batch_size=1, shuffle=False)
                ext_dl = DataLoader(self.external_data, batch_size=1, shuffle=False)
                dls += [train_dl, ext_dl]
            obss_all, hits_all = [], []
            for _, obss, hits in self.train_dataloader:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc = np.array([(x,y) for x,y in zip(np.array(hits_all) == 1, obss_all)], dtype=[('hit',bool),('time',float)])
            for dl in dls:
                preds_all, preds_all_brier = [], []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    if 'survloss': # surv loss
                        interp = interpolate.interp1d(bins, preds, axis=-1, kind='quadratic')
                        concordance_time = 24
                        preds = interp(concordance_time) # approx @ 24
                        preds_all_brier += list(interp(bins))
                        preds_all += [preds[0]]
                    else:
                        preds_all += list(np.array(preds)[0])
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0] == 1]
                # print(preds_all[:10])
                preds_all = [-p for p in preds_all]
                c_index = concordance_index_censored(hits_all, obss_all, preds_all)
                test_struc = np.array([(x,y) for x,y in zip(np.array(hits_all) == 1, obss_all)], dtype=[('hit',bool),('time',float)])
                bins[-1] = max(obss_all)-1
                bins[0] = min(obss_all)
                # print('train', len(obss_all), obss_all)
                # sys.exit()
                bs = integrated_brier_score(train_struc, test_struc, preds_all_brier, bins)
                cis += [c_index]
                bss += [bs]
            # self.c_index.append(c_index)
        # print(cis)
        # print(bss)
        return cis, bss

    def concord(self, load=True, all=False):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        cis = []
        bss = []
        bins = [0, 24, 48, 108][:self.config['out_dim']+1]
        with torch.no_grad():
            dls = [self.test_dataloader, self.valid_dataloader]
            if all:
                train_dl = DataLoader(self.train_data, batch_size=1, shuffle=False)
                ext_dl = DataLoader(self.external_data, batch_size=1, shuffle=False)
                dls += [train_dl, ext_dl]
            obss_all, hits_all = [], []
            for _, obss, hits in self.train_dataloader:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc = make_struc_array(hits_all, obss_all)
            for dl in dls:
                preds_all = []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    preds_all.append(preds)
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0]]
                preds_all = np.concatenate(preds_all,axis=0)
                c_index = concordance_index_censored(np.asarray(hits_all) == 1, obss_all, -preds_all[:,1])
                interp = interpolate.PchipInterpolator(bins, preds_all, axis=1)
                new_bins = bins[:-1].copy() + [min(max(obss_all),108)-1]
                preds_all_brier = interp(new_bins)
                test_struc = make_struc_array(hits_all, obss_all)
                bs = integrated_brier_score(train_struc, test_struc, preds_all_brier, new_bins)
                cis += [c_index]
                bss += [bs]
        # print(cis)
        # print(bss)
        return cis, bss

    def test(self):
        self.model.eval()
        dls = [self.train_dataloader, self.valid_dataloader, self.test_dataloader]
        dls = [self.train_dataloader]
        names = ['train dataset', 'valid dataset', 'test dataset']
        target_names = ['class ' + str(i) for i in range(2)]
        for dl, n in zip(dls, names):
            preds_all = []
            temp = []
            labels_all = []
            with torch.no_grad():
                for inputs, _, labels in dl:
                    # here only use 1 patch
                    inputs, labels = inputs.cuda(), labels.float().cuda()
                    preds_all += torch.round(self.model(inputs).view(-1)).cpu().tolist()
                    temp += self.model(inputs).view(-1).cpu().tolist()
                    labels_all += labels.cpu().tolist()
                print()
                x = inputs
                n = x.shape[0]
                x1 = x[:,:,:60].flatten().view(n, -1) #in here shape[0] is the batch size
                x2 = x[:,:,60:].flatten().view(n, -1) #in here shape[0] is the batch size
                x1 = self.model.map1(x1)
                x2 = self.model.map2(x2)
                x = torch.stack((x1, x2), dim=0)
                print('1')
                print(x)
                print(x.shape)
                # sys.exit()

                # prepend class token
                cls_token = self.model.cls_token.repeat(1, n, 1)
                x = torch.cat([cls_token, x], dim=0)

                # x = self.pos_encoder1(x)
                x = self.model.pos_encoder2(x)
                x = self.model.t_encoder(x)
                print('3')
                print(x)
                print(x.shape)
                # sys.exit()
                # x = x.view(b_size, -1)
                # print('before linear', x.shape)
                x = self.model.t_decoder(x)
                print('4')
                print(x)
                print(x.shape)
                # x = self.a(x)
                # x = self.t_decoder2(x.squeeze().permute(1,0))
                x = self.model.da(x)
                # print('last', x.shape)
                # sys.exit()
                print('5')
                print(x)
                print(x.shape)
                x = x[0]
            print(n)

    def overlay_prepare(self, load=True, all=False):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        bins = [0, 24, 48, 108]
        with torch.no_grad():
            dls = [self.test_dataloader, DataLoader(self.external_data, batch_size=1, shuffle=False)]
            infos = [self.test_data, self.external_data]
            names = ['ADNI', 'NACC']
            obss_all, hits_all = [], []
            dfs = []
            
            for dl, info, name in zip(dls, infos, names):
                preds_all = []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    # interp = interpolate.interp1d(bins, preds, axis=-1, kind='quadratic')
                    # concordance_time = 24
                    # preds_all_brier += list(interp(bins))
                    # preds_all += [interp(concordance_time)[0]]# approx @ 24
                    preds_all += [list(np.array(preds)[0])]
                    obss_all += [np.array(obss)[0]]
                    hits_all += [float(np.array(hits)[0] == 1)]
                preds_all = np.asarray(preds_all)
                
                d = {}
                d['RID'] = info.fileIDs
                d['Dataset'] = [name]*len(info.fileIDs)
                d['0'] = preds_all[:,0]
                d['24'] = preds_all[:,1]
                d['48'] = preds_all[:,2]
                d['108'] = preds_all[:,3]
                d['TIMES'] = obss_all
                d['PROGRESSES'] = hits_all
                dfs += [pd.DataFrame(data=d)]
        return dfs

    def overlay_prepare_with_train(self, load=True, all=False):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        bins = [0, 24, 48, 108]
        with torch.no_grad():
            dls = [DataLoader(self.train_data, batch_size=1, shuffle=False, drop_last=True), self.test_dataloader, DataLoader(self.external_data, batch_size=1, shuffle=False)]
            infos = [self.train_data, self.test_data, self.external_data]
            names = ['ADNI_train', 'ADNI', 'NACC']
            obss_all, hits_all = [], []
            dfs = []
            
            for dl, info, name in zip(dls, infos, names):
                preds_all = []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    # interp = interpolate.interp1d(bins, preds, axis=-1, kind='quadratic')
                    # concordance_time = 24
                    # preds_all_brier += list(interp(bins))
                    # preds_all += [interp(concordance_time)[0]]# approx @ 24
                    preds_all += [list(np.array(preds)[0])]
                    obss_all += [np.array(obss)[0]]
                    hits_all += [float(np.array(hits)[0] == 1)]
                preds_all = np.asarray(preds_all)
                d = {}
                d['RID'] = info.fileIDs
                d['Dataset'] = [name]*len(info.fileIDs)
                d['0'] = preds_all[:,0]
                d['24'] = preds_all[:,1]
                d['48'] = preds_all[:,2]
                d['108'] = preds_all[:,3]
                d['TIMES'] = obss_all
                d['PROGRESSES'] = hits_all
                dfs += [pd.DataFrame(data=d)]
        return dfs
class Pre_ViT_Wrapper:
    def __init__(self, config, exp_idx, num_fold, seed, model_name):
        self.gpu = 1

        self.config = config
        self.data_dir = config['data_dir']
        self.lr = config['lr']
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.seed = seed
        self.model_name = model_name
        self.metric = config['metric']
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        torch.manual_seed(seed)
        self.prepare_dataloader(config['batch_size'], self.data_dir)

        # in_size = 121*145*121
        vector_len = config['out_dim']
        self.targets = list(range(vector_len))
        self.model = _Pre_Model(config).cuda()
        if self.gpu != 1:
            self.model = self.model.cpu()

        if self.metric == 'Standard':
            self.criterion = nn.BCELoss().cuda()
            # self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
            # self.criterion = nn.MSELoss().cuda()
        else:
            self.criterion = sur_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)

    def prepare_dataloader(self, batch_size, data_dir):
        train_data = ViT_Data(data_dir, self.exp_idx, stage='train', seed=self.seed, name=self.model_name, fold=self.num_fold)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        self.train_data = train_data
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
        if self.gpu != 1:
            self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)

        valid_data = ViT_Data(data_dir, self.exp_idx, stage='valid', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

        test_data  = ViT_Data(data_dir, self.exp_idx, stage='test', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = ViT_Data(data_dir, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))

        Data_dir_NACC = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test/"
        external_data = ViT_Data(Data_dir_NACC, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold, external=True)
        self.external_data = external_data

    def train(self, epochs):
        print('training...')
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for self.epoch in range(epochs):
            train_loss = self.train_model_epoch()
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
            if self.epoch % (epochs//10) == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                self.test()
                # sys.exit()

                end.record()
                torch.cuda.synchronize()

                print('{}th epoch validation loss [{}] ='.format(self.epoch, self.config['metric']), '%.3f' % (val_loss), '|| train_loss = %.3f' % (train_loss), '|| time(s) =', start.elapsed_time(end)//1000)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric.item())
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def train_model_epoch(self):
        self.model.train(True)
        total_loss = []
        for inputs, _, hits in self.train_dataloader:
            # self.test()

            inputs, hits = inputs.cuda(), hits.float().cuda()
            if self.gpu != 1:
                inputs, hits = inputs.cpu(), hits.cpu()
            self.model.zero_grad()
            preds = self.model(inputs)
            # preds = self.model(inputs).view(-1)
            # print(hits.shape)
            # print(self.model(inputs).shape)
            # sys.exit()
            # print(preds)
            hits = F.one_hot(hits.long(), num_classes=2)

            if self.metric == 'Standard':
                loss = self.criterion(preds, hits.float())
            else:
                loss = self.criterion(preds, obss, hits)
            total_loss += [loss.item()]

            # torch.use_deterministic_algorithms(False)
            loss.backward()
            # torch.use_deterministic_algorithms(True)
            # clip = 0.5
            # nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            # print(self.model(inputs).view(-1))


            # for p in self.model.parameters():
                # print(p.data)
            # self.test()
            # sys.exit()
            # print('hits', hits[:10])
        return np.mean(total_loss)

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            loss = []
            for inputs, _, hits in self.valid_dataloader:
                # here only use 1 patch
                if self.gpu == 1:
                    inputs, hits = inputs.cuda(), hits.float().cuda()
                preds = self.model(inputs)
                hits = F.one_hot(hits.long(), num_classes=2).float()
                loss += [self.criterion(preds, hits).item()]

        return np.mean(loss)

    def save_checkpoint(self, loss):
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

    def test(self):
        self.model.eval()
        dls = [self.train_dataloader, self.valid_dataloader, self.test_dataloader]
        dls = [self.train_dataloader]
        names = ['train dataset', 'valid dataset', 'test dataset']
        target_names = ['class ' + str(i) for i in range(2)]
        for dl, name in zip(dls, names):
            preds_all = []
            temp = []
            labels_all = []
            with torch.no_grad():
                for inputs, _, labels in dl:
                    # here only use 1 patch
                    inputs, labels = inputs.cuda(), labels.float().cuda()
                    # preds_all += torch.round(self.model(inputs).view(-1)).cpu().tolist()
                    preds_all += torch.argmax(self.model(inputs), dim=1).cpu().tolist()
                    # print(preds_all)
                    # print(self.model(inputs).shape)
                    # sys.exit()
                    temp += self.model(inputs).cpu().tolist()
                    # temp += self.model(inputs).view(-1).cpu().tolist()
                    labels_all += labels.cpu().tolist()
                print(self.model(inputs))
                print(torch.argmax(self.model(inputs), dim=1).cpu().tolist())
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1], target_names=target_names, zero_division=0, output_dict=False)
            print(name)
            print(report)

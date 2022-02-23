import csv
import numpy as np
import torch
import torch.nn as nn
import os, sys
import json
from scipy.stats import zscore, sem 
import random
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from statistics import mean
#from utils import read_csv2

from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve
import matplotlib.pyplot as plt

accu_all = []
f1_all = []
sens_all = []
spec_all = []

def calculate_metrics(y_true, y_pred):
    #Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    #Calculate f1
    f1 = f1_score(y_true, y_pred)

    #Calculate sensitivity & specificity
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]==1:
            true_pos += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
            false_pos += 1
        if y_true[i]==y_pred[i]==0:
            true_neg += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
            false_neg += 1

    sensitivity = true_pos/(true_pos+false_neg)
    specificity = true_neg/(true_neg+false_pos)

    return accuracy, f1, sensitivity, specificity
      
def sem_196(list):
    ci_95 = 1.96*sem(list)
    
    return ci_95


def read_csv_demographics(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        age, mmse, abeta, tau, ptau = [], [], [], [], []
        for r in reader:
            if str(int(float(r['RID']))) != '4274':
                age += [str(int(float(r['AGE'])))]
                mmse += [str(int(float(r['MMSCORE_mmse'])))]
                abeta += [str(int(float(r['abeta'])))]
                tau += [str(int(float(r['tau'])))]
                ptau += [str(int(float(r['ptau'])))]
            
    return age, mmse, abeta, tau, ptau

def read_csv2(filename):
    with open(filename, 'r') as f:
        # reader = csv.reader(f)
        # your_list = list(reader)
        reader = csv.DictReader(f)
        fileIDs, labels = [], []
        for r in reader:
            if str(int(float(r['RID']))) != '4274':
                fileIDs += [str(int(float(r['RID'])))]
                labels += [int(float(r['PROGRESSES']))]
    # fileIDs = [str(int(float(a[-12]))) for a in your_list[1:]]
    # fileIDs = ['0'*(4-len(f))+f for f in fileIDs]
    # labels = [int(float(a[-24])) for a in your_list[1:]]
    fileIDs = ['0'*(4-len(f))+f for f in fileIDs]
    
    return fileIDs, labels

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config


def write_raw_score(f, preds, labels):
    preds = preds.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')

class _MLP(nn.Module):
    "MLP that only use DPMs from fcn"
    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(fil_num)
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 2)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()

    def forward(self, X):
        X = self.bn1(X)
        out = self.do1(X)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out



class MLP_Data(Dataset):
    def __init__(self, Data_dir, seed, exp_idx, stage):
        #TODO:
        # implement 5-fold cross validation
        # zscore continuous data and one-hot encode categorical
        random.seed(seed)
        self.csv_directory = '/home/mjadick/mri-pet/metadata/data_processed/'
        fileIDs, labels = read_csv2(self.csv_directory + 'fdg_mri3_mci_csf_amyloid_2yr_pruned_final.csv')
            ###^check to make sure we're using the same files (mri?)
        age, mmse, abeta, tau, ptau = read_csv_demographics(self.csv_directory + 'fdg_mri3_mci_csf_amyloid_2yr_pruned_final.csv')
        n = len(fileIDs)
        self.data_l = [[float(abeta[i]),float(tau[i]),float(ptau[i])] for i in range(n)]
        self.labels_l = np.array(labels)
        
        
        age = zscore(list(map(int, age)))
        mmse = zscore(list(map(int, mmse)))
        abeta = zscore(list(map(int, abeta)))
        tau = zscore(list(map(int, tau)))
        ptau = zscore(list(map(int, ptau)))

        idxs = list(range(len(fileIDs)))
        random.shuffle(idxs)
        self.fileIDs = np.array(fileIDs)

        ratio = [0.6, 0.8, 1]
        length = len(self.labels_l)
        split = [int(r*length) for r in ratio]

        self.data_l = torch.FloatTensor(self.data_l)
        self.labels_l = np.array(labels)

        if stage == 'train':
            self.index_list = idxs[:split[0]]
            self.data = self.data_l #[:split[0]]
            self.labels = self.labels_l #[:split[0]]
        elif stage == 'valid':
            self.index_list = idxs[split[0]:split[1]]
            self.data = self.data_l #[split[0]:split[1]]
            self.labels = self.labels_l #[split[0]:split[1]]
        elif stage == 'test':
            self.index_list = idxs[split[1]:split[2]]
            self.data = self.data_l #[split[1]:split[2]]
            self.labels = self.labels_l #[split[1]:split[2]]


    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        #x = [float(self.data[i][idx]) for i in range(5)]
##        print(idx)
##        idx = self.index_list[idx]
##        print(self.data)
##        sys.exit()
        idx = self.index_list[idx]
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


class MLP_Wrapper:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, exp_idx, model_name, lr, metric, roi_threshold, roi_count=200, choice='count', mode=None, yr=2):
        self.seed = seed
        self.choice = choice
        self.exp_idx = exp_idx
        self.model_name = model_name
        # self.roi_count = roi_count
        # self.roi_threshold = roi_threshold
        self.eval_metric = metric
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.Data_dir = './DPMs/mlp_exp{}/'.format(exp_idx)
        # self.yr = yr
        self.prepare_dataloader(seed, batch_size, balanced, self.Data_dir)
        self.criterion = nn.CrossEntropyLoss()
        torch.manual_seed(seed)
        self.model = _MLP(in_size=self.in_size, fil_num=fil_num, drop_rate=drop_rate).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))

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

    def prepare_dataloader(self,seed, batch_size, balanced, Data_dir):
        train_data = MLP_Data('Data_dir',seed, self.exp_idx, stage = 'train')
        valid_data = MLP_Data('Data_dir',seed, self.exp_idx, stage = 'valid')
        test_data = MLP_Data('Data_dir',seed, self.exp_idx, stage = 'test')

        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.in_size = 3

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = 0
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            valid_matrix, report = self.valid_model_epoch()
            v_score = report[self.eval_metric]['f1-score']
            if self.epoch % 30 == 0:
                print('{}th epoch validation f1 score:'.format(self.epoch), '%.4f' % v_score, '\t[weighted, average]:', '%.4f' % report['weighted avg']['f1-score'], '%.4f' % report['macro avg']['f1-score'])
            self.save_checkpoint(valid_matrix, report)
        return self.optimal_valid_metric

    def train_model_epoch(self):
        self.model.train(True)

        for inputs, labels in self.train_dataloader:            
            inputs, labels = inputs, labels

##            print('train')                  ###
##            print(inputs,labels)

            
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
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1], target_names=target_names, zero_division=0, output_dict=True)
            valid_matrix = confusion_matrix(y_true=labels_all, y_pred=preds_all, labels=[0,1])
        return valid_matrix, report

    def test(self, repe_idx):
        CSV = False
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        accu_list = []
        with torch.no_grad():
            for stage in ['train', 'valid', 'test']: #, 'AIBL', 'NACC'
                data = MLP_Data('Data_dir',self.seed, self.exp_idx, stage=stage)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                #f = open(self.checkpoint_dir + 'raw_score_{}_{}.txt'.format(stage, repe_idx), 'w')
                preds_all = []
                labels_all = []
                #inputs_all = []
                # a = []
                # b = []
                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs, labels
                    preds = self.model(inputs)
                    #write_raw_score(f, preds, labels)
                    preds = [np.argmax(p).item() for p in preds]
                    preds_all += preds
                    labels_all += labels
                    labels_all = [int(tensor) for tensor in labels_all]
                    #inputs_all += inputs
                if CSV:
                    rids = list(data.fileIDs[data.index_list])
                    filename = '{}_{}_{}_{}'.format('0', repe_idx, stage, 'golden') #fcn_mlp_scan_stage
                    with open('mlp_f_csvs/'+filename+'.csv', 'w') as f:
                        wr = csv.writer(f)
                        wr.writerows([['label']+labels_all]+[['prediction']+preds_all]+[['RID']+rids])
                               
                #f.close()
                if stage == 'test':
                    print('i did it')
                    accuracy, f1, sensitivity, specificity = calculate_metrics(labels_all, preds_all)
                    accu_all.append(accuracy)
                    f1_all.append(f1)
                    sens_all.append(sensitivity)
                    spec_all.append(specificity)
                    
                target_names = ['class ' + str(i) for i in range(4)]
                print('REPORTING FOR ' + stage)
                print(len(preds_all))
                report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1], target_names=target_names, zero_division=0, output_dict=True)
                print(classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1], target_names=target_names, zero_division=0))
                #plot(inputs_all, labels_all, preds_all)

                accu_list.append(report[self.eval_metric]['f1-score'])


            print(accu_all)
            print(f1_all)
            print(sens_all)
            print(spec_all)
                
        return accu_list


def mlp_main(mlp_repeat, model_name, mlp_setting):
    print('Evaluation metric: {}'.format(mlp_setting['metric']))
    for repe_idx in range(mlp_repeat):
        mlp = MLP_Wrapper(fil_num         = mlp_setting['fil_num'],
                            drop_rate       = mlp_setting['drop_rate'],
                            batch_size      = mlp_setting['batch_size'],
                            balanced        = mlp_setting['balanced'],
                            roi_threshold   = mlp_setting['roi_threshold'],
                            exp_idx         = repe_idx,
                            seed            = repe_idx,
                            model_name      = model_name,
                            lr              = mlp_setting['learning_rate'],
                            metric          = mlp_setting['metric'],
                            yr              = mlp_setting['yr'])
        mlp.train(epochs = mlp_setting['train_epochs'])
        mlp.test(repe_idx)

###PLOTTING PEFORMANCE CURVES###

##def plot(x_test, y_test, y_score):
##    average_precision = average_precision_score(y_test, y_score)
##    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
##
##    disp = plot_precision_recall_curve(x_test, y_test)
##    disp.ax_.set_title('2-class Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))    
        
##def roc_plot_perfrom_table(txt_file=None, mode=['fcn', 'fcn_gan'], fcn_repeat=5, mlp_repeat=1):
##    roc_info, pr_info = {}, {}
##    aucs, apss = {}, {}
##    dlist = ['valid', 'test'] #['test', 'AIBL', 'NACC']
##    for m in mode:
##        roc_info[m], pr_info[m], aucs[m], apss[m] = {}, {}, {}, {}
##        for ds in dlist:
##            Scores, Labels = [], []
##            for exp_idx in range(fcn_repeat):
##                for repe_idx in range(mlp_repeat):
##                    labels, scores = read_raw_score('checkpoint_dir/{}_exp{}/raw_score_{}_{}.txt'.format(m, exp_idx, ds, repe_idx))
##                    Scores.append(scores)
##                    Labels.append(labels)
##            # scores = np.array(Scores).mean(axis=0)
##            # labels = Labels[0]
##            # filename = '{}_{}_mean'.format(m, ds)
##            # with open(filename+'.csv', 'w') as f1:
##            #     wr = csv.writer(f1)
##            #     wr.writerows([[s] for s in scores])
##            # with open(filename+'_l.csv', 'w') as f2:
##            #     wr = csv.writer(f2)
##            #     wr.writerows([[l] for l in labels])
##            #     # f.write(' '.join(map(str,scores))+'\n'+' '.join(map(str,labels)))
##
##            roc_info[m][ds], aucs[m][ds] = get_roc_info(Labels, Scores)
##            pr_info[m][ds], apss[m][ds] = get_pr_info(Labels, Scores)
##
##    plt.style.mplstyle('fivethirtyeight')
##    plt.rcParams['axes.facecolor'] = 'w'
##    plt.rcParams['figure.facecolor'] = 'w'
##    plt.rcParams['savefig.facecolor'] = 'w'
##
##    # convert = {'fcn':"1.5T", 'fcn_gan':"1.5T*", 'fcn_aug':'1.5T Aug'}
##
##    # roc plot
##    fig, axes_ = plt.subplots(1, 2, figsize=[18, 6], dpi=100)
##    axes = dict(zip(dlist, axes_))
##    lines = ['-', '-.', '--']
##    hdl_crv = {m:{} for m in mode}
##    for i, ds in enumerate(dlist):
##        title = ds
##        i += 1
##        for j, m in enumerate(mode):
##            hdl_crv[m][ds] = plot_curve(curve='roc', **roc_info[m][ds], ax=axes[ds],
##                                    **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': lines[j],
##                                        'title': title})
##
##    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=roc_info, neo_lgd_hdl=None, set1=aucs[mode[0]], set2=aucs[mode[1]])
##    fig.savefig('./plot/roc.png', dpi=300)
##
##    # specificity sensitivity plot
##    fig, axes_ = plt.subplots(1, 2, figsize=[18, 6], dpi=100)
##    axes = dict(zip(dlist, axes_))
##    lines = ['-', '-.', '--']
##    hdl_crv = {m:{} for m in mode}
##    for i, ds in enumerate(dlist):
##        title = ds
##        i += 1
##        for j, m in enumerate(mode):
##            hdl_crv[m][ds] = plot_curve(curve='sp_se', **roc_info[m][ds], ax=axes[ds],
##                                    **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': lines[j],
##                                        'title': title})
##
##    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=roc_info, neo_lgd_hdl=None, set1=aucs[mode[0]], set2=aucs[mode[1]])
##    fig.savefig('./plot/ss.png', dpi=300)
##
##    # pr plot
##    fig, axes_ = plt.subplots(1, 2, figsize=[18, 6], dpi=100)
##    axes = dict(zip(dlist, axes_))
##    hdl_crv = {m: {} for m in mode}
##    for i, ds in enumerate(dlist):
##        title = ds
##        i += 1
##        for j, m in enumerate(mode):
##            hdl_crv[m][ds] = plot_curve(curve='pr', **pr_info[m][ds], ax=axes[ds],
##                                    **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': lines[j],
##                                        'title': title})
##
##    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=pr_info, neo_lgd_hdl=None, set1=apss[mode[0]], set2=apss[mode[1]])
##    fig.savefig('./plot/pr.png', dpi=300)


if __name__ == "__main__":
    print('hello')
    mlp_repeat = 25

    print('running MLP classifiers')

    mlp_config = read_json('./mlp_config.json')

    m_name = 'mica_mlp'

    # Model using only MRI scans
    mlp_main(mlp_repeat, m_name, mlp_config['mlp'])
    print('-'*100)

    filename = 'averages_csf'
    with open('mlp_averages/'+filename+'.csv','w') as f:
        wr = csv.writer(f)
        wr.writerows([['accuracy']+accu_all]+[['f1']+f1_all]+[['sensitivity']+sens_all]+[['specificity']+spec_all])
        wr.writerows([['avg_accuracy']+mean(accu_all)]+[['avg_f1']+mean(f1_all)]+[['avg_sensitivity']+mean(sens_all)]+[['avg_specificity']+mean(spec_all)])
        wr.writerows([['sd_accuracy']+sem_196(accu_all)]+[['sd_f1']+sem_196(f1_all)]+[['sd_sensitivity']+sem_196(sens_all)]+[['sd_specificity']+sem_196(spec_all)])

    sys.exit()

import numpy as np
import torch
import os, sys
import json
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from scipy import interpolate
from loss_functions import sur_loss, cox_loss_orig, \
    cox_loss_discrete
from datas import ParcellationDataCSF, ParcellationData, \
    ParcellationDataNacc, CoxData
from simple_models import _MLP, _MLP_Surv, _CNN
import logging
import shap
import pandas as pd

logging.basicConfig(filename='logs/mlp_training_scores.log',
                    level=logging.DEBUG)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'

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

class MLP_Wrapper_Meta:
    def __init__(self, exp_idx,
                 model_name, lr, weight_decay, model, model_kwargs,
                 dataset_kwargs = {},
                 criterion=cox_loss_orig, dataset=ParcellationData,
                 dataset_external=ParcellationDataNacc):
        self.seed = exp_idx
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.device = DEVICE
        self.dataset = dataset
        self.dataset_kwargs = dataset_kwargs
        self.lr = lr
        self.weight_decay = weight_decay
        self.c_index = []
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.prepare_dataloader(exp_idx)
        self.criterion = criterion
        torch.manual_seed(exp_idx)
        self.model = model(in_size=self.in_size, **model_kwargs).float()
        self.model.to(self.device)

    def save_checkpoint(self, loss):
        # if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:

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
            torch.save(self.model.state_dict(),
                       '{}{}_{}.pth'.format(
                    self.checkpoint_dir, self.model_name, self.optimal_epoch)
                       )

    def prepare_dataloader(self, seed):
        train_data = self.dataset(seed=seed, stage = 'train',
                                  **self.dataset_kwargs)
        valid_data = self.dataset(seed=seed, stage = 'valid',
                                  **self.dataset_kwargs)
        test_data = self.dataset(seed=seed, stage = 'test',
                                 **self.dataset_kwargs)
        all_data = self.dataset(seed=seed, stage='all', **self.dataset_kwargs)
        self.train_dataloader = DataLoader(train_data, batch_size=len(
                train_data), shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=len(
                valid_data), shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=len(test_data),
                                          shuffle=False)
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data),
                                         shuffle=False)
        self.in_size = train_data.data.shape[1]
        self.all_data = all_data
        self.train_data = train_data
        self.test_data = test_data

    def train(self, epochs):
        self.val_loss = []
        self.train_loss = []
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(
                0.5, 0.999), weight_decay=self.weight_decay)
        for self.epoch in range(epochs):
                self.train_model_epoch_all(self.optimizer)
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                self.val_loss.append(val_loss)
                if self.epoch % 300 == 0:
                    print('{}: {}th epoch validation score:'.format(
                            self.model_name, self.epoch),
                          '%.4f' % (val_loss))
                    logging.debug('{}: {}th epoch validation score:{:.4f}'.format(
                            self.model_name, self.epoch, val_loss))

        print('Best model saved at the {}th epoch; cox-based loss: {}'.format(
                self.optimal_epoch, self.optimal_valid_metric))
        self.optimal_path = '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)
        print('Location: '.format(self.optimal_path))
        return self.optimal_valid_metric

    def train_all(self, epochs):
        self.train_loss = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(
                0.5, 0.999), weight_decay=self.weight_decay)
        for self.epoch in range(epochs):
            self.train_model_epoch_all(self.optimizer)
            if self.epoch % 30 == 0:
                print(self.train_loss[-1])
        self.optimal_path = '{}{}_{}.pth'.format(self.checkpoint_dir,
                                                 self.model_name, self.epoch)
        torch.save(self.model.state_dict(),
                   self.optimal_path
                   )
        print('Location: '.format(self.optimal_path))

    def train_model_epoch(self, optimizer):
        self.model.train(True)
        for inputs, obss, hits in self.train_dataloader:
            if torch.sum(hits) == 0:
                continue
            self.model.zero_grad()
            preds = self.model(inputs.to(self.device))
            loss = self.criterion(preds.to('cpu'), obss, hits)
            loss.backward()
            optimizer.step()
            self.train_loss.append(loss.detach().numpy())

    def train_model_epoch_all(self, optimizer):
        self.model.train(True)
        for inputs, obss, hits in self.all_dataloader:
            if torch.sum(hits) == 0:
                continue
            self.model.zero_grad()
            preds = self.model(inputs.to(self.device))
            loss = self.criterion(preds.to('cpu'), obss, hits)
            loss.backward()
            optimizer.step()
            self.train_loss.append(loss.detach().numpy())

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            for data, obss, hits in self.valid_dataloader:
                preds = self.model(data.to(self.device))
                loss = self.criterion(preds.to('cpu'), obss,
                                      hits)
        return loss

    def plot_val_loss(self):
        title = self.model_name + ' ' + str(self.exp_idx)
        fig, ax = plt.subplots()
        ax.plot(list(range(len(self.val_loss))), self.val_loss, label=title +
                                                                      ' valid')
        ax.plot(list(range(len(self.train_loss))), self.train_loss,
                label=title + ' train')

        ax.set(xlabel='Step #', ylabel='CPH loss', title=title)
        ax.legend()
        fig.savefig(title.replace(' ','_') + '.png')

    def plot_train_loss(self):
        title = self.model_name + ' ' + str(self.exp_idx)
        fig, ax = plt.subplots()
        ax.plot(list(range(len(self.train_loss))), self.train_loss,
                label=title + ' train')

        ax.set(xlabel='Step #', ylabel='CPH loss', title=title)
        ax.legend()
        fig.savefig(title.replace(' ','_') + '.png')

    def test_data_optimal_epoch(self, external_data=False, return_preds=False):
        if external_data:
            dataset = ParcellationDataNacc(self.seed, stage='all')
            dataloader = DataLoader(dataset, batch_size=len(dataset),
                                     shuffle=False)
        else:
            dataloader = self.test_dataloader
            dataset = self.test_data
        with torch.no_grad():
            self.model.train(False)
            for data, obss, hits in dataloader:
                preds = self.model(data.to(self.device))
        c_index = concordance_index_censored(hits.numpy() == 1, obss.numpy(),
                  preds.to('cpu').numpy().squeeze())
        if return_preds:
            return c_index, preds.to('cpu').numpy().squeeze(), dataset.fileIDs
        return c_index[0]


    def test_surv_data_optimal_epoch(self, bins, concordance_time=24,
                                     external_data=False, return_preds = False):
        if external_data:
            dataset = ParcellationDataNacc(self.seed, stage='all')
            dataloader = DataLoader(dataset, batch_size=len(dataset),
                                     shuffle=False)
        else:
            dataset = self.test_data
            dataloader = self.test_dataloader
        with torch.no_grad():
            self.model.train(False)
            for data, obss, hits in dataloader:
                preds = self.model(data.to(self.device)).to('cpu')
        preds_raw = np.concatenate((np.ones((preds.shape[0],1)),
                          np.cumprod(preds.numpy(), axis=1)), axis=1)
        interp = interpolate.interp1d(bins, preds_raw, axis=-1,
                                      kind='quadratic')
        preds = interp(concordance_time)
        c_index = concordance_index_censored(hits.numpy() == 1, obss.numpy(),
                                             -preds)
        if return_preds:
            return c_index[0], preds_raw, dataset.fileIDs, interp
        return c_index[0]


class CoxLossUtility:
    def __init__(self, model_name, mlp_repeat=5,
                 fname='test_model.txt'):
        self._json_props = read_json(
                'mlp_config.json')
        self.mlp_setting = self._json_props[model_name]
        self.model_name = model_name
        self.c_index = []
        self.fname = fname
        self.mlp_repeat = mlp_repeat
        self.mlps = self._create_mlps()

    def plot_val_loss(self):
        for mlp in self.mlps:
            mlp.plot_val_loss()

    def train(self):
        for mlp in self.mlps:
            mlp.train(epochs=self.mlp_setting['train_epochs'])

    def get_c_index(self):
        for mlp in self.mlps:
            self.c_index.append(mlp.test_data_optimal_epoch())
            write_c_index(self.c_index[-1], self.model_name, self.fname,
                          mlp.optimal_path)
            print(f'C-index: {self.c_index[-1]}')
        print(self.c_index)
        print(f'{np.mean(self.c_index)} +/- {np.std(self.c_index)}')
        with open(self.fname, 'a') as fi:
            fi.write(f'\nC-idx avg: {np.mean(self.c_index)} +/-'
                     f' {np.std(self.c_index)}\n\n')

    def get_c_index_external(self):
        preds = np.asarray([])
        labels = np.asarray([])
        exp_no = np.asarray([])
        for idx, mlp in enumerate(self.mlps):
            c_idx, pred, label = mlp.test_data_optimal_epoch(
                    external_data=True, return_preds=True)
            self.c_index.append(c_idx)
            preds = np.concatenate([preds, pred], axis=0)
            labels = np.concatenate([labels, label], axis=0)
            exp_no = np.concatenate([exp_no, np.ones(pred.shape)*idx])
            write_c_index(c_idx, self.model_name, self.fname,
                          mlp.optimal_path)
            print(f'C-index: {self.c_index[-1]}')
        write_preds_labels(preds, labels, exp_no, self.model_name)

    def _create_mlps(self):
        mlps = []
        for repe_idx in range(self.mlp_repeat):
            mlps.append(
                MLP_Wrapper_Meta(
                   repe_idx,
                   model_name=self.model_name,
                   lr=self.mlp_setting['learning_rate'],
                   weight_decay=self.mlp_setting['weight_decay'],
                   model=eval(self.mlp_setting['model']),
                   model_kwargs={
                       'fil_num'  : self.mlp_setting["fil_num"],
                       'drop_rate': self.mlp_setting["drop_rate"]
                   },
                   criterion=eval(self.mlp_setting['criterion']),
                   dataset=eval(self.mlp_setting['dataset']),
                   dataset_external=eval(self.mlp_setting['dataset_external'])
                   )
            )
        return mlps

class SurLossUtility:
    def __init__(self, model_name="mlp_csf_sur_loss", mlp_repeat=1,
                 fname='test_model.txt'):
        self._json_props = read_json(
                'mlp_config.json')
        self.mlp_setting = self._json_props[model_name]
        self.model_name = model_name
        self.bins = self.mlp_setting["bins"]
        if len(self.bins) == 0:
            self._compute_bins()
        self.c_index = []
        self.fname = fname
        self.mlp_repeat = mlp_repeat
        self.mlps = self._create_mlps()

    def _compute_bins(self):
        metadata_csv = pd.read_csv(self._json_props['datadir'] +
                                   self._json_props['metadata_fi'])
        time_to_progress_quantiles = metadata_csv.query('PROGRESSES == 1')[
            'TIME_TO_PROGRESSION'].quantile([0.25, 0.5, 0.75, 1],
                                         interpolation='higher')
        self.bins = np.concatenate(
                [np.asarray([0]), time_to_progress_quantiles.to_numpy()])
        print(self.bins)

    def plot_val_loss(self):
        for mlp in self.mlps:
            mlp.plot_val_loss()

    def train(self):
        for mlp in self.mlps:
            mlp.train(epochs=self.mlp_setting['train_epochs'])

    def get_c_index(self, concordance_time=24):
        for mlp in self.mlps:
            self.c_index.append(mlp.test_surv_data_optimal_epoch(self.bins,
                                                                 concordance_time))
            write_c_index(self.c_index[-1], self.model_name, self.fname,
                              mlp.optimal_path)
            print(f'C-index: {self.c_index[-1]}')
        print(self.c_index)
        print(f'{np.mean(self.c_index)} +/- {np.std(self.c_index)}')
        with open(self.fname, 'a') as fi:
            fi.write(f'C-idx avg: {np.mean(self.c_index)} +/-'
                     f' {np.std(self.c_index)}\n\n')

    def get_c_index_external(self, concordance_time=24):
        preds = []
        labels = []
        exp_no = []
        for idx, mlp in enumerate(self.mlps):
            c_idx, pred, label, model = mlp.test_surv_data_optimal_epoch(
                    self.bins, concordance_time,
                    external_data=True, return_preds=True)
            self.c_index.append(c_idx)
            preds.append(pred)
            labels.append(label)
            exp_no.append(np.ones(pred.shape)*idx)
            write_c_index(c_idx, self.model_name, self.fname + '_NACC',
                          mlp.optimal_path)
            print(f'C-index: {self.c_index[-1]}')
        print(self.c_index)
        print(f'{np.mean(self.c_index)} +/- {np.std(self.c_index)}')
        with open(self.fname, 'a') as fi:
            fi.write(f'C-idx avg: {np.mean(self.c_index)} +/-'
                     f' {np.std(self.c_index)}\n\n')
        labels = np.concatenate(labels, axis=0).reshape((-1,1))
        preds = np.concatenate(preds, axis=0)
        exp_no = np.asarray(exp_no)
        preds, labels, exp_no, bins = _format_labels(
                preds, labels, exp_no, self.bins.reshape((1,-1)))
        write_preds_labels(preds, labels, exp_no, self.model_name, bins=bins)

    def _create_mlps(self):
        mlps = []
        for repe_idx in range(self.mlp_repeat):
            mlps.append(
                    MLP_Wrapper_Meta(
                           repe_idx,
                           model_name=self.model_name,
                           lr=self.mlp_setting['learning_rate'],
                           weight_decay=self.mlp_setting['weight_decay'],
                           model=eval(self.mlp_setting['model']),
                           model_kwargs={
                                   'fil_num'  : self.mlp_setting["fil_num"],
                                   'drop_rate': self.mlp_setting["drop_rate"],
                                   'output_shape': len(self.bins)-1
                           },
                           criterion=lambda x,y,z: eval(self.mlp_setting[
                               'criterion'])(x,y,z,torch.Tensor([self.bins])),
                           dataset=eval(self.mlp_setting['dataset']),
                           dataset_external=eval(self.mlp_setting[
                                                   'dataset_external'])
                           )
            )
        return mlps

def write_c_index(c_index_list, curr_model_name, fname, pth):
    with open(fname, 'a') as fi:
        fi.write(curr_model_name + '\n')
        fi.write('Optimal path: ' + pth + '\n')
        if type(c_index_list) == tuple:
            fi.write(', '.join(str(x) for x in c_index_list) + '\n')
        else:
            fi.write(f'C index = {str(c_index_list)}' + '\n')
	
def _format_labels(predictions, labels, exp_no, bins):
    labels = np.repeat(labels, predictions.shape[1], axis=1)
    bins = np.repeat(bins, predictions.shape[0], axis=0)
    predictions = np.reshape(predictions, (-1, 1), order='C').squeeze()
    labels = np.reshape(labels, (-1,1), order='C').squeeze()
    exp_no = np.reshape(exp_no, (-1,1), order='C').squeeze()
    bins = np.reshape(bins, (-1, 1), order='C').squeeze()
    return predictions, labels, exp_no, bins

def write_preds_labels(preds, labels, exp_no, fname, **kwargs):
    _json_props = read_json(
            'mlp_config.json')
    df = pd.DataFrame(data={'Predictions': preds, 'Experiment':
        exp_no.astype(int), **kwargs},
                      index=labels)
    df.to_csv(_json_props['data_dir'] + fname + '.csv', index_label='RID')

def main():
    sur = SurLossUtility('mlp_parcellation_sur_loss')
    sur.train()
    # sur.get_c_index()
    # sur.get_c_index_external()
    # sur = SurLossUtility('mlp_csf_sur_loss')
    # sur.train()
    # sur.get_c_index()

    w = sur.mlps[0]
    
    for inputs, obss, hits in w.train_dataloader:
        print('producing shap plots')
        # inputs = pd.DataFrame(inputs.numpy(), columns = ["abeta", "tau","ptau"])
        shap_dir = 'shap/'

        background = inputs[:inputs.shape[0]//2] #does enpty entry as background work?
        test = inputs[inputs.shape[0]//2:]
        e = shap.DeepExplainer(w.model, background)
        shap_values = e.shap_values(test)
        contri = np.mean(np.abs(np.array(shap_values)), axis=1)
        # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        # test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

        # shap.image_plot(shap_values, test.numpy())
        # shap.image_plot(shap_numpy, -test_numpy)
        fig, ax = plt.subplots()
        for i, c in enumerate(contri):
            if sur.model_name == 'mlp_csf_sur_loss':
                xs = ["abeta", "tau","ptau"]
                ax.bar(xs, c, color='b')
            else:
                # xs = list('feature '+str(i) for i in range(inputs.shape[1]))
                xs = np.array([str(i) for i in range(inputs.shape[1])])
                
                c_arg = np.argsort(-c)
                xs = xs[c_arg]
                c = c[c_arg]
                k = 10
                
                ax.bar(xs[:k], c[:k], color='b')
            ax.set_title('contribution for class '+str(i))
            plt.savefig(shap_dir+sur.model_name+'_shap_'+str(i)+'.jpg')
            plt.cla()
            
    print('OK')
    
main()

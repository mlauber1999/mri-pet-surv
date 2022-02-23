# main file for vision transformer
# Created: 6/16/2021
# Status: ok
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python vit_main.py

import sys
import torch

import numpy as np
import pandas as pd

from networks import ViT_Wrapper
from utils import read_json
from packaging import version

if version.parse(torch.__version__) >= version.parse("1.8.0"):
    torch.use_deterministic_algorithms(True)
else:
    torch.set_deterministic(True)



def ViT(num_exps, model_name, config, Wrapper):
    print('Evaluation metric: {}'.format(config['metric']))
    c_te, c_tr, c_ex = [], [], []
    b_te, b_tr, b_ex = [], [], []
    for exp_idx in range(num_exps):
        print('*'*50)
        vit = Wrapper(config          = config,
                      exp_idx         = exp_idx,
                      num_fold        = num_exps,
                      seed            = 1000*exp_idx,
                      model_name      = model_name)
        # vit.load('./checkpoint_dir/{}_exp{}/'.format('pre_Linear', 0), fixed=False)
        # sys.exit()
        vit.train(epochs = config['train_epochs'])
        # cnn.check('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0))
        out = vit.concord(all=True)
        c_te += [out[0][0][0]]
        c_tr += [out[0][2][0]]
        c_ex += [out[0][3][0]]
        b_te += [out[1][0]]
        b_tr += [out[1][2]]
        b_ex += [out[1][3]]
        # cnn.shap()
        # print(c_te)
        # print(c_tr)
        # print('exit')
    print('CI test: %.3f+-%.3f' % (np.mean(c_te), np.std(c_te)))
    print('CI train: %.3f+-%.3f' % (np.mean(c_tr), np.std(c_tr)))
    print('CI external: %.3f+-%.3f' % (np.mean(c_ex), np.std(c_ex)))
    print('BS test: %.3f+-%.3f' % (np.mean(b_te), np.std(b_te)))
    print('BS train: %.3f+-%.3f' % (np.mean(b_tr), np.std(b_tr)))
    print('BS external: %.3f+-%.3f' % (np.mean(b_ex), np.std(b_ex)))

def ViT_dfs(num_exps, model_name, config, Wrapper):
    print('Evaluation metric: {}'.format(config['metric']))
    dfss = []
    for exp_idx in range(num_exps):
        print('*'*50)
        vit = Wrapper(config          = config,
                      exp_idx         = exp_idx,
                      num_fold        = num_exps,
                      seed            = 1000*exp_idx,
                      model_name      = model_name)
        dfss += [vit.overlay_prepare(load=True)]
    df_adni = dfss[0][0]
    df_nacc = dfss[0][1]
    for i in range(1, 5):
        df_adni = df_adni.append(dfss[i][0], ignore_index=True)
        df_nacc = df_nacc.append(dfss[i][1], ignore_index=True)
    df_nacc = df_nacc.groupby(['RID', 'Dataset']).mean().reset_index()
    # print(df_adni.sort_values(by=['RID'], ignore_index=True))
    # sys.exit()
    df = df_adni.append(df_nacc, ignore_index=True)
    df.to_csv('SViT.csv')
    

def ViT_dfs_raw(num_exps, model_name, config, Wrapper):
    print('Evaluation metric: {}'.format(config['metric']))
    dfss = []
    exp_idxx = {'ADNI': [], 'NACC': []}
    for exp_idx in range(num_exps):
        print('*'*50)
        vit = Wrapper(config          = config,
                      exp_idx         = exp_idx,
                      num_fold        = num_exps,
                      seed            = 1000*exp_idx,
                      model_name      = model_name)
        df = vit.overlay_prepare(load=True)
        dfss += [df]
        exp_idxx['ADNI'] += [[1*exp_idx]*len(df[0])]
        exp_idxx['NACC'] += [[1*exp_idx]*len(df[1])]
    df_adni = dfss[0][0]
    df_adni['Exp'] = exp_idxx['ADNI'][0]
    df_nacc = dfss[0][1]
    df_nacc['Exp'] = exp_idxx['NACC'][0]
    for i in range(1, 5):
        dfss_adni = dfss[i][0].copy()
        dfss_adni['Exp'] = exp_idxx['ADNI'][i]
        dfss_nacc = dfss[i][1].copy()
        dfss_nacc['Exp'] = exp_idxx['NACC'][i]
        df_adni = df_adni.append(dfss_adni, ignore_index=True)
        df_nacc = df_nacc.append(dfss_nacc, ignore_index=True)
    
    # print(df_adni.sort_values(by=['RID'], ignore_index=True))
    # sys.exit()
    df = df_adni.append(df_nacc, ignore_index=True)
    df.to_csv('SViT_raw.csv')

def ViT_dfs_raw_train(num_exps, model_name, config, Wrapper):
    print('Evaluation metric: {}'.format(config['metric']))
    dfss = []
    exp_idxx = {'ADNI': [], 'NACC': [], 'ADNI_train': []}
    for exp_idx in range(num_exps):
        print('*'*50)
        vit = Wrapper(config          = config,
                      exp_idx         = exp_idx,
                      num_fold        = num_exps,
                      seed            = 1000*exp_idx,
                      model_name      = model_name)
        df = vit.overlay_prepare_with_train(load=True)
        dfss += [df]
        exp_idxx['ADNI'] += [[1*exp_idx]*len(df[1])]
        exp_idxx['NACC'] += [[1*exp_idx]*len(df[2])]
        exp_idxx['ADNI_train'] += [[1*exp_idx]*len(df[0])]
    
    df_adni_train = dfss[0][0]
    df_adni_train['Exp'] = exp_idxx['ADNI_train'][0]
    df_adni = dfss[0][1]
    df_adni['Exp'] = exp_idxx['ADNI'][0]
    df_nacc = dfss[0][2]
    df_nacc['Exp'] = exp_idxx['NACC'][0]
    for i in range(1, 5):
        dfss_adni_train = dfss[i][0].copy()
        dfss_adni_train['Exp'] = exp_idxx['ADNI_train'][i]
        dfss_adni = dfss[i][1].copy()
        dfss_adni['Exp'] = exp_idxx['ADNI'][i]
        dfss_nacc = dfss[i][2].copy()
        dfss_nacc['Exp'] = exp_idxx['NACC'][i]
        df_adni = df_adni.append(dfss_adni, ignore_index=True)
        df_nacc = df_nacc.append(dfss_nacc, ignore_index=True)
        df_adni_train = df_adni_train.append(dfss_adni_train, ignore_index=True)
    
    # print(df_adni.sort_values(by=['RID'], ignore_index=True))
    # sys.exit()
    df = df_adni.append(df_nacc, ignore_index=True).append(df_adni_train, ignore_index=True)
    df.to_csv('SViT_raw.csv')
    return df

def _validate_df(df):
    for exp in range(5):
        train_truth = df.query('Exp == @exp and Dataset == \'ADNI_train\'').copy()[['RID','TIMES','PROGRESSES']]
        other_exp = np.setdiff1d(list(range(5)),[exp])
        train_test = df.query('Exp in @other_exp and Dataset == \'ADNI\'').copy()[['RID','TIMES','PROGRESSES']]
        train_truth = train_truth.sort_values('RID').reset_index(drop=True)
        train_test = train_test.sort_values('RID').reset_index(drop=True)

def main(run=False):
    print('-'*100)
    print('Running vision transformer (ViT)')
    config = read_json('./config.json')['vit']
    num_exps = 5
    model_name = 'vit_{}'.format(config['mapping'])
    if run:
        ViT(num_exps, model_name, config, ViT_Wrapper)
    # ViT_dfs(num_exps, model_name, config, ViT_Wrapper)
    ViT_dfs_raw_train(num_exps, model_name, config, ViT_Wrapper)
    print('-'*100)
    print('OK')

def train():
    print('-'*100)
    print('Running vision transformer (ViT)')
    config = read_json('./config.json')['vit']
    num_exps = 5
    model_name = 'vit_{}'.format(config['mapping'])
    # ViT_dfs(num_exps, model_name, config, ViT_Wrapper)
    ViT_dfs_raw_train(num_exps, model_name, config, ViT_Wrapper)
    print('-'*100)
    print('OK')

if __name__ == "__main__":
    #main()
    train()
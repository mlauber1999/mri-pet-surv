# main file for vision transformer
# Created: 6/16/2021
# Status: ok
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python pre_main.py

import sys
import torch

import numpy as np

from networks import Pre_ViT_Wrapper
from utils import read_json


def ViT(num_exps, model_name, config, Wrapper):
    print('Evaluation metric: {}'.format(config['metric']))
    for exp_idx in range(num_exps):
        print('*'*50)
        vit = Wrapper(config          = config,
                      exp_idx         = exp_idx,
                      num_fold        = num_exps,
                      seed            = 1000*exp_idx,
                      model_name      = model_name)
        vit.train(epochs = config['train_epochs'])
        vit.test()

def main():
    torch.use_deterministic_algorithms(True)
    print('-'*100)
    print('Pre training model for ViT')
    config = read_json('./config.json')['pre']
    num_exps = 1
    model_name = 'pre_{}'.format(config['mapping'])
    print(model_name)
    ViT(num_exps, model_name, config, Pre_ViT_Wrapper)
    print('-'*100)
    print('OK')


if __name__ == "__main__":
    main()

# classifier baselines: applying the previous cnn (vanilla cnn, with our tuned parameters) for baseline estimation
import torch.nn as nn
import sys
sys.path.insert(1, './plot/')
from plot import report_table,roc_plot_perfrom_table
from networks import CNN_Wrapper, FCN_Wrapper, MLP_Wrapper
from utils import read_json, cross_set

import torch
torch.set_deterministic(True)


def cnn_main(repe_time, model_name, cnn_setting):
    for exp_idx in range(repe_time):
        cnn = CNN_Wrapper(fil_num        = cnn_setting['fil_num'],
                         drop_rate       = cnn_setting['drop_rate'],
                         batch_size      = cnn_setting['batch_size'],
                         balanced        = cnn_setting['balanced'],
                         Data_dir        = cnn_setting['Data_dir'],
                         exp_idx         = exp_idx,
                         seed            = 1000,
                         model_name      = model_name,
                         metric          = 'accuracy')
        cnn.train(lr     = cnn_setting['learning_rate'],
                  epochs = cnn_setting['train_epochs'])
        cnn.test()

def fcn_main(fcn_repeat, model_name, augment, fcn_setting):
    print('Evaluation metric: {}'.format(fcn_setting['metric']))
    for exp_idx in range(fcn_repeat):
        fcn = FCN_Wrapper(fil_num        = fcn_setting['fil_num'],
                        drop_rate       = fcn_setting['drop_rate'],
                        batch_size      = fcn_setting['batch_size'],
                        balanced        = fcn_setting['balanced'],
                        Data_dir        = fcn_setting['Data_dir'],
                        patch_size      = fcn_setting['patch_size'],
                        lr              = fcn_setting['learning_rate'],
                        exp_idx         = exp_idx,
                        num_fold        = fcn_repeat,
                        seed            = 1000,
                        model_name      = model_name,
                        metric          = fcn_setting['metric'],
                        augment         = augment,
                        dim             = fcn_setting['input_dim'],
                        yr              = fcn_setting['yr'])
        # fcn.train(epochs = fcn_setting['train_epochs'])
        # fcn.test_and_generate_DPMs(epoch=640)
        # fcn.test_and_generate_DPMs(single_dim=False)
        fcn.test_and_generate_DPMs(root = '/data2/MRI_PET_DATA/processed_images_final_cumulative/')
        # plot_heatmap('/home/sq/gan2020/DPMs/fcn_exp', 'fcn_heatmap', exp_idx=exp_idx, figsize=(9, 4))

def mlp_main(fcn_repeat, mlp_repeat, model_name, mode, mlp_setting):
    print('Evaluation metric: {}'.format(mlp_setting['metric']))
    for exp_idx in range(fcn_repeat):
        for repe_idx in range(mlp_repeat):
            mlp = MLP_Wrapper(fil_num         = mlp_setting['fil_num'],
                                drop_rate       = mlp_setting['drop_rate'],
                                batch_size      = mlp_setting['batch_size'],
                                balanced        = mlp_setting['balanced'],
                                roi_threshold   = mlp_setting['roi_threshold'],
                                exp_idx         = exp_idx,
                                seed            = repe_idx*exp_idx,
                                mode            = mode,
                                model_name      = model_name,
                                lr              = mlp_setting['learning_rate'],
                                metric          = mlp_setting['metric'],
                                yr              = mlp_setting['yr'])
            mlp.train(epochs = mlp_setting['train_epochs'])
            mlp.test(repe_idx)


if __name__ == "__main__":
    mlp_repeat = 5
    fcn_repeat = 5
    print('running FCN classifiers')

    cnn_config = read_json('./cnn_config.json')
    # cnn_main(repeat, 'cnn', cnn_config['cnn'])

    # # Testing model using only MRI scans
    # fcn_main(fcn_repeat, 'fcn_mri_test', False, cnn_config['fcn_test'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_test_mlp', 'mri_test_', cnn_config['mlp'])

    fcn_names = ['fcn_mri', 'fcn_amyloid', 'fcn_fdg']
    mlp_names = ['fcn_mri_mlp', 'fcn_amyloid_mlp', 'fcn_fdg_mlp']
    scan_names = ['mri', 'amyloid', 'fdg']
    modes = []

    print('-'*100)
    for f_name, m_name, s_name in zip(fcn_names, mlp_names, scan_names):
        modes += [m_name]
        # Model using only MRI scans
        fcn_main(fcn_repeat, f_name, False, cnn_config['fcn'])
        # mlp_main(fcn_repeat, mlp_repeat, m_name, s_name, cnn_config['mlp'])
        print('-'*100)

    # # Model using both MRI & amyloid PET scans
    # fcn_main(fcn_repeat, 'fcn_mri_amyloid', False, cnn_config['fcn_dual'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_amyloid_mlp', 'mri_amyloid_', cnn_config['mlp'])
    #
    # # Model using both MRI & fdg PET scans
    # fcn_main(fcn_repeat, 'fcn_mri_fdg', False, cnn_config['fcn_dual'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_fdg_mlp', 'mri_fdg_', cnn_config['mlp'])
    #
    # # Model using both anyloid & fdg PET scans
    # fcn_main(fcn_repeat, 'fcn_amyloid_fdg', False, cnn_config['fcn_dual'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_amyloid_fdg_mlp', 'amyloid_fdg_', cnn_config['mlp'])
    #
    # # Model using both MRI & amyloid & fdg scans
    # fcn_main(fcn_repeat, 'fcn_mri_amyloid_fdg', False, cnn_config['fcn_tri'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_amyloid_fdg_mlp', 'mri_amyloid_fdg_', cnn_config['mlp'])


    # mlp_names = ['fcn_mri_mlp', 'fcn_amyloid_mlp']
    # mlp_names = ['fcn_mri_mlp']
    # mlp_names = ['fcn_mri_test_mlp']
    # mlp_names = ['fcn_mri_mlp', 'fcn_amyloid_mlp', 'fcn_fdg_mlp', 'fcn_mri_amyloid_mlp', 'fcn_mri_fdg_mlp', 'fcn_amyloid_fdg_mlp', 'fcn_mri_amyloid_fdg_mlp']
    report_table(mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    # roc_plot_perfrom_table(mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    # cross_set(mode=modes, fcn_repeat=mlp_repeat, mlp_repeat=fcn_repeat)

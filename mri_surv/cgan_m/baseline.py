# classifier baselines: applying the previous cnn (vanilla cnn, with our tuned parameters) for baseline estimation
import torch.nn as nn
import sys
sys.path.insert(1, './plot/')
from plot import report_table,roc_plot_perfrom_table
from networks import CNN_Wrapper, FCN_Wrapper, MLP_Wrapper, FCN_Cox_Wrapper, MLP_Cox_Wrapper, AE_Cox_Wrapper, CNN_Cox_Wrapper, CNN_Cox_Wrapper_Pre
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

def fcn_main(fcn_repeat, model_name, augment, fcn_setting, Wrapper):
    print('Evaluation metric: {}'.format(fcn_setting['metric']))
    for exp_idx in range(fcn_repeat):
        fcn = Wrapper(fil_num        = fcn_setting['fil_num'],
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
        fcn.train(epochs = fcn_setting['train_epochs'])
        # fcn.test_and_generate_DPMs(epoch=640)
        # fcn.test_and_generate_DPMs(single_dim=False)
        # fcn.test_and_generate_DPMs(root = '/data2/MRI_PET_DATA/processed_images_final_cumulative/')
        fcn.test_and_generate_DPMs(root = '/data2/MRI_PET_DATA/processed_images_final_cumulative/Cox/')
        # plot_heatmap('/home/sq/gan2020/DPMs/fcn_exp', 'fcn_heatmap', exp_idx=exp_idx, figsize=(9, 4))

def ae_main(fcn_repeat, model_name, augment, fcn_setting, Wrapper):
    print('Evaluation metric: {}'.format(fcn_setting['metric']))
    for exp_idx in range(fcn_repeat):
        fcn = Wrapper(fil_num        = fcn_setting['fil_num'],
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
        fcn.train(epochs = fcn_setting['train_epochs'])
        # fcn.test_and_generate_DPMs(epoch=640)
        # fcn.test_and_generate_DPMs(single_dim=False)
        # fcn.test_and_generate_DPMs(root = '/data2/MRI_PET_DATA/processed_images_final_cumulative/')
        fcn.test_and_generate_DPMs(root = '/data2/MRI_PET_DATA/processed_images_final_cumulative/Cox/', single_dim=False)
        # plot_heatmap('/home/sq/gan2020/DPMs/fcn_exp', 'fcn_heatmap', exp_idx=exp_idx, figsize=(9, 4))

def cnn_cox_main(fcn_repeat, model_name, fcn_setting, Wrapper):
    print('Evaluation metric: {}'.format(fcn_setting['metric']))
    for exp_idx in range(fcn_repeat):
        cnn = Wrapper(fil_num        = fcn_setting['fil_num'],
                        drop_rate       = fcn_setting['drop_rate'],
                        batch_size      = fcn_setting['batch_size'],
                        balanced        = fcn_setting['balanced'],
                        Data_dir        = fcn_setting['Data_dir'],
                        patch_size      = fcn_setting['patch_size'],
                        lr              = fcn_setting['learning_rate'],
                        exp_idx         = exp_idx,
                        num_fold        = fcn_repeat,
                        seed            = 1000*exp_idx,
                        # seed            = 1000,
                        model_name      = model_name,
                        metric          = fcn_setting['metric'],
                        dim             = fcn_setting['input_dim'],
                        yr              = fcn_setting['yr'])
        cnn.train(epochs = fcn_setting['train_epochs'])
        # cnn.test()
        cnn.concord()
        # cnn.predict_plot()
        # cnn.predict_plot_general()
        # cnn.predict_plot_scatter()

        # plot_heatmap('/home/sq/gan2020/DPMs/fcn_exp', 'fcn_heatmap', exp_idx=exp_idx, figsize=(9, 4))

def cnn_pre_cox_main(fcn_repeat, model_name, fcn_setting, Wrapper):
    print('Evaluation metric: {}'.format(fcn_setting['metric']))
    for exp_idx in range(fcn_repeat):
        cnn = Wrapper(fil_num        = fcn_setting['fil_num'],
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
                        dim             = fcn_setting['input_dim'],
                        yr              = fcn_setting['yr'])
        # cnn.train(epochs = fcn_setting['train_epochs'])
        cnn.concord()
        # cnn.predict_plot()
        # cnn.predict_plot_general()
        # cnn.predict_plot_scatter()


def mlp_main(fcn_repeat, mlp_repeat, model_name, mode, mlp_setting, MWrapper):
    print('Evaluation metric: {}'.format(mlp_setting['metric']))
    for exp_idx in range(fcn_repeat):
        for repe_idx in range(mlp_repeat):
            mlp = MWrapper(fil_num         = mlp_setting['fil_num'],
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
            mlp.predict_plot(repe_idx)
            mlp.predict_plot_general(repe_idx)
            # mlp.test(repe_idx)


if __name__ == "__main__":
    config = read_json('./cnn_config.json')

    mlp_repeat = 1
    fcn_repeat = 3
    cox=True
    if cox:
        FWrapper = FCN_Cox_Wrapper
        AWrapper = AE_Cox_Wrapper
        CWrapper = CNN_Cox_Wrapper
        PWrapper = CNN_Cox_Wrapper_Pre
        MWrapper = MLP_Cox_Wrapper
        fcn_config = config['fcn_cox']
        ae_config = config['ae_cox']
        cnn_config = config['cnn_cox']
        cnn_pre_config = config['cnn_cox_pre']
        mlp_config = config['mlp_cox']
    else:
        FWrapper = FCN_Wrapper
        MWrapper = MLP_Wrapper
        fcn_config = config['fcn']
        mlp_config = config['mlp']
    print('running FCN classifiers')

    # cnn_main(repeat, 'cnn', config['cnn'])

    # # Testing model using only MRI scans
    # fcn_main(fcn_repeat, 'fcn_mri_test', False, config['fcn_test'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_test_mlp', 'mri_test_', config['mlp'])

    fcn_names = ['fcn_mri', 'fcn_amyloid', 'fcn_fdg']
    mlp_names = ['fcn_mri_mlp', 'fcn_amyloid_mlp', 'fcn_fdg_mlp']
    scan_names = ['mri', 'amyloid', 'fdg']
    modes = []

    print('-'*100)
    for f_name, m_name, s_name in zip(fcn_names, mlp_names, scan_names):
        modes += [m_name]
        # Model using only MRI scans
        # fcn_main(fcn_repeat, f_name, False, fcn_config, Wrapper=FWrapper)
        # ae_main(fcn_repeat, f_name, False, ae_config, Wrapper=AWrapper)
        # cnn_pre_cox_main(fcn_repeat, f_name, cnn_pre_config, Wrapper=PWrapper)
        cnn_cox_main(fcn_repeat, f_name, cnn_config, Wrapper=CWrapper)
        # if cox:
            # mlp_main(fcn_repeat, mlp_repeat, m_name, [s_name], mlp_config, MWrapper=MWrapper) #for mri only now
        # else:
        #     mlp_main(fcn_repeat, mlp_repeat, m_name, s_name, mlp_config, MWrapper=MWrapper)
        print('-'*100)
        break
    sys.exit()
    print(modes)
    # # Model using both MRI & amyloid PET scans
    # fcn_main(fcn_repeat, 'fcn_mri_amyloid', False, config['fcn_dual'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_amyloid_mlp', 'mri_amyloid_', config['mlp'])
    #
    # # Model using both MRI & fdg PET scans
    # fcn_main(fcn_repeat, 'fcn_mri_fdg', False, config['fcn_dual'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_fdg_mlp', 'mri_fdg_', config['mlp'])
    #
    # # Model using both anyloid & fdg PET scans
    # fcn_main(fcn_repeat, 'fcn_amyloid_fdg', False, config['fcn_dual'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_amyloid_fdg_mlp', 'amyloid_fdg_', config['mlp'])
    #
    # # Model using both MRI & amyloid & fdg scans
    # fcn_main(fcn_repeat, 'fcn_mri_amyloid_fdg', False, config['fcn_tri'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_amyloid_fdg_mlp', 'mri_amyloid_fdg_', config['mlp'])


    # mlp_names = ['fcn_mri_mlp', 'fcn_amyloid_mlp']
    # mlp_names = ['fcn_mri_mlp']
    # mlp_names = ['fcn_mri_test_mlp']
    # mlp_names = ['fcn_mri_mlp', 'fcn_amyloid_mlp', 'fcn_fdg_mlp', 'fcn_mri_amyloid_mlp', 'fcn_mri_fdg_mlp', 'fcn_amyloid_fdg_mlp', 'fcn_mri_amyloid_fdg_mlp']
    if cox:
        report_table(txt_file='report.txt', mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat, out=2)
    else:
        report_table(txt_file='report.txt', mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    # roc_plot_perfrom_table(mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    # cross_set(mode=modes, fcn_repeat=mlp_repeat, mlp_repeat=fcn_repeat)

# Main file:
#   Train the conditional GAN
#   Train 2 classifiers with original data and generated data
#   Report results


from baseline import fcn_main, mlp_main, read_json
from networks import FCN_GAN
from plot import report_table, roc_plot_perfrom_table
import os


def gan_main():
    # epoch = 5555 for 2000, 3-1-1
    # 2000*0.8*417/(151*0.6)
    # epoch = 7413 for 2000, 4-1-0
    gan = FCN_GAN('./gan_config_optimal.json', 0)
    gan.train()
    # gan.generate(dataset_name=['ADNI'], epoch=8000, visualize=True)
    # gan.epoch = 800
    # gan.plot_MRI(epoch=800)

    # gan.generate_DPMs(epoch=3780)
    # gan.fcn.test_and_generate_DPMs(stages=['AIBL'])

    return gan

if __name__ == "__main__":
    repeat = 5

    cnn_config = read_json('./cnn_config.json')

    # # Model using only MRI scans
    # fcn_main(1, 'fcn_mri', False, cnn_config['fcn_mri'])
    # mlp_main(1, 5, 'fcn_mri_mlp', 'mri_', cnn_config['mlp'])
    #
    # # Model using only PET scans
    # fcn_main(1, 'fcn_pet', False, cnn_config['fcn_pet'])
    # mlp_main(1, 5, 'fcn_pet_mlp', 'pet_', cnn_config['mlp'])
    #
    # # Model using both MRI & PET scans
    # fcn_main(1, 'fcn_dual', False, cnn_config['fcn_dual'])
    # mlp_main(1, 5, 'fcn_dual_mlp', 'dual_', cnn_config['mlp'])

    # cnn_main(repeat, 'cnn', cnn_config['cnn'])
    # fcn_main(1, 'fcn', False, cnn_config['fcn'])
    # fcn_main(5, 'fcn', False, cnn_config['fcn'])
    # mlp_main(1, 5, 'fcn_mlp', '', cnn_config['mlp']) # train 5 mlp models with random seeds on generated DPMs from FCN-GAN
    print('baseline OK.')

    gan = gan_main()       # train FCN-GAN; generate images*; generate DPMs for mlp and plot MCC heatmap
    print('GAN OK.')
    # fcn_main(1, 'fcn_gan', False, cnn_config['fcn_gan'])
    # mlp_main(1, 5, 'fcn_gan_mlp', 'gan_', cnn_config['mlp'])
    # fcn_main(1, 'fcn_gan_dual', False, cnn_config['fcn_gan_dual'])
    # mlp_main(1, 5, 'fcn_gan_dual_mlp', 'gan_dual_', cnn_config['mlp'])
    #       (exp_time, repe_time, model_name, mode, mlp_setting)

    # gan.eval_iqa_orig()
    # gan.eval_iqa_gene(epoch=390)
    # gan.eval_iqa_orig(names=['valid'])
    # get_best()
    # train_plot(gan.iqa_hash) # plot image quality, accuracy change as function of time; scatter plots between variables

    report_table(mode=['fcn_mri_mlp', 'fcn_pet_mlp', 'fcn_dual_mlp', 'fcn_gan_dual_mlp'], mlp_repeat=repeat)  # print mlp performance table
    # roc_plot_perfrom_table(mode=['fcn_mri_mlp', 'fcn_pet_mlp', 'fcn_dual_mlp', 'fcn_gan_dual_mlp'])           # plot roc and pr curve
    # roc_plot_perfrom_table(mode=['fcn_mlp', 'fcn_gan_mlp'])           # plot roc and pr curve
    # report_table(mode=['fcn_mlp', 'fcn_gan_mlp'], mlp_repeat=repeat)  # print mlp performance table

    # gan.pick_time()

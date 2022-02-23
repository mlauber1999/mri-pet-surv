from utils_stat import get_roc_info, get_pr_info, load_neurologist_data, calc_neurologist_statistics, read_raw_score
import matplotlib.pyplot as plt
from utils_plot import plot_curve, plot_neorologist
from time import time
from sklearn.metrics import confusion_matrix, classification_report
import collections
from tabulate import tabulate
import os, sys
from matplotlib.patches import Rectangle
from scipy import stats, ndimage
import csv
import numpy as np
from nilearn.image.image import mean_img
from nilearn.plotting import plot_img, show
from nilearn.image import resample_to_img, resample_img
import regex as re
from subprocess import call

# define function to plot everything that needs to be plotted
def plot_mri_tau(mri, tau, pt, save=True):
    fig, ax = plt.subplots(2, 1)
    img1 = plot_img(mri, axes=ax[0], draw_cross=False)
    img1.title('MRI' + ' ' + pt)
    mean_tau = mean_img(tau)
    img2 = plot_img(mean_tau, axes=ax[1], draw_cross=False)
    img2.title('MRI+PET' + ' ' + pt)
    if save:
        fig.savefig(pt + '.pdf', orientation='landscape')
    return fig, ax

# define function to plot everything that needs to be plotted
def plot_mri_tau_overlay(mri, tau, pt, save=True):
    fig = plot_img(mri, cmap="gray")
    fig.title('PET+MRI on MRI' + ' ' + pt)
    fig.add_overlay(tau, alpha=0.5, cmap="jet")
    if save:
        plt.savefig(pt + '.pdf', orientation='landscape')
    return fig

def p_val(o, g):
    t, p = stats.ttest_ind(o, g, equal_var = False)
    # print(o, g, p)
    return p

def plot_legend(axes, crv_lgd_hdl, crv_info, neo_lgd_hdl, set1, set2):
    m_name = list(crv_lgd_hdl.keys())
    ds_name = list(crv_lgd_hdl[m_name[0]].keys())

    hdl = collections.defaultdict(list)
    val = collections.defaultdict(list)

    if neo_lgd_hdl:
        for ds in neo_lgd_hdl:
            hdl[ds] += neo_lgd_hdl[ds]
            val[ds] += ['Neurologist', 'Avg. Neurologist']

    # convert = {'cnn':"1.5T", 'cnn_gan':"1.5T*", 'cnn_aug':'1.5T Aug'}

    for ds in ds_name:
        for m in m_name:
            # print(m, ds)
            hdl[ds].append(crv_lgd_hdl[m][ds])
            val[ds].append('{} AUC: {:.3f}$\pm${:.3f}'.format(m, crv_info[m][ds]['auc_mean'], crv_info[m][ds]['auc_std']))
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        # val[ds].append('p-value: {:.4f}'.format(p_val(set1[ds], set2[ds])))

        axes[ds].legend(hdl[ds]+[extra], val[ds],
                        facecolor='w', prop={"weight":'bold', "size":17},  # frameon=False,
                        bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
                        loc='lower left')

def stat_metric(reports, names):
    precision, recall, f1 = [],[],[]
    names += ['macro avg', 'weighted avg']
    r_base = {}
    for n in names:
        r_base[n] = {'precision':[], 'recall':[], 'f1-score':[]}
    r_base['accuracy'] = []

    for r in reports:
        # report['weighted avg']['f1-score'], '%.4f' % report['macro avg']['f1-score']
        for n in names:
            r_base[n]['precision'] += [r[n]['precision']]
            r_base[n]['recall'] += [r[n]['recall']]
            r_base[n]['f1-score'] += [r[n]['f1-score']]
        if 'accuracy' in r:
            r_base['accuracy'] += [r['accuracy']]
        else:
            print('skiping case')
    for n in names:
        r_base[n]['precision'] = '{0:.4f}+/-{1:.4f}'.format(np.mean(r_base[n]['precision']), np.std(r_base[n]['precision']))
        r_base[n]['recall'] = '{0:.4f}+/-{1:.4f}'.format(np.mean(r_base[n]['recall']), np.std(r_base[n]['recall']))
        r_base[n]['f1-score'] = '{0:.4f}+/-{1:.4f}'.format(np.mean(r_base[n]['f1-score']), np.std(r_base[n]['f1-score']))
    r_base['accuracy'] = '{0:.4f}+/-{1:.4f}'.format(np.mean(r_base['accuracy']), np.std(r_base['accuracy']))

    return r_base

def report_table(txt_file=None, mode=['cnn'], fcn_repeat=0, mlp_repeat=0):

    table = collections.defaultdict(dict)

    # dlist = ['valid', 'test', 'AIBL', 'NACC']
    dlist = ['valid', 'test']

    for i, ds in enumerate(dlist):
        for m in mode:
            Matrix = []
            reports = []
            for exp_idx in range(fcn_repeat):
                for repe_idx in range(mlp_repeat):
                    if 'fcn' in m:
                        labels, preds = read_raw_score('checkpoint_dir/{}_exp{}/raw_score_{}_{}.txt'.format(m, exp_idx, ds, repe_idx))
                    else:
                        labels, preds = read_raw_score('checkpoint_dir/{}_exp{}/raw_score_{}.txt'.format(m, repe_idx, ds))
                    preds = [np.argmax(p) for p in preds]
                    target_names = ['class ' + str(i) for i in range(4)]
                    reports += [classification_report(y_true=labels, y_pred=preds, labels=[0,1,2,3], target_names=target_names, zero_division=0, output_dict=True)]
            r = stat_metric(reports, target_names)
            accuracy = r.pop('accuracy')

            headers = [m+' '+ds, 'precision', 'recall', 'f1-score']
            values = [[n]+[r[n][h] for h in headers[1:]] for n in r.keys()]
            print('accuracy:', accuracy)
            print(tabulate(values, headers=headers))

            if txt_file:
                with open(txt_file, 'a') as f:
                    line = tabulate(values, headers=headers)
                    f.write(str(line) + '\n')


def roc_plot_perfrom_table(txt_file=None, mode=['fcn', 'fcn_gan'], fcn_repeat=5, mlp_repeat=1):
    roc_info, pr_info = {}, {}
    aucs, apss = {}, {}
    dlist = ['valid', 'test'] #['test', 'AIBL', 'NACC']
    for m in mode:
        roc_info[m], pr_info[m], aucs[m], apss[m] = {}, {}, {}, {}
        for ds in dlist:
            Scores, Labels = [], []
            for exp_idx in range(fcn_repeat):
                for repe_idx in range(mlp_repeat):
                    labels, scores = read_raw_score('checkpoint_dir/{}_exp{}/raw_score_{}_{}.txt'.format(m, exp_idx, ds, repe_idx))
                    Scores.append(scores)
                    Labels.append(labels)
            # scores = np.array(Scores).mean(axis=0)
            # labels = Labels[0]
            # filename = '{}_{}_mean'.format(m, ds)
            # with open(filename+'.csv', 'w') as f1:
            #     wr = csv.writer(f1)
            #     wr.writerows([[s] for s in scores])
            # with open(filename+'_l.csv', 'w') as f2:
            #     wr = csv.writer(f2)
            #     wr.writerows([[l] for l in labels])
            #     # f.write(' '.join(map(str,scores))+'\n'+' '.join(map(str,labels)))

            roc_info[m][ds], aucs[m][ds] = get_roc_info(Labels, Scores)
            pr_info[m][ds], apss[m][ds] = get_pr_info(Labels, Scores)

    plt.style.use('fivethirtyeight')
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['figure.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'

    # convert = {'fcn':"1.5T", 'fcn_gan':"1.5T*", 'fcn_aug':'1.5T Aug'}

    # roc plot
    fig, axes_ = plt.subplots(1, 2, figsize=[18, 6], dpi=100)
    axes = dict(zip(dlist, axes_))
    lines = ['-', '-.', '--']
    hdl_crv = {m:{} for m in mode}
    for i, ds in enumerate(dlist):
        title = ds
        i += 1
        for j, m in enumerate(mode):
            hdl_crv[m][ds] = plot_curve(curve='roc', **roc_info[m][ds], ax=axes[ds],
                                    **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': lines[j],
                                        'title': title})

    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=roc_info, neo_lgd_hdl=None, set1=aucs[mode[0]], set2=aucs[mode[1]])
    fig.savefig('./plot/roc.png', dpi=300)

    # specificity sensitivity plot
    fig, axes_ = plt.subplots(1, 2, figsize=[18, 6], dpi=100)
    axes = dict(zip(dlist, axes_))
    lines = ['-', '-.', '--']
    hdl_crv = {m:{} for m in mode}
    for i, ds in enumerate(dlist):
        title = ds
        i += 1
        for j, m in enumerate(mode):
            hdl_crv[m][ds] = plot_curve(curve='sp_se', **roc_info[m][ds], ax=axes[ds],
                                    **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': lines[j],
                                        'title': title})

    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=roc_info, neo_lgd_hdl=None, set1=aucs[mode[0]], set2=aucs[mode[1]])
    fig.savefig('./plot/ss.png', dpi=300)

    # pr plot
    fig, axes_ = plt.subplots(1, 2, figsize=[18, 6], dpi=100)
    axes = dict(zip(dlist, axes_))
    hdl_crv = {m: {} for m in mode}
    for i, ds in enumerate(dlist):
        title = ds
        i += 1
        for j, m in enumerate(mode):
            hdl_crv[m][ds] = plot_curve(curve='pr', **pr_info[m][ds], ax=axes[ds],
                                    **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': lines[j],
                                        'title': title})

    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=pr_info, neo_lgd_hdl=None, set1=apss[mode[0]], set2=apss[mode[1]])
    fig.savefig('./plot/pr.png', dpi=300)


if __name__ == "__main__":
    roc_plot_perfrom_table()

import numpy as np
import pandas as pd
import os
from collections import defaultdict
from typing import Union, Tuple, List
from statistics.utilities import CONFIG, bootstrap_ci, pchip_interpolator, \
    benjamini_hochberg_correct
from statistics.clustered_mlp_output_wrappers import load_metadata_survival, \
    load_metadata_mlp_pivot
from statistics.statistics_formatters import \
    kaplan_meier_estimator, KaplanMeierPairwise
from statistics.dataframe_validation import \
    MlpPivotClusterSchema, DataFrame, MlpPivotSchema, MlpSurvivalSchema, pa
from lifelines.statistics import \
    survival_difference_at_fixed_point_in_time_test as surv_fixed_point
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import lifelines

plt.style.use('./statistics/styles/style.mplstyle')

@pa.check_types
def survival_probability_smooth(mlp_pivot: DataFrame[MlpPivotClusterSchema]) -> \
        pd.DataFrame:
    """Up-samples the survival probability data for plotting purposes

    Args:
        step (int, optional): step size for interpolation in units of months. Defaults to 1.
    """
    survival_values = []
    bins = CONFIG['bins']
    bins_float = [np.float(bin) for bin in bins]
    new_axis = eval(CONFIG['new_bins'])
    new_bins = np.asarray([str(int(x)) for x in new_axis])
    for _, row in mlp_pivot.iterrows():
        bin_values = row[bins]
        interpolator = pchip_interpolator(bins_float, bin_values)
        interp_vals = interpolator(new_axis)
        survival_values.append(np.clip(interp_vals, a_min=0, a_max=1))
    survival_values = np.asarray(survival_values)
    mlp_pivot.drop(columns=bins, inplace=True)
    for bin in new_bins:
        mlp_pivot[bin] = np.nan
    mlp_pivot.loc[:, new_bins] = survival_values
    return mlp_pivot

@pa.check_types
def predict_survival_by_group(df: pd.DataFrame,
                              group_variable='Cluster Idx',
                              pred_top_and_bot=True) -> defaultdict:
    survival_data = defaultdict(dict)
    for group, sub_df in df.groupby(group_variable):
        kmf = kaplan_meier_estimator(sub_df, label=group)
        survival_data[group]['kmf'] = kmf
        if pred_top_and_bot:
            pred = sub_df[eval(CONFIG['new_bins']).astype(str)].to_numpy()
            top_pred = bootstrap_ci(pred, (0.025, 0.9750), 10000)
            survival_data[group]['pred_top'] = top_pred[0]
            survival_data[group]['pred_bot'] = top_pred[1]
    return survival_data

@pa.check_types
def plot_raw_kmf_by_group(mlp_pivot: DataFrame[MlpPivotClusterSchema],
                          group_variable='Cluster Idx',
                          dataset='ADNI'):
    datas = mlp_pivot.query('Dataset == @dataset').copy()
    survival_data = predict_survival_by_group(
            datas, group_variable, pred_top_and_bot=False)
    fig, ax = plt.subplots()
    line = {}
    ncolors = len(cc.coolwarm)
    for cluster, value in survival_data.items():
        color = cc.bmy[ncolors // (int(cluster)+1) - 1]
        line[int(cluster)] = \
            value['kmf'].plot_survival_function(
                    ax=ax, ci_alpha=0.2, color=color)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Probability of survival')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 108])
    ax.legend([line[i].get_lines()[i] for i in line.keys()],
                [f'Subtype {i}' for i in line.keys()])
    os.makedirs(
            os.path.split(CONFIG['survival_plots'][dataset])[0], exist_ok=True)
    plt.savefig(
                CONFIG['survival_plots'][dataset],
                dpi=300)
    plt.close()

def predict_survival_nested_group(mlp_pivot: pd.DataFrame,
                                  first_group='Dataset',
                                  second_group='Cluster Idx') -> dict:
    survival_data = {}
    for group, sub_df in mlp_pivot.groupby(first_group):
        survival_data[group] = predict_survival_by_group(sub_df.copy(), second_group)
    return survival_data

def plot_survival_data_overlay_by_group(survival_data_dict: dict, label='NACC'):
    for key, value in survival_data_dict.items():
        fig, ax = plt.subplots()
        polys = plt.fill_between(eval(CONFIG['new_bins']), value['pred_bot'], value['pred_top'], alpha=0.2, axes=ax,
            facecolor=sns.color_palette('pastel')[2])
        line = value['kmf'].plot_survival_function(ax=ax, ci_alpha=0.2, color=sns.color_palette('pastel')[3])
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Probability of survival')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 108])
        ax.legend([polys, line.get_lines()[0]], ['Predicted survival','Kaplan Meier Estimate'])
        os.makedirs(os.path.split(CONFIG['survival_plots_overlay'][label])[0], exist_ok=True)
        plt.savefig(
                '{}{}.svg'.format(CONFIG['survival_plots_overlay'][label], key),
                dpi=300)
        plt.close()

def kmf_statistics_by_group(mlp_pivot: DataFrame[MlpPivotSchema],
                            group_variable='Cluster Idx', dataset='ADNI'):
    datas = mlp_pivot.query('Dataset == @dataset').copy()
    datas['Cluster Idx'] = datas['Cluster Idx'].astype('category')
    survival_data = KaplanMeierPairwise(datas, dataset, group_variable)
    return survival_data

# def write_pairwise_comparisons(*args):
#     _str = "\n" + "-"*50 + "\n" + "-"*50 + "\n"
#     _str = _str.join(args)
#     with open(CONFIG['survival_statistics'],'w') as fi:
#         fi.write(_str)

def _dataframe_bh_correct_wrapper(df: pd.DataFrame):
    p_value_corrected = \
        benjamini_hochberg_correct(
            df['p'].to_numpy()
            )
    df['p_correct'] = p_value_corrected
    return df

def write_pairwise_comparisons(tbl: pd.DataFrame):
    with open(CONFIG['survival_statistics'],'w') as fi:
        tbl = tbl.reset_index()
        tbl = tbl.groupby(['Dataset','Time']).apply(_dataframe_bh_correct_wrapper).reset_index(drop=True)
        tbl.sort_values(['Dataset', 'Time'], inplace=True)
        tbl = tbl[['Dataset','Time','Clusters','test_statistic','p','p_correct','n_censored_A','n_obs_A','n_censored_B','n_obs_B','test_name',]]
        fi.write(tabulate(tbl, headers=tbl.columns, showindex=False, tablefmt='fancygrid'))

@pa.check_types
def compare_surv_different_times(
        km_dataset: DataFrame[MlpPivotClusterSchema], time) -> defaultdict:
    clog = lambda s: np.log(-np.log(s)) # from https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/statistics.py
    survival_data = defaultdict(dict)
    for ds, sub_df_dataset in km_dataset.groupby('Dataset'):
        for cluster, sub_df in sub_df_dataset.groupby('Cluster Idx'):
            kmf = kaplan_meier_estimator(sub_df, label=cluster)
            survival_data[ds][cluster] = kmf
    results = defaultdict(lambda : defaultdict(dict))
    for dataset, dataset_dict in survival_data.items():
        for t in time:
            # iterate through pairs right here
            clusters = list(dataset_dict.keys())
            for idx1 in range(len(clusters)):
                for idx2 in range(idx1):
                    pair = f'{clusters[idx2]}x{clusters[idx1]}'
                    sA_t = dataset_dict[clusters[idx1]].predict(t)
                    sB_t = dataset_dict[clusters[idx2]].predict(t)
                    results[dataset][pair][t] = surv_fixed_point(t, dataset_dict[clusters[idx1]],
                                                dataset_dict[clusters[idx2]], point_estimate=f'log(-log(x)) difference ' + \
                                                f'{clusters[idx1]}-{clusters[idx2]}: '
                                                f'{clog(sA_t)-clog(sB_t)}')
    return results

class FormattedStatisticalResult(lifelines.statistics.StatisticalResult):
    def __init__(self, results: lifelines.statistics.StatisticalResult,
                indices : Tuple):
        self._kwargs = dict(test_name=results.test_name,
                        null_distribution=results.null_distribution,
                        n_censored_A=sum(results.fitterA.event_table['censored']),
                        n_obs_A=results.fitterA.event_table.loc[0,'at_risk'],
                        n_censored_B=sum(results.fitterB.event_table['censored']),
                        n_obs_B=results.fitterB.event_table.loc[0,'at_risk'])
        super().__init__(
                        p_value=results.p_value,
                        test_statistic=results.test_statistic,
                        name=results.name,
                        **self._kwargs
                        )
        self.specs = indices

    def to_table(self):
        mi = pd.MultiIndex.from_tuples((self.specs,), names=('Dataset', 'Clusters','Time'))
        tbl = self.summary.copy()
        df = pd.DataFrame(index=mi,
            data=tbl.to_numpy(),
            columns=tbl.columns
        )
        for kwarg, value in self._kwargs.items():
            df.loc[mi, kwarg] = value
        return df

def formatted_statistical_result_stack(nested_dict: Union[defaultdict, dict],
                                       labels: Tuple=()):
    tables = []
    for key, value in nested_dict.items():
        if type(value) == lifelines.statistics.StatisticalResult:
            result = FormattedStatisticalResult(value, indices=tuple(list(
                    labels) + [key]))
            tables.append(result.to_table())
        else:
            f = formatted_statistical_result_stack(
                    value,
                    tuple(list(labels) + [key])
            )
            tables.append(f)
    tables = pd.concat(tables, axis=0)
    return tables

def main():
    mlp_pivot : DataFrame[MlpPivotClusterSchema] = load_metadata_mlp_pivot()
    plot_raw_kmf_by_group(mlp_pivot, dataset='ADNI')
    plot_raw_kmf_by_group(mlp_pivot, dataset='NACC')
    results = compare_surv_different_times(mlp_pivot, [24, 48, 96])
    df = formatted_statistical_result_stack(results)
    write_pairwise_comparisons(df)
    mlp_pivot_smoothed = survival_probability_smooth(mlp_pivot)


    surv_predictions = predict_survival_nested_group(mlp_pivot_smoothed)
    print(surv_predictions)
    plot_survival_data_overlay_by_group(surv_predictions['NACC'], 'NACC')
    plot_survival_data_overlay_by_group(surv_predictions['ADNI'], 'ADNI')

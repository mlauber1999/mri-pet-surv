import pandas as pd
import numpy as np
import plotly.express as px
import re
from tabulate import tabulate
import os
import itertools
from statistics.utilities import CONFIG, get_interpolator, to_datetime, \
    benjamini_hochberg_correct
from statistics.clustered_mlp_output_wrappers import \
    load_metadata_pathology_mlp_pivot
from icecream import ic
from scipy.stats import chi2_contingency, fisher_exact
from statistics.dataframe_validation import DataFrame, MlpPathologySchema

def dump_pathology_table(mlp, label, index: bool):
    os.makedirs(os.path.split(CONFIG['pathology_csv_long'])[0], exist_ok=True)
    mlp.to_csv('{}{}.csv'.format(CONFIG['pathology_csv_long'], label), index=index)

def _modify_factor_levels(tbl: pd.DataFrame, factor: str) -> pd.DataFrame:
    replacements = CONFIG['path_config']['group_changes'][factor]
    if not bool(replacements):
        return tbl
    tbl[factor].replace(replacements, inplace=True)
    cat_order = CONFIG['path_config']['category_orders_new'][factor]
    cat_type = pd.CategoricalDtype(categories=cat_order, ordered=True)
    tbl[factor] = tbl[factor].astype(cat_type)
    cat_order = [x for x in cat_order if x in tbl[factor].cat.categories]
    tbl[factor] = tbl[factor].cat.reorder_categories(
            cat_order
    )
    return tbl

def _format_tbl_for_sunburst(
        pathology_mlp_df: DataFrame[MlpPathologySchema],
        factor: str):
    pathology_mlp_df[factor] = pathology_mlp_df[factor].astype('category')
    if factor == 'PROGRESSES':
        tbl_temp = pathology_mlp_df[["PROGRESSES", 'Cluster Idx']].copy()
        tbl_temp['PROGRESSES'] = tbl_temp['PROGRESSES'].apply(str)
    else:
        tbl_temp = pathology_mlp_df[
            ["PROGRESSES", factor, 'Cluster Idx']].copy()
    tbl_temp = tbl_temp.loc[
               [~np.isnan(x) if type(x) is float
                else True for x in tbl_temp[factor]],:]
    tbl_temp = _modify_factor_levels(tbl_temp, factor)
    if factor == 'PROGRESSES':
        tbl_temp = tbl_temp.groupby(['Cluster Idx', 'PROGRESSES']).agg(
                 len).reset_index()
    else:
        tbl_temp = tbl_temp.groupby(['Cluster Idx', factor, 'PROGRESSES']).agg(len).reset_index()
        tbl_temp['PROGRESSES'] = tbl_temp['PROGRESSES'].apply(str)
        tbl_temp = _modify_factor_levels(tbl_temp, 'PROGRESSES')
    tbl_temp.rename(columns={0: 'Counts'}, inplace=True)
    tbl_temp['Cluster Idx'] = tbl_temp['Cluster Idx'].apply(str)
    tbl_temp = _modify_factor_levels(tbl_temp, 'Cluster Idx')
    tbl_temp['Cluster Idx'] = tbl_temp['Cluster Idx'].cat.reorder_categories(
        CONFIG['path_config']['category_orders_new']['Cluster Idx']
    )
    tbl_temp.rename(columns={factor: CONFIG['path_config']['column_maps'][factor], 'PROGRESSES': 'Progresses'},
                    inplace=True)
    return tbl_temp

def _add_severity(tbl: pd.DataFrame, factor: str) -> pd.DataFrame:
    tbl['Cluster Idx'] = tbl['Cluster Idx'].astype('str')
    cat_order = tbl[factor].cat.categories
    d = {j: i/(len(cat_order)-1) for i,j in enumerate(cat_order)}
    tbl['Severity'] = tbl[factor].astype('str').map(d)
    tbl['Cluster Idx'] = tbl['Cluster Idx'].astype('category')
    return tbl

def _standardize_factor(factor) -> str:
    return re.sub("\?|\s|-|:","",factor)

def _progression_stats(tbl):
    tbl = tbl.groupby(['Cluster Idx','Progresses']).agg(np.sum).reset_index()
    tbl = pd.crosstab(index=tbl['Cluster Idx'], columns=tbl['Progresses'], values=tbl['Counts'], aggfunc=np.sum)
    result = chi2_contingency(tbl.to_numpy())
    st = {'Chi2': result[0], 'pval': result[1], 'dof': result[2], 'n': np.sum(result[3].reshape(-1,1))}
    return st

def sunburst_survival_plot_one_level(pathology_mlp_df: DataFrame[
    MlpPathologySchema], factor: str):
    tbl_progresses = _format_tbl_for_sunburst(pathology_mlp_df, factor)
    factor = CONFIG['path_config']['column_maps'][factor]
    tbl_all = tbl_progresses[['Cluster Idx', factor, 'Counts']].groupby(
            ['Cluster Idx', factor]).agg(np.sum).reset_index()
    tbl_all = _add_severity(tbl_all, factor)
    fig = px.sunburst(tbl_all, path=['Cluster Idx', factor],
                       values='Counts', color="Severity")
    n = np.sum(tbl_progresses.Counts)
    fig.update_layout({'title': f'{factor}, n={n}'},font=dict(
            size=28,
        ))
    fig.update_traces(insidetextorientation='radial')
    predict_str = 'nopredict_'
    fig.write_image(
        '{}{}{}_singlefactor.svg'.format(
        CONFIG['pathology_plot_prefix'],
        predict_str,
        _standardize_factor(factor).lower()
        ))
    dump_pathology_table(tbl_progresses, _standardize_factor(factor).lower(),
                         index=False)
    st : pd.DataFrame = sunburst_statistics(tbl_all, factor,
                                            _standardize_factor(factor))
    st2 : dict = _progression_stats(tbl_progresses.drop(columns=[factor]))
    return fig, st, n, st2

def sunburst_plots_for_factors_one_level(mlp_pathology:
    DataFrame[MlpPathologySchema]):
    os.makedirs(os.path.split(CONFIG['pathology_plot_prefix'])[0], exist_ok=True)
    with open(CONFIG['chi2_stats_fi'], 'w') as fi:
        st = []
        st_surv = []
        for key in CONFIG['path_config']['column_maps'].keys():
            if key == 'Cluster Idx' or key == 'PROGRESSES':
                continue
            fig, st_stub, n, st_stub2 = sunburst_survival_plot_one_level(
                    mlp_pathology, key)
            st_stub['Factor'] = key
            st.append(st_stub)
            st_surv.append(pd.Series(st_stub2).to_frame(name=key).T)
        st = pd.concat(st, axis=0)
        st_surv = pd.concat(st_surv, axis=0)
        fi.write(tabulate(st))
        fi.write('\n'*10 + r'Differences in Survival:' + '\n')
        fi.write(tabulate(st_surv))

def fisher_exact_pairwise(ct) -> pd.DataFrame:
    if ct.shape[1] > 2:
        return pd.DataFrame(columns=['Comparison','OddsRatio','P-value(BH-corrected)','N'])
    n_subgroups = ct.shape[0]
    idx = ct.index.categories
    print(idx)
    label = []
    _or = []
    _p = []
    _n = []
    for pair in itertools.combinations(list(range(n_subgroups)), 2):
        index_1 = idx[pair[0]]
        index_2 = idx[pair[1]]
        tbl = ct.loc[[index_1, index_2],:].to_numpy()
        st = fisher_exact(tbl)
        label.append(f'{index_1}_vs_{index_2}')
        _or.append(st[0])
        _p.append(st[1])
        _n.append(np.sum(tbl.reshape(-1)))
    _p_corrected = benjamini_hochberg_correct(_p)
    df = pd.DataFrame(data={'Comparison': label, 'OddsRatio': _or, 'P-value(BH-corrected)': _p_corrected,
        'N': _n
    })
    return df

def sunburst_statistics(df: pd.DataFrame, factor: str, reg: str):
    df.rename(columns={factor: reg, 'Cluster Idx': 'ClusterIdx'}, inplace=True)
    df = df.drop(columns=["Severity"])
    ct = pd.crosstab(index=df['ClusterIdx'], columns=df[reg], values=df['Counts'], aggfunc=np.nansum)
    st = fisher_exact_pairwise(ct)
    return st

def main():
    df = load_metadata_pathology_mlp_pivot()
    sunburst_plots_for_factors_one_level(df)
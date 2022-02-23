from preprocessing import move_nii_files
from preprocessing.cohorts import \
    ADNICollection, NaccCollection
from preprocessing.move_nii_files import \
    find_and_move_mri, find_and_move_mri_ad, find_and_move_unused_mri
from preprocessing import move_nacc_files
from process_parcellations import make_imaging_sheet
from statistics import survival_analysis, mlp_analysis, \
    clustering, demographic_statistics, brain_visualization, \
    generate_pathology_mris, shap_plots, plotting, \
    reviewerstats_subregion, pathology_analysis
import random
import argparse
import os
import pandas as pd

######################################

def create_csv_time(suffix = '', CollectionClass = ADNICollection):
    '''
    creates initial csv file with merged data, curated for MCI visits with CSF and MRI
    and also with time to progression
    '''
    p = CollectionClass()
    table = p.get_progression_data_time_to_progress_ad()
    for col in table.columns:
            if any([type(x) == list for x in table[col]]):
                    table.loc[:,col] = p.stringify(table[col])
    table.to_csv('metadata/data_processed'
                    '/merged_dataframe_cox' + suffix +
                    '.csv')
    return table

def create_csv_time_unused(suffix = ''):
    p = ADNICollection()
    table = p.tbl_merged.copy()
    table = table[['RID','DX','EXAMDATE','VISCODE2','VISCODE3','Phase',
                   'EXAMDATE_mri3', 'PTGENDER_demo', 'AGE', 'MMSCORE_mmse',
                   'abeta','tau','ptau']]
    not_nan_inds = [not pd.isna(x) if type(x) is not list else True for x
                    in table.EXAMDATE_mri3]
    table = table.loc[not_nan_inds, :]
    for col in table.columns:
            if any([type(x) == list for x in table[col]]):
                    table.loc[:,col] = p.stringify(table[col])
    table.set_index('RID', drop=True, inplace=True)
    table.to_csv('metadata/data_processed'
                    '/merged_dataframe_all_cox' + suffix +
                    '.csv', index_label='RID')

def consolidate_images_noqc(move=True):
    find_and_move_mri(move=move)

def consolidate_dummy_data(move=True):
    dummy_df = find_and_move_unused_mri(move=move)
    dummy_df.to_csv('metadata/data_processed'
                    '/merged_dataframe_unused_cox_pruned.csv')

def consolidate_images_ad(move=True):
    find_and_move_mri_ad(move=move)

def consolidate_images_nacc(move=True):
    move_nacc_files.find_and_move_mri(move, save=True)

##########################################

class ProcessImagesMRI(object):
    def __init__(self, suffix='cox_noqc'):
        self.prefix = 'matlab -nodisplay -r \"addpath(genpath(\'.\'));'
        self.prefix += "addpath(genpath(\'/usr/local/spm\'));"
        self.suffix = suffix
        self.basedir = '/data2/MRI_PET_DATA/processed_images_final_{}/'.format(
            self.suffix)
        self.realign = self.prefix + "realign_all_niftis(\'{}\'," \
                                     "\'_{}\');exit\"".format(
            self.basedir, suffix)
        self.process = self.prefix + "process_files_cox(\'_{}\');exit\"".format(
            suffix)
        self.parcellate = self.prefix + "batch_mrionly_job(\'_{}\');exit\"".format(
            suffix)

    def __call__(self):
        os.system(self.realign)
        os.system(self.process)
        os.system(self.parcellate)

class ProcessImagesMRIAD(object):
    def __init__(self, suffix='cox_noqc_AD'):
        self.prefix = 'matlab -nodisplay -r \"addpath(genpath(\'.\'));'
        self.prefix += "addpath(genpath(\'/usr/local/spm\'));"
        self.suffix = suffix
        self.basedir = '/data2/MRI_PET_DATA/processed_images_final_{}/'.format(
            self.suffix)
        self.realign = self.prefix + "realign_all_niftis(\'{}\'," \
                                     "\'_{}\');exit\"".format(
            self.basedir, suffix)
        self.parcellate = self.prefix + "batch_mrionly_job(\'_{}\');exit\"".format(
            suffix)

    def __call__(self):
        os.system(self.realign)
        # os.system(self.process)
        os.system(self.parcellate)

class ProcessImagesMRIDummy(object):
    def __init__(self, suffix='unused_cox'):
        self.prefix = 'matlab -nodisplay -r \"addpath(genpath(\'.\'));'
        self.prefix += "addpath(genpath(\'/usr/local/spm\'));"
        self.suffix = suffix
        self.basedir = '/data2/MRI_PET_DATA/processed_images_final_{}/'.format(
            self.suffix)
        self.realign = self.prefix + "realign_all_niftis_unused(\'{}\'," \
                                     "\'_{}\');exit\"".format(self.basedir, suffix)
        self.process = self.prefix + "process_files_cox_unused(\'_{}\');exit\"".format(
            suffix)
        self.parcellate = self.prefix + "batch_mrionly_unused_job(\'_{}\');exit\"".format(suffix)
        self.parcellate_errors = self.prefix + "batch_mrionly_job_errors(\'_{}\');exit\"".format(suffix)

    def move_nii(self):
        orig_dir = self.basedir + 'ADNI_MRI_nii_recenter_' + self.suffix + '/'
        new_dir = self.basedir + 'ADNI_MRI_nii_recenter_NL_' + self.suffix
        os.makedirs(new_dir, exist_ok=True)
        dat = pd.read_csv('metadata/data_processed/merged_dataframe_unused_cox_pruned.csv', dtype={'RID': str})
        for _, row in dat.iterrows():
            if row.DX == 'NL':
                os.system(f'rsync -v {orig_dir}{row.FILE_CODE}.nii {new_dir}/')
    
    def process_errors(self):
        os.system(self.parcellate_errors)

    def __call__(self):
        os.system(self.realign)
        os.system(self.process)
        self.move_nii()
        os.system(self.parcellate)

class ProcessImagesMRINacc(object):
    def __init__(self, suffix='cox_test'):
        self.prefix = 'matlab -nodisplay -r \"addpath(genpath(\'.\'));'
        self.prefix += "addpath(genpath(\'/usr/local/spm\'));"
        self.suffix = suffix
        self.basedir = '/data2/MRI_PET_DATA/processed_images_final_{}/'.format(
            self.suffix)
        self.realign = self.prefix + \
            "realign_all_niftis_nacc(\'{}\'," "\'_{}\');exit\"".format(
            self.basedir, suffix)
        self.process = self.prefix + \
            "process_files_cox(\'_{}\', \'^(NACC[0-9]+).*\.nii$\',1);exit\"".format(suffix)
        self.parcellate = self.prefix + "batch_mrionly_job_nacc(\'_{}\');exit\"".format(
            suffix)

    def __call__(self):
        os.system(self.realign)
        os.system(self.process)
        os.system(self.parcellate)

def make_imaging_sheet_main():
    make_imaging_sheet.main()

def dem_stats():
    demographic_statistics.main()

def cluster():
    clustering.main()

def survival_for_subtype():
    survival_analysis.main()

def mlp_plots_and_stats():
    mlp_analysis.main()

def make_stats_and_plots(_plot=False):
    make_imaging_sheet.main()
    demographic_statistics.main()
    clustering.main()
    survival_analysis.main()
    mlp_analysis.main()
    # reviewerstats_subregion.main()
    pathology_analysis.main()
    if _plot:
        brain_visualization.main()
        shap_plots.main()
        plotting.main()


if __name__ == '__main__':
    random.seed(10)
    parser = argparse.ArgumentParser('Enter in the time frame in y')
    parser.add_argument('--makecsv', dest='csv',default=0, type=bool)
    parser.add_argument('--moverawimages', dest='moveraw', default=0, type=bool)
    parser.add_argument('--extractimg', dest='extractimg',default=0, type=bool)
    parser.add_argument('--plot_examples', dest='plot_examples', default=0, type=bool)
    parser.add_argument('--stats', dest='stats', default=0, type=bool)
    parser.add_argument('--process_image', dest='process_image', default=0, type=bool)
    parser.add_argument('--test', dest='test', default=0, type=bool)
    parser.add_argument('--run_mlp', dest='run_mlp', default=0, type=bool)
    args = parser.parse_args()

    if args.csv == 1:
        if args.test == 1:
            suffix = '_test'
            _class = NaccCollection
        else:
            _class = ADNICollection
            suffix = '_noqc'
        table = create_csv_time(suffix, _class)
    if args.extractimg == 1:
        if args.test == 1:
            consolidate_images_nacc(args.moveraw == 1)
        else:
            consolidate_images_noqc(args.moveraw == 1)

    ## everything above this line
    if args.process_image == 1:
        suffix = 'cox'
        if args.test == 1:
            suffix += '_test'
            p = ProcessImagesMRINacc(suffix)
        else:
            suffix += '_noqc'
            p = ProcessImagesMRI(suffix)
        p()
    if args.stats == 1:
        make_stats_and_plots()
    if args.run_mlp == 1:
        raise NotImplementedError
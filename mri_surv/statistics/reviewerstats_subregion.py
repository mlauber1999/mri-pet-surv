import pandas as pd
import numpy as np

def main():
    #start from mri_pet working directory
    csv = pd.read_csv('./metadata/data_raw/MCIADSubtypeAssessment.csv')

    # Need to create a tree of a loop

    lobes = ['mesial_temp', 'temporal_lobe_other','insula', 'frontal', 'cingulate','occipital','parietal']

    #Create a dictionary where keys=lobes, values=subregions
    lobe_dict = {'mesial_temp': {'mesial_temp': ['hippocampus','amygdala',
                                'parahippocampus','entorhinal']},
                'temporal_lobe_other': {'temporal': ['fusiform',
                                                    'inftemporalgyrus']},
                'insula': {'insula': ['insula']},
                'frontal': {'frontal': ['subcallosum','supfrontalgyrus']},
                'cingulate': {'cingulate': ['cingulate']},
                'occipital': {'occipital': ['occipital']},
                'parietal': {'parietal': ['angular']}
    }

    # --> coming up with a sum of the _l and _r subregions for each
    #loop over every row
    for index, row in csv.iterrows():
        for lobe in lobes:
            # lobe_sum = [] #once filled, this is the sum of l_sum and r_sum
            
            #walk down each row and create a weighted sum across the entire lobe
            #and also the l and r
            #add columns for each lobe... l_sum, r_sum, lobe_sum
            # --> l_sum = the sum over all the l subregions
            # --> r_sum = the sum over all the r subregions
            # --> lobe_sum = the sum over l and right subregions (all)
            l_sum, r_sum = np.nan, np.nan
            if row[lobe] == 0:
                # append a 0 for l_sum, r_sum, and lobe_sum
                l_sum, r_sum = 0, 0
                #lobe_sum = 0
            else:
                for subregion in lobe_dict[lobe].keys(): #list of the subregions
                    #get a l_sum and r_sum for each subregion, 
                    # append to the *_sum_subregions for this lobe
                    if not np.isnan(row['l_'+subregion]):
                        l_sum = row['l_' + subregion]
                    else:
                        l_sum = row[lobe]
                        print(f'Imputing value for record no {row.record_no}, '
                            f'{subregion}, l')

                    if not np.isnan(row['r_' + subregion]):
                        r_sum = row['r_'+subregion]
                    else:
                        r_sum = row[lobe]
                        print(f'Imputing value for record no {row.record_no}, '
                            f'{subregion}, r')
                    # assert(not np.any([np.isnan(x) for x in l_sum]))
                    # assert(not np.any([np.isnan(x) for x in r_sum]))
            lobe_avg = (l_sum+r_sum)/(len(lobe_dict[lobe])*2)
            csv.loc[index,lobe+'_l_sum'] = l_sum
            csv.loc[index,lobe+'_r_sum'] = r_sum  
            csv.loc[index,lobe+'_avg'] = lobe_avg
            
            # if index < 15:
            #     print('avg',lobe_avg,lobe)
    print(csv)
    print(csv.head())
    # print(csv['mesial_temp_avg'])
    csv.to_csv('./metadata/data_raw/MCIADSubtypeAssessment_weighted.csv')

if __name__ == '__main__':
    pass
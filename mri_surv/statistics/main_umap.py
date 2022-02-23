load = False

if load:

    import pandas as pd
    import seaborn as sns

    df = pd.read_csv('./results/emb.csv')
    g = sns.scatterplot(data=df, x='emb_0', y='emb_1', hue='type', s=16, palette='tab20')
    g.set(yticklabels=[], xticklabels=[], xlabel=None, ylabel=None)
    g.axis('off')
    fig = g.get_figure()
    fig.tight_layout()
    fig.savefig('./figures/emb.svg')

else:

    import pandas as pd
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt
    import tqdm
    import os
    import umap

    # dataframe
    lst_seq = []
    df_info = pd.read_csv('./metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv')
    df_info.sort_values(by='RID')
    col_mri_fname = df_info['MRI_fname'].to_numpy().tolist()
    for mri_fname in col_mri_fname:
        hint = os.path.normpath(mri_fname).split(os.path.sep)[6]
        if 'MPRAGE' in hint or 'MP-RAGE' in hint:
            lst_seq.append('MPRAGE')
        elif 'SPGR' in hint:
            lst_seq.append('IR-(F)SPGR')
        else:
            raise RuntimeError('Cannot determine scan type \"{}\"'.format(hint))

    dir_ = '/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/'
    lst_fn = list(os.listdir(dir_))
    lst_fn = sorted(lst_fn)

    imgs = []
    img_ids = []
    # for i, fn in tqdm.tqdm(enumerate(lst_fn[:100])):
    for i, fn in tqdm.tqdm(enumerate(lst_fn)):
        fn = os.path.join(dir_, fn)
        img = np.array(nib.load(fn).dataobj)
        img[np.isnan(img)] = 0
        img = img.flatten()
        imgs.append(img)
        img_ids.append(np.repeat(i, len(img)))

    imgs = np.stack(imgs)

    reducer = umap.UMAP(random_state=3227, verbose=True)
    reducer.fit(imgs)
    emb = reducer.transform(imgs)

    df = pd.DataFrame()
    df['RID'] = df_info['RID']
    df['type'] = lst_seq
    df['emb_0'] = emb[:, 0]
    df['emb_1'] = emb[:, 1]
    # df['feature_1'] = img_ids
    # df['feature_2'] = np.stack(img_ids)

    df.to_csv('./results/emb.csv', index='RID')

    
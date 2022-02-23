% main script for processing images
%% recenter all images at origin
realign_all_niftis
%% store MRIs so they don't get overridden
mkdir('/data2/MRI_PET_dir('/data2/MRI_PET_DATA/processed_images_final_{}yr/ADNI_MRI_nii_recentered_cat12_{}yr'.format(n_years))DATA/processed_images_final/ADNI_MRI_nii_recentered_cat12');
!rsync -a /data2/MRI_PET_DATA/processed_images_final/ADNI_MRI_nii_recenter/* /data2/MRI_PET_DATA/processed_images_final/ADNI_MRI_nii_recentered_cat12
%% initial processing and normalization to MNI space + masking
process_files
%%
cat12_slice_and_dice_all_job
%% now map anatomical maps to new space
basedir = '/data2/MRI_PET_DATA/processed_images_final/';
mkdir([basedir 'ADNI_MRI_nii_recentered_copy']);
mkdir([basedir 'ADNI_MRI_nii_recentered_copy/mri_atlas']);
system(['rsync -ar ' basedir 'ADNI_MRI_nii_recenter_original/* ' basedir 'ADNI_MRI_nii_recentered_copy']);
system(['rsync -ar ' basedir 'ADNI_MRI_nii_recentered_cat12/mri_atlas/neuromorphometrics*.nii ' basedir 'ADNI_MRI_nii_recentered_copy/mri_atlas/']);
system(['rsync -ar ' basedir 'ADNI_MRI_nii_recentered_cat12/label ' basedir 'ADNI_MRI_nii_recentered_copy']);
%%
normalize_atlas

%%
system(['rsync -ar ' basedir 'ADNI_MRI_nii_recentered_cat12/label ' basedir 'atlas_normalized_fdg']);
system(['rsync -ar ' basedir 'ADNI_MRI_nii_recentered_cat12/label ' basedir 'atlas_normalized_amyloid']);
system(['rsync -ar ' basedir 'brain_stripped/*_amyloid.nii ' basedir 'atlas_normalized_amyloid']);
system(['rsync -ar ' basedir 'brain_stripped/*_fdg.nii ' basedir 'atlas_normalized_fdg']);
system(['rsync -ar ' basedir 'brain_stripped/*_mri.nii ' basedir 'atlas_normalized_fdg']);
system(['rsync -ar ' basedir 'brain_stripped/*_mri.nii ' basedir 'atlas_normalized_amyloid']);


%% recenter all images at origin

outputfolder = '/data2/MRI_PET_DATA/processed_images_final_1yr';
realign_all_niftis(outputfolder,'_1yr')

%%
mkdir('/data2/MRI_PET_DATA/processed_images_final_1yr/ADNI_MRI_nii_recentered_cat12_1yr');
!rsync -a /data2/MRI_PET_DATA/processed_images_final_1yr/ADNI_MRI_nii_recenter_1yr/* /data2/MRI_PET_DATA/processed_images_final_1yr/ADNI_MRI_nii_recentered_cat12_1yr

%%
process_files('_1yr')

%%
cat12_slice_and_dice_all_job('_1yr')
%% now map anatomical maps to new space
basedir = '/data2/MRI_PET_DATA/processed_images_final_1yr/';
mkdir([basedir 'ADNI_MRI_nii_recentered_copy_1yr']);
mkdir([basedir 'ADNI_MRI_nii_recentered_copy_1yr/mri_atlas']);
system(['rsync -ar ' basedir 'ADNI_MRI_nii_recenter_original_1yr/* ' basedir 'ADNI_MRI_nii_recentered_copy_1yr']);
system(['rsync -ar ' basedir 'ADNI_MRI_nii_recentered_cat12_1yr/mri_atlas/neuromorphometrics*.nii ' basedir 'ADNI_MRI_nii_recentered_copy_1yr/mri_atlas/']);
system(['rsync -ar ' basedir 'ADNI_MRI_nii_recentered_cat12_1yr/label ' basedir 'ADNI_MRI_nii_recentered_copy_1yr']);
%% recenter all images at origin

outputfolder = '/data2/MRI_PET_DATA/processed_images_final_3yr';
realign_all_niftis(outputfolder,'_3yr')

%%
mkdir('/data2/MRI_PET_DATA/processed_images_final_3yr/ADNI_MRI_nii_recentered_cat12_3yr');
!rsync -a /data2/MRI_PET_DATA/processed_images_final_3yr/ADNI_MRI_nii_recenter_3yr/* /data2/MRI_PET_DATA/processed_images_final_3yr/ADNI_MRI_nii_recentered_cat12_3yr

%%
process_files('_3yr')
%%
cat12_slice_and_dice_all_job('_3yr')
%% now map anatomical maps to new space
basedir = '/data2/MRI_PET_DATA/processed_images_final_3yr/';
mkdir([basedir 'ADNI_MRI_nii_recentered_copy_3yr']);
mkdir([basedir 'ADNI_MRI_nii_recentered_copy_3yr/mri_atlas']);
system(['rsync -ar ' basedir 'ADNI_MRI_nii_recenter_original_3yr/* ' basedir 'ADNI_MRI_nii_recentered_copy_3yr']);
system(['rsync -ar ' basedir 'ADNI_MRI_nii_recentered_cat12_3yr/mri_atlas/neuromorphometrics*.nii ' basedir 'ADNI_MRI_nii_recentered_copy_3yr/mri_atlas/']);
system(['rsync -ar ' basedir 'ADNI_MRI_nii_recentered_cat12_3yr/label ' basedir 'ADNI_MRI_nii_recentered_copy_3yr']);
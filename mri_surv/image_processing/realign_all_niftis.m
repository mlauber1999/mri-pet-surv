%script to move niftis to a new folder once they are realigned
function rids = realign_all_niftis(outputfolder, suffix)
%% main
if nargin < 1
    outputfolder = '/data2/MRI_PET_DATA/processed_images_final';
end
if nargin < 2
    suffix = '';
end

MRI_folder_init = ['/data2/MRI_PET_DATA/raw_data/MRI_nii' suffix filesep]; %goes through this folder 
fnames = dir(MRI_folder_init); 
fnames = {fnames.name};
rids = arrayfun(@(x) regexp(x,'^([0-9]{4}).*\.nii$','tokens'), fnames, 'uniformoutput', false);
rids = [rids{:}];
rids = [rids{:}];
rids_mri = [rids{:}];

rids = rids_mri;

MRI_folder_new = [outputfolder '/ADNI_MRI_nii_recenter' suffix];
% moves the niftis to this new folder after realigning them 

if ~exist(outputfolder,'dir')
    mkdir(outputfolder)
end
if ~exist(MRI_folder_new,'dir')
    mkdir(MRI_folder_new);
end

realign_nifti(MRI_folder_init, MRI_folder_new, rids, 'mri');

function normalize_atlas(nyrs)
addpath(genpath('/home/mfromano/spm/spm12/'));

% create folders
BASE_DIR = ['/data2/MRI_PET_DATA/processed_images_final' nyrs '/'];
MRI_folder_new = [BASE_DIR 'ADNI_MRI_nii_recentered_normalized_cat12' nyrs '/'];
PET_folder_init = ['/data2/MRI_PET_DATA/raw_data/AMYLOID_nii' nyrs filesep];

% get file names for raw PET scans
fnames = dir(PET_folder_init);
fnames = {fnames.name};
rids = arrayfun(@(x) regexp(x,'^([0-9]{4}).*\.nii$','tokens'), fnames, 'uniformoutput', false);
rids = [rids{:}];
rids = [rids{:}];
rids_pet = [rids{:}];

MRI_folder_init = ['/data2/MRI_PET_DATA/raw_data/MRI_nii' nyrs filesep];
fnames = dir(MRI_folder_init);
fnames = {fnames.name};
rids = arrayfun(@(x) regexp(x,'^([0-9]{4}).*\.nii$','tokens'), fnames, 'uniformoutput', false);
rids = [rids{:}];
rids = [rids{:}];
rids_mri = [rids{:}];

% get rids in both folders
rids = intersect(rids_pet, rids_mri);

spm('defaults', 'PET');
spm_jobman('initcfg');
for i=1:numel(rids)
    r = rids{i};
    matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = {BASE_DIR};
    matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'tmp2';
    matlabbatch{2}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {[BASE_DIR 'ADNI_AMYLOID_nii_recenter' nyrs]};
    matlabbatch{2}.cfg_basicio.file_dir.file_ops.file_fplist.filter = ['^' r '_amyloid.nii$'];
    matlabbatch{2}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
    matlabbatch{3}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {MRI_folder_new};
    matlabbatch{3}.cfg_basicio.file_dir.file_ops.file_fplist.filter = ['^' r '_mri.nii$'];
    matlabbatch{3}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
    matlabbatch{4}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {[MRI_folder_new 'mri_atlas']};
    matlabbatch{4}.cfg_basicio.file_dir.file_ops.file_fplist.filter = ['^neuromorphometrics_' r '_mri.nii$'];
    matlabbatch{4}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
    % move raw amyloid, raw mri, and neuromorphometrics file (not warped
    % neuromorphometrics file) into a temp directory
    matlabbatch{5}.cfg_basicio.file_dir.file_ops.file_move.files(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^' r '_amyloid.nii$)'], substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
    matlabbatch{5}.cfg_basicio.file_dir.file_ops.file_move.action.copyto(1) = cfg_dep('Make Directory: Make Directory ''tmp2''', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
    matlabbatch{6}.cfg_basicio.file_dir.file_ops.file_move.files(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^' r '_mri.nii$)'], substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
    matlabbatch{6}.cfg_basicio.file_dir.file_ops.file_move.action.copyto(1) = cfg_dep('Make Directory: Make Directory ''tmp2''', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
    matlabbatch{7}.cfg_basicio.file_dir.file_ops.file_move.files(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^neuromorphometrics_' r '_mri.nii$)'], substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
    matlabbatch{7}.cfg_basicio.file_dir.file_ops.file_move.action.copyto(1) = cfg_dep('Make Directory: Make Directory ''tmp2''', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));

    % coregister the MRI to the PET. Carry along the neuromorphometrics map
    % as a reference
    matlabbatch{8}.spm.spatial.coreg.estimate.ref(1) = cfg_dep('Move/Delete Files: Moved/Copied Files', substruct('.','val', '{}',{5}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
    matlabbatch{8}.spm.spatial.coreg.estimate.source(1) = cfg_dep('Move/Delete Files: Moved/Copied Files', substruct('.','val', '{}',{6}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
    matlabbatch{8}.spm.spatial.coreg.estimate.other(1) = cfg_dep('Move/Delete Files: Moved/Copied Files', substruct('.','val', '{}',{7}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
    matlabbatch{8}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    matlabbatch{8}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
    matlabbatch{8}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{8}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
    
    % normalize the atlas to normalized mri's space using nearest neighbor
    % interpolation
    matlabbatch{9}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {[BASE_DIR 'ADNI_MRI_nii_recenter_amyloid' nyrs]};
    matlabbatch{9}.cfg_basicio.file_dir.file_ops.file_fplist.filter = ['^y_' r '_mri.nii$'];
    matlabbatch{9}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
    matlabbatch{10}.cfg_basicio.file_dir.file_ops.file_move.files(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^y_' r '_mri.nii$)'], substruct('.','val', '{}',{9}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
    matlabbatch{10}.cfg_basicio.file_dir.file_ops.file_move.action.copyto(1) = cfg_dep('Make Directory: Make Directory ''tmp2''', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
    matlabbatch{11}.cfg_basicio.file_dir.file_ops.cfg_file_split.name = ['coregistered_atlas_' r];
    matlabbatch{11}.cfg_basicio.file_dir.file_ops.cfg_file_split.files(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.','val', '{}',{8}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
    matlabbatch{11}.cfg_basicio.file_dir.file_ops.cfg_file_split.index = {2};
    matlabbatch{12}.spm.spatial.normalise.write.subj.def(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^y_' r '_mri.nii$)'], substruct('.','val', '{}',{9}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
    matlabbatch{12}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep(['File Set Split: coregistered_atlas_' r ' (1)'], substruct('.','val', '{}',{11}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('{}',{1}));
    matlabbatch{12}.spm.spatial.normalise.write.woptions.bb = [NaN NaN NaN
                                                               NaN NaN NaN];
    matlabbatch{12}.spm.spatial.normalise.write.woptions.vox = [NaN NaN NaN];
    matlabbatch{12}.spm.spatial.normalise.write.woptions.interp = 0;
    matlabbatch{12}.spm.spatial.normalise.write.woptions.prefix = 'w';
    matlabbatch{13}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = {BASE_DIR};
    matlabbatch{13}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'atlas_normalized_amyloid';
    % write the now normalized atlas to the amyloid/mri directory
    matlabbatch{14}.cfg_basicio.file_dir.file_ops.file_move.files(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{12}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
    matlabbatch{14}.cfg_basicio.file_dir.file_ops.file_move.action.moveto(1) = cfg_dep('Make Directory: Make Directory ''atlas_normalized_amyloid''', substruct('.','val', '{}',{13}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
%    
%     
%     matlabbatch{15}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {[MRI_folder_new 'mri_atlas']};
%     matlabbatch{15}.cfg_basicio.file_dir.file_ops.file_fplist.filter = ['^neuromorphometrics_' r '_mri.nii$'];
%     matlabbatch{15}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
%     matlabbatch{16}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {MRI_folder_new};
%     matlabbatch{16}.cfg_basicio.file_dir.file_ops.file_fplist.filter = ['^' r '_mri.nii$'];
%     matlabbatch{16}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
%     matlabbatch{17}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {[BASE_DIR 'ADNI_FDG_nii_recenter' nyrs]};
%     matlabbatch{17}.cfg_basicio.file_dir.file_ops.file_fplist.filter = ['^' r '_fdg.nii$'];
%     matlabbatch{17}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
%     matlabbatch{18}.cfg_basicio.file_dir.file_ops.file_move.files(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^neuromorphometrics_' r '_mri.nii$)'], substruct('.','val', '{}',{15}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
%     matlabbatch{18}.cfg_basicio.file_dir.file_ops.file_move.action.copyto(1) = cfg_dep('Make Directory: Make Directory ''tmp2''', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
%     matlabbatch{19}.cfg_basicio.file_dir.file_ops.file_move.files(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^' r '_mri.nii$)'], substruct('.','val', '{}',{16}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
%     matlabbatch{19}.cfg_basicio.file_dir.file_ops.file_move.action.copyto(1) = cfg_dep('Make Directory: Make Directory ''tmp2''', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
%     matlabbatch{20}.cfg_basicio.file_dir.file_ops.file_move.files(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^' r '_fdg.nii$)'], substruct('.','val', '{}',{17}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
%     matlabbatch{20}.cfg_basicio.file_dir.file_ops.file_move.action.copyto(1) = cfg_dep('Make Directory: Make Directory ''tmp2''', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
%     
%     
%     matlabbatch{21}.spm.spatial.coreg.estimate.ref(1) = cfg_dep('Move/Delete Files: Moved/Copied Files', substruct('.','val', '{}',{20}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
%     matlabbatch{21}.spm.spatial.coreg.estimate.source(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^' r '_mri.nii$)'], substruct('.','val', '{}',{19}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
%     matlabbatch{21}.spm.spatial.coreg.estimate.other(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^neuromorphometrics_' r '_mri.nii$)'], substruct('.','val', '{}',{18}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
%     matlabbatch{21}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
%     matlabbatch{21}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
%     matlabbatch{21}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
%     matlabbatch{21}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
%     
%     matlabbatch{22}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {[BASE_DIR 'ADNI_MRI_nii_recenter_fdg' nyrs]};
%     matlabbatch{22}.cfg_basicio.file_dir.file_ops.file_fplist.filter = ['^y_' r '_mri.nii$'];
%     matlabbatch{22}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
%     matlabbatch{23}.cfg_basicio.file_dir.file_ops.file_move.files(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^y_' r '_mri.nii$)'], substruct('.','val', '{}',{22}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
%     matlabbatch{23}.cfg_basicio.file_dir.file_ops.file_move.action.copyto(1) = cfg_dep('Make Directory: Make Directory ''tmp2''', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
%     matlabbatch{24}.cfg_basicio.file_dir.file_ops.cfg_file_split.name = ['coregistered_atlas_' r];
%     matlabbatch{24}.cfg_basicio.file_dir.file_ops.cfg_file_split.files(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.','val', '{}',{21}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
%     matlabbatch{24}.cfg_basicio.file_dir.file_ops.cfg_file_split.index = {2};
%     matlabbatch{25}.spm.spatial.normalise.write.subj.def(1) = cfg_dep(['File Selector (Batch Mode): Selected Files (^y_' r '_mri.nii$)'], substruct('.','val', '{}',{22}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
%     matlabbatch{25}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep(['File Set Split: coregistered_atlas_' r ' (1)'], substruct('.','val', '{}',{24}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('{}',{1}));
%     matlabbatch{25}.spm.spatial.normalise.write.woptions.bb = [NaN NaN NaN
%                                                                NaN NaN NaN];
%     matlabbatch{25}.spm.spatial.normalise.write.woptions.vox = [NaN NaN NaN];
%     matlabbatch{25}.spm.spatial.normalise.write.woptions.interp = 0;
%     matlabbatch{25}.spm.spatial.normalise.write.woptions.prefix = 'w';
%     matlabbatch{26}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = {BASE_DIR};
%     matlabbatch{26}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'atlas_normalized_fdg';
%     matlabbatch{27}.cfg_basicio.file_dir.file_ops.file_move.files(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{25}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
%     matlabbatch{27}.cfg_basicio.file_dir.file_ops.file_move.action.moveto(1) = cfg_dep('Make Directory: Make Directory ''atlas_normalized_fdg''', substruct('.','val', '{}',{26}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
    %delete temporary directory
    matlabbatch{28}.cfg_basicio.file_dir.dir_ops.dir_move.dir(1) = cfg_dep('Make Directory: Make Directory ''tmp2''', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
    matlabbatch{28}.cfg_basicio.file_dir.dir_ops.dir_move.action.delete = true;
    spm_jobman('run_nogui', matlabbatch);
end
end
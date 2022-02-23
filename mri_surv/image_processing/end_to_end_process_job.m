%-----------------------------------------------------------------------
% Job saved on 11-Jan-2021 16:08:59 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {'/data2/MRI_PET_DATA/processed_images_final_cumulative/ADNI_MRI_nii_recentered_cat12_cumulative/mri'};
matlabbatch{1}.cfg_basicio.file_dir.file_ops.file_fplist.filter = '^wm2239_mri.nii$';
matlabbatch{1}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
matlabbatch{2}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {'/data2/MRI_PET_DATA/processed_images_final_cumulative/ADNI_AMYLOID_nii_recenter_cumulative'};
matlabbatch{2}.cfg_basicio.file_dir.file_ops.file_fplist.filter = '^2239_amyloid.nii$';
matlabbatch{2}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
matlabbatch{3}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {'/data2/MRI_PET_DATA/processed_images_final_cumulative/ADNI_FDG_nii_recenter_cumulative'};
matlabbatch{3}.cfg_basicio.file_dir.file_ops.file_fplist.filter = '^2239_fdg.nii$';
matlabbatch{3}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
matlabbatch{4}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = {'/data2/MRI_PET_DATA/processed_images_final_cumulative'};
matlabbatch{4}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'tmp_coregister';
matlabbatch{5}.cfg_basicio.file_dir.file_ops.file_fplist.dir = {'/data2/MRI_PET_DATA/processed_images_final_cumulative/ADNI_MRI_nii_recentered_cat12_cumulative'};
matlabbatch{5}.cfg_basicio.file_dir.file_ops.file_fplist.filter = '^y_2399_mri.nii$';
matlabbatch{5}.cfg_basicio.file_dir.file_ops.file_fplist.rec = 'FPList';
matlabbatch{6}.cfg_basicio.file_dir.file_ops.file_move.files(1) = cfg_dep('File Selector (Batch Mode): Selected Files (^wm2239_mri.nii$)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{6}.cfg_basicio.file_dir.file_ops.file_move.files(2) = cfg_dep('File Selector (Batch Mode): Selected Files (^2239_amyloid.nii$)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{6}.cfg_basicio.file_dir.file_ops.file_move.files(3) = cfg_dep('File Selector (Batch Mode): Selected Files (^2239_fdg.nii$)', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{6}.cfg_basicio.file_dir.file_ops.file_move.files(4) = cfg_dep('File Selector (Batch Mode): Selected Files (^y_2399_mri.nii$)', substruct('.','val', '{}',{5}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{6}.cfg_basicio.file_dir.file_ops.file_move.action.moveto(1) = cfg_dep('Make Directory: Make Directory ''tmp_coregister''', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
matlabbatch{7}.cfg_basicio.file_dir.file_ops.cfg_file_split.name = 'moved_files';
matlabbatch{7}.cfg_basicio.file_dir.file_ops.cfg_file_split.files(1) = cfg_dep('Move/Delete Files: Moved/Copied Files', substruct('.','val', '{}',{6}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{7}.cfg_basicio.file_dir.file_ops.cfg_file_split.index = {
                                                                     1
                                                                     2
                                                                     3
                                                                     4
                                                                     }';
matlabbatch{8}.spm.spatial.normalise.write.subj.def(1) = cfg_dep('File Set Split: moved_files (4)', substruct('.','val', '{}',{7}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('{}',{4}));
matlabbatch{8}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep('File Selector (Batch Mode): Selected Files (^2239_amyloid.nii$)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{8}.spm.spatial.normalise.write.subj.resample(2) = cfg_dep('File Selector (Batch Mode): Selected Files (^2239_fdg.nii$)', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{8}.spm.spatial.normalise.write.woptions.bb = [NaN NaN NaN
                                                          NaN NaN NaN];
matlabbatch{8}.spm.spatial.normalise.write.woptions.vox = [NaN NaN NaN];
matlabbatch{8}.spm.spatial.normalise.write.woptions.interp = 4;
matlabbatch{8}.spm.spatial.normalise.write.woptions.prefix = 'w';
matlabbatch{9}.cfg_basicio.file_dir.file_ops.file_fplist.dir = '<UNDEFINED>';
matlabbatch{9}.cfg_basicio.file_dir.file_ops.file_fplist.filter = '<UNDEFINED>';
matlabbatch{9}.cfg_basicio.file_dir.file_ops.file_fplist.rec = '<UNDEFINED>';

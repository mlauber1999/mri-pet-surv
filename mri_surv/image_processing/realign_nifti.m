%this script realigns all nifti files
%version 1 of this in zip file for Mike's abstract, download v1 file, should have everything in place for pet coregistration 
function fnames = realign_nifti(raw_nifti_folder, folder_name, rid_list, modality)
if ~exist(folder_name,'dir')
    mkdir(folder_name);
end

listing = dir(raw_nifti_folder);
fnames = {};
for i=1:length(listing)
    if ~listing(i).isdir && strcmp(listing(i).name(end-3:end), '.nii')
        rid = regexp(listing(i).name,'^([0-9]{4}).*\.nii$','tokens');
        fname = join([listing(i).folder filesep listing(i).name],'');
        system(['rsync -a ' fname ' ' folder_name '/../tmp/']);
        if ismember(rid{1},rid_list)
            Vo = SetOriginToCenter([folder_name '/../tmp/' listing(i).name]); %set origin to center realigns them 
            [data] = spm_read_vols(Vo);
            Vo.fname = [folder_name filesep rid{1}{1} '_' modality '.nii'];
            fnames = [fnames, Vo.fname]; %file you get from the core 
            if ~exist(Vo.fname, 'file')
                spm_write_vol(Vo, data); %writes the new volume at a new location 
            end
        else
            disp(['file ' listing(i).name ' not in list of RIDs'])
        end
    else
        disp(['skipping ' listing(i).folder filesep listing(i).name])
    end

end

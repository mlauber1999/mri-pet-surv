%this script realigns all nifti files
%version 1 of this in zip file for Mike's abstract, download v1 file, should have everything in place for pet coregistration 
function fnames = realign_nifti(raw_nifti_folder, folder_name, rid_list, modality)
%creates function named realign_nifit with inputs raw_nifit_folder, folder_name, and modality and output fnames (filenames) 
if ~exist(folder_name,'dir')
%if folder_name exists in the directory, make one 
    mkdir(folder_name);
end

listing = dir(raw_nifti_folder);
%define var listing with the value of the all the files in the directory raw_nifti_folder 
fnames = {};
%creates an empty cell array assigned to var fnames 
%array is numbers, cell array is characters 
for i=1:length(listing) %for loop that runs as many times as the length of all the files in the raw_nifti_folder (assigned to var listing) 
    if ~listing(i).isdir && strcmp(listing(i).name(end-3:end), '.nii') 
    %if the file is not a directory, do a string comparision to see if the last 4 char of the file is the same as .nii 
    %if listing is not an existing directory, 
    %strcmp is string comparison 
    %with dir you have .isdir 
    %for the ith file within folder, if the listing(folder name) is not a directory, do 
    %&& means and, you can also use single & but && is computationally faster  
    
        rid = regexp(listing(i).name,'^([0-9]{4}).*\.nii$','tokens'); %searching wihtin listing name to meet the search criteria 
        %i think 'tokens' saves this pattern 
        fname = join([listing(i).folder filesep listing(i).name],''); %creating a new file name that is the file name + the folder name 
        %filesep is likely a shortcut to seperate the fle and folder name 
        system(['rsync -a ' fname ' ' folder_name '/../tmp/']); 
        %system calls os to do the specified command 
        %rysnc is copying and syncing data from one computer to another 
        %rsync -a means archive, syncs directories recursively, preserve symbolic links, modification times, groups, ownership, and permission
        %this whole command is telling the os to sync the filename and folder name and creates a temporary file/directory for it
        if ismember(rid{1},rid_list) %if the first rid is on the rid_list:
        %ismember(a,b) returns true if a is found within b 
            Vo = SetOriginToCenter([folder_name '/../tmp/' listing(i).name]); %set origin to center realigns them 
            [data] = spm_read_vols(Vo); %tells spm to read the volumes 
            Vo.fname = [folder_name filesep rid{1}{1} '_' modality '.nii']; %v0.name creates a variable within the variable (structure within a structure) and defines the name for v0.filename
            fnames = [fnames, Vo.fname]; %file you get from the core 
            %this is then filling in the empty cell array created above
            %has columns for fnames and one for vo.name
            if ~exist(Vo.fname, 'file') %if vo.name doesnt exist, tells spm to write the volume 
                spm_write_vol(Vo, data); %writes the new volume at a new location 
            end
        else
            disp(['file ' listing(i).name ' not in list of RIDs']) %display the file name and tell you its not listed in the rids 
        end
    else
        disp(['skipping ' listing(i).folder filesep listing(i).name]) %tells you that its skipping this file 
    end

end

close all; 
clearvars;

addpath('../CREATE_REFERENCES_CUBS/')
% --- load parameters in memory
run('set_parameters')
% --- load data
files_A1 = loadData.load_files(p.PATH_TO_ANNOTATION_A1);
files_A1_bis = loadData.load_files(p.PATH_TO_ANNOTATION_A1_BIS);
files_A2 = loadData.load_files(p.PATH_TO_ANNOTATION_A2);
files_A3 = loadData.load_files(p.PATH_TO_ANNOTATION_A3);
% --- get patient name
for k=1:size(files_A1, 1)
	names_patient{k}=files_A1(k).name;
end
% --- initialization
id_LI=0;
id_MA=0;
id_rect=0;
inc=1;
% --- process patient
while size(names_patient, 2)>1
%     size(names_patient, 2)
    % --- interpolation
    % --- A1 expert
    [LI_A1, MA_A1] = interpolation.load_annotation(p.PATH_TO_ANNOTATION_A1, names_patient{inc});
    [LI_int_A1, MA_int_A1, borders_A1]=interpolation.interfaces_interpolation_pchip(LI_A1, MA_A1);
    % --- A1 bis expert
    [LI_A1_bis, MA_A1_bis] = interpolation.load_annotation(p.PATH_TO_ANNOTATION_A1_BIS, names_patient{inc});
    [LI_int_A1_bis, MA_int_A1_bis, borders_A1_bis]=interpolation.interfaces_interpolation_pchip(LI_A1_bis, MA_A1_bis);
    % --- A2 expert
    [LI_A2, MA_A2] = interpolation.load_annotation(p.PATH_TO_ANNOTATION_A2, names_patient{inc});
    [LI_int_A2, MA_int_A2, borders_A2]=interpolation.interfaces_interpolation_pchip(LI_A2, MA_A2);  
    % --- load the matching image
    path_to_image = p.PATH_TO_IMAGES;
    name_=split(names_patient{inc}, '-');
    name_=name_{1};
    image_name=strcat(path_to_image, name_, '.tiff');
    image=loadData.load_image(image_name);
    % --- we modify the image
    image=modify_image(LI_int_A1, MA_int_A1, LI_A1, MA_A1, image, 'A1');
    image=modify_image(LI_int_A1_bis, MA_int_A1_bis, LI_A1_bis, MA_A1_bis, image, 'A1_bis');
    image=modify_image(LI_int_A2, MA_int_A2, LI_A2, MA_A2, image, 'A2');
    % --- we compute the union and the intersection
    left_border = [borders_A2.border_left, borders_A1_bis.border_left, borders_A1.border_left];
    right_border = [borders_A2.border_right, borders_A1_bis.border_right, borders_A1.border_right];
    % --- retrieve union and intersection borders
    borders_union.border_right=max(right_border);
    borders_union.border_left=min(left_border);
    borders_intersection.border_right=min(right_border);
    borders_intersection.border_left=max(left_border);
    % --- we save the data
    path_to_save_data = p.PATH_RES ;
    save_image(image, path_to_save_data, name_);
    save_borders(path_to_save_data, borders_intersection, borders_union, borders_A1, borders_A1_bis, borders_A2, name_);
    save_references(LI_int_A1, MA_int_A1, path_to_save_data, name_, 'A1', image);
    save_references(LI_int_A1_bis, MA_int_A1_bis, path_to_save_data, name_, 'A1_bis', image);
    save_references(LI_int_A2, MA_int_A2, path_to_save_data, name_, 'A2', image);
    % --- update names_patient    
    LI_name = strcat(name_, '-LI.txt');
    MA_name = strcat(name_, '-MA.txt');
    % --- we remove the processed patient
    for k=1:1:size(names_patient, 2)
        disp(['current patient: ' num2str(k)]);
        if strcmp(names_patient{k}, LI_name)
            id_LI=k;
            break
        end
    end
    names_patient(id_LI)=[];
    for k=1:1:size(names_patient, 2)
        if strcmp(names_patient{k}, MA_name)
            id_MA=k;
            break
        end
    end
    names_patient(id_MA)=[];
end

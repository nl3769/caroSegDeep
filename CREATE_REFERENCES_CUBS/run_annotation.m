close all; 
clear all;

% --- we get parameters
run('set_parameters')
 
path_annotations_A1 = p.PATH_TO_ANNOTATION_A1 ;
path_annotations_A1_bis = p.PATH_TO_ANNOTATION_A1_BIS ;
path_annotations_A2 = p.PATH_TO_ANNOTATION_A2 ;
path_annotations_A3 = p.PATH_TO_ANNOTATION_A3 ;


files_A1 = load_data.loadFiles(path_annotations_A1);
files_A1_bis = load_data.loadFiles(path_annotations_A1_bis);
files_A2 = load_data.loadFiles(path_annotations_A2);
files_A3 = load_data.loadFiles(path_annotations_A3);

annotationNb=size(files_A1);


for k=1:size(files_A1, 1)
        namesPatient{k}=files_A1(k).name;
end


id_LI=0;
id_MA=0;
id_rect=0;

inc=1;

while size(namesPatient, 2)>1
    
    size(namesPatient, 2)
    fileName = namesPatient{inc} 

    
    % --- interpolation
    % --- A1 expert
    [LI_A1, MA_A1] = interpolation.loadAnnoation(path_annotations_A1, fileName);
    [LI_int_A1, MA_int_A1, borders_A1]=interpolation.InterfacesInterpolationPchip(LI_A1, MA_A1);
    % --- A1 bis expert
    [LI_A1_bis, MA_A1_bis] = interpolation.loadAnnoation(path_annotations_A1_bis, fileName);
    [LI_int_A1_bis, MA_int_A1_bis, borders_A1_bis]=interpolation.InterfacesInterpolationPchip(LI_A1_bis, MA_A1_bis);
    % --- A2 expert
    [LI_A2, MA_A2] = interpolation.loadAnnoation(path_annotations_A2, fileName);
    [LI_int_A2, MA_int_A2, borders_A2]=interpolation.InterfacesInterpolationPchip(LI_A2, MA_A2);  
    
    % --- load the matching image
    pathToImage = p.PATH_TO_IMAGES;
    tmp=split(fileName, '-');
    tmp=tmp{1};
    imageName=strcat(pathToImage, tmp, '.tiff');
    image=load_data.loadImage(imageName);

    % --- we modify the image
    image=modified_image(LI_int_A1, MA_int_A1, LI_A1, MA_A1, image, 'A1');
    image=modified_image(LI_int_A1_bis, MA_int_A1_bis, LI_A1_bis, MA_A1_bis, image, 'A1_bis');
    image=modified_image(LI_int_A2, MA_int_A2, LI_A2, MA_A2, image, 'A2');

    % --- we compute the union and the intersection
    left_border = [borders_A2.border_left, borders_A1_bis.border_left, borders_A1.border_left];
    right_border = [borders_A2.border_right, borders_A1_bis.border_right, borders_A1.border_right];
    
    % --- retrieve union and intersection borders
    borders_union.border_right=max(right_border);
    borders_union.border_left=min(left_border);
    borders_intersection.border_right=min(right_border);
    borders_intersection.border_left=max(left_border);

    % --- we save the data
    pathToSaveData = p.PATH_RES ;
    name = strcat(tmp);
    save_image(image, pathToSaveData, name);
    save_borders(pathToSaveData, borders_intersection, borders_union, borders_A1, borders_A1_bis, borders_A2, name);
    save_references(LI_int_A1, MA_int_A1, pathToSaveData, name, 'A1', image);
    save_references(LI_int_A1_bis, MA_int_A1_bis, pathToSaveData, name, 'A1_bis', image);
    save_references(LI_int_A2, MA_int_A2, pathToSaveData, name, 'A2', image);
    
    % --- update namesPatient    
    LIName = strcat(tmp, '-LI.txt');
    MAName = strcat(tmp, '-MA.txt');

    % --- we remove the processed patient
    for k=1:1:size(namesPatient, 2)
        k
        if strcmp(namesPatient{k}, LIName)
            id_LI=k;
            break
        end
    end
    namesPatient(id_LI)=[];
    
    for k=1:1:size(namesPatient, 2)
        if strcmp(namesPatient{k}, MAName)
            id_MA=k;
            break
        end
    end
    
    namesPatient(id_MA)=[];
    
end

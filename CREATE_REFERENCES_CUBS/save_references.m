function[] = save_references(LI, MA, path, name, expert, image)
    
    % --- we save the segmentation    
    widthImage=size(image, 2);
    segLI=zeros(widthImage, 1);
    segLI(LI(:,1))=LI(:,2);
    
    segMA=zeros(widthImage, 1);
    segMA(MA(:,1))=MA(:, 2);

    % --- we first save the LI interface
    seg=segLI;
    pathToSaveSegLI=strcat(path, 'CONTOURS/', expert, '/', name, '_IFC3_A1', '.mat');
    save(pathToSaveSegLI, 'seg');
    
    seg = segMA ;
    pathToSaveSegMA=strcat(path, 'CONTOURS/', expert, '/', name, '_IFC4_A1', '.mat');
    save(pathToSaveSegMA, 'seg');

end


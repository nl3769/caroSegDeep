function[] = save_references(LI, MA, path, name, expert, image)
    % Saves the segmentation in .mat file
    
    widthImage=size(image, 2);
    segLI=zeros(widthImage, 1);
    segLI(LI(:,1))=LI(:,2);
    
    segMA=zeros(widthImage, 1);
    segMA(MA(:,1))=MA(:, 2);
    % --- saves the LI interface
    seg=segLI;
    pathToSaveSegLI=fullfile(path, 'CONTOURS', expert, name, '_IFC3_A1', '.mat');
    save(pathToSaveSegLI, 'seg');
    % --- saves the MA interface
    seg = segMA ;
    pathToSaveSegMA=fullfile(path, 'CONTOURS', expert, name, '_IFC4_A1', '.mat');
    save(pathToSaveSegMA, 'seg');

end

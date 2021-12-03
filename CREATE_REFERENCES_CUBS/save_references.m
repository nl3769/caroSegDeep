function[] = save_references(LI, MA, path, name, expert, image)
    % Saves the segmentation in .mat file
    
    width_image=size(image, 2);
    seg_LI=zeros(width_image, 1);
    seg_LI(LI(:,1))=LI(:,2);
    
    seg_MA=zeros(width_image, 1);
    seg_MA(MA(:,1))=MA(:, 2);
    % --- saves the LI interface
    seg=seg_LI;
    path_save_seg_LI=fullfile(path, 'CONTOURS', expert, [name '_IFC3_A1.mat']);
    save(path_save_seg_LI, 'seg');
    % --- saves the MA interface
    seg = seg_MA ;
    path_save_seg_MA=fullfile(path, 'CONTOURS', expert, [name '_IFC4_A1.mat']);
    save(path_save_seg_MA, 'seg');

end

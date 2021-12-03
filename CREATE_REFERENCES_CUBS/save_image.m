function[] = save_image(image, path_to_save, name)
    % Saves the image
    pathToSaveImage=fullfile(path_to_save, 'IMAGES', [name '.tiff']);
    imwrite(image,pathToSaveImage,'tiff');
end

function[] = save_image(image, path_to_save, name)
    % --- we save the image
    pathToSaveImage=strcat(path_to_save, 'IMAGES/', name, '.tiff');
    imwrite(image,pathToSaveImage,'tiff');
end
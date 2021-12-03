function[] = save_borders(path, borders_intersection, borders_union, border_A1, border_A1_bis, border_A2, name)
    % Saves the borders .mat file
    % --- we save the borders (intersection)
    path_save_borders=fullfile(path, 'BORDERS', 'BORDERS_INTERSECTION', [name '_borders.mat']);
    border_right=borders_intersection.border_right;
    border_left=borders_intersection.border_left;
    save(path_save_borders, 'border_right', 'border_left');
    % --- we save the borders (union)
    path_save_borders=fullfile(path, 'BORDERS', 'BORDERS_UNION', [name '_borders.mat']);
    border_right=borders_union.border_right;
    border_left=borders_union.border_left;
    save(path_save_borders, 'border_right', 'border_left');
    % --- we save the borders (A1)
    path_save_borders=fullfile(path, 'BORDERS', 'BORDERS_A1', [name '_borders.mat']);
    border_right=border_A1.border_right;
    border_left=border_A1.border_left;
    save(path_save_borders, 'border_right', 'border_left');
    % --- we save the borders (A1 BIS)
    path_save_borders=fullfile(path, 'BORDERS', 'BORDERS_A1_BIS', [name '_borders.mat']);
    border_right=border_A1_bis.border_right;
    border_left=border_A1_bis.border_left;
    save(path_save_borders, 'border_right', 'border_left');
    % --- we save the borders (A2)
    path_save_borders=fullfile(path, 'BORDERS', 'BORDERS_A2', [name '_borders.mat']);
    border_right=border_A2.border_right;
    border_left=border_A2.border_left;
    save(path_save_borders, 'border_right', 'border_left');
    
end

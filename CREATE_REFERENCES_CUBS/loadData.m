
classdef loadData
   
    methods (Static)

        function [files]=load_files(path)
            % loads control points 
            filePattern=fullfile(path, '*.txt');
            files=dir(filePattern);

        end

        function [img]=load_image(path)
             % loads image
            img = imread(path);
        end
        
        
   end
   
end


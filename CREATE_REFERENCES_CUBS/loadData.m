
classdef loadData
   
    methods (Static)

        function [files]=load_files(path)

            filePattern=fullfile(path, '*.txt');
            % ---- we load the file in the directory
            files=dir(filePattern);

        end

        function [img]=load_image(path)
            img = imread(path);
        end
        
        
   end
   
end


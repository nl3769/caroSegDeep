classdef interpolation
    
    methods (Static)
    
        function[LI, MA]=load_annotation(path, fileName)
            % load annotaion         
            name=split(fileName, '-');
            name=name{1};            
            path_LI=fullfile(path, [name '-LI.txt']);
            path_MA=fullfile(path, [name '-MA.txt']);
            LI=load(path_LI);
            MA=load(path_MA);                       
        end       
 
        function[LI_int, MA_int, borders]=interfaces_interpolation_makima(LI, MA)
            % apply makima interpolation
            min_LI_x=round(min(LI(:,1)));
            max_LI_x=round(max(LI(:,1)));               
            min_MA_x=round(min(MA(:,1)));
            max_MA_x=round(max(MA(:,1)));
            x_LI=min_LI_x:1:max_LI_x;
            x_MA=min_MA_x:1:max_MA_x;
            y_LI = makima(LI(:,1),LI(:,2),x_LI);
            y_MA = makima(MA(:,1),MA(:,2),x_MA);
            LI_int=[x_LI; y_LI].';
            MA_int=[x_MA; y_MA].';            
            borders.border_right=min(max_LI_x, max_MA_x);
            borders.border_left=max(min_LI_x, min_MA_x);            
        end
        
        function[LI_int, MA_int, borders]=interfaces_interpolation_pchip(LI, MA)
            % apply pchip interpolation
            min_LI_x=round(min(LI(:,1)));
            max_LI_x=round(max(LI(:,1)));
            min_MA_x=round(min(MA(:,1)));
            max_MA_x=round(max(MA(:,1)));
            x_LI=min_LI_x:1:max_LI_x;
            x_MA=min_MA_x:1:max_MA_x;
            y_LI = pchip(LI(:,1),LI(:,2),x_LI);
            y_MA = pchip(MA(:,1),MA(:,2),x_MA);
            LI_int=[x_LI; y_LI].';
            MA_int=[x_MA; y_MA].';            
            borders.border_right=min(max_LI_x, max_MA_x);
            borders.border_left=max(min_LI_x, min_MA_x);            
        end        
    end    
end

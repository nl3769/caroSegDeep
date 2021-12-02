classdef interpolation
    
    methods (Static)
    
        function[LI, MA]=load_annotation(path, fileName)
            % load annotaion         
            tmp=split(fileName, '-');
            tmp=tmp{1};            
            pathToLI=strcat(path, tmp, '-LI.txt');
            pathToMA=strcat(path, tmp, '-MA.txt');
            LI=load(pathToLI);
            MA=load(pathToMA);                       
        end       
 
        function[LI_int, MA_int, borders]=interfaces_interpolation_makima(LI, MA)
            % apply makima interpolation
            minLI_x=round(min(LI(:,1)));
            maxLI_x=round(max(LI(:,1)));               
            minMA_x=round(min(MA(:,1)));
            maxMA_x=round(max(MA(:,1)));
            xLI=minLI_x:1:maxLI_x;
            xMA=minMA_x:1:maxMA_x;
            yLI = makima(LI(:,1),LI(:,2),xLI);
            yMA = makima(MA(:,1),MA(:,2),xMA);
            LI_int=[xLI; yLI].';
            MA_int=[xMA; yMA].';            
            borders.border_right=min(maxLI_x, maxMA_x);
            borders.border_left=max(minLI_x, minMA_x);            
        end
        
        function[LI_int, MA_int, borders]=interfaces_interpolation_pchip(LI, MA)
            % apply pchip interpolation
            minLI_x=round(min(LI(:,1)));
            maxLI_x=round(max(LI(:,1)));
            minMA_x=round(min(MA(:,1)));
            maxMA_x=round(max(MA(:,1)));
            xLI=minLI_x:1:maxLI_x;
            xMA=minMA_x:1:maxMA_x;
            yLI = pchip(LI(:,1),LI(:,2),xLI);
            yMA = pchip(MA(:,1),MA(:,2),xMA);
            LI_int=[xLI; yLI].';
            MA_int=[xMA; yMA].';            
            borders.border_right=min(maxLI_x, maxMA_x);
            borders.border_left=max(minLI_x, minMA_x);            
        end        
    end    
end

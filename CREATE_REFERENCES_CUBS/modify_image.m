function[img]=modify_image(LI_int,MA_int, LI, MA, img, expert)
    % Adds the interpolated curve for visual inspection
    switch expert
       case 'A1'
              % ---- here I add the interface
            for k=1:1:size(LI_int, 1)

                img(round(LI_int(k,2)), LI_int(k,1), 1)=255;
                img(round(LI_int(k,2)), LI_int(k,1), 2)=0;
                img(round(LI_int(k,2)), LI_int(k,1), 3)=0; 

            end

            for k=1:1:size(MA_int, 1)
                img(round(MA_int(k,2)), MA_int(k,1), 1)=0;
                img(round(MA_int(k,2)), MA_int(k,1), 2)=255;
                img(round(MA_int(k,2)), MA_int(k,1), 3)=0; 
            end

            % ---- here I add the point defined by the experts
            for k=1:1:size(MA, 1)
                img(round(MA(k,2)), round(MA(k,1)), 1)=255;
                img(round(MA(k,2)), round(MA(k,1)), 2)=0;
                img(round(MA(k,2)), round(MA(k,1)), 3)=0; 
            end

            for k=1:1:size(LI, 1)
                img(round(LI(k,2)), round(LI(k,1)), 1)=0;
                img(round(LI(k,2)), round(LI(k,1)), 2)=255;
                img(round(LI(k,2)), round(LI(k,1)), 3)=0; 
            end
    
        case 'A1_bis'
            % ---- Adds the interface
            for k=1:1:size(LI_int, 1)

                img(round(LI_int(k,2)), LI_int(k,1), 1)=0;
                img(round(LI_int(k,2)), LI_int(k,1), 2)=255;
                img(round(LI_int(k,2)), LI_int(k,1), 3)=0; 

            end

            for k=1:1:size(MA_int, 1)
                img(round(MA_int(k,2)), MA_int(k,1), 1)=255;
                img(round(MA_int(k,2)), MA_int(k,1), 2)=0;
                img(round(MA_int(k,2)), MA_int(k,1), 3)=0; 
            end

            % ---- Adds the interface
            for k=1:1:size(MA, 1)
                img(round(MA(k,2)), round(MA(k,1)), 1)=0;
                img(round(MA(k,2)), round(MA(k,1)), 2)=255;
                img(round(MA(k,2)), round(MA(k,1)), 3)=0; 
            end

            for k=1:1:size(LI, 1)
                img(round(LI(k,2)), round(LI(k,1)), 1)=255;
                img(round(LI(k,2)), round(LI(k,1)), 2)=0;
                img(round(LI(k,2)), round(LI(k,1)), 3)=0; 
            end
        case 'A2'
            % ---- Adds the interface
            for k=1:1:size(LI_int, 1)

                img(round(LI_int(k,2)), LI_int(k,1), 1)=0;
                img(round(LI_int(k,2)), LI_int(k,1), 2)=0;
                img(round(LI_int(k,2)), LI_int(k,1), 3)=255; 

            end

            for k=1:1:size(MA_int, 1)
                img(round(MA_int(k,2)), MA_int(k,1), 1)=0;
                img(round(MA_int(k,2)), MA_int(k,1), 2)=0;
                img(round(MA_int(k,2)), MA_int(k,1), 3)=255; 
            end

            % ---- Adds the interface
            for k=1:1:size(MA, 1)
                img(round(MA(k,2)), round(MA(k,1)), 1)=255;
                img(round(MA(k,2)), round(MA(k,1)), 2)=255;
                img(round(MA(k,2)), round(MA(k,1)), 3)=0; 
            end

            for k=1:1:size(LI, 1)
                img(round(LI(k,2)), round(LI(k,1)), 1)=255;
                img(round(LI(k,2)), round(LI(k,1)), 2)=255;
                img(round(LI(k,2)), round(LI(k,1)), 3)=0; 
            end

    end

    
end

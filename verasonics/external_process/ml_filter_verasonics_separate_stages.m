function ml_filter_verasonics_separate_stages(data)
    persistent stage1

    if isempty(stage1)
        % Load the first stage model (U-Net) if not already loaded
        load('custom_onnx_unet113.mat');
        stage1 = custom_onnx_unet113;
    end
    
    persistent stage2

    if isempty(stage2)
        % Load the second stage model (CycleGAN) if not already loaded
        load('custom_onnx_cyclegan.mat');
        stage2 = custom_onnx_cyclegan;
    end
    
    % Preprocess the input data
    bmode = abs(imresize(data, [512,512]));
    bmode = db(bmode/max(bmode, [], 'all'));
    bmode(bmode<-50) = -50;
    bmode = (bmode-min(bmode, [], 'all'))/(max(bmode,[], 'all')-min(bmode, [], 'all'))*2-1;
    
    % Predict using the first stage model
    stage1_output = predict(stage1, bmode, 'Acceleration','auto');
    % Predict using the second stage model
    stage2_output = predict(stage2, stage1_output, 'Acceleration','auto');

    persistent myImg1
    persistent myImg2
    persistent myImg3
    if (isempty(myImg1) || ~isprop(myImg1, 'CData')) || (isempty(myImg2) || ~isprop(myImg2, 'CData')) || (isempty(myImg3) || ~isprop(myImg3, 'CData'))
        % Create a figure with three subplots if not already created
        figure('Position', [100,100, 1600, 600]);
        subtightplot(1,3,1, [0.01, 0.03])
        myImg1 = imagesc(zeros(512,512));
        colorbar()
        caxis([-1 1])
        colormap('gray')
        axis image
        title('PW Image')
        
        subtightplot(1,3,2, [0.01, 0.03])
        myImg2 = imagesc(zeros(512,512));
        colorbar()
        caxis([-1 1])
        colormap('gray')
        axis image
        title('Stage 1')
        
        subtightplot(1,3,3, [0.01, 0.03])
        myImg3 = imagesc(zeros(512,512));
        colorbar()
        caxis([-1 1])
        colormap('gray')
        axis image
        title('Stage 2')
    end
    
    % Update the images with the new data
    myImg1.CData = bmode;
    myImg2.CData = stage1_output;
    myImg3.CData = stage2_output;

    % Refresh the figure
    drawnow limitrate;

end
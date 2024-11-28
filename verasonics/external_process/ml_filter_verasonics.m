function ml_filter_verasonics(data)
    persistent model

    if isempty(model)
        % Load the combined model (U-Net and CycleGAN) if not already loaded
        load('combined_model.mat');
        model = combined_model;
    end
    
    % Preprocess the input data
    bmode = abs(imresize(data, [512,512]));
    bmode = db(bmode/max(bmode, [], 'all'));
    bmode(bmode<-50) = -50;
    bmode = (bmode-min(bmode, [], 'all'))/(max(bmode,[], 'all')-min(bmode, [], 'all'))*2-1;
    
    % Predict using the combined model
    img = predict(model, bmode, 'Acceleration','auto');

    persistent myImg
    if (isempty(myImg) || ~isprop(myImg, 'CData'))
        % Create a figure if not already created
        figure;
        myImg = imagesc(zeros(512,512));
        colorbar()
        caxis([-1 1])
        colormap('gray')
        axis image
    end
    
    % Update the image with the new data
    myImg.CData = img;
    drawnow limitrate;

end
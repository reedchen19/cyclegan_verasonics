function ml_filter_verasonics(data)
    persistent model

    if isempty(model)
        load('combined_model.mat'); % stage 1 and 2 combined together
        model = combined_model;
    end
    
    bmode = abs(imresize(data, [512,512]));
    bmode = db(bmode/max(bmode, [], 'all'));
    bmode(bmode<-50) = -50;
    bmode = (bmode-min(bmode, [], 'all'))/(max(bmode,[], 'all')-min(bmode, [], 'all'))*2-1;
    
    img = predict(model, bmode, 'Acceleration','auto');

    persistent myImg
    if (isempty(myImg) || ~isprop(myImg, 'CData'))
        figure;
        myImg = imagesc(zeros(512,512));
        colorbar()
        caxis([-1 1])
        colormap('gray')
        axis image
    end
    
    myImg.CData = img;
    drawnow limitrate;

end
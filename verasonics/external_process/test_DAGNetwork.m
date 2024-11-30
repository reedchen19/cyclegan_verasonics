% load models and test image
load('custom_onnx_unet113.mat');
stage1 = custom_onnx_unet113;

load('custom_onnx_cyclegan.mat');
stage2 = custom_onnx_cyclegan;

load('combined_model.mat');
combined = combined_model;

load("test_input_img.mat");
bmode = (data-min(data, [], 'all'))/(max(data,[], 'all')-min(data, [], 'all'))*2-1;

% inference
stage1_output = predict(stage1, bmode, 'Acceleration','auto');
stage2_output = predict(stage2, stage1_output, 'Acceleration','auto');
combined_output = predict(combined, bmode, 'Acceleration','auto');

% plot outputs
figure()
imagesc(stage1_output);
colorbar()
caxis([-1 1])
colormap('gray')
axis image
title('Stage 1')

figure()
imagesc(stage2_output);
colorbar()
caxis([-1 1])
colormap('gray')
axis image
title('Stage 2')

figure()
imagesc(combined_output);
colorbar()
caxis([-1 1])
colormap('gray')
axis image
title('Combined')
% Import the U-Net ONNX model
onnx_layers = importONNXLayers('unet_113.onnx', 'ImportWeights', 1);

% Customize the imported U-Net ONNX model
custom_unet_lg = onnx_layers;
custom_unet_lg = removeLayers(custom_unet_lg, 'StatefulPartitionedCall|model|conv2d|BiasAdd__6');
custom_unet_lg = removeLayers(custom_unet_lg, 'StatefulPartitionedCall|model|conv2d_18|BiasAdd__140');
input_layer = imageInputLayer([512 512 1], 'Name',  'input', 'DataAugmentation', 'none', 'Normalization', 'none', 'AverageImage', []);
custom_unet_lg = replaceLayer(custom_unet_lg, 'Input_input_1', [input_layer]);
custom_unet_lg = connectLayers(custom_unet_lg, 'input', 'StatefulPartitionedCall|model|conv2d|BiasAdd');

% Replace transposed convolution layers with MATLAB's implementation
tp0 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(27).Weights, 'Bias', onnx_layers.Layers(27).Bias, 'name', 'tp0');
tp1 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(33).Weights, 'Bias', onnx_layers.Layers(33).Bias, 'name', 'tp1');
tp2 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(39).Weights, 'Bias', onnx_layers.Layers(39).Bias, 'name', 'tp2');
tp3 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(45).Weights, 'Bias', onnx_layers.Layers(45).Bias, 'name', 'tp3');
custom_unet_lg = replaceLayer(custom_unet_lg, custom_unet_lg.Layers(26).Name, tp0);
custom_unet_lg = replaceLayer(custom_unet_lg, custom_unet_lg.Layers(32).Name, tp1);
custom_unet_lg = replaceLayer(custom_unet_lg, custom_unet_lg.Layers(38).Name, tp2);
custom_unet_lg = replaceLayer(custom_unet_lg, custom_unet_lg.Layers(44).Name, tp3);

% Plot the customized U-Net model
figure(1)
plot(custom_unet_lg)

%% add cyclegan
% Import the CycleGAN ONNX model
onnx_layers = importONNXLayers('cyclegan.onnx', 'ImportWeights', 1);

% Customize the imported CycleGAN ONNX model
custom_cyclegan_lg = onnx_layers;
custom_cyclegan_lg = removeLayers(custom_cyclegan_lg, 'StatefulPartitionedCall|model_1|conv2d_19|BiasAdd__6');
custom_cyclegan_lg = removeLayers(custom_cyclegan_lg, 'StatefulPartitionedCall|model_1|conv2d_37|BiasAdd__140');
input_layer = imageInputLayer([512 512 1], 'Name',  'input_1', 'DataAugmentation', 'none', 'Normalization', 'none', 'AverageImage', []);
custom_cyclegan_lg = replaceLayer(custom_cyclegan_lg, 'Input_input_2', [input_layer]);
custom_cyclegan_lg = connectLayers(custom_cyclegan_lg, 'input_1', 'StatefulPartitionedCall|model_1|conv2d_19|BiasAdd');

% Replace transposed convolution layers with MATLAB's implementation
tp0 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(27).Weights, 'Bias', onnx_layers.Layers(27).Bias, 'name', 'tp0_1');
tp1 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(33).Weights, 'Bias', onnx_layers.Layers(33).Bias, 'name', 'tp1_1');
tp2 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(39).Weights, 'Bias', onnx_layers.Layers(39).Bias, 'name', 'tp2_1');
tp3 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(45).Weights, 'Bias', onnx_layers.Layers(45).Bias, 'name', 'tp3_1');
custom_cyclegan_lg = replaceLayer(custom_cyclegan_lg, custom_cyclegan_lg.Layers(26).Name, tp0);
custom_cyclegan_lg = replaceLayer(custom_cyclegan_lg, custom_cyclegan_lg.Layers(32).Name, tp1);
custom_cyclegan_lg = replaceLayer(custom_cyclegan_lg, custom_cyclegan_lg.Layers(38).Name, tp2);
custom_cyclegan_lg = replaceLayer(custom_cyclegan_lg, custom_cyclegan_lg.Layers(44).Name, tp3);

% Plot the customized CycleGAN model
figure(2)
plot(custom_cyclegan_lg)

%%
% Combine the U-Net and CycleGAN models
combined_lg = addLayers(custom_unet_lg, custom_cyclegan_lg.Layers);
figure(3)
plot(combined_lg)

% Connect the layers of the combined model
combined_lg = connectLayers(combined_lg, combined_lg.Layers(55).Name, [combined_lg.Layers(95).Name '/in2']);
combined_lg = connectLayers(combined_lg, combined_lg.Layers(60).Name, [combined_lg.Layers(89).Name '/in2']);
combined_lg = connectLayers(combined_lg, combined_lg.Layers(65).Name, [combined_lg.Layers(83).Name '/in2']);
combined_lg = connectLayers(combined_lg, combined_lg.Layers(70).Name, [combined_lg.Layers(77).Name '/in2']);

% Plot the combined model
figure(4)
plot(combined_lg)

%%
% Remove unnecessary layers and add the output regression layer
combined_lg = removeLayers(combined_lg, combined_lg.Layers(51).Name);
combined_lg = connectLayers(combined_lg, combined_lg.Layers(50).Name, combined_lg.Layers(51).Name);

output_reg = regressionLayer('Name', 'out');
combined_lg = addLayers(combined_lg, output_reg);
combined_lg = connectLayers(combined_lg, combined_lg.Layers(end-1).Name, 'out');

% Plot the final combined model
figure(5)
plot(combined_lg)

% Assemble the combined model
combined_model = assembleNetwork(combined_lg);

% Save the final customized ONNX model
save('verasonics/external_process/2019_DAGNetworks/combined_model.mat', "combined_model");
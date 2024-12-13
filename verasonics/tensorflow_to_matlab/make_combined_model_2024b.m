% Import the U-Net ONNX model
onnx_layers = importONNXLayers('unet_113.onnx', 'ImportWeights', 1);

% Customize the imported U-Net ONNX model
custom_unet_lg = onnx_layers;
custom_unet_lg = removeLayers(custom_unet_lg, 'ReshapeLayer1051');
custom_unet_lg = removeLayers(custom_unet_lg, 'ReshapeLayer1087');
custom_unet_lg = removeLayers(custom_unet_lg, 'OutputLayer_conv2d_18');
input_layer = imageInputLayer([512 512 1], 'Name',  'input', 'DataAugmentation', 'none', 'Normalization', 'none', 'AverageImage', []);
custom_unet_lg = replaceLayer(custom_unet_lg, 'input_1', [input_layer]);
custom_unet_lg = connectLayers(custom_unet_lg, 'input', 'StatefulPartitio_123');

% Replace transposed convolution layers with MATLAB's implementation
tp0 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(27).StatefulPartitio_64), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(27).StatefulPartitio_62), [1, 1, 16]), 'name', 'tp0');
tp1 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(33).StatefulPartitio_67), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(33).StatefulPartitio_65), [1, 1, 16]), 'name', 'tp1');
tp2 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(39).StatefulPartitio_70), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(39).StatefulPartitio_68), [1, 1, 16]), 'name', 'tp2');
tp3 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(45).StatefulPartitio_73), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(45).StatefulPartitio_71), [1, 1, 16]), 'name', 'tp3');
custom_unet_lg = replaceLayer(custom_unet_lg, custom_unet_lg.Layers(26).Name, tp0, "ReconnectBy", "order");
custom_unet_lg = replaceLayer(custom_unet_lg, custom_unet_lg.Layers(32).Name, tp1, "ReconnectBy", "order");
custom_unet_lg = replaceLayer(custom_unet_lg, custom_unet_lg.Layers(38).Name, tp2, "ReconnectBy", "order");
custom_unet_lg = replaceLayer(custom_unet_lg, custom_unet_lg.Layers(44).Name, tp3, "ReconnectBy", "order");

% Plot the customized U-Net model
figure(1)
plot(custom_unet_lg)

%% add cyclegan
% Import the CycleGAN ONNX model
onnx_layers = importONNXLayers('cyclegan.onnx', 'ImportWeights', 1);

% Customize the imported CycleGAN ONNX model
custom_cyclegan_lg = onnx_layers;
custom_cyclegan_lg = removeLayers(custom_cyclegan_lg, 'ReshapeLayer1051');
custom_cyclegan_lg = removeLayers(custom_cyclegan_lg, 'ReshapeLayer1087');
custom_cyclegan_lg = removeLayers(custom_cyclegan_lg, 'OutputLayer_conv2d_37');
input_layer = imageInputLayer([512 512 1], 'Name',  'input_1', 'DataAugmentation', 'none', 'Normalization', 'none', 'AverageImage', []);
custom_cyclegan_lg = replaceLayer(custom_cyclegan_lg, 'input_2', [input_layer]);
custom_cyclegan_lg = connectLayers(custom_cyclegan_lg, 'input_1', 'StatefulPartitio_100');

% Replace transposed convolution layers with MATLAB's implementation
tp0 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(27).StatefulPartitio_64), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(27).StatefulPartitio_62), [1, 1, 16]), 'name', 'tp0');
tp1 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(33).StatefulPartitio_67), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(33).StatefulPartitio_65), [1, 1, 16]), 'name', 'tp1');
tp2 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(39).StatefulPartitio_70), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(39).StatefulPartitio_68), [1, 1, 16]), 'name', 'tp2');
tp3 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(45).StatefulPartitio_73), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(45).StatefulPartitio_71), [1, 1, 16]), 'name', 'tp3');
custom_cyclegan_lg = replaceLayer(custom_cyclegan_lg, custom_cyclegan_lg.Layers(26).Name, tp0, "ReconnectBy", "order");
custom_cyclegan_lg = replaceLayer(custom_cyclegan_lg, custom_cyclegan_lg.Layers(32).Name, tp1, "ReconnectBy", "order");
custom_cyclegan_lg = replaceLayer(custom_cyclegan_lg, custom_cyclegan_lg.Layers(38).Name, tp2, "ReconnectBy", "order");
custom_cyclegan_lg = replaceLayer(custom_cyclegan_lg, custom_cyclegan_lg.Layers(44).Name, tp3, "ReconnectBy", "order");

% Plot the customized CycleGAN model
figure(2)
plot(custom_cyclegan_lg)

%%
% Combine the U-Net and CycleGAN models
for i = 1:numel(custom_cyclegan_lg.Layers)
    oldName = custom_cyclegan_lg.Layers(i).Name;
    newName = ['cyclegan_' oldName];
    newLayer = custom_cyclegan_lg.Layers(i);
    newLayer.Name = newName;
    custom_cyclegan_lg = replaceLayer(custom_cyclegan_lg, oldName, newLayer);
end
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
save('../external_process/2024b_DAGNetworks/combined_model.mat', "combined_model");
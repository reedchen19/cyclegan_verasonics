% Import the ONNX model
onnx_layers = importONNXLayers('cyclegan.onnx', 'ImportWeights', 1);

% Plot the imported ONNX layers
plot(onnx_layers)
findPlaceholderLayers(onnx_layers)

%%

% Customize the imported ONNX model
custom_onnx = onnx_layers;
custom_onnx = removeLayers(custom_onnx, 'ReshapeLayer1051')
custom_onnx = removeLayers(custom_onnx, 'ReshapeLayer1087')
custom_onnx = removeLayers(custom_onnx, 'OutputLayer_conv2d_37')
input_layer = imageInputLayer([512 512 1], 'Name',  'input', 'DataAugmentation', 'none', 'Normalization', 'none', 'AverageImage', []);
custom_onnx = replaceLayer(custom_onnx, 'input_2', [input_layer]);
custom_onnx = connectLayers(custom_onnx, 'input', 'StatefulPartitio_100');

% Plot the customized ONNX layers
plot(custom_onnx)
findPlaceholderLayers(custom_onnx)
%%

% Replace transposed convolution layers with MATLAB's implementation
tp0 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(27).StatefulPartitio_64), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(27).StatefulPartitio_62), [1, 1, 16]), 'name', 'tp0')
tp1 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(33).StatefulPartitio_67), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(33).StatefulPartitio_65), [1, 1, 16]), 'name', 'tp1')
tp2 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(39).StatefulPartitio_70), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(39).StatefulPartitio_68), [1, 1, 16]), 'name', 'tp2')
tp3 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(45).StatefulPartitio_73), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(45).StatefulPartitio_71), [1, 1, 16]), 'name', 'tp3')
custom_onnx = replaceLayer(custom_onnx, custom_onnx.Layers(26).Name, tp0, "ReconnectBy", "order")
custom_onnx = replaceLayer(custom_onnx, custom_onnx.Layers(32).Name, tp1, "ReconnectBy", "order")
custom_onnx = replaceLayer(custom_onnx, custom_onnx.Layers(38).Name, tp2, "ReconnectBy", "order")
custom_onnx = replaceLayer(custom_onnx, custom_onnx.Layers(44).Name, tp3, "ReconnectBy", "order")

% Add and connect the output regression layer
output_reg = regressionLayer('Name', 'out')
custom_onnx = addLayers(custom_onnx, output_reg)
custom_onnx = connectLayers(custom_onnx, custom_onnx.Layers(50).Name, 'out')

% Assemble the customized ONNX model
custom_onnx_cyclegan = assembleNetwork(custom_onnx)

% Plot the final customized ONNX model
figure()
plot(custom_onnx_cyclegan)

% Save the final customized ONNX model
save('../external_process/2024b_DAGNetworks/custom_onnx_cyclegan.mat', "custom_onnx_cyclegan");
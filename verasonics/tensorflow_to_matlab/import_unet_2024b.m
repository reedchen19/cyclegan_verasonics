% Import the ONNX model
onnx_layers = importONNXLayers('unet_113.onnx', 'ImportWeights', 1)

% Plot the imported ONNX layers
plot(onnx_layers)
findPlaceholderLayers(onnx_layers)

% Customize the imported ONNX model
custom_onnx2 = onnx_layers
custom_onnx2 = removeLayers(custom_onnx2, 'ReshapeLayer1051')
custom_onnx2 = removeLayers(custom_onnx2, 'ReshapeLayer1087')
custom_onnx2 = removeLayers(custom_onnx2, 'OutputLayer_conv2d_18')
input_layer = imageInputLayer([512 512 1], 'Name',  'input', 'DataAugmentation', 'none', 'Normalization', 'none', 'AverageImage', [])
custom_onnx2 = replaceLayer(custom_onnx2, 'input_1', [input_layer])
custom_onnx2 = connectLayers(custom_onnx2, 'input', 'StatefulPartitio_123')

plot(onnx_layers)
findPlaceholderLayers(onnx_layers)

% Replace transposed convolution layers with MATLAB's implementation
tp0 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(27).StatefulPartitio_64), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(27).StatefulPartitio_62), [1, 1, 16]), 'name', 'tp0')
tp1 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(33).StatefulPartitio_67), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(33).StatefulPartitio_65), [1, 1, 16]), 'name', 'tp1')
tp2 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(39).StatefulPartitio_70), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(39).StatefulPartitio_68), [1, 1, 16]), 'name', 'tp2')
tp3 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', permute(extractdata(onnx_layers.Layers(45).StatefulPartitio_73), [2,1,3,4]), 'Bias', reshape(extractdata(onnx_layers.Layers(45).StatefulPartitio_71), [1, 1, 16]), 'name', 'tp3')
custom_onnx2 = replaceLayer(custom_onnx2, custom_onnx2.Layers(26).Name, tp0, "ReconnectBy", "order")
custom_onnx2 = replaceLayer(custom_onnx2, custom_onnx2.Layers(32).Name, tp1, "ReconnectBy", "order")
custom_onnx2 = replaceLayer(custom_onnx2, custom_onnx2.Layers(38).Name, tp2, "ReconnectBy", "order")
custom_onnx2 = replaceLayer(custom_onnx2, custom_onnx2.Layers(44).Name, tp3, "ReconnectBy", "order")

% Add and connect the output regression layer
output_reg = regressionLayer('Name', 'out')
custom_onnx2 = addLayers(custom_onnx2, output_reg)
custom_onnx2 = connectLayers(custom_onnx2, custom_onnx2.Layers(50).Name, 'out')


% Assemble the customized ONNX model
custom_onnx_unet113 = assembleNetwork(custom_onnx2)

% Plot the final customized ONNX model
figure()
plot(custom_onnx2)

% Save the final customized ONNX model
save('verasonics/external_process/2024b_DAGNetworks/custom_onnx_unet113.mat', "custom_onnx_unet113");








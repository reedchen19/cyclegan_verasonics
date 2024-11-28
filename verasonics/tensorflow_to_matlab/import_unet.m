% Import the ONNX model
onnx_layers = importONNXLayers('./unet_113.onnx', 'ImportWeights', 1)

% Plot the imported ONNX layers
plot(onnx_layers)
findPlaceholderLayers(onnx_layers)

% Import the Keras model
keras_layers = importKerasLayers('./unet_113.h5')

% Customize the imported ONNX model
custom_onnx2 = onnx_layers
custom_onnx2 = removeLayers(custom_onnx2, 'StatefulPartitionedCall|model|conv2d|BiasAdd__6')
custom_onnx2 = removeLayers(custom_onnx2, 'StatefulPartitionedCall|model|conv2d_18|BiasAdd__140')
custom_onnx2 = replaceLayer(custom_onnx2, 'Input_input_1', [keras_layers.Layers(1)])
custom_onnx2 = connectLayers(custom_onnx2, 'input_1', 'StatefulPartitionedCall|model|conv2d|BiasAdd')

% Replace transposed convolution layers with MATLAB's implementation
tp0 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(27).Weights, 'Bias', onnx_layers.Layers(27).Bias, 'name', 'tp0')
tp1 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(33).Weights, 'Bias', onnx_layers.Layers(33).Bias, 'name', 'tp1')
tp2 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(39).Weights, 'Bias', onnx_layers.Layers(39).Bias, 'name', 'tp2')
tp3 = transposedConv2dLayer([3,3], 16, 'NumChannels', 16, 'Stride', [2,2], 'Cropping', 'same', 'Weights', onnx_layers.Layers(45).Weights, 'Bias', onnx_layers.Layers(45).Bias, 'name', 'tp3')
custom_onnx2 = replaceLayer(custom_onnx2, custom_onnx2.Layers(26).Name, tp0)
custom_onnx2 = replaceLayer(custom_onnx2, custom_onnx2.Layers(32).Name, tp1)
custom_onnx2 = replaceLayer(custom_onnx2, custom_onnx2.Layers(38).Name, tp2)
custom_onnx2 = replaceLayer(custom_onnx2, custom_onnx2.Layers(44).Name, tp3)

% Add and connect the output regression layer
output_reg = regressionLayer('Name', 'out')
custom_onnx2 = addLayers(custom_onnx2, output_reg)
custom_onnx2 = connectLayers(custom_onnx2, custom_onnx2.Layers(50).Name, 'out')

% Assemble the customized ONNX model
custom_onnx_unet113 = assembleNetwork(custom_onnx2)







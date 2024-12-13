# CycleGAN Verasonics

This repository contains the code for the paper [**"Emulating Clinical Quality Muscle B-mode Ultrasound Images from Plane-Wave Images Using a Two-Stage Machine Learning Model"**](https://doi.org/10.48550/arXiv.2412.05758). The code covers model training in Python and Verasonics implementation in MATLAB.

![plane wave image processing with ML](figure7.png)

## Repository Structure

- **`unet/`** and **`cyclegan/`**: These folders contain:
  - Datasets, models, trainers, and weights for the two-stage model: U-Net (stage 1) and CycleGAN (stage 2). Note that the U-Net and CycleGAN share the same model.
- **`verasonics/`**: This folder includes:
  - Code for converting U-Net and CycleGAN models from ONNX to MATLAB DAGNetworks.
  - Code to incorporate these DAGNetworks into Verasonics External Process objects, enabling real-time image enhancement on the Verasonics system.

---

### Verasonics Implementation

To implement the models on the Verasonics Vantage system, which uses a MATLAB interface, models trained in **Python 3.7** with **TensorFlow 2.9.1** were imported into **MATLAB R2019a/R2019b/R2024b** with the **Deep Learning Toolbox** and **Deep Learning Toolbox Converter for ONNX Model Format**. Due to compatibility and import challenges, several steps were taken:

1. **Model Format Conversion**:
   - **MATLAB R2019a** includes the `importKerasLayers` method for importing Keras HDF5 models. However, the method did not support several TensorFlow layers.
   - To overcome this, models were first saved in the **SavedModel** format with TensorFlow, then converted to **ONNX** format using `tf2onnx` and ONNX opset-9. 
      - `python -m tf2onnx.convert --saved-model ./unet/unet_113_savedwith37/ --opset 9 --output ./verasonics/tensorflow_to_matlab/unet_113.onnx`
      - `python -m tf2onnx.convert --saved-model ./cyclegan/long_01_ckpt-122/ --opset 9 --output ./verasonics/tensorflow_to_matlab/cyclegan.onnx`

2. **ONNX Model Import to MATLAB**:
   - ONNX models were imported into MATLAB using MATLAB’s `importONNXLayers` function. 
   - Due to differences between TensorFlow and MATLAB's layer implementations, additional corrections were made:
     - **Conv2DTranspose Layer**: TensorFlow’s `Conv2DTranspose` layer pads the input, whereas MATLAB’s implementation of the transposed convolution layer crops the output. To address this difference, the `Conv2DTranspose` layers were replaced with MATLAB’s implementation while preserving the original kernel weights.
     - **Reshape Layer**: ONNX models expect image data in `[size, channels, size]` format, whereas TensorFlow and MATLAB expect `[size, size, channels]`. As a result, `importONNXLayers` added incorrect reshape layers after the input layer and before the output layer. These reshape layers were removed, and a correctly formatted input layer (`[size, size, channels]`) and an output regression layer were added.

3. **Assembly of MATLAB DAGNetworks**:
   - Due to the two-stage nature of the model, three DAGNetworks were created:
     - **Stage 1**: A network for the first stage only,
     - **Stage 2**: A network for the second stage only,
     - **Combined**: A two-stage network connecting the output of stage 1 to the input of stage 2.
   - MATLAB code (for both R2019 and R2024b) used to import the ONNX models into MATLAB and assemble the DAGNetworks are located at:
     - `verasonics/tensorflow_to_matlab/import_unet_20xx.m`
     - `verasonics/tensorflow_to_matlab/import_cyclegan_20xx.m`
     - `verasonics/tensorflow_to_matlab/make_combined_model_20xx.m`
   - The corresponding DAGNetworks are saved at:
     - `verasonics/external_process/20xx_DAGNetworks/custom_onnx_unet113.mat`
     - `verasonics/external_process/20xx_DAGNetworks/custom_onnx_cyclegan.mat`
     - `verasonics/external_process/20xx_DAGNetworks/combined_model.mat`
   - `verasonics/external_process/test_DAGNetwork.m` can be used to load and test the saved DAGNetworks.

> **Note**: This code was successfully tested on MATLAB R2019a, R2019b, and R2024b. Version-specific MATLAB files are designated with either `2019` (tested with R2019a and R2019b) or `2024b` (tested with R2024b) in the filename. The differences between the version-specific MATLAB files are due to changes to the Deep Learning Toolbox,  . Later MATLAB versions support a wider range of TensorFlow layers with `importTensorFlowNetwork` and `importTensorFlowLayers`, which may simplify some of these steps. However, the Verasonics Vantage system in this study supported only MATLAB R2019a, thus `importONNXLayers` was used instead.

---

### Integrating the Model into Verasonics Acquisition Sequences

1. A **Verasonics External Process object** was created that reads **plane-wave intensity data** from the `ImageBuffer`.
   
2. The External Process object calls a MATLAB function that:
    - Loads the DAGNetwork as a persistent variable,
    - Resizes the intensity data to 512x512 and passes the data through the DAGNetwork.
    - Displays the enhanced output image on a persistent MATLAB figure, allowing for real-time image enhancement on the Verasonics system.
   
   These MATLAB functions called by the External Process object are located at:
    - `verasonics/external_process/ml_filter_verasonics.m` displays only the output of the combined model
    - `verasonics/external_process/ml_filter_verasonics_separate_stages.m` displays the input image, the output of the first stage U-Net, and the output of the second stage CycleGAN

--- 

This repository provides a framework for enhancing ultrasound imaging with machine learning models directly the Verasonics, enabling real-time image reconstruction with seamless integration.

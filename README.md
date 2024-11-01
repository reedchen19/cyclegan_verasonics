# CycleGAN Verasonics

This repository contains the code for the paper **"Emulating Clinical Quality Muscle B-mode Ultrasound Images from Plane-Wave Images Using a Two-Stage Machine Learning Model"**. The code covers model training in Python and Verasonics implementation in MATLAB.

![plane wave image processing with ML](figure7.png)

## Repository Structure

- **`unet/`** and **`cyclegan/`**: These folders contain:
  - Datasets, models, trainers, and weights for the two-stage model: U-Net (stage 1) and CycleGAN (stage 2). Note that the U-Net and CycleGAN share the same model.
- **`verasonics/`**: This folder includes:
  - Code for converting U-Net and CycleGAN models from ONNX to MATLAB DAGNetworks.
  - Code to incorporate these DAGNetworks into Verasonics External Process objects, enabling real-time image enhancement on the Verasonics system.

---

### Verasonics Implementation

To implement the models on the Verasonics Vantage system, which uses a MATLAB interface, models trained in **Python 3.7** with **TensorFlow 2.9.1** were imported into **MATLAB R2019a**. Due to compatibility and import challenges, several steps were taken:

1. **Model Format Conversion**:
   - **MATLAB R2019a** includes the `importKerasLayers` method for importing Keras HDF5 models. However, the method did not support several TensorFlow layers.
   - To overcome this, models were first saved in the **SavedModel** format with TensorFlow, then converted to **ONNX** format using `tf2onnx` and ONNX opset-9.

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

> **Note**: Later MATLAB versions support a wider range of TensorFlow layers with `importTensorFlowNetwork` and `importTensorFlowLayers`, which may simplify some of these steps. However, the Verasonics Vantage system in this study supported only MATLAB R2019a.

---

### Integrating the Model into Verasonics Acquisition Sequences

1. A **Verasonics External Process object** was created that reads **plane-wave intensity data** from the `ImageBuffer`.
   
2. The External Process object calls a MATLAB function that:
    - Loads the DAGNetwork as a persistent variable,
    - Resizes the intensity data to 512x512 and passes the data through the DAGNetwork.
    - Displays the enhanced output image on a persistent MATLAB figure, allowing for real-time image enhancement on the Verasonics system.

--- 

This repository provides a framework for enhancing ultrasound imaging with machine learning models directly the Verasonics, enabling real-time image reconstruction with seamless integration.

# ML_QuickDraw
- This is a ML way to recognize the easy drawing and the model is small/efficient enough to deployment on edge device such as MCU by tflite. 
- The dataset is from [The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset/tree/master) and [Quick, Draw! Doodle Recognition Challenge](https://www.kaggle.com/competitions/quickdraw-doodle-recognition/overview).
- The notebooks are reference from [Greyscale MobileNet](https://www.kaggle.com/code/gaborfodor/greyscale-mobilenet-lb-0-892).

## 1. First step
### a. Install virtual env  
- If you haven't installed [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise), please follow these steps to install Python virtual environment and ***choose `NuEdgeWise_env`***.
- Skip if you have already done it.
### b. Running
- The `Shuffle_CSVs.ipynb`, `MobileNet.ipynb`, `MobileNet_finetune.ipynb` will help you prepare data, train the model, and finally convert it to a TFLite.

## 2. Work Flow
### a.
### 1. Data prepare
- Users can utilize `classfication.ipynb` to download easy datasets, prepare their custom datasets (or even download from other open-source platforms like Kaggle).
- `classfication.ipynb` will prepare the user's chosen dataset folder, supporting a general structure where the folder names correspond to class labels.

### 2. Training
- `classfication.ipynb` offers some attributes for training configuration.
- The strategy of this image classification training is [transfer learning & fine-tunning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- The output is tflite model.

### 3. Test
- Use `classfication.ipynb` to test the tflite model.

### 4. Deployment
- Utilize `classfication.ipynb` to convert the TFLite model to Vela and generate C source/header files.
- Also support Label source/header files converting.
- The `cmd.ipynb` notebook will demonstrate how to use the script located in `datasets\gen_rgb_cpp.py` to convert an image to a bytes source file.


# ML_QuickDraw
- This is a ML way to recognize the easy drawing and the model is small/efficient enough to deployment on edge device such as MCU by tflite. 
- The dataset is from [The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset/tree/master) and [Quick, Draw! Doodle Recognition Challenge](https://www.kaggle.com/competitions/quickdraw-doodle-recognition/overview).
- The notebooks are reference from [Greyscale MobileNet](https://www.kaggle.com/code/gaborfodor/greyscale-mobilenet-lb-0-892).
- The notebooks lead users step by step to create the model or you can use pre-train model directly.

## 1. First step
### a. Install virtual env  
- If you haven't installed [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise), please follow these steps to install Python virtual environment and ***choose `NuEdgeWise_env`***.
- Skip if you have already done it.
### b. Running
- The `Shuffle_CSVs.ipynb`, `MobileNet.ipynb`, `MobileNet_finetune.ipynb` will help you prepare data, train the model, and finally convert it to a TFLite.

## 2. Work Flow - training from scratch
### a. Download the Dataset
- Register on Kaggle and download the `train_simplified` dataset from [quickdraw-doodle-recognition dataset](https://www.kaggle.com/competitions/quickdraw-doodle-recognition/data?select=train_simplified).
- This dataset is the same as the [Simplified Drawing files (.ndjson)](https://github.com/googlecreativelab/quickdraw-dataset#preprocessed-dataset) but it's provided in CSV format and includes 340 categories.
### b. Data prepare
- Users can make use of the `Shuffle_CSVs.ipynb` notebook to shuffle either the entire dataset or a specific portion of it.
- This shuffling process is important and assists us in easily splitting the large amount of data into training and validation datasets.

### c. Training
- The `MobileNet.ipynb` guides you through the steps of training the model, validating it, and converting it to tflite. It also provides various attributes for configuring the training process.
- It provides MobileNet v1 to v3 with varying widths. However, based on our experiments, we have found that MobileNetv1 is superior in terms of accuracy, model size, and training time.

### d. Test
- Use `MobileNet.ipynb` to test the tflite model.

### e. Deployment
- Utilize `MobileNet.ipynb` to convert Keras model to the TFLite model.
- Use `/tflu/gen_model_cpp_no_vela.bat` to convert the TFLite model into a C source file. Remember to update `/vela/variables_no_vela.bat` with the location of your model.

## 3. Work Flow - transfer learning and fine tune
- Understanding this method:  [transfer learning & fine-tunning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- Use Case 1: If you're interested in using specific categories from the 340 available categories and prefer not to start the training process from the beginning.
- Use Case 2: If you possess your own data (in a format grayscale PNG with matched pixel dimensions, e.g., 64x64) and wish to integrate it with the original dataset to train a new model.
### a. Data prepare
- Users can use `convert2png.py` to transform the entire or a portion of the CSV vector dataset (previously downloaded as train_simplified/) into an image dataset, generating training, validation, and test datasets.
- Users can incorporate their own data into the respective category (label) folder within the image dataset. For instance, it can be added to `dataset\{USER_DEFINED_FOLDER_NAME}\train\{THE_CATEGORY}` or `dataset\{USER_DEFINED_FOLDER_NAME}\validation\{THE_CATEGORY}`.

### b. Training
- "The `MobileNet_finetune.ipynb` guides you through the process of using transfer learning and fine-tuning step by step. It also provides various attributes for configuring the training.
- You will need a pre-trained model located in the /pretrain_model directory.
- It provides MobileNet v1 to v3 with various widths. However, based on our experiments, MobileNetv1 has shown better results in terms of accuracy, model size, and training time.

### c. Test
- Use `MobileNet_finetune.ipynb` to test the tflite model.

### d. Deployment
- Utilize ``MobileNet_finetune.ipynb` to convert Keras model to the TFLite model.
- Use `/tflu/gen_model_cpp_no_vela.bat` to convert the TFLite model into a C source file. Remember to update `/vela/variables_no_vela.bat` with the location of your model.

## 4. Work Flow - Optimization
- Please check the `MobileNet.ipynb` and `MobileNet_finetune.ipynb`.
- Currently, clustering is beneficial only on devices equipped with NPU+Vela compiler.

## 5. Inference code on Device
- MCU: 
- MPU: 

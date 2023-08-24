set IMAGE_SRC_DIR=..\samples
set IMAGE_SRC_WIDTH=320
set IMAGE_SRC_HEIGHT=320

::set LABEL_SRC_FILE=..\labels\labels_mobilenet_v2_1.0_224.txt
::set GEN_LABEL_FILE_NAME=Labels

set MODEL_SRC_DIR=..\workspace\340_mobilenetv2_035\tflite
set MODEL_SRC_FILE=mobilenetv2_035_340_int8quant.tflite
::The vela OPTIMISE_FILE should be SRC_FILE_NAME + _vela

set TEMPLATES_DIR=Tool\tflite2cpp\templates

set GEN_SRC_DIR=..\workspace\340_mobilenetv2_035\tflite
set GEN_INC_DIR=generated\include











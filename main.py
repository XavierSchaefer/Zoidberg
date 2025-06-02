import os
import sys

# 1) Redirige le file-descriptor 2 (stderr) vers /dev/null
#    => supprime tous les messages C/C++ (CUDA, XLA, cuDNN, cuBLAS, etc.)
# sys.stderr.flush()
# devnull_fd = os.open(os.devnull, os.O_RDWR)
# os.dup2(devnull_fd, sys.stderr.fileno())

# 2) Désactive ensuite les logs de TF/Python (infos et warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'    # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'    # désactive le oneDNN info


# from src.randomForest import randomForestModel
# from src.DenseNeuronalNetwork import DenseNeuronalNetworkModel
# from src.EfficientNetV2 import EfficientNetV2
# from src.CNN import ConvolutionalNeuralNetworkModel
from src.CNNV2 import ConvolutionalNeuralNetworkModelV2
from src.CNNV3 import ConvolutionalNeuralNetworkV3
from src.analyse.CNNAnalyser import analyseModel
ROOT = os.getcwd()  # /mnt/.../Projet/code
LOAD_MODEL = False
MODEL_NAME = "CNNV3"
FILE_NAME = "CNNV3_50epochs_0.001lr_4patience_42seed.keras"

def main():
  # randomForestModel()
  # DenseNeuronalNetworkModel()
  # ConvolutionalNeuralNetworkModel()
  # ConvolutionalNeuralNetworkModelV2()   
  # EfficientNetV2()


  if(LOAD_MODEL):
    model_path = os.path.join(ROOT,"src", "models", MODEL_NAME, FILE_NAME)
    test_path  = os.path.join(ROOT, "dataset", "initial", "test")
    analyseModel(model_path, test_path)
  else:
    ConvolutionalNeuralNetworkV3()
main()

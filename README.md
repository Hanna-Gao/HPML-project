# 6998 HPML Final Project

## Description

The overarching goal of this project is to enhance an existing Lightweight facial emotion recognition system on GitHub by optimizing its data processing, model performance, and deployment efficiency. We will retrain the model with the existing model and we aim to achieve higher recognition accuracy and better runtime performance by several optimization techniques. 

## Outline of the code respository

./dataset: dataset in csv format used for training\
./src: all source code for training and optimizing\
---PlotNumWorkers.py\
---model.py: define network structure\
---trainNumWorkers.py\
---trainHPtuning.py\
---trainOptimalParam.py\
---trainOriginal.py\
./outputs: raw output saved in numpy array or text files\
./plots: images of result plots\
./trained: save trained model checkpoints\

## Retrain

1. Prepare the dataset: please see [here](./dataset/README.md)

2. Hardware Requirements:
   * CPU: 8 vCPU, 4 cores, 30 GB memory
   * GPU: 1 NVIDIA T4 GPU
     
3. Install Software:
   * PyTorch
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url 	https://download.pytorch.org/whl/cu116
```
  * Wandb
```
pip install wandb
```

4. Retrain
   * Data Loading Optimization: number of workers in DataLoader as command line parameter
```
cd src
python trainNumWorkers.py -n 2
```
   * Hyperparameter Tuning
```
cd src
python trainHPtuning.py
```
   * Train model with the optimal set of Hyperparmeters
```
cd src
python trainOptimalParam.py
```

## Results
1. Dataloading optimization:
   <img src="./plots/NumWorkersvsTime.png">
   
3. Hyperparameter Tuning:
   <img src="./plots/HPaccuracy.png">
   Link to Weights and Biase Project: https://wandb.ai/6998/6998-proj2?workspace=user-qg2205 


## Reference

* Base Model: https://github.com/yoshidan/pytorch-facial-expression-recognition

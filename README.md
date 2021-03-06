# ND-Adam

This repository contains the code for the paper [Normalized Direction-preserving Adam](https://arxiv.org/abs/1709.04546).  ND-Adam is a tailored version of Adam for training DNNs, which bridges the generalization gap between Adam and SGD.

The code is based on a [TensorFlow implementation](https://github.com/tensorflow/models/tree/master/research/resnet) and a [PyTorch implementation](https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch) of wide residual networks.

## Usage
#### PyTorch
##### CIFAR-10
```
## ND-Adam
main.py --depth 28 --width 10 --dataroot ../cifar10 --save ./logs/resnet_model --ngpu 1 --gpu_id 0
```
##### CIFAR-100
```
## ND-Adam
main.py --depth 28 --width 10 --dataset CIFAR100 --dataroot ../cifar100 --save ./logs/resnet_model --ngpu 1 --gpu_id 0
```

#### TensorFlow
##### CIFAR-10
```
# Training
## ND-Adam
resnet_main.py --bnsoftmax_scale 2.5 --train_data_path=cifar10/data_batch* --log_root=./resnet_model --train_dir=./resnet_model/train --dataset=cifar10 --num_gpus=1

## Adam
resnet_main.py --optimizer adam --bnsoftmax_scale 2.5 --train_data_path=cifar10/data_batch* --log_root=./resnet_model --train_dir=./resnet_model/train --dataset=cifar10 --num_gpus=1

## SGD with momentum
resnet_main.py --optimizer mom --init_learning_rate 0.1 --weight_decay 0.001 --bnsoftmax_scale 2.5 --train_data_path=cifar10/data_batch* --log_root=./resnet_model --train_dir=./resnet_model/train --dataset=cifar10 --num_gpus=1

# Evaluation
resnet_main.py --bnsoftmax_scale 2.5 --eval_data_path=cifar10/test_batch.bin --log_root=./resnet_model --eval_dir=./resnet_model/test --mode=eval --dataset=cifar10 --num_gpus=0
```
##### CIFAR-100

Change `--train_data_path`,  `--eval_data_path`, and `--dataset` accordingly,  and replace `--bnsoftmax_scale 2.5` with `--bnsoftmax_scale 1`.
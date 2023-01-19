# TWNs

## Classification Task

First, you should  move into the directory `cd cls`

You can run the following code to train on MNIST dataset

`python main.py --data mnist --data_path <path/to/mnist> --model_type lenet5 --epochs 100 --lr-epochs 15 --lr 1e-3`

You can run the following code to train on CIFAR-10 dataset

`python main.py --data cifar10 --data_path <path/to/cifar10> --model_type vgg7 --epochs 150 --lr-epochs 30 --lr 0.1`

You can run the following code to train on CIFAR-100 dataset

`python main.py --data cifar100 --data_path <path/to/cifar100> --model_type vgg7 --epochs 150 --lr-epochs 30 --lr 0.1`

Your can run the following code to train on ImageNet dataset

`python main.py --data imagenet --data_path <path/to/imagenet> --model_type restnet18 --epochs 150 --lr-epochs 15 --lr 0.1`


## Detection Task

First, you should  move into the directory `cd det`

Your can run the following code to train on VOC dataset

`python train.py --data data/VOC.yaml --weights '' --cfg yolov5s_TWN.yaml`

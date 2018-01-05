# Improved-WGAN
A Theano implementation of the paper "[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)" based on the original [Tensorflow implementation](https://github.com/igul222/improved_wgan_training)

## Requirements
[Theano v0.9](http://deeplearning.net/software/theano/)

[NeuralNet](https://github.com/justanhduc/neuralnet)

## Results
The MNIST images are obtained at iter 20k (the original code runs for 200k). The CIFAR images are generated at iter 18k.

![MNIST images](https://github.com/justanhduc/Improved-WGAN/blob/master/results/mnist.jpg)
![CIFAR images](https://github.com/justanhduc/Improved-WGAN/blob/master/results/cifar.jpg)

## Usages
To train WGAN on MNIST using the current training scheme

```
python train_wgan_mnist.py
```

To train WGAN on CIFAR using the current training scheme

```
python train_wgan_cifar.py
```

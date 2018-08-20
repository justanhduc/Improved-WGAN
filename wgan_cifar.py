import neuralnet as nn
from neuralnet import Model
from neuralnet import read_data
from neuralnet.utils import DataManager, inference

import numpy as np
from theano import tensor as T


class WGAN(Model):
    def __init__(self, config_file, **kwargs):
        super(WGAN, self).__init__(config_file, **kwargs)

        self.name = self.config['model']['name']
        self.noise_input_shape = self.config['model']['noise_input_shape']
        self.image_input_shape = self.config['model']['image_input_shape']
        self.output_shape = self.config['model']['output_shape']
        self.augmentation = self.config['model']['augmentation']
        self.dim = self.config['model']['dim']
        self.LAMBDA = self.config['model']['lambda']
        self.critic_iters = self.config['model']['critic_iters']
        self.noise_tensor_shape = tuple([None] + [self.noise_input_shape])
        self.image_tensor_shape = tuple([None] + [self.image_input_shape[2]] + self.image_input_shape[:2])

        self.gen = nn.Sequential(input_shape=self.noise_tensor_shape, layer_name='generator')
        self.gen.append(nn.FullyConnectedLayer(self.gen.output_shape, 4 * 4 * 4 * self.dim, activation='linear',
                                               layer_name='gen1'))
        self.gen.append(nn.BatchNormLayer(self.gen.output_shape, layer_name='gen1_bn', axes='spatial', no_scale=False,
                                          activation='relu'))

        shape = (-1, 4*self.dim, 4, 4)
        self.gen.append(nn.ReshapingLayer(self.gen.output_shape, shape, 'gen2_reshape'))

        shape = self.gen.output_shape
        self.gen.append(nn.TransposedConvolutionalLayer(shape, 2 * self.dim, 5, (2 * shape[2], 2 * shape[3]),
                                                        layer_name='gen2', padding='half', activation='linear'))
        self.gen.append(nn.BatchNormLayer(self.gen.output_shape, 'gen2_bn', axes='per-activation', no_scale=False,
                                          activation='relu'))

        shape = self.gen.output_shape
        self.gen.append(nn.TransposedConvolutionalLayer(shape, self.dim, 5, (2 * shape[2], 2 * shape[3]),
                                                        layer_name='gen3', padding='half', activation='linear'))
        self.gen.append(nn.BatchNormLayer(self.gen.output_shape, 'gen3_bn', axes='per-activation', no_scale=False,
                                          activation='relu'))

        shape = self.gen.output_shape
        self.gen.append(nn.TransposedConvolutionalLayer(shape, 3, 5, (2 * shape[2], 2 * shape[3]),
                                                        layer_name='gen4', padding='half', activation='tanh'))
        self.model.append(self.gen)

        self.dis = nn.Sequential(input_shape=self.gen.output_shape, layer_name='discriminator')
        self.dis.append(nn.ConvolutionalLayer(self.dis.output_shape, self.dim, 5, stride=(2, 2), activation='lrelu',
                                              layer_name='disc1'))
        self.dis.append(nn.ConvolutionalLayer(self.dis.output_shape, 2*self.dim, 5, stride=(2, 2), activation='lrelu',
                                              layer_name='disc2'))
        self.dis.append(nn.ConvolutionalLayer(self.dis.output_shape, 4*self.dim, 5, stride=(2, 2), activation='lrelu',
                                              layer_name='disc3'))
        self.dis.append(nn.FullyConnectedLayer(self.dis.output_shape, 1, activation='linear', layer_name='disc4'))
        self.model.append(self.dis)

    def inference(self, input):
        output = self.dis(self.gen(input))
        return output


class DataManager2(DataManager):
    def __init__(self, config_file, placeholders):
        super(DataManager2, self).__init__(config_file, placeholders)
        self.load_data()

    def load_data(self):
        X_train, _, X_test, _ = read_data.load_dataset(self.path)
        self.dataset = np.float32(2.*((X_train / 255.) - 0.5))
        self.data_size = X_train.shape[0]

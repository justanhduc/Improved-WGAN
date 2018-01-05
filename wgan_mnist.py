from neuralnet import layers
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
        self.CRITIC_ITERS = self.config['model']['critic_iters']
        self.noise_tensor_shape = tuple([None] + [self.noise_input_shape])
        self.image_tensor_shape = tuple([None] + [self.image_input_shape[2]] + self.image_input_shape[:2])

        self.network = {'generator': [], 'discriminator': []}
        subnet = 'generator'
        self.network[subnet].append(layers.FullyConnectedLayer(self.noise_tensor_shape, 4 * 4 * 4 * self.dim,
                                                               activation='relu', layer_name='gen1'))
        # self.network[subnet].append(layers.BatchNormLayer(self.network[subnet][-1].output_shape, layer_name='gen1_bn',
        #                                                   axes='spatial', no_scale=False, activation='relu'))

        shape = (-1, 4*self.dim, 4, 4)
        self.network[subnet].append(layers.ReshapingLayer(self.network[subnet][-1].output_shape, shape, 'gen2_reshape'))
        shape = self.network[subnet][-1].output_shape
        self.network[subnet].append(layers.TransposedConvolutionalLayer(shape, (4*self.dim, 2*self.dim, 5, 5),
                                                                        (shape[0], 2*self.dim, 2*shape[2], 2*shape[3]),
                                                                        layer_name='gen2', padding='half', activation='relu'))
        # self.network[subnet].append(layers.BatchNormLayer(self.network[subnet][-1].output_shape, 'gen2_bn',
        #                                                   axes='per-activation', no_scale=False, activation='relu'))
        self.network[subnet].append(layers.SlicingLayer(self.network[subnet][-1].output_shape, (7, 7), layer_name="gen2_slicing"))

        shape = self.network[subnet][-1].output_shape
        self.network[subnet].append(layers.TransposedConvolutionalLayer(shape, (2*self.dim, self.dim, 5, 5),
                                                                        (shape[0], self.dim, 2*shape[2], 2*shape[3]),
                                                                        layer_name='gen3', padding='half', activation='relu'))
        # self.network[subnet].append(layers.BatchNormLayer(self.network[subnet][-1].output_shape, 'gen3_bn',
        #                                                   axes='per-activation', no_scale=False, activation='relu'))

        shape = self.network[subnet][-1].output_shape
        self.network[subnet].append(layers.TransposedConvolutionalLayer(shape, (self.dim, self.output_shape[2], 5, 5),
                                                                        (shape[0], self.output_shape[2], 2 * shape[2], 2 * shape[3]),
                                                                        layer_name='gen4', padding='half', activation='sigmoid'))

        subnet = 'discriminator'
        self.network[subnet].append(layers.ConvolutionalLayer(self.network['generator'][-1].output_shape,
                                                              (self.dim, self.network['generator'][-1].output_shape[1], 5, 5),
                                                              stride=(2, 2), activation='lrelu', layer_name='disc1'))
        self.network[subnet].append(layers.ConvolutionalLayer(self.network[subnet][-1].output_shape,
                                                              (2*self.dim, self.dim, 5, 5), stride=(2, 2), activation='lrelu',
                                                              layer_name='disc2'))
        self.network[subnet].append(layers.ConvolutionalLayer(self.network[subnet][-1].output_shape,
                                                              (4*self.dim, 2*self.dim, 5, 5), stride=(2, 2), activation='lrelu',
                                                              layer_name='disc3'))
        self.network[subnet].append(layers.FullyConnectedLayer(self.network[subnet][-1].output_shape, 1,
                                                               activation='linear', layer_name='disc4'))
        super(WGAN, self).get_all_params()
        super(WGAN, self).get_trainable()
        super(WGAN, self).get_regularizable()

    def inference_generator(self, input):
        output = inference(input, self.network['generator'])
        return output

    def inference_discriminator(self, input):
        output = inference(input, self.network['discriminator'])
        return output

    def inference(self, input):
        output = self.inference_discriminator(self.inference_generator(input))
        return output


class DataManager2(DataManager):
    def __init__(self, config_file, placeholders):
        super(DataManager2, self).__init__(config_file, placeholders)

    def load_data(self):
        import os
        import urllib
        import gzip
        import pickle
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

        if not os.path.isfile(self.path + '/mnist.pkl.gz'):
            print("Couldn't find MNIST dataset in %s, downloading..." % self.path)
            urllib.request.urlretrieve(url, self.path)

        with gzip.open(self.path + '/mnist.pkl.gz', 'rb') as f:
            train_data, _, _ = pickle.load(f, encoding='latin-1')
            f.close()
        imgs, _ = train_data
        self.training_set = np.reshape(imgs.astype('float32'), (-1, 1, 28, 28))
        self.num_train_data = imgs.shape[0]

    def generator(self, stage='train'):
        index = np.arange(0, self.num_train_data)
        np.random.shuffle(index)
        data = self.training_set[index]
        for i in range(0, self.num_train_data, self.batch_size):
            yield data[i:i+self.batch_size]

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
                                                               activation='linear', layer_name='gen1'))
        self.network[subnet].append(layers.BatchNormLayer(self.network[subnet][-1].output_shape, layer_name='gen1_bn',
                                                          axes='spatial', no_scale=False, activation='relu'))

        shape = (-1, 4*self.dim, 4, 4)
        self.network[subnet].append(layers.ReshapingLayer(self.network[subnet][-1].output_shape, shape, 'gen2_reshape'))
        shape = self.network[subnet][-1].output_shape
        self.network[subnet].append(layers.TransposedConvolutionalLayer(shape, (4*self.dim, 2*self.dim, 5, 5),
                                                                        (shape[0], 2*self.dim, 2*shape[2], 2*shape[3]),
                                                                        layer_name='gen2', padding='half', activation='linear'))
        self.network[subnet].append(layers.BatchNormLayer(self.network[subnet][-1].output_shape, 'gen2_bn',
                                                          axes='per-activation', no_scale=False, activation='relu'))

        shape = self.network[subnet][-1].output_shape
        self.network[subnet].append(layers.TransposedConvolutionalLayer(shape, (2*self.dim, self.dim, 5, 5),
                                                                        (shape[0], self.dim, 2*shape[2], 2*shape[3]),
                                                                        layer_name='gen3', padding='half', activation='linear'))
        self.network[subnet].append(layers.BatchNormLayer(self.network[subnet][-1].output_shape, 'gen3_bn',
                                                          axes='per-activation', no_scale=False, activation='relu'))

        shape = self.network[subnet][-1].output_shape
        self.network[subnet].append(layers.TransposedConvolutionalLayer(shape, (self.dim, 3, 5, 5),
                                                                        (shape[0], 3, 2 * shape[2], 2 * shape[3]),
                                                                        layer_name='gen4', padding='half', activation='tanh'))

        subnet = 'discriminator'
        self.network[subnet].append(layers.ConvolutionalLayer(self.network['generator'][-1].output_shape,
                                                              (self.dim, 3, 5, 5), stride=(2, 2), activation='lrelu',
                                                              layer_name='disc1'))
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
        X_train, _, X_test, _ = read_data.load_dataset(self.path)
        self.training_set = 2.*((X_train / 255.) - 0.5)
        self.num_train_data = X_train.shape[0]

    def augment_minibatches(self, minibatches, *args):
        """
        Randomly augments images by horizontal flipping with a probability of
        `flip` and random translation of up to `trans` pixels in both directions.
        """
        flip, trans = args
        for batch in minibatches:
            if self.no_target:
                inputs = batch
            else:
                inputs, targets = batch

            batchsize, c, h, w = inputs.shape
            if flip:
                coins = np.random.rand(batchsize) < flip
                inputs = [inp[:, :, ::-1] if coin else inp
                          for inp, coin in zip(inputs, coins)]
                if not trans:
                    inputs = np.asarray(inputs)
            outputs = inputs
            if trans:
                outputs = np.empty((batchsize, c, h, w), inputs[0].dtype)
                shifts = np.random.randint(-trans, trans, (batchsize, 2))
                for outp, inp, (x, y) in zip(outputs, inputs, shifts):
                    if x > 0:
                        outp[:, :x] = 0
                        outp = outp[:, x:]
                        inp = inp[:, :-x]
                    elif x < 0:
                        outp[:, x:] = 0
                        outp = outp[:, :x]
                        inp = inp[:, -x:]
                    if y > 0:
                        outp[:, :, :y] = 0
                        outp = outp[:, :, y:]
                        inp = inp[:, :, :-y]
                    elif y < 0:
                        outp[:, :, y:] = 0
                        outp = outp[:, :, :y]
                        inp = inp[:, :, -y:]
                    outp[:] = inp
            yield outputs, targets if not self.no_target else outputs

    def generator(self, stage='train'):
        index = np.arange(0, self.num_train_data)
        np.random.shuffle(index)
        data = self.training_set[index]
        for i in range(0, self.num_train_data, self.batch_size):
            yield data[i:i+self.batch_size].astype('float32')

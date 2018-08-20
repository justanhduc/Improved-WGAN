import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import theano
from theano import tensor as T

import neuralnet as nn
from wgan_cifar import WGAN, DataManager2

srng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=2546)


def train(config_file, *args):
    net = WGAN(config_file)

    noise = srng.normal((net.batch_size, net.noise_input_shape), dtype='float32')
    real_data = T.tensor4('input_real', theano.config.floatX)

    image_ = theano.shared(np.zeros((net.batch_size, net.image_input_shape[2], net.image_input_shape[0],
                                     net.image_input_shape[1]), 'float32'))

    #build train ops
    net.set_training_status(True)
    fake_data = net.gen(noise)
    disc_real = net.dis(real_data)
    disc_fake = net.dis(fake_data)

    #cost
    gen_cost = -T.mean(disc_fake)
    disc_cost = T.mean(disc_fake) - T.mean(disc_real)

    if args[0] == 'wgan-gp':
        #gradient penalty
        alpha = srng.uniform((net.batch_size, ), dtype='float32')
        diff = fake_data - real_data
        interpolates = real_data + (alpha.dimshuffle(0, 'x', 'x', 'x') * diff)
        gradients = T.grad(T.sum(net.dis(interpolates)), [interpolates])[0]
        slopes = T.sqrt(T.sum(T.sqr(gradients), axis=(1, 2, 3)))
        gradient_penalty = T.mean(T.sqr(slopes-1))
        disc_cost += net.LAMBDA * gradient_penalty
    elif args[0] == 'wgan':
        w_clip_updates = [(p, T.clip(p, -.01, .01)) for p in net.dis.trainable]
        clip_weights = nn.function([], updates=w_clip_updates, name='clip weights')
    elif args[0] == 'dcgan':
        raise NotImplementedError
    else:
        raise NotImplementedError

    gen_updates = nn.adam(gen_cost, net.gen.trainable, net.learning_rate, net.beta1, net.beta2)
    disc_updates = nn.adam(disc_cost, net.dis.trainable, net.learning_rate, net.beta1, net.beta2)

    #compile Theano train funcs
    train_gen = nn.function([], gen_cost, updates=gen_updates, name='train generator')
    train_disc = nn.function([], disc_cost, updates=disc_updates, givens={real_data: image_},
                             name='train discriminator')

    #build test op and func
    net.set_training_status(False)
    fixed_noise = T.as_tensor_variable(np.random.normal(size=(net.validation_batch_size, 128)).astype('float32'))
    gen_data = net.gen(fixed_noise)
    test_gen = nn.function([], gen_data, name='test generator')

    #training scheme
    dm = DataManager2(config_file, image_)
    mon = nn.monitor.Monitor(config_file)
    num_train_batches = (dm.data_size // net.batch_size) // net.critic_iters
    epoch = 0
    print('Training...')
    while epoch < net.n_epochs:
        epoch += 1
        batches = dm.get_batches(epoch, net.n_epochs)

        for it in range(num_train_batches):
            iteration = (epoch - 1.) * num_train_batches + it + 1

            #update generator
            training_gen_cost = train_gen()
            if np.isnan(training_gen_cost):
                raise ValueError('Training failed due to NaN cost')
            mon.plot('training gen cost', training_gen_cost)

            #update discriminator
            for i in range(net.critic_iters):
                batches.__next__()
                training_disc_cost = train_disc()
                if args[0] == 'wgan':
                    clip_weights()
                mon.plot('training disc cost', training_disc_cost)

            if iteration % net.validation_frequency == 0:
                gen_images = test_gen()
                mon.save_image('generated image', gen_images / 2 + .5)
                mon.flush()
            mon.tick()
    print('Training finished!')


if __name__ == '__main__':
    train('wgan_cifar.config', 'wgan-gp')

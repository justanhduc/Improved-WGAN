import time
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from wgan_cifar import WGAN, DataManager2
from neuralnet import monitor
from neuralnet.optimization import Adam

srng = RandomStreams(seed=2546)


def train(config_file, *args):
    net = WGAN(config_file)

    noise = srng.normal((net.batch_size, net.noise_input_shape), dtype='float32')
    real_data = T.tensor4('input_real', theano.config.floatX)

    placeholder_image = theano.shared(np.zeros((net.batch_size, net.image_input_shape[2], net.image_input_shape[0],
                                                net.image_input_shape[1]), 'float32'))

    #build train ops
    net.set_training_status(True)
    fake_data = net.inference_generator(noise)
    disc_real = net.inference_discriminator(real_data)
    disc_fake = net.inference_discriminator(fake_data)

    gen_params = [param for layer in net.network['generator'] for param in layer.trainable]
    disc_params = [param for layer in net.network['discriminator'] for param in layer.trainable]

    #cost
    gen_cost = -T.mean(disc_fake)
    disc_cost = T.mean(disc_fake) - T.mean(disc_real)

    if args[0] == 'wgan-gp':
        #gradient penalty
        alpha = srng.uniform((net.batch_size, ), dtype='float32')
        diff = fake_data - real_data
        interpolates = real_data + (alpha.dimshuffle(0, 'x', 'x', 'x') * diff)
        gradients = T.grad(T.sum(net.inference_discriminator(interpolates)), [interpolates])[0]
        slopes = T.sqrt(T.sum(T.sqr(gradients), axis=(1, 2, 3)))
        gradient_penalty = T.mean(T.sqr(slopes-1))
        disc_cost += net.LAMBDA * gradient_penalty
    elif args[0] == 'wgan':
        w_clip_updates = [(p, T.clip(p, -.01, .01)) for p in disc_params]
        clip_weights = net.compile([], updates=w_clip_updates, name='clip weights')
    elif args[0] == 'dcgan':
        raise NotImplementedError
    else:
        raise NotImplementedError

    opt = Adam(net.learning_rate, net.beta1, net.beta2)
    gen_updates = opt.get_updates(gen_params, T.grad(gen_cost, gen_params))
    disc_updates = opt.get_updates(disc_params, T.grad(disc_cost, disc_params))

    #compile Theano train funcs
    train_gen = net.compile([], gen_cost, updates=gen_updates, name='train generator')
    train_disc = net.compile([], disc_cost, updates=disc_updates, givens={real_data: placeholder_image},
                             name='train discriminator')

    #build test op and func
    net.set_training_status(False)
    fixed_noise = T.as_tensor_variable(np.random.normal(size=(net.validation_batch_size, 128)).astype('float32'))
    gen_data = net.inference_generator(fixed_noise)
    test_gen = net.compile([], gen_data, name='test generator')

    #training scheme
    data_manager = DataManager2(config_file, placeholder_image)
    mon = monitor.Monitor(config_file)
    num_train_batches = (data_manager.num_train_data // net.batch_size) // net.CRITIC_ITERS
    epoch = 0
    print('Training...')
    while epoch < net.n_epochs:
        epoch += 1
        batches = data_manager.get_batches(epoch, net.n_epochs)

        for it in range(num_train_batches):
            iteration = (epoch - 1.) * num_train_batches + it + 1
            training_gen_cost = train_gen()
            if np.isnan(training_gen_cost):
                raise ValueError('Training failed due to NaN cost')
            mon.plot('training gen cost', training_gen_cost)

            for i in range(net.CRITIC_ITERS):
                b = batches.__next__()
                data_manager.update_input(b)
                training_disc_cost = train_disc()
                if args[0] == 'wgan':
                    clip_weights()
                mon.plot('training disc cost', training_disc_cost)

            if iteration % net.validation_frequency == 0:
                gen_images = test_gen()
                mon.save_image('generated image', gen_images / 2 + .5)
                mon.flush()
            mon.tick()


if __name__ == '__main__':
    train('wgan_cifar.config', 'wgan-gp')

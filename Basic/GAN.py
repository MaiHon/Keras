from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, time, datetime, random
import numpy as np


# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator
# import tensorflow as tf
# from keras import Model
# from keras.layers import Layer
# from keras import backend as K
# from keras.optimizers import Adam, RMSprop
# from keras.callbacks import ModelCheckpoint 
# from keras.initializers import RandomNormal
# from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, UpSampling2D
# from keras.layers import Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
# from keras.losses import BinaryCrossentropy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, UpSampling2D
from tensorflow.keras.layers import Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.losses import BinaryCrossentropy


class GAN():
    def __init__(self,
        start_epoch,
        input_dim,
        discriminator_conv_filters,
        discriminator_conv_kernel_size,
        discriminator_conv_strides,
        discriminator_batch_norm_momentum,
        discriminator_activation,
        discriminator_dropout_rate,
        generator_initial_dense_layer_size,
        generator_upsample,
        generator_conv_filters,
        generator_conv_kernel_size,
        generator_conv_strides,
        generator_batch_norm_momentum,
        generator_activation,
        generator_dropout_rate,
        learning_rate,
        z_dim,
        ):

        self.name = 'gan'

        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate

        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        
        self.loss_obj = BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, epsilon=1e-6)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, epsilon=1e-6)
        self.lambda_value = 10
        self.z_dim = z_dim

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = '/Users/mah/Desktop/GAN/logs/gradient_tape/' + self.current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)
        self.epoch = start_epoch

        self.d_total_losses = []
        self.d_real_losses  = []
        self.d_fake_losses  = []
        self.g_total_losses = []
        self.gen_losses     = []
        self.l1_losses      = []

        self._build_discriminator()
        self._build_generator()

    def get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha = 0.2)
        else:
            layer = Activation(activation)
        return layer

    def _build_discriminator(self):
        discriminator_input = Input(shape=self.input_dim, name='discriminator_input')
        x = discriminator_input

        for i in range(self.n_layers_discriminator):
            x = Conv2D(
                filters = self.discriminator_conv_filters[i]
                , kernel_size = self.discriminator_conv_kernel_size[i]
                , strides = self.discriminator_conv_strides[i]
                , padding = 'same'
                , name = 'discriminator_conv_' + str(i)
                , kernel_initializer = self.weight_init
                )(x)

            if self.discriminator_batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum = self.discriminator_batch_norm_momentum)(x)

            x = self.get_activation(self.discriminator_activation)(x)

            if self.discriminator_dropout_rate:
                x = Dropout(rate = self.discriminator_dropout_rate)(x)

        x = Flatten()(x)
        
        discriminator_output = Dense(1, activation='sigmoid', kernel_initializer = self.weight_init)(x)
        self.discriminator = Model(discriminator_input, discriminator_output)


    def _build_generator(self):
        generator_input = Input(shape=(self.z_dim,), name='generator_input')
        x = generator_input
        x = Dense(np.prod(self.generator_initial_dense_layer_size), kernel_initializer = self.weight_init)(x)

        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
        x = self.get_activation(self.generator_activation)(x)
        x = Reshape(self.generator_initial_dense_layer_size)(x)

        if self.generator_dropout_rate:
            x = Dropout(rate = self.generator_dropout_rate)(x)

        for i in range(self.n_layers_generator):

            if self.generator_upsample[i] == 2:
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters = self.generator_conv_filters[i]
                    , kernel_size = self.generator_conv_kernel_size[i]
                    , padding = 'same'
                    , name = 'generator_conv_' + str(i)
                    , kernel_initializer = self.weight_init
                )(x)
            else:

                x = Conv2DTranspose(
                    filters = self.generator_conv_filters[i]
                    , kernel_size = self.generator_conv_kernel_size[i]
                    , padding = 'same'
                    , strides = self.generator_conv_strides[i]
                    , name = 'generator_conv_' + str(i)
                    , kernel_initializer = self.weight_init
                    )(x)
            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
                x = self.get_activation(self.generator_activation)(x)
            else:
                x = Activation('tanh')(x)

        generator_output = x
        self.generator = Model(generator_input, generator_output)


    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_obj(y_true=K.ones_like(real_output), y_pred=real_output)
        fake_loss = self.loss_obj(y_true=K.zeros_like(fake_output), y_pred=fake_output)
        total_loss = real_loss + fake_loss
        
        return total_loss, real_loss, fake_loss

    def generator_loss(self, fake_output, fake, target):
        fake_loss = self.loss_obj(y_true=K.ones_like(fake_output), y_pred=fake_output)
        l1_loss   = K.mean(K.abs(target - fake))

        total_loss = fake_loss + self.lambda_value * l1_loss
        return total_loss, fake_loss, l1_loss

    def train_step(self, _step, imag, target):
        with tf.GradientTape(watch_accessed_variables=False) as gen_tape, tf.GradientTape(watch_accessed_variables=False) as dis_tape:
            gen_tape.watch(self.generator.trainable_variables)
            dis_tape.watch(self.discriminator.trainable_variables)


            fake_img = self.generator(imag, training=True)
            disc_real_output = self.discriminator([imag, target], training=True)
            disc_generated_output = self.discriminator([imag, fake_img], training=True)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
            gen_loss = self.generator_loss(disc_generated_output, fake_img, target)
        discriminator_gradients = dis_tape.gradient(disc_loss[0], self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
        generator_gradients = gen_tape.gradient(gen_loss[0], self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        with self.train_summary_writer.as_default():
            tf.summary.scalar('D Total Loss', disc_loss[0].numpy(), step=_step)
            tf.summary.scalar('D Real Loss', disc_loss[1].numpy(), step=_step)
            tf.summary.scalar('D Fake Loss', disc_loss[2].numpy(), step=_step)

            tf.summary.scalar('G Total Loss', gen_loss[0].numpy(), step=_step)
            tf.summary.scalar('G Fake Loss', gen_loss[1].numpy(), step=_step)
            tf.summary.scalar('G L1 Loss', gen_loss[2].numpy(), step=_step)
        return gen_loss, disc_loss

    def train(self, epochs, batchS, train_steps_for_epoch, test_steps, train_dir, train_datagen, test_flow):
        result_num = 1
        steps      = 0
        for epoch in range(self.epoch, epochs):
            avg_d_total_loss  = 0
            avg_d_real_loss   = 0
            avg_d_fake_loss   = 0
            avg_g_total_loss  = 0
            avg_gen_loss      = 0
            avg_l1_loss       = 0

            random_seed = random.randint(1, 100)
            train_flow = train_datagen.flow_from_directory(train_dir, target_size = self.input_dim[:2],
                                                batch_size=batchS, seed = random_seed,
                                                color_mode='grayscale', class_mode=None)

            for imag, target in tqdm(train_flow, total=train_steps_for_epoch):
                g_loss, d_loss = self.train_step(steps, imag, target)

                self.d_total_losses.append(d_loss[0])
                self.d_real_losses.append(d_loss[1])
                self.d_fake_losses.append(d_loss[2])
                self.g_total_losses.append(g_loss[0])
                self.gen_losses.append(g_loss[1])
                self.l1_losses.append(g_loss[2])

                avg_d_total_loss += d_loss[0]
                avg_d_real_loss  += d_loss[1]
                avg_d_fake_loss  += d_loss[2]
                avg_g_total_loss += g_loss[0]
                avg_l1_loss      += g_loss[1]
                avg_gen_loss     += g_loss[2]
                steps            += 1

                if steps%train_steps_for_epoch == 0 and steps is not 0:
                    break


            print("Epoch: {}".format(epoch+1))
            print("[Generator Loss] Total: {} (Gen Loss: {}, L1 Loss: {})".format(avg_g_total_loss/steps, avg_gen_loss/steps, avg_l1_loss/steps))
            print("[Discriminator Loss] {} (Real: {}, Fake: {})".format(avg_d_total_loss/steps, avg_d_real_loss/steps, avg_d_fake_loss/steps))

            if (epoch+1)%2 == 0:
                date = time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time()))
                self.discriminator.save("/Users/mah/Desktop/GAN/models/discriminator/{}_{}_discriminator.h5".format(date, str(epoch).zfill(4)))
                self.generator.save("/Users/mah/Desktop/GAN/models/generator/{}_{}_generator.h5".format(date, str(epoch).zfill(4)))

                self.discriminator.save("/Users/mah/Desktop/GAN/models/discriminator/weights.h5")
                self.generator.save("/Users/mah/Desktop/GAN/models/generator/weights.h5")


            print("Test During the Training")
            test_step = 0
            for test_img in tqdm(test_flow, total=test_steps):
                if test_step > test_steps:
                    break

                result   = self.generator.predict(test_img)
                test_img = tf.squeeze(test_img, 0)
                result   = tf.squeeze(result, 0)

                plt.figure(figsize=(15, 15))
                plt.subplot(1, 2, 1)
                plt.title("Original")
                plt.imshow(tf.keras.preprocessing.image.array_to_img(test_img))
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.title("Predicted")
                plt.imshow(tf.keras.preprocessing.image.array_to_img(result))
                plt.axis('off')
              
                savePath = os.path.join("/Users/mah/Desktop/GAN/Result_imgs", str(epoch+1).zfill(4))
                if not os.path.exists(savePath):
                    os.mkdir(savePath)

                plt.savefig(os.path.join(savePath, str(result_num).zfill(6)) + '.png')
                result_num += 1
                plt.close()

                test_step+=1
            print("Done")

    def load_weights(self, filepath):
        dis_weights_path = os.path.join(filepath, 'discriminator/weights.h5')
        gen_weights_path = os.path.join(filepath, 'generator/weights.h5')

        self.discriminator.load_weights(dis_weights_path)
        self.generator.load_weights(gen_weights_path)



if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    config = tf.compat.v1.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    model = GAN(input_dim=(28, 28, 1),
                start_epoch = 0,
                discriminator_conv_filters=[64, 64, 128, 128],
                discriminator_conv_kernel_size=[5, 5, 5, 5],
                discriminator_conv_strides=[2, 2, 2, 1],
                discriminator_batch_norm_momentum=None,
                discriminator_activation='relu',
                discriminator_dropout_rate=0.4,
                generator_initial_dense_layer_size=(7, 7, 64),
                generator_upsample=[2, 2, 1, 1],
                generator_conv_filters=[128, 64, 64, 1],
                generator_conv_kernel_size=[5, 5, 5, 5],
                generator_conv_strides=[1, 1, 1, 1],
                generator_batch_norm_momentum=0.9,
                generator_activation='relu',
                generator_dropout_rate=None,
                learning_rate = 4e-4,
                z_dim=100)

    model.generator.summary()
    model.discriminator.summary()

    batchS = 32
    train_dir = ""
    test_dir  = ""

    train_imgs = load_img(train_dir)
    test_imgs  = load_img(test_dir)
    train_steps_for_epoch = int(len(list(train_imgs))/batchS)
    test_steps = int(len(list(test_imgs))/batchS)

    train_args = dict(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        resacle=1/255.,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range = [0.5, 1.0],
        horizontal_flip=True,
        vertical_flip = True
    )

    train_datagen = ImageDataGenerator(**train_args)
    test_datagen = ImageDataGenerator(rescale=1/255.)
    test_flow     = test_datagen.flow_from_directory(test_dir, target_size = (28, 28),
                                                    color_mode='grayscale',
                                                    batch_size=1, class_mode=None, shuffle=False)

    GAN.train(100, batchS, train_steps_for_epoch, test_steps, train_dir, train_datagen, test_flow)
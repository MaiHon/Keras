from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, MaxPooling2D, Input
from tensorflow.keras.layers import LeakyReLU, Dropout, Flatten, Conv2DTranspose, Reshape
from tensorflow.keras.initializers import RandomNormal, he_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Model

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import datetime
import time
from tqdm import tqdm

from IPython import display

class GAN_MNIST():
    def __init__(self,
                start_epoch,
                learning_rate,
                z_dim):

                self.name = 'gan'
                self.loss_obj = BinaryCrossentropy(from_logits=True)
                self.generator_optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, epsilon=1e-6)
                self.discriminator_optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, epsilon=1e-6)
                self.z_dim = z_dim

                self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.train_log_dir = '/Users/mah/Desktop/GAN/logs/Gradient_tape/' + self.current_time + '/train'
                self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

                self.weight_init = RandomNormal(mean=0., stddev=0.02)
                self.epoch = start_epoch

                self.d_total_losses = []
                self.g_total_losses = []

                self.seed = tf.random.normal([16, self.z_dim])

                self._build_discriminator()
                self._build_generator()

    def _build_discriminator(self):
        discriminator_input = Input(shape=(28, 28, 1), name="Discriminator_input")
        x = discriminator_input

        x = Conv2D(64, (5, 5), strides=(2, 2), padding='same', name="Disc_First")(x)
        x = LeakyReLU(name="Disc_First_LearkyReLU")(x)
        x = Dropout(.3, name="Disc_First_Dropout")(x)

        x = Conv2D(128, (5, 5), strides=(2, 2), padding='same', name="Disc_Second")(x)
        x = LeakyReLU(name="Disc_Second_LearkyReLU")(x)
        x = Dropout(.3, name="Disc_Second_Dropout")(x)

        x = Flatten()(x)
        out = Dense(1)(x) # ==> BinarayCrossentropy(from_logits=True)
        self.discriminator = Model(inputs=discriminator_input, outputs=out, name="Discriminator")

    def _build_generator(self):
        generator_input = Input(shape=(self.z_dim, ), name="Generator_input")
        x = generator_input

        x = Dense(7*7*256, use_bias=False, name="Gen_First")(x)
        x = BatchNormalization(name="Gen_First_BathNormalization")(x)
        x = LeakyReLU(name="Gen_First_LearkyReLU")(x)
        x = Reshape((7, 7, 256))(x)
        print(x.shape)
        print(type(x.shape))
        assert x.shape.as_list() == [None, 7, 7, 256]

        x = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False, name="Gen_Second")(x)
        x = BatchNormalization(name="Gen_Second_BathNormalization")(x)
        x = LeakyReLU(name="Gen_Second_LearkyReLU")(x)
        assert x.shape.as_list() == [None, 7, 7, 128]

        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False, name="Gen_Thrid")(x)
        x = BatchNormalization(name="Gen_Third_BathNormalization")(x)
        x = LeakyReLU(name="Gen_Third_LearkyReLU")(x)
        assert x.shape.as_list() == [None, 14, 14, 64]

        x = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, name="Gen_Forth")(x)
        x = BatchNormalization(name="Gen_Forth_BathNormalization")(x)
        out = LeakyReLU(name="Gen_Forth_LearkyReLU")(x)
        assert x.shape.as_list() == [None, 28, 28, 1]

        self.generator = Model(inputs=generator_input, outputs=out, name="Generator")

    def discriminator_loss(self, real_out, fake_out):
        real_loss = self.loss_obj(tf.ones_like(real_out), real_out)
        fake_loss = self.loss_obj(tf.zeros_like(fake_out), fake_out)

        return real_loss + fake_loss

    def generator_loss(self, fake_out):
        return self.loss_obj(tf.ones_like(fake_out), fake_out)

    def train_step(self, _step, images):
        batch = images.shape[0]

        noise = tf.random.normal([batch, self.z_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        with self.train_summary_writer.as_default():
            tf.summary.scalar('D Total Loss', disc_loss.numpy(), step=_step)
            tf.summary.scalar('G Total Loss', gen_loss.numpy(), step=_step)
        return gen_loss, disc_loss

    def generate_and_save_images(self, epoch, test_input):
        predictions = self.generator(test_input, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('./results/image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()
        # plt.show()

    # 에포크 숫자를 사용하여 하나의 이미지를 보여줍니다.
    def display_image(self, epoch_no):
        return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

    def make_gif(self):
        anim_gif = './DCGAN_mnist.gif'
        with imageio.get_writer(anim_gif, mode='I') as writer:
            filenames = glob.glob('./results/image*.png')
            filenames = sorted(filenames)
            last = -1

            for i, filename in enumerate(filenames):
                frame = 2*(i**0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

        import IPython
        if IPython.version_info > (6,2,0,''):
            display.Image(filename=anim_gif)

    def load_weights(self):
        self.generator.load_weights("/Users/mah/Desktop/GAN/models/generator/weights.h5")
        self.discriminator.load_weights("/Users/mah/Desktop/GAN/models/discriminator/weights.h5")

    def train(self, dataset, epochs, step_per_epoch):
        step = 1
        avg_gen_loss = 0
        avg_dis_loss = 0

        for epoch in range(self.epoch, epochs):
            start = time.time()

            for image_batch in tqdm(dataset, total=step_per_epoch):
                gen_loss, dis_loss = self.train_step(step, image_batch)
                avg_gen_loss += gen_loss
                avg_dis_loss += dis_loss
                step += 1
            
            print('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
            print("Avg Gan Loss: {}\nAvg Dis Loss: {}".format(avg_gen_loss/step, avg_dis_loss/step))

            # GIF를 위한 이미지를 바로 생성합니다.
            display.clear_output(wait=True)
            self.generate_and_save_images(epoch, self.seed)

            if (epoch)%5 == 0:
                date = time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time()))
                self.discriminator.save("/Users/mah/Desktop/GAN/models/discriminator/{}_{}_discriminator.h5".format(date, str(epoch).zfill(4)))
                self.generator.save("/Users/mah/Desktop/GAN/models/generator/{}_{}_generator.h5".format(date, str(epoch).zfill(4)))

                self.discriminator.save("/Users/mah/Desktop/GAN/models/discriminator/weights.h5")
                self.generator.save("/Users/mah/Desktop/GAN/models/generator/weights.h5")
            print()
            

        # 마지막 에포크가 끝난 후 생성합니다.
        display.clear_output(wait=True)
        self.generate_and_save_images(epochs, self.seed)
        self.make_gif()


if __name__ == "__main__":

    # GPU setting if possible
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

    # preparing datasete
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data("./mnisst_dataset")
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')

    # stanardize image from -1 to 1
    train_images = (train_images - 127.5) / 127.5

    # Train params
    buffer_s = 60000
    batch_size = 256
    steps_per_epoch = int(buffer_s/batch_size)

    total_epoch = 50
    z_dim = 100
    learning_rate = 3e-4
    start_epoch =6

    # Datasetting
    train_datasets = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_s).batch(batch_size)

    # Model Define
    model = GAN_MNIST(start_epoch, learning_rate, z_dim)
    model.generator.summary()
    model.discriminator.summary()
    # model.load_weights()
    
    # Train
    model.train(train_datasets, total_epoch, steps_per_epoch)
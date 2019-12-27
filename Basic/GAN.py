from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, time, datetime, random
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, UpSampling2D, MaxPooling2D
from tensorflow.keras.layers import Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.losses import BinaryCrossentropy


class GAN():
    def __init__(self,
        start_epoch,
        input_dim,
        learning_rate,
        z_dim,
        ):

        self.name = 'gan'

        self.input_dim = input_dim
        
        self.loss_obj = BinaryCrossentropy()
        self.generator_optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, epsilon=1e-6)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, epsilon=1e-6)
        self.lambda_value = 10
        self.z_dim = z_dim

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = 'C:\\Users\\H\\Desktop\\GAN\\logs\\Gradient_tape\\' + self.current_time + '\\train'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
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

    def _build_discriminator(self, output_n=1):
        discriminator_input = Input(shape=self.input_dim, name='discriminator_input')

        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(discriminator_input)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool')(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool')(x)

        x         = Flatten()(x)
        x         = Dense(1024,      activation="relu")(x)
        out       = Dense(output_n,   activation='sigmoid')(x)
        self.discriminator = Model(discriminator_input, out, name="discriminator")


    def _build_generator(self):
        generator_input = Input(shape=(self.z_dim,), name='generator_input')
        '''
        noise_shape : the dimension of the input vector for the generator
        img_shape   : the dimension of the output
        '''
        ## latent variable as input
        d = Dense(1024, activation="relu")(generator_input) 
        # d = Dense(1024, activation="relu")(d) 
        d = Dense(128*8*8, activation="relu")(d)
        d = Reshape((8,8,128))(d)
        
        d = Conv2DTranspose(128, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False)(d)
        d = Conv2D( 64  , ( 1 , 1 ) , activation='relu' , padding='same', name="block_4")(d) ## 16,16
        d = Conv2DTranspose(32, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False)(d)
        d = Conv2D( 64  , ( 1 , 1 ) , activation='relu' , padding='same', name="block_5")(d) ## 32,32
        d = Conv2DTranspose(32, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False)(d)
        d = Conv2D( 128  , ( 1 , 1 ) , activation='relu' , padding='same', name="block_5")(d) ## 64,64
        d = Conv2DTranspose(32, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False)(d)
        d = Conv2D( 256  , ( 1 , 1 ) , activation='relu' , padding='same', name="block_5")(d) ## 128,128
        
        img = Conv2D( 3 , ( 1 , 1 ) , activation='sigmoid' , padding='same', name="final_block")(d) ## 128, 128
        self.generator = Model(generator_input, img, name="generator")


    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_obj(y_true=K.ones_like(real_output), y_pred=real_output)
        fake_loss = self.loss_obj(y_true=K.zeros_like(fake_output), y_pred=fake_output)
        total_loss = real_loss + fake_loss
        
        return total_loss

    def generator_loss(self, fake_output):
        fake_loss = self.loss_obj(y_true=K.ones_like(fake_output), y_pred=fake_output)
        return fake_loss

    def train_step(self, _step, batch_size, imag, target):
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            fake_imag = self.generator(noise, training=True)

            disc_real_output = self.discriminator(imag, training=True)
            disc_generated_output = self.discriminator(fake_imag, training=True)

            gen_loss = self.generator_loss(disc_generated_output)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        discriminator_gradients = dis_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        with self.train_summary_writer.as_default():
            tf.summary.scalar('D Total Loss', disc_loss.numpy(), step=_step)
            tf.summary.scalar('G Total Loss', gen_loss.numpy(), step=_step)
        return gen_loss, disc_loss

    def sample_images(self, epoch, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        fig, axs = plt.subplots(r, c, figsize=(15,15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                img = array_to_img(gen_imgs[cnt, :, :, :])
                axs[i,j].imshow(img)
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "sample_{}.png".format(epoch)))
        plt.close()

    def train(self, epochs, batchS, train_steps_for_epoch, train_flow):
        result_num = 1
        steps      = 0

        for epoch in range(self.epoch, epochs):
            avg_d_total_loss  = 0
            avg_d_real_loss   = 0
            avg_d_fake_loss   = 0
            avg_g_total_loss  = 0
            avg_gen_loss      = 0
            avg_l1_loss       = 0

            for imag, target in tqdm(train_flow, total=train_steps_for_epoch):
                batch_size = imag.shape[0]
                g_loss, d_loss = self.train_step(steps, batch_size, imag, target)

                self.d_total_losses.append(d_loss)
                self.gen_losses.append(g_loss)

                avg_d_total_loss += d_loss
                avg_gen_loss     += g_loss
                steps            += 1

                if steps%train_steps_for_epoch == 0 and steps is not 0:
                    break


            print("Epoch: {}".format(epoch))
            print("[Generator Loss] Total: {}".format(avg_gen_loss/steps))
            print("[Discriminator Loss] {}".format(avg_d_total_loss/steps))

            if (epoch)%2 == 0:
                date = time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time()))
                self.discriminator.save("C:\\Users\\H\\Desktop\\GAN\\models\\discriminator\\{}_{}_discriminator.h5".format(date, str(epoch).zfill(4)))
                self.generator.save("C:\\Users\\H\\Desktop\\GAN\\models\\generator\\{}_{}_generator.h5".format(date, str(epoch).zfill(4)))

                self.discriminator.save("C:\\Users\\H\\Desktop\\GAN\\models\\discriminator\\weights.h5")
                self.generator.save("C:\\Users\\H\\Desktop\\GAN\\models\\generator\\weights.h5")


            print("Test During the Training")
            self.sample_images(epoch, "C:\\Users\\H\\Desktop\\GAN\\result\\")

            print("Done")

    def load_weights(self, filepath):
        dis_weights_path = os.path.join(filepath, 'discriminator\\weights.h5')
        gen_weights_path = os.path.join(filepath, 'generator\\weights.h5')

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

    model = GAN(input_dim=(128, 128, 3),
                start_epoch = 10,
                learning_rate = 3e-4,
                z_dim=100)

    model.load_weights("C:\\Users\\H\\Desktop\\GAN\\models")
    model.generator.summary()
    model.discriminator.summary()

    batchS = 128
    train_dir = "C:\\Users\\H\\Desktop\\GAN\\train"

    train_args = dict(
        rescale=1/255.,
        # rotation_range=30,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # brightness_range = [0.5, 1.0],
        # horizontal_flip=True,
        # vertical_flip = True,
        dtype=tf.float32
    )

    train_datagen = ImageDataGenerator(**train_args)
    train_flow = train_datagen.flow_from_directory(train_dir, target_size = (128, 128),
                                                batch_size=batchS, seed = 1,
                                                color_mode='rgb', class_mode='input')

    # import cv2
    # for f, s in train_flow:

    #     f[0] = cv2.cvtColor(f[0], cv2.COLOR_RGB2BGR)
    #     s[0] = cv2.cvtColor(s[0], cv2.COLOR_RGB2BGR)

    #     cv2.imshow("f", f[0])
    #     cv2.imshow("s", s[0])
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         cv2.destroyAllWindows()
    #         break
        
    model.train(epochs=100, batchS=batchS, train_steps_for_epoch=200, train_flow=train_flow)
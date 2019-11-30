from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

from keras import backend as K
from keras.layers import Layer
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras import Model
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam



class Sampling(Layer):
    def __init__(self):
        super(Sampling, self).__init__()

    def call(self, args):
        mean, log_var = args
        eps = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
        return mean + K.exp(log_var/2) * eps


class VAE(object):
    def __init__(self,
                input_dim,
                Encoder_layer_num,
                Encoder_filters,
                Encoder_kernels,
                Encoder_strides,
                latent_dim,
                Decoder_layer_num,
                Decoder_filters,
                Decoder_kernels,
                Decoder_strides,
                Decoder_batch_norm = False,
                Decoder_dropout = False
                ):
        self.input_dim = input_dim
        self.size      = self.input_dim[0]
        self.latent_dim = latent_dim
        self.n_layers_encoder = Encoder_layer_num
        self.encoder_conv_filters = Encoder_filters
        self.encoder_conv_kernels = Encoder_kernels
        self.encoder_conv_strides = Encoder_strides

        self.n_layers_decoder =Decoder_layer_num
        self.use_batch_norm = Decoder_batch_norm
        self.use_dropout = Decoder_dropout
        self.decoder_conv_filters = Decoder_filters
        self.decoder_conv_kernels = Decoder_kernels
        self.decoder_conv_strides = Decoder_strides
        
        self._build()

    def _build(self):
        encoder_input = Input(shape=self.input_dim, name='Encoder_Input')
        x = encoder_input

        for _ in range(self.n_layers_encoder):
            encoder_conv = Conv2D(
                filters = self.encoder_conv_filters[_],
                kernel_size = self.encoder_conv_kernels[_],
                strides = self.encoder_conv_strides[_],
                padding = 'same',
                name = "encoder_conv_" + str(_)
            )

            x = encoder_conv(x)
            x = LeakyReLU()(x)

        shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)

        self.mean = Dense(self.latent_dim, name='Mean')(x)
        self.log_var = Dense(self.latent_dim, name="Log_Var")(x)
        self.encoder_mu_log_var = Model(inputs = encoder_input, outputs= [self.mean, self.log_var])

        
        encoder_output = Sampling()([self.mean, self.log_var])
        self.encoder = Model(inputs = encoder_input, outputs = encoder_output, name = "Encoder")


        decoder_input = Input(shape=(self.latent_dim,), name='Decodeer_Input')
        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for _ in range(self.n_layers_decoder):
            decoder_convT = Conv2DTranspose(
                filters = self.decoder_conv_filters[_],
                kernel_size = self.decoder_conv_kernels[_],
                strides = self.decoder_conv_strides[_],
                padding = 'same',
                name = "Decoder_convT_" + str(_)
            )

            x = decoder_convT(x)
            if _ < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate = 0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x
        self.decoder = Model(inputs = decoder_input, outputs = decoder_output, name="Decoder")
        
        # Define AE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output, name='AE')

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mean) - K.exp(self.log_var), axis = 1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return  r_loss + kl_loss

        optimizer = Adam(lr=learning_rate, beta_1=0.5, epsilon=1e-5)
        self.model.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])

    # def train(self, x_train, batch_size, epochs):
    #     self.model.fit(     
    #         x_train,
    #         x_train,
    #         batch_size = batch_size,
    #         shuffle = True,
    #         epochs = epochs
    #     )
    
    def train(self, x_train, batch_size, epochs, save_folder, initial_epoch = 0):
        checkpoint_filepath=os.path.join(save_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only = True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(save_folder, 'weights/weights_{}.h5'.format(self.size)), save_weights_only = True, verbose=1)
        callbacks_list = [checkpoint1, checkpoint2]

        self.model.fit(     
            x_train
            , x_train
            , batch_size = batch_size
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
            , callbacks = callbacks_list
        )
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save(self, saveDir):
        curtime = time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time()))
        saveDir = os.path.join(saveDir, curtime)
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        self.model.save(saveDir + "/model.h5", overwrite=True)

    def test(self, trainSet):
        fig = plt.figure(figsize=(30,10))
        nplot = [10, 10]
        latent_space = []

        for j in range(0, nplot[0], 2):
            for i in range(nplot[1]):
                count = np.random.randint(0, len(trainSet), 1)
                ax = fig.add_subplot(nplot[0], nplot[1], j * nplot[1] + i + 1)
                ax.imshow(trainSet[count[0]])
                ax.axis('off')

                latent_space.append(self.encoder.predict(np.expand_dims(trainSet[count[0]], 0)))
                ax = fig.add_subplot(nplot[0], nplot[1], (j+1) * nplot[1] + i + 1)
                ax.imshow(np.squeeze(self.model.predict(np.expand_dims(trainSet[count[0]], 0)), 0))
                ax.axis('off')             
        plt.show()
        plt.close()

        return latent_space

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf


# config=tf.compat.v1.ConfigProto(log_device_placement=True, allow_soft_placement=True)
# # config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# tf.compat.v1.enable_eager_execution(config=config)


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

    print("Training with GPU? ==> [", tf.test.is_built_with_gpu_support(), "]")
    img_shape = (128, 128, 3)
    VAE = VAE(
            img_shape,
            4,
            [32, 32, 64, 128],
            [3, 3, 3, 3],
            [1, 2, 2, 1],
            200,
            4,
            [128, 64, 32, 3],
            [3, 3, 3, 3],
            [1, 2, 2, 1],
            Decoder_batch_norm=True,
            Decoder_dropout=True
    )

    print(VAE.decoder.summary())
    print(VAE.encoder.summary())


    VAE.compile(5e-5, 10000)


    """
        Data Loading from tensorflow keras mnist dataset
        ==> divide by 255. to standardise image value 0 to 1
    """
    import os
    from tqdm import tqdm
    data_dir = '/Users/mah/Desktop/img_align_celeba/'
    whole_imgs = np.sort(os.listdir(data_dir))

    base = 0
    train_num  = 120000
    train_imgs = whole_imgs[base:base+train_num]
    test_imgs  = whole_imgs[train_num:int(train_num*1.1)]

    def setData(img_Dir, trainSet, testSet, shapes):
        train = []
        test  = []
        for i, name in tqdm(enumerate(trainSet), total=len(trainSet)):
            image = load_img(img_Dir + "/" + name,
                            target_size=shapes[:2])
            image = img_to_array(image) / 255.
            train.append(image)
        
        for i, name in tqdm(enumerate(testSet), total=len(testSet)):
            image = load_img(img_Dir + "/" + name,
                            target_size=shapes[:2])
            image = img_to_array(image) / 255.
            test.append(image)

        return np.array(train), np.array(test)

    print("Dataset Preparing", end='\t')
    trainSet, testSet = setData(data_dir, train_imgs, test_imgs, img_shape)
    print("Train data shape: {}".format(trainSet.shape))
    print("Done")

    print("Dataset Checking")
    fig = plt.figure(figsize=(30,10))
    nplot = 7
    for count in range(1, nplot):
        ax = fig.add_subplot(1,nplot,count)
        ax.imshow(trainSet[count])
    plt.show()
    plt.close()

    print("Start Training with Device: {}".format(tf.test.gpu_device_name()))
    VAE.load_weights("/Users/mah/Desktop/CelebA_Model/weights/weights_128.h5") 
    VAE.train(trainSet, 8, 50, "/Users/mah/Desktop/CelebA_Model/")
    VAE.save("/Users/mah/Desktop/CelebA_Model/")
    VAE.test(testSet)
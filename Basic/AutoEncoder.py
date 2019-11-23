from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, Flatten, Activation, Conv2DTranspose, Input, Reshape
from tensorflow.keras.optimizers import Adam


class AutoEncoder(object):
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
                Decoder_strides
                ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_layers_encoder = Encoder_layer_num
        self.encoder_conv_filters = Encoder_filters
        self.encoder_conv_kernels = Encoder_kernels
        self.encoder_conv_strides = Encoder_strides

        self.n_layers_decoder =Decoder_layer_num
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
        encoder_output = Dense(self.latent_dim, name = "Encoder_Output")(x)
        self.encoder = Model(inputs = encoder_input, outputs = encoder_output)


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
                x = LeakyReLU()(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x
        self.decoder = Model(inputs = decoder_input, outputs = decoder_output)
        
        # Define AE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output, name='AE')

    def compile(self, learning_rate):
        self.learning_rate = learning_rate

        optimizer = Adam(lr=learning_rate, epsilon=1e-6)

        def mse_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

        self.model.compile(optimizer=optimizer, loss = mse_loss, metrics=['accuracy'])

    def train(self, x_train, batch_size, epochs, print_every_n_batches = 100, initial_epoch = 0):
        self.model.fit(     
            x_train
            , x_train
            , batch_size = batch_size
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
        )

AE = AutoEncoder((28, 28, 1),
                4,
                [32, 64, 64, 64],
                [3, 3, 3, 3],
                [1, 2, 2, 1],
                2,
                4,
                [64, 64, 32, 1],
                [3, 3, 3, 3],
                [1, 2, 2, 1])
AE.compile(5e-4)

print(AE.decoder.summary())
print(AE.encoder.summary())

"""
    Data Loading from tensorflow keras mnist dataset
    ==> divide by 255. to standardise image value 0 to 1
"""
from tensorflow.keras.datasets.mnist import load_data
(x_train, y_train), (x_test, y_test) = load_data()
x_train = np.array(x_train, dtype=np.float32) / 255.
x_test = np.array(x_test, dtype=np.float32) / 255.
AE.train(tf.expand_dims(x_train, -1), 256, 10)



n_to_show = 10      # Test sample number
np.random.seed(88)  # To fix sample result
example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]
z_points = AE.encoder.predict(tf.expand_dims(example_images, -1))
reconst_images = AE.decoder.predict(z_points)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(n_to_show):
    img = example_images[i].squeeze()
    ax = fig.add_subplot(2, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, str(np.round(z_points[i],1)), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img, cmap='gray_r')

for i in range(n_to_show):
    img = reconst_images[i].squeeze()
    ax = fig.add_subplot(2, n_to_show, i+n_to_show+1)
    ax.axis('off')
    ax.imshow(img, cmap='gray_r')
plt.show()


"""
    Visualizing Latent Space
"""
n_to_show = 5000
grid_size = 15
figsize = 10
example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]
example_labels = y_test[example_idx]
z_points = AE.encoder.predict(tf.expand_dims(example_images, -1))

min_x = min(z_points[:, 0])
max_x = max(z_points[:, 0])
min_y = min(z_points[:, 1])
max_y = max(z_points[:, 1])

plt.figure(figsize=(figsize, figsize))
plt.scatter(z_points[:, 0] , z_points[:, 1], cmap='rainbow', c=example_labels, alpha=0.5, s=2)
plt.colorbar()
plt.show()
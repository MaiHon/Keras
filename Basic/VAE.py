from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.layers import Layer
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras import Model
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

        ### COMPILATION
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

    def train(self, x_train, batch_size, epochs):
        self.model.fit(     
            x_train,
            x_train,
            batch_size = batch_size,
            shuffle = True,
            epochs = epochs
        )

config = tf.compat.v1.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth= True
tf.compat.v1.enable_eager_execution(config=config)

print("Training with GPU? ==> [", tf.test.is_built_with_gpu_support(), "]")
VAE = VAE(
        (28, 28, 1),
        4,
        [32, 64, 128, 128],
        [3, 3, 3, 3],
        [1, 2, 2, 1],
        2,
        4,
        [128, 64, 32, 1],
        [3, 3, 3, 3],
        [1, 2, 2, 1]
)

print(VAE.decoder.summary())
print(VAE.encoder.summary())


VAE.compile(1e-3, 1000)

"""
    Data Loading from tensorflow keras mnist dataset
    ==> divide by 255. to standardise image value 0 to 1
"""
from tensorflow.keras.datasets.mnist import load_data
(x_train, y_train), (x_test, y_test) = load_data()
x_train = np.array(x_train, dtype=np.float32) / 255.
x_test = np.array(x_test, dtype=np.float32) / 255.
VAE.train(np.expand_dims(x_train, -1), 256, 10)



n_to_show = 10      # Test sample number
np.random.seed(88)  # To fix sample result
example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]
z_points = VAE.encoder.predict(np.expand_dims(example_images, -1))
reconst_images = VAE.decoder.predict(z_points)

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
from scipy.stats import norm

n_to_show = 5000
grid_size = 15
fig_height = 7
fig_width = 15

example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]
example_labels = y_test[example_idx]

z_points = VAE.encoder.predict(np.expand_dims(example_images, -1))
p_points = norm.cdf(z_points)

fig = plt.figure(figsize=(fig_width, fig_height))

ax = fig.add_subplot(1, 2, 1)
plot_1 = ax.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c=example_labels
            , alpha=0.5, s=2)
plt.colorbar(plot_1)

ax = fig.add_subplot(1, 2, 2)
plot_2 = ax.scatter(p_points[:, 0] , p_points[:, 1] , cmap='rainbow' , c=example_labels
            , alpha=0.5, s=5)

plt.show()



n_to_show = 5000
grid_size = 20
figsize = 10

example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]
example_labels = y_test[example_idx]

z_points = VAE.encoder.predict(np.expand_dims(example_images, -1))

plt.figure(figsize=(figsize, figsize))
plt.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels
            , alpha=0.5, s=2)
plt.colorbar()

x = norm.ppf(np.linspace(0.01, 0.99, grid_size))
y = norm.ppf(np.linspace(0.01, 0.99, grid_size))
xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
z_grid = np.array(list(zip(xv, yv)))

reconst = VAE.decoder.predict(z_grid)
plt.scatter(z_grid[:, 0] , z_grid[:, 1], c = 'black'#, cmap='rainbow' , c= example_labels
            , alpha=1, s=2)
plt.show()

fig = plt.figure(figsize=(figsize, figsize))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_size**2):
    ax = fig.add_subplot(grid_size, grid_size, i+1)
    ax.axis('off')
    ax.imshow(reconst[i, :,:,0], cmap = 'Greys')
plt.show()
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from keras import backend as K
from keras.layers import Layer
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras import Model
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from VAE_CelebA import VAE

class ImageLabelLoader():
    def __init__(self, image_folder, target_size):
        self.image_folder = image_folder
        self.target_size = target_size

    def build(self, att, batch_size, label = None):

        data_gen = ImageDataGenerator(rescale=1./255)
        if label:
            data_flow = data_gen.flow_from_dataframe(
                att
                , self.image_folder
                , x_col='image_id'
                , y_col=label
                , target_size=self.target_size 
                , class_mode='raw'
                , batch_size=batch_size
                , shuffle=True
            )
        else:
            data_flow = data_gen.flow_from_dataframe(
                att
                , self.image_folder
                , x_col='image_id'
                , target_size=self.target_size 
                , class_mode='input'
                , batch_size=batch_size
                , shuffle=True
            )

        return data_flow



if __name__ == "__main__":
    import os
    from tqdm import tqdm

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

    img_shape = (128, 128, 3)
    data_dir = 'C:\\Users\H\Desktop\img_align_celeba\\'
    whole_imgs = np.sort(os.listdir(data_dir))

    base = 0
    train_num  = 20000
    train_imgs = whole_imgs[base:base+train_num]
    test_imgs  = whole_imgs[train_num:int(train_num*1.1)]

    def setData(img_Dir, trainSet, testSet, shapes):
        train = []
        test  = []
        for i, name in tqdm(enumerate(trainSet), total=len(trainSet)):
            image = load_img(img_Dir + "\\" + name,
                            target_size=shapes[:2])
            image = img_to_array(image) / 255.
            train.append(image)
        
        for i, name in tqdm(enumerate(testSet), total=len(testSet)):
            image = load_img(img_Dir + "\\" + name,
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

    # print(VAE.decoder.summary())
    # print(VAE.encoder.summary())
    # VAE.compile(5e-5, 10000)


    VAE.load_weights("C:\\Users\\H\\Desktop\\CelebA_Model\\weights\weights_128.h5")
    z_test = VAE.test(testSet)
    z_test = np.array(z_test)

    x = np.linspace(-3, 3, 100)

    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    from scipy.stats import norm
    for i in range(len(z_test)):
        ax = fig.add_subplot(10, 10, i+1)
        ax.hist(np.squeeze(z_test[i], 0), density=True, bins = 20)
        ax.axis('off')
        ax.text(0.5, -0.35, str(i+1), fontsize=10, ha='center', transform=ax.transAxes)
        ax.plot(x, norm.pdf(x))
    plt.show()
    plt.close()



    n_to_show =  30
    z_new = np.random.normal(size=(n_to_show, VAE.latent_dim))

    reconstruced = VAE.decoder.predict_on_batch(np.array(z_new))
    fig = plt.figure(figsize=(18, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for _ in range(n_to_show):
        ax = fig.add_subplot(3, 10, _+1)
        ax.imshow(reconstruced[_, :, :, :])
        ax.axis('off')
    plt.show()
    plt.close()


    att = pd.read_csv(os.path.join("C:\\Users\H\Desktop\CelebA_Model", 'list_attr_celeba.csv'))
    imageLoader = ImageLabelLoader('C:\\Users\H\Desktop\img_align_celeba\\', img_shape[:2])
    data_flow_generic = imageLoader.build(att, n_to_show)

    def get_vector_from_label(label, batch_size):
        data_flow_label = imageLoader.build(att, batch_size, label = label)

        origin = np.zeros(shape = VAE.latent_dim, dtype = 'float32')
        current_sum_POS = np.zeros(shape = VAE.latent_dim, dtype = 'float32')
        current_n_POS = 0
        current_mean_POS = np.zeros(shape =VAE.latent_dim, dtype = 'float32')

        current_sum_NEG = np.zeros(shape = VAE.latent_dim, dtype = 'float32')
        current_n_NEG = 0
        current_mean_NEG = np.zeros(shape = VAE.latent_dim, dtype = 'float32')

        current_vector = np.zeros(shape = VAE.latent_dim, dtype = 'float32')
        current_dist = 0

        print('label: ' + label)
        print('images : POS move : NEG move :distance : 𝛥 distance')
        while(current_n_POS < 10000):

            batch = next(data_flow_label)
            im = batch[0]
            attribute = batch[1]

            z = VAE.encoder.predict(np.array(im))

            z_POS = z[attribute==1]
            z_NEG = z[attribute==-1]

            if len(z_POS) > 0:
                current_sum_POS = current_sum_POS + np.sum(z_POS, axis = 0)
                current_n_POS += len(z_POS)
                new_mean_POS = current_sum_POS / current_n_POS
                movement_POS = np.linalg.norm(new_mean_POS-current_mean_POS)

            if len(z_NEG) > 0: 
                current_sum_NEG = current_sum_NEG + np.sum(z_NEG, axis = 0)
                current_n_NEG += len(z_NEG)
                new_mean_NEG = current_sum_NEG / current_n_NEG
                movement_NEG = np.linalg.norm(new_mean_NEG-current_mean_NEG)

            current_vector = new_mean_POS-new_mean_NEG
            new_dist = np.linalg.norm(current_vector)
            dist_change = new_dist - current_dist


            print(str(current_n_POS)
                + '    : ' + str(np.round(movement_POS,3))
                + '    : ' + str(np.round(movement_NEG,3))
                + '    : ' + str(np.round(new_dist,3))
                + '    : ' + str(np.round(dist_change,3))
                )

            current_mean_POS = np.copy(new_mean_POS)
            current_mean_NEG = np.copy(new_mean_NEG)
            current_dist = np.copy(new_dist)

            if np.sum([movement_POS, movement_NEG]) < 0.08:
                current_vector = current_vector / current_dist
                print('Found the ' + label + ' vector')
                break

        return current_vector   


    BATCH_SIZE = 500
    attractive_vec = get_vector_from_label('Attractive', BATCH_SIZE)
    mouth_open_vec = get_vector_from_label('Mouth_Slightly_Open', BATCH_SIZE)
    smiling_vec = get_vector_from_label('Smiling', BATCH_SIZE)
    lipstick_vec = get_vector_from_label('Wearing_Lipstick', BATCH_SIZE)
    young_vec = get_vector_from_label('High_Cheekbones', BATCH_SIZE)
    male_vec = get_vector_from_label('Male', BATCH_SIZE)
    eyeglasses_vec = get_vector_from_label('Eyeglasses', BATCH_SIZE)
    blonde_vec = get_vector_from_label('Blond_Hair', BATCH_SIZE)

    def add_vector_to_images(feature_vec):
        n_to_show = 5
        factors = [-4,-3,-2,-1,0,1,2,3,4]

        example_batch = next(data_flow_generic)
        example_images = example_batch[0]
        example_labels = example_batch[1]

        z_points = VAE.encoder.predict(example_images)

        fig = plt.figure(figsize=(18, 10))
        counter = 1
        for i in range(n_to_show):

            img = example_images[i].squeeze()
            sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
            sub.axis('off')        
            sub.imshow(img)

            counter += 1

            for factor in factors:

                changed_z_point = z_points[i] + feature_vec * factor
                changed_image = VAE.decoder.predict(np.array([changed_z_point]))[0]

                img = changed_image.squeeze()
                sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
                sub.axis('off')
                sub.imshow(img)

                counter += 1
        
        plt.show()
        plt.close()


    print('Attractive Vector')
    add_vector_to_images(attractive_vec)

    print('Mouth Open Vector')
    add_vector_to_images(mouth_open_vec)

    print('Smiling Vector')
    add_vector_to_images(smiling_vec)

    print('Lipstick Vector')
    add_vector_to_images(lipstick_vec)

    print('Young Vector')
    add_vector_to_images(young_vec)

    print('Male Vector')
    add_vector_to_images(male_vec)

    print('Eyeglasses Vector')
    add_vector_to_images(eyeglasses_vec)

    print('Blond Vector')
    add_vector_to_images(blonde_vec)
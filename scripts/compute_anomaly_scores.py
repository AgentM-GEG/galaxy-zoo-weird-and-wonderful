# This python script illustrates how to load the wGAN + encoder checkpoints and apply them to a set of images
# the output of this script is a file that contains the filename and their corresponding anomaly scores 
# computed by the wGAN framework.


### load the necessary modules.
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from tqdm import tqdm
from IPython import display
from astropy.io import ascii
print(tf.__version__)
import sys
import os
import tqdm

# Global hyperparameters
AUTOTUNE=tf.data.experimental.AUTOTUNE
lambda_weight = 0.2  # 0 means its only izi else izif


# a function to load an image and do some additional processing on it.
def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3) #color images , images are already converted to 0-1 range so we do not need to do again
    img = tf.image.convert_image_dtype(img, tf.float32) 
    #crop center to get 96x96x3
    img = tf.image.resize_with_crop_or_pad(img,96,96)
    return img

# function to a process a file path (that is to an image)
def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

# a function to setup the randomized buffer amongst a list of various image files.
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat() #repeat forever
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# A tensorflow dataset is constructed based on a input folder (here named images). Expecting png files.
list_ds = tf.data.Dataset.list_files(str('images/*.png'), shuffle=True)
N_TRAIN_POINTS = len(list_ds)
print(f'Number of images :{N_TRAIN_POINTS}')

# loading the generator, discriminator, and the encoder saved model checkpoints
generator = tf.keras.models.load_model("model_checkpoints/generator_MagLim.h5")
discriminator = tf.keras.models.load_model("model_checkpoints/discriminator_MagLim.h5")
encoder = tf.keras.models.load_model('model_checkpoints/encoder_MagLim.h5')

# extracting the intermediate layer (flattened discriminator features)
layer_name = 'flatten'

# defining a new model that goes from an image and outputs the intermediate discriminator features.
inter_discriminator_fts = tf.keras.Model(inputs=discriminator.input,
                                       outputs=discriminator.get_layer(layer_name).output)


# looping through the list of files to compute the anomaly scores.
counter = 0
with open("/home/fortson/manth145/codes/GZ2-GAN/HSC-GAN/anomaly_scores_MagLim.csv", "w") as f_out:
    writeout = ['# counter', 'filename', 'anomaly_score', 'image_score', 'feature_score']
    f_out.write(','.join(str(item) for item in writeout))
    f_out.write('\n')
    for f in tqdm.tqdm(list_ds):
        # process the image path
        image = process_path(f)
        
        # extract the latent vector for that image.
        z_hat = encoder(image.numpy().reshape(-1, 96, 96, 3))
        
        # pass the z_hat to the generator, which generates an image.
        generate_encoded = generator(z_hat)
        
        # extract the intermediate discriminator features for the actual image
        real_fts = inter_discriminator_fts(image.numpy().reshape(-1, 96, 96, 3))
        
        # extract the intermediate discriminator features for the generated image
        generated_fts = inter_discriminator_fts(generate_encoded.numpy().reshape(-1, 96, 96, 3))
        
        # compute the similarity/dissimilarity between the real and generated intermediate features (feature score)
        feature_residual = tf.reduce_sum(tf.reduce_mean(tf.pow(tf.subtract(real_fts, generated_fts),2),axis=0)) 
        
        # compute the residual between the original input image and the generated image.
        residual = tf.reduce_sum(tf.reduce_mean(tf.pow(tf.subtract(image, generate_encoded),2), axis=0))

        # compute the anomaly score as a combination of the feature and image score.
        ana_score = (1-lambda_weight)*residual + lambda_weight*feature_residual
        
        # writing the results to the file.
        writeout = [counter, os.path.basename(f.numpy().decode('UTF-8')).strip('.png'), ana_score.numpy(), residual.numpy(), feature_residual.numpy()]
        f_out.write(','.join(str(item) for item in writeout))
        f_out.write('\n')
        
        # also record the latent space information.
        if counter == 0:
            m_z_hat = z_hat.numpy().reshape(1,128)
        else:
            m_z_hat = np.vstack((m_z_hat, z_hat.numpy().reshape(1,128)))

        counter+=1

# save the latent space information into a separate file.
with open('z_hat_hsc_MagLim.npy', 'wb') as fz_hat:
    np.save(fz_hat, m_z_hat)
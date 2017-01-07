import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf

MEAN_IMAGE = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) # to be subtracted from every image
                                                                      # required for VGG-19 model

def img_load(path):

    image = scipy.misc.imread(path)
    image = np.reshape(image,(1,)+image.shape)
    image = image - MEAN_IMAGE                 # required by vgg model
    
    return image

def img_save(image,path):

    image = image + MEAN_IMAGE
    image = image[0]
    image = np.clip(image,0,255).astype('uint8')
    scipy.misc.imsave(path,image)

def generate_noise(shape,noise_ratio = 0.6,seed_image = None):
    
    noise = np.random.uniform(-20,20,shape).astype("float32")
    noise = noise*noise_ratio + seed_image*(1-noise_ratio)

    return noise


def content_loss(p,x):
    '''p and x are layers'''
    dims = [int(dim) for dim in list(p.get_shape())]
    N = dims[3]
    M = dims[1]*dims[2]

    return (1/(4*N*M)) * tf.reduce_sum(tf.pow(x-p,2))


def style_loss(a,x,layers,style_weights):

    def _style_loss_(a,x):

        def gram_matrix(F,N,M):

            f = tf.reshape(F,(M,N))
            return tf.matmul(tf.transpose(f),f)

        dims = [int(dim) for dim in list(a.get_shape())]
        N = dims[3]
        M = dims[1]*dims[2]

        A = gram_matrix(a,N,M)
        G = gram_matrix(x,N,M)

        return (1/(4* N**2 * M**2)) * tf.reduce_sum(tf.pow(G-A,2))

    loss = 0
    for layer,sweights in zip(layers,style_weights):
        loss += sweights * _style_loss_(a[layer],x[layer])

    return loss
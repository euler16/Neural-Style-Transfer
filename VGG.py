import scipy.misc
import scipy.io
import numpy as np
import tensorflow as tf

imHEIGHT  = 600
imWEIGHT  = 800
imCHANNEL = 3

class VGG(dict):

    vgg = None
    path = None

    def __init__(self,path):

        if VGG.path == None:
            VGG.path = path

        if VGG.vgg == None:
            VGG.vgg = scipy.io.loadmat(self.path)
            VGG.vgg = VGG.vgg['layers'][0]

    def weights(self,layer,name):

        # W = self.vgg[0][layer][0][0][0][0][0]
        # b = self.vgg[0][layer][0][0][0][0][1]
        # layer_name = self.vgg[0][layer][0][0][-2]
        # assert layer_name == name, print(layer_name)
    
        # return W, b
        layer = VGG.vgg[layer]
        tmplayer = layer[0][0]
        layer_name = layer[0][0][0][0]
        params = layer[0][0][2][0]
        weights = params[0]
        biases = params[1]

        return weights, biases



    def conv(self,prev,layer,layer_name):

        weights, biases = self.weights(layer,layer_name)
        weights = tf.constant(weights)
        biases = tf.constant(np.reshape(biases,(biases.size)))

        return tf.add(tf.nn.conv2d(prev,filter = weights,strides = [1,1,1,1],padding = 'SAME'),biases)


    def relu(self,conv_layer):

        return tf.nn.relu(conv_layer)


    def avg_pool(self,layer):
        return tf.nn.avg_pool(layer,ksize = [1,2,2,1], strides = [1,2,2,1],padding = 'SAME')


    def init(self,input = np.zeros((1,600,800,3),dtype = np.float32),type = 'var'):

        if not input.dtype == np.float32:
            input = input.astype('float32')
            
        assert input.dtype == np.float32, print("data-type of input should be float32")

        if type == 'const':
            self['input']   = tf.placeholder_with_default(input,input.shape)
        elif type == 'var':
            self['input']   = tf.Variable(input)

        self['conv1_1'] = self.conv(self['input'],0,'conv1_1')
        self['conv1_1'] = self.relu(self['conv1_1'])
        
        self['conv1_2'] = self.conv(self['conv1_1'],2,'conv1_2')
        self['conv1_2'] = self.relu(self['conv1_2'])

        self['pool1'] = self.avg_pool(self['conv1_2'])

        self['conv2_1'] = self.conv(self['pool1'],5,'conv2_1')
        self['conv2_1'] = self.relu(self['conv2_1'])

        self['conv2_2'] = self.conv(self['conv2_1'],7,'conv2_2')
        self['conv2_2'] = self.relu(self['conv2_2'])

        self['pool2'] = self.avg_pool(self['conv2_2'])

        self['conv3_1'] = self.conv(self['pool2'],10,'conv3_1')
        self['conv3_1'] = self.relu(self['conv3_1'])

        self['conv3_2'] = self.conv(self['conv3_1'],12,'conv3_2')
        self['conv3_2'] = self.relu(self['conv3_2'])

        self['conv3_3'] = self.conv(self['conv3_2'],14,'conv3_3')
        self['conv3_3'] = self.relu(self['conv3_3'])

        self['conv3_4'] = self.conv(self['conv3_3'],16,'conv3_4')
        self['conv3_4'] = self.relu(self['conv3_4'])

        self['pool3'] = self.avg_pool(self['conv3_4'])

        self['conv4_1'] = self.conv(self['pool3'],19,'conv4_1')
        self['conv4_1'] = self.relu(self['conv4_1'])

        self['conv4_2'] = self.conv(self['conv4_1'],21,'conv4_2')
        self['conv4_2'] = self.relu(self['conv4_2'])

        self['conv4_3'] = self.conv(self['conv4_2'],23,'conv4_3')
        self['conv4_3'] = self.relu(self['conv4_3'])

        self['conv4_4'] = self.conv(self['conv4_3'],25,'conv4_4')
        self['conv4_4'] = self.relu(self['conv4_4'])

        self['pool4'] = self.avg_pool(self['conv4_4'])

        self['conv5_1'] = self.conv(self['pool4'],28,'conv5_1')
        self['conv5_1'] = self.relu(self['conv5_1'])

        self['conv5_2'] = self.conv(self['conv5_1'],30,'conv5_2')
        self['conv5_2'] = self.relu(self['conv5_2'])

        self['conv5_3'] = self.conv(self['conv5_2'],32,'conv5_3')
        self['conv5_3'] = self.relu(self['conv5_3'])

        self['conv5_4'] = self.conv(self['conv5_3'],34,'conv5_4')
        self['conv5_4'] = self.relu(self['conv5_4'])

        self['pool5'] = self.avg_pool(self['conv5_4']) 



if __name__ == '__main__':
    vgg = VGG("imagenet-vgg-verydeep-19.mat")
    vgg.init()                        
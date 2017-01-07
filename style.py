from utilities import *
from VGG import VGG
import tensorflow as tf
import os

# NEURAL STYLE TRANSFER ALGORITHM

# Constants
EPOCH = 5000
LEARNING_RATE = 2.0
ALPHA = 1     # content ratio
BETA = 100    # style ratio
NOISE_RATIO = 0.6

STYLE_LAYER   = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']
CONTENT_LAYER = ['conv4_2']

STYLE_WEIGTHS = [1/len(STYLE_LAYER) for i in range(len(STYLE_LAYER))] 


OUTPUT_DIR = "output/"

VGG_path     = "imagenet-vgg-verydeep-19.mat"
content_path = "content/dr.jpg"
style_path   = "style/StarryNight.jpg"

imHEIGHT   = 176
imWIDTH    = 162
imCHANNELS = 3


content_image = img_load(content_path)
style_image = img_load(style_path)
noise = generate_noise((1,imHEIGHT,imWIDTH,imCHANNELS),NOISE_RATIO,content_image)

P = VGG(VGG_path)  # content
X = VGG(VGG_path)  # noise/output
A = VGG(VGG_path)  # style

P.init(input = style_image,type = 'const')
X.init(input = noise,type = 'var')
A.init(input = content_image,type = 'const')


lossContent = content_loss(P['conv4_2'],X['conv4_2'])
lossStyle   = style_loss(A,X,STYLE_LAYER,STYLE_WEIGTHS)

loss  = ALPHA * lossContent + BETA * lossStyle

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:

	sess.run(tf.initialize_all_variables())

	for epoch in range(EPOCH):

		sess.run(optimizer)

		if epoch%100 == 0:
			print("ITERATION: ",epoch)
			print("loss: ",sess.run(loss))

			if not os.path.exists(OUTPUT_DIR):
				os.mkdir(OUTPUT_DIR)

			image = sess.run(X['input'])
			x_name = OUTPUT_DIR + "/output"+str(epoch)+".png"
			img_save(image,x_name)

		else:
			print(epoch)
 # -- coding: UTF-8 --

import math
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
#%matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = 'your path  of caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import os
import os.path

caffe.set_mode_gpu()
#需要改为自己的模型
model_def = caffe_root + 'test/deploy.prototxt'
model_weights = caffe_root + 'test/caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)




# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)


# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

mae=0
rootdir = 'your path of the test dataset'
for i in os.listdir(rootdir):
    predict_label = 0
    image = caffe.io.load_image(rootdir+'/'+i)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image# copy the image data into the memory allocated for the net
    output = net.forward()### perform classification
    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
    for j in range(100):
    	predict_label += output_prob[j]*j
    print 'predicted value is:',i, predict_label 
    mae += abs(float(label)-predict_label)#label is image i's
print 'mae=',mae
avr_mae = mae / float('total number of this test dataset')
print 'avr_mae=',avr_mae










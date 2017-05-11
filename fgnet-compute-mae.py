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
caffe_root = 'your path of caffe root'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

import os
import os.path
#if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
 #   print 'CaffeNet found.'
#else:
 #   print 'Downloading pre-trained CaffeNet model...'
  #  !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet



caffe.set_mode_cpu()
#需要改为自己的模型
model_def = caffe_root + 'your path of deploy.prototxt'

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)
my_mae=0
# the number of the test set from 1 to 82
count_num=[15, 16, 12, 9, 11, 12, 9, 15, 7, 7, 14, 15, 12, 10, 13, 13, 13, 11, 10, 13, 12, 13, 12, 11, 12, 11, 11, 11, 13, 11, 13, 12, 11, 13, 14, 13, 12, 13, 14, 14, 10, 13, 11, 10, 13, 13, 14, 16, 10, 8, 11, 11, 13, 13, 8, 8, 10, 11, 9, 12, 13, 12, 10, 6, 15, 12, 10, 10, 11, 11, 13, 14, 16, 16, 10, 18, 16, 16, 14, 14, 12, 11]
# the total number of the test set from 1 to 82
count_num_all=987
for tt in range(1,83):
    model_weights = 'your path of caffemodel of each person'

    net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

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
    rootdir = 'your path of test set for the person'
    for i in os.listdir(rootdir):
        predict_label = 0
        s=i.split('.')
        label = int(s[0][4:6])
        image = caffe.io.load_image(rootdir+'/'+i)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image# copy the image data into the memory allocated for the net
        output = net.forward()### perform classification
        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
        for j in range(100):
    	    predict_label += output_prob[j]*j
        print 'predicted value is:',i, predict_label 
        mae += abs(float(label)-predict_label)
    my_mae += mae
mae_mean = my_mae / float(count_num_all)
print 'mae=',mae_mean










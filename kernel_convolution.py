	# Feb 11, 2020
# This started as ~/public_html/other/scantron/scantron_new_join_studentcsv.py
# I want to try doing yellowjacket detection/location by convolution instead of neural networks.
# So I will be hacking out almost all of this code.
# Leaving just the loading of the big image and the small kernel image
# doing the convolution and writing out a detection image and data.

import numpy as np
import scipy
from scipy import signal,misc
import matplotlib.pyplot as plt
from time import time
from skimage import measure
import sys
import os
from datetime import datetime
import pandas
import glob
from PIL import Image, ImageDraw
#from jpegtran import JPEGImage
import imageio# for imread

threshold = .9 #.80 #.85 
preconvthreshold = 232

kernel_path = 'kernel_imgs/b&w_cardinal8'
image_path = 'test_imgs/b&w_cardinal8_composites'
colors = ['red','orange','yellow','lime','green','blue','purple','magenta']
#colors = ['red','yellow','green','purple']

# Process kernels
kernelfiles = sorted(glob.glob(kernel_path+'/*.png'))	# Select all PNG files in the specified folder
kernels = [np.array(imageio.imread(kernelfile),dtype=float) for kernelfile in kernelfiles]	# Save the image to an array
kernels = [kernel[:,:,:3].sum(axis=2)/3 if len(kernel.shape)==3 else kernel for kernel in kernels]	# allow for RGB or RGBA image
kernels = [kernel-kernel.mean() for kernel in kernels]	# Normalize kernels (so avg color is 0)
for i,kernel in enumerate(kernels):
	print('Kernel', i, 'has Shape', kernel.shape, '|', kernelfiles[i])

# Process Image files
imagefiles = glob.glob(image_path+'/*.png')	# Select all PNG files in the specified folder

def now(): return str(datetime.now())[:19].replace(' ','_').replace(':','-')
timestamp = now()


# note the flip because convolution is not the same as correlation!

def centroids(img, kernel):

	global threshold

	#print(img.shape,kernel.shape)

	#print('Convolution...')
	a = signal.fftconvolve(img, kernel, mode='same')
	#print('done.')

	# Look at convolution
	#plt.imshow(a,cmap='gist_ncar')#'gray')
	#plt.colorbar()
	#plt.show()

	# Thresholding: All pixels above threshold get set to 1
	a = (a - a.min()) / a.max()		# Normalize to [0,1]
	above = a>threshold 
	a = np.zeros(a.shape,dtype=int)
	a[above] = 1

	# Find connected components of thresholded image
	#labelarray = measure.label(a, neighbors=8, background=0)
	labelarray = measure.label(a, connectivity=2, background=0)
	labels = np.unique(labelarray)

	# Find centroid of each connected component
	yellowjackets = []
	for label in labels[1:]:
		x,y = np.where(labelarray==label)
		yellowjackets.append([x.mean(), y.mean()])

	return yellowjackets


for nimage,imagefile in enumerate(imagefiles):

	img = imageio.imread(imagefile)
	print('Image', nimage, 'has Shape',img.shape)

	# allow for RGB or RGBA image
	if len(img.shape)==3: 
		img = img[:,:,:3].sum(axis=2)/3

	# pre-convolution thresholding
	#above = form > preconvthreshold
	#img[:,:] = 0
	#img[above] = 255

	rgbArray = np.zeros((img.shape[0],img.shape[1],3), 'uint8')
	rgbArray[..., 0] = img
	rgbArray[..., 1] = img
	rgbArray[..., 2] = img
	imgd = Image.fromarray(rgbArray)
	draw = ImageDraw.Draw(imgd)

	for kernel,color in zip(kernels,colors):
		# Use convolution with each kernel to match yellowjackets
		yjs = centroids(img, kernel )

		# Draw a dot on each detected yellowjacket
		r = 5 	# Radius of ellipses
		for yj in yjs:
			ix = int(yj[0]+0.5)
			iy = int(yj[1]+0.5)
			draw.ellipse((iy-r, ix-r, iy+r, ix+r), fill = color, outline =None)

	#timestamp = 'timehere'
	print('Writing image', nimage, 'with dots...')
	validation = imagefile[:-4]+'.validation.jpg' #_'+timestamp+'.jpg')
	imgd.save(validation)



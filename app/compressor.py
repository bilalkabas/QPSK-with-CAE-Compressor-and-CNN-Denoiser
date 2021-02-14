import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import gridspec
from matplotlib.pyplot import figure
import tkinter as tk
from tkinter import filedialog
import numpy as np
from numpy import genfromtxt
from keras.models import Model, load_model
import os.path
import time

# Clear past data
def dataClear ():
	dir = os.getcwd() + "/data"
	for f in os.listdir(dir):
		os.remove(os.path.join(dir, f))

# Global variables
img_tx = plt.imread('img/2.jpg')
img_tx_temp = img_tx
noise_level = 0.0
bypass_labview = 1
labview = 0

def chooseImage (event):
	global img_tx
	global img_tx_temp
	global noise_level

	# Open choose file dialog
	root = tk.Tk()
	root.withdraw()
	file_path = filedialog.askopenfilename(initialdir=(os.getcwd() + "/test-set"))
	img_tx = plt.imread(file_path)

	# Load the image from specified path
	img_tx_temp = img_tx
	plt.subplot2grid((30, 40), (11, 2), colspan=17, rowspan=17)
	plt.imshow(img_tx.reshape(28,28))
	plt.axis('off')
	plt.gray()

	# Classification
	classify()
	dataClear()

	# Show question mark in place of received image
	img = plt.imread('img/question-mark.png')
	plt.subplot2grid((30, 40), (11, 20), colspan=17, rowspan=17)
	plt.imshow(img)
	plt.gray()
	plt.axis('off')
	
	# Clear previous prediction results
	plt.subplot2grid((30, 40), (7, 31), colspan=2, rowspan=1)
	plt.cla()
	plt.axis('off')
	plt.subplot2grid((30, 40), (7, 34), colspan=2, rowspan=1)
	plt.cla()
	plt.axis('off')

	# Reset noise level
	noise_level = 0.0
	nl = '%.1f' % (noise_level)
	nl_ax = plt.subplot2grid((30, 40), (4, 32), colspan=2, rowspan=3)
	nl_ax.text(1.4, 0.97, nl, verticalalignment='bottom', 
		horizontalalignment='left', color='#eee', fontsize=20)
	plt.axis('off')

	# Delete data and show plots
	# dataClear()
	plt.show()


def compress (event):
	global img_tx
	global bypass_labview
	dataClear()
	
	img_tx = img_tx.reshape(1,28,28,1)
	encoder = load_model('models/encoder.h5')
	encoded = encoder.predict(img_tx)
	# Normalize data between 0-255
	normalized = ((encoded-np.min(encoded))/(np.max(encoded)-np.min(encoded))*255).astype(np.uint8)
	# Convert to binary
	binary = np.unpackbits(normalized)
	# Save binary data to csv file
	binary = np.ndarray.flatten(binary)
	if bypass_labview == 1:
		np.savetxt("data/demod.csv", binary, delimiter=",")
	else:
		np.savetxt("data/encoded.csv", binary, delimiter=",")

	# Check if demodulated data exists
	file_path = 'data/demod.csv'
	while not os.path.exists(file_path):
		time.sleep(1)

	if os.path.isfile(file_path):
		# Decoding
		decoder = load_model('models/decoder.h5')
		demod_bin = genfromtxt(file_path, delimiter=',')
		demod_bin = np.reshape(demod_bin, (int(len(demod_bin)/8), 8)).astype(int)
		demod_dec = np.packbits(demod_bin)
		demod_float = demod_dec.astype('float32')
		encoded = np.reshape(demod_float, (1,4,4,8))
		decoded = decoder.predict(encoded/50)

		# Show decoded image
		plt.subplot2grid((30, 40), (11, 20), colspan=17, rowspan=17)
		plt.imshow(decoded.reshape(28,28))
		plt.axis('off')
		plt.gray()

		# Classification
		model = load_model('models/classification.h5')
		pre = model.predict(decoded.reshape(1,28,28,1))
		pre = ((pre-np.min(pre))/(np.max(pre)-np.min(pre))*255).astype(np.uint8)
		pre = ((100/np.sum(pre))*pre).astype(np.uint8)
		index = np.argmax(pre)

		# Show prediction results
		print("The image predicted as '" + str(index) + "' with", str(pre[0][index]) + "% probability.")
		predict = plt.subplot2grid((30, 40), (7, 31), colspan=2, rowspan=1)
		predict.text(0.60, -2.5, (str(pre[0][index]) + '%'), verticalalignment='bottom', 
			horizontalalignment='right', color='#111', fontsize=20)
		plt.axis('off')
		predict = plt.subplot2grid((30, 40), (7, 34), colspan=2, rowspan=1)
		predict.text(0.70, -2.5, index, verticalalignment='bottom', 
			horizontalalignment='right', color='#eee', fontsize=20)
		plt.axis('off')
		plt.show()


def classify ():
	# Load model and predict
	global img_tx
	model = load_model('models/classification.h5')
	img_tx = img_tx.astype('float32') / 255.
	img_tx = np.reshape(img_tx, (1, 28, 28, 1))
	pre = model.predict(img_tx)
	pre = ((pre-np.min(pre))/(np.max(pre)-np.min(pre))*255).astype(np.uint8)
	pre = ((100/np.sum(pre))*pre).astype(np.uint8)
	index = np.argmax(pre)
	
	# Show prediction results    
	print("The image predicted as '" + str(index) + "' with", str(pre[0][index]) + "% probability.")
	predict = plt.subplot2grid((30, 40), (7, 13), colspan=2, rowspan=1)
	predict.text(0.60, -2.5, (str(pre[0][index]) + '%'), verticalalignment='bottom', 
		horizontalalignment='right', color='#111', fontsize=20)
	plt.axis('off')
	predict = plt.subplot2grid((30, 40), (7, 16), colspan=2, rowspan=1)
	predict.text(0.70, -2.5, index, verticalalignment='bottom', 
		horizontalalignment='right', color='#eee', fontsize=20)
	plt.axis('off')


def increaseNoise (event):
	global noise_level
	global img_tx
	global img_tx_temp
	if noise_level <= 0.9 and noise_level >= 0:
		noise_level = noise_level + 0.1
		nl = '%.1f' % (noise_level)
		nl_ax = plt.subplot2grid((30, 40), (4, 32), colspan=2, rowspan=3)
		nl_ax.text(1.4, 0.97, nl, verticalalignment='bottom', 
			horizontalalignment='left', color='#eee', fontsize=20)
		plt.axis('off')
		# Add noise
		plt.subplot2grid((30, 40), (11, 2), colspan=17, rowspan=17)
		img_tx = img_tx.reshape(28,28)
		img_tx = img_tx_temp + noise_level*255*np.random.normal(0, 1, img_tx.shape)
		plt.imshow(img_tx)
		img_tx = img_tx.astype('float32') / 255.
		plt.axis('off')
		plt.gray()
		plt.show()


def decreaseNoise (event):
	global noise_level
	global img_tx
	global img_tx_temp
	if noise_level <= 1 and noise_level >= 0.1:
		noise_level = noise_level - 0.1
		nl = '%.1f' % (noise_level)
		nl_ax = plt.subplot2grid((30, 40), (4, 32), colspan=2, rowspan=3)
		nl_ax.text(1.4, 0.97, nl, verticalalignment='bottom', 
			horizontalalignment='left', color='#eee', fontsize=20)
		plt.axis('off')
		# Add noise
		plt.subplot2grid((30, 40), (11, 2), colspan=17, rowspan=17)
		img_tx = img_tx.reshape(28,28)
		img_tx = img_tx_temp + noise_level*255*np.random.normal(0, 1, img_tx.shape)
		plt.imshow(img_tx)
		img_tx = img_tx.astype('float32') / 255.
		plt.axis('off')
		plt.gray()
		plt.show()


def bypassLabview (event):
	global bypass_labview
	global labview

	if bypass_labview == 0:
		bypass_labview = 1
		labview.color='#ffa8a8'
	else:
		bypass_labview = 0
		labview.color='#a8ffad'
	plt.show()


# Figure window configurations
mpl.rcParams['toolbar'] = 'None'
figure(num=None, figsize=(10, 7.5), dpi=80, facecolor='#000', edgecolor='k')
fig = plt.gcf()
fig.canvas.set_window_title('CAE Compressor & Denoiser')

# Show transmitted image
plt.subplot2grid((30, 40), (11, 2), colspan=17, rowspan=17)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.imshow(img_tx.reshape(28,28))
plt.gray()
plt.axis('off')

# Show 'transmitted' label
nl_ax = plt.subplot2grid((30, 40), (5, 2), colspan=2, rowspan=3)
nl_ax.text(0, 0, 'Transmitted', verticalalignment='bottom', 
			horizontalalignment='left', color='#666', fontsize=20)
plt.axis('off')

# Show 'Received' label
nl_ax = plt.subplot2grid((30, 40), (5, 20), colspan=2, rowspan=3)
nl_ax.text(0, 0, 'Received', verticalalignment='bottom', 
			horizontalalignment='left', color='#666', fontsize=20)
plt.axis('off')


# Received image
img = plt.imread('img/question-mark.png')
plt.subplot2grid((30, 40), (11, 20), colspan=17, rowspan=17)
plt.imshow(img)
plt.gray()
plt.axis('off')

# Start
button1 = plt.imread('img/button1.png')
axButton1 = plt.subplot2grid((30, 40), (2, 2), colspan=8, rowspan=3)
axButton1 = Button(ax=axButton1,
		  label='',
		  image=button1)
plt.axis('off')

# Choose an image
button2 = plt.imread('img/button2.png')
axButton2 = plt.subplot2grid((30, 40), (2, 10), colspan=16, rowspan=3)
axButton2 = Button(ax=axButton2,
		  label='',
		  image=button2)
plt.axis('off')

# Transmitted image prediction
img = plt.imread('img/info.png')
ax3 = plt.subplot2grid((30, 40), (8, 2), colspan=17, rowspan=3)
p = plt.imshow(img)
plt.axis('off')

# Received image prediction
img = plt.imread('img/info.png')
ax3 = plt.subplot2grid((30, 40), (8, 20), colspan=17, rowspan=3)
p = plt.imshow(img)
plt.axis('off')

# AWGN
img = plt.imread('img/increase.png')
increase = plt.subplot2grid((30, 40), (2, 26), colspan=2, rowspan=3)
increase = Button(ax=increase,
		  label='',
		  image=img)
plt.axis('off')

img = plt.imread('img/decrease.png')
decrease = plt.subplot2grid((30, 40), (2, 28), colspan=2, rowspan=3)
decrease = Button(ax=decrease,
		  label='',
		  image=img)
plt.axis('off')

# AWGN title
ax3 = plt.subplot2grid((30, 40), (4, 30), colspan=2, rowspan=3)
ax3.text(0.17, 0.97, 'AWGN:', verticalalignment='bottom', 
	horizontalalignment='left', color='#eee', fontsize=20)
plt.axis('off')

# AWGN value
ax3 = plt.subplot2grid((30, 40), (4, 32), colspan=2, rowspan=3)
ax3.text(1.42, 0.97, noise_level, verticalalignment='bottom', 
	horizontalalignment='left', color='#eee', fontsize=20)
plt.axis('off')

# Bypass LabVIEW
labview = plt.subplot2grid((30, 40), (1, 34), colspan=3, rowspan=1)
labview = Button(ax=labview,
		  color='#ff9c9c',
		  label='LabVIEW')

# Additional elements
img = plt.imread('img/bar.png')
ax4 = plt.subplot2grid((30, 40), (28, 2), colspan=17, rowspan=1)
p = plt.imshow(img)
plt.axis('off')
img = plt.imread('img/bar.png')
ax5 = plt.subplot2grid((30, 40), (28, 20), colspan=17, rowspan=1)
p = plt.imshow(img)
plt.axis('off')

# Callbacks
axButton1.on_clicked(compress)
axButton2.on_clicked(chooseImage)
increase.on_clicked(increaseNoise)
decrease.on_clicked(decreaseNoise)
labview.on_clicked(bypassLabview)

# Classify the transmitted image for the first time
classify()

plt.show()
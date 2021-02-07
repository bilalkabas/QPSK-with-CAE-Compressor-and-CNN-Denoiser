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
import cv2
import time

# Clear metadata
def dataClear ():
	dir = os.getcwd() + "/data"
	for f in os.listdir(dir):
		os.remove(os.path.join(dir, f))

px = 100
img_tx = cv2.resize(plt.imread('test-set/cifar10/cifar10_6.jpg'), (px, px))
img_tx_temp = img_tx
noise_level = 0.0
font_size = 21
org_px = 28

def chooseImage (event):
	global img_tx
	global img_tx_temp
	global noise_level
	global org_px
	
	# Open choose file dialog
	root = tk.Tk()
	root.withdraw()
	file_path = filedialog.askopenfilename(initialdir=(os.getcwd() + "/test-set/cifar10"))
	img_tx = cv2.resize(plt.imread(file_path), (px,px))
	org_px = (plt.imread(file_path)).shape[0]

	# Load the image from specified path
	img_tx_temp = img_tx
	plt.subplot2grid((30, 60), (11, 2), colspan=17, rowspan=17)
	plt.imshow(cv2.resize(img_tx.reshape(px,px), (org_px,org_px)))
	plt.axis('off')
	plt.gray()

	# Clear previous prediction results
	plt.subplot2grid((30, 60), (7, 13), colspan=2, rowspan=1)
	plt.cla()
	plt.axis('off')
	plt.subplot2grid((30, 60), (7, 16), colspan=2, rowspan=1)
	plt.cla()
	plt.axis('off')
	plt.subplot2grid((30, 60), (7, 31), colspan=2, rowspan=1)
	plt.cla()
	plt.axis('off')
	plt.subplot2grid((30, 60), (7, 34), colspan=2, rowspan=1)
	plt.cla()
	plt.axis('off')
	plt.subplot2grid((30, 60), (7, 49), colspan=2, rowspan=1)
	plt.cla()
	plt.axis('off')
	plt.subplot2grid((30, 60), (7, 52), colspan=2, rowspan=1)
	plt.cla()
	plt.axis('off')

	# Classification
	classify()
	dataClear()

	# Show question mark in place of received image
	img = plt.imread('img/question-mark.png')
	plt.subplot2grid((30, 60), (11, 20), colspan=17, rowspan=17)
	plt.imshow(img)
	plt.gray()
	plt.axis('off')

	# Show question mark in place of filtered image
	img = plt.imread('img/question-mark.png')
	plt.subplot2grid((30, 60), (11, 38), colspan=17, rowspan=17)
	plt.imshow(img)
	plt.gray()
	plt.axis('off')

	# Reset noise level
	noise_level = 0.0
	nl = '%.1f' % (noise_level)
	nl_ax = plt.subplot2grid((30, 60), (4, 32), colspan=2, rowspan=3)
	nl_ax.text(1.4, 0.97, nl, verticalalignment='bottom', 
		horizontalalignment='left', color='#eee', fontsize=font_size)
	plt.axis('off')

	# Delete data and show plots
	plt.show()


def transmit (event):
	global img_tx
	global org_px

	dataClear()
	encoded = img_tx.reshape(1,px,px,1)
	# Normalize data between 0-255
	normalized = ((encoded-np.min(encoded))/(np.max(encoded)-np.min(encoded))*255).astype(np.uint8)
	# Convert to binary
	binary = np.unpackbits(normalized)
	# Save binary data to csv file
	binary = np.ndarray.flatten(binary)
	np.savetxt("data/encoded.csv", binary, delimiter=",")

	# Check if demodulated data exists
	file_path = 'data/demod.csv'
	while not os.path.exists(file_path):
		time.sleep(1)

	if os.path.isfile(file_path):
		# Decoding
		demod_bin = genfromtxt(file_path, delimiter=',')
		demod_bin = np.reshape(demod_bin, (int(len(demod_bin)/8), 8)).astype(int)
		demod_dec = np.packbits(demod_bin)
		# demod_float = demod_dec.astype('float32') / 30
		decoded = np.reshape(demod_dec, (1,px,px,1))

		# Show received image
		plt.subplot2grid((30, 60), (11, 20), colspan=17, rowspan=17)
		plt.imshow(cv2.resize(decoded.reshape(px,px), (org_px,org_px)))
		plt.axis('off')
		plt.gray()
		
		# Filtering
		denoiser = load_model('models/denoiser.h5')
		denoiser_out = denoiser.predict(decoded/255.)
		denoiser_out = cv2.resize(denoiser_out.reshape(px,px), (org_px,org_px))
		
		# Show denoiser output
		plt.subplot2grid((30, 60), (11, 38), colspan=17, rowspan=17)
		plt.imshow(denoiser_out.reshape(org_px,org_px))
		plt.axis('off')

		if org_px == 28: 
			# Classify received image
			classification = load_model('models/classification.h5')
			pre = classification.predict(cv2.resize(decoded.reshape(px,px), (org_px,org_px)).reshape(1,org_px,org_px,1))
			pre = ((pre-np.min(pre))/(np.max(pre)-np.min(pre))*255).astype(np.uint8)
			pre = ((100/np.sum(pre))*pre).astype(np.uint8)
			index = np.argmax(pre)

			# Show prediction results
			print("The image predicted as '" + str(index) + "' with", str(pre[0][index]) + "% probability.")
			predict = plt.subplot2grid((30, 60), (7, 31), colspan=2, rowspan=1)
			predict.text(0.60, -2.5, (str(pre[0][index]) + '%'), verticalalignment='bottom', 
				horizontalalignment='right', color='#111', fontsize=font_size)
			plt.axis('off')
			predict = plt.subplot2grid((30, 60), (7, 34), colspan=2, rowspan=1)
			predict.text(0.70, -2.5, index, verticalalignment='bottom', 
				horizontalalignment='right', color='#eee', fontsize=font_size)
			plt.axis('off')

			# Classify filtered image
			pre = classification.predict(denoiser_out.reshape(1,org_px,org_px,1))
			pre = ((pre-np.min(pre))/(np.max(pre)-np.min(pre))*255).astype(np.uint8)
			pre = ((100/np.sum(pre))*pre).astype(np.uint8)
			index = np.argmax(pre)

			# Show prediction results for filtered image
			predict = plt.subplot2grid((30, 60), (7, 49), colspan=2, rowspan=1)
			predict.text(0.60, -2.5, (str(pre[0][index]) + '%'), verticalalignment='bottom', 
				horizontalalignment='right', color='#111', fontsize=font_size)
			plt.axis('off')
			predict = plt.subplot2grid((30, 60), (7, 52), colspan=2, rowspan=1)
			predict.text(0.70, -2.5, index, verticalalignment='bottom', 
				horizontalalignment='right', color='#eee', fontsize=font_size)
			plt.axis('off')

		plt.show()


def classify ():
	global org_px
	if org_px == 28:
		# Load model and predict
		global img_tx
		model = load_model('models/classification.h5')
		img_tx = img_tx.astype('float32') / 255.
		img_tx = np.reshape(img_tx, (1, px, px, 1))
		pre = model.predict((cv2.resize(img_tx.reshape(px,px), (org_px,org_px))).reshape(1,org_px,org_px,1))
		pre = ((pre-np.min(pre))/(np.max(pre)-np.min(pre))*255).astype(np.uint8)
		pre = ((100/np.sum(pre))*pre).astype(np.uint8)
		index = np.argmax(pre)
		
		# Show prediction results    
		print("The image predicted as '" + str(index) + "' with", str(pre[0][index]) + "% probability.")

		predict = plt.subplot2grid((30, 60), (7, 13), colspan=2, rowspan=1)
		predict.text(0.60, -2.5, (str(pre[0][index]) + '%'), verticalalignment='bottom', 
			horizontalalignment='right', color='#111', fontsize=font_size)
		plt.axis('off')

		predict = plt.subplot2grid((30, 60), (7, 16), colspan=2, rowspan=1)
		predict.text(0.70, -2.5, index, verticalalignment='bottom', 
			horizontalalignment='right', color='#eee', fontsize=font_size)
		plt.axis('off')

	
def increaseNoise (event):
	global noise_level
	global img_tx
	global img_tx_temp
	global org_px

	if noise_level <= 0.9 and noise_level >= 0:
		noise_level = noise_level + 0.1
		nl = '%.1f' % (noise_level)
		nl_ax = plt.subplot2grid((30, 60), (4, 32), colspan=2, rowspan=3)
		nl_ax.text(1.4, 0.97, nl, verticalalignment='bottom', 
			horizontalalignment='left', color='#eee', fontsize=font_size)
		plt.axis('off')
		# Add noise
		plt.subplot2grid((30, 60), (11, 2), colspan=17, rowspan=17)
		img_tx = img_tx.reshape(px,px)
		img_tx = img_tx_temp + noise_level*255*np.random.normal(0, 1, img_tx.shape)
		img_tx = np.clip(img_tx, 0, 255)
		plt.imshow(cv2.resize((img_tx), (org_px,org_px)))
		img_tx = img_tx.astype('float32') / 255.
		plt.axis('off')
		plt.gray()
		plt.show()


def decreaseNoise (event):
	global noise_level
	global img_tx
	global img_tx_temp
	global org_px

	if noise_level <= 1 and noise_level >= 0.1:
		noise_level = noise_level - 0.1
		nl = '%.1f' % (noise_level)
		nl_ax = plt.subplot2grid((30, 60), (4, 32), colspan=2, rowspan=3)
		nl_ax.text(1.4, 0.97, nl, verticalalignment='bottom', 
			horizontalalignment='left', color='#eee', fontsize=font_size)
		plt.axis('off')
		# Add noise
		plt.subplot2grid((30, 60), (11, 2), colspan=17, rowspan=17)
		img_tx = img_tx.reshape(px,px)
		img_tx = img_tx_temp + noise_level*255*np.random.normal(0, 1, img_tx.shape)
		img_tx = np.clip(img_tx, 0, 255)
		plt.imshow(cv2.resize(img_tx, (org_px,org_px)))
		img_tx = img_tx.astype('float32') / 255.
		plt.axis('off')
		plt.gray()
		plt.show()


# Figure window configurations
mpl.rcParams['toolbar'] = 'None'
figure(num=None, figsize=(15, 7.5), dpi=80, facecolor='#000', edgecolor='k')
fig = plt.gcf()
fig.canvas.set_window_title('Denoiser')

# Show transmitted image
plt.subplot2grid((30, 60), (11, 2), colspan=17, rowspan=17)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.imshow(cv2.resize(img_tx.reshape(px,px), (org_px,org_px)))
plt.gray()
plt.axis('off')

# Show 'transmitted' label
nl_ax = plt.subplot2grid((30, 60), (5, 2), colspan=2, rowspan=3)
nl_ax.text(0, 0, 'Transmitted', verticalalignment='bottom', 
			horizontalalignment='left', color='#666', fontsize=font_size)
plt.axis('off')

# Show 'Received' label
nl_ax = plt.subplot2grid((30, 60), (5, 20), colspan=2, rowspan=3)
nl_ax.text(0, 0, 'Received', verticalalignment='bottom', 
			horizontalalignment='left', color='#666', fontsize=font_size)
plt.axis('off')

# Original image
img = plt.imread('img/question-mark.png')
plt.subplot2grid((30, 60), (11, 2), colspan=17, rowspan=17)
plt.imshow(img)
plt.gray()
plt.axis('off')

# Received image
img = plt.imread('img/question-mark.png')
plt.subplot2grid((30, 60), (11, 20), colspan=17, rowspan=17)
plt.imshow(img)
plt.gray()
plt.axis('off')

# Show 'Filtered' label
nl_ax = plt.subplot2grid((30, 60), (5, 38), colspan=2, rowspan=3)
nl_ax.text(0, 0, 'Filtered', verticalalignment='bottom', 
			horizontalalignment='left', color='#666', fontsize=font_size)
plt.axis('off')

# Filtered image
img = plt.imread('img/question-mark.png')
plt.subplot2grid((30, 60), (11, 38), colspan=17, rowspan=17)
plt.imshow(img)
plt.gray()
plt.axis('off')

# Start
button1 = plt.imread('img/button1.png')
axButton1 = plt.subplot2grid((30, 60), (2, 2), colspan=8, rowspan=3)
axButton1 = Button(ax=axButton1,
		  label='',
		  image=button1)
plt.axis('off')

# Choose an image
button2 = plt.imread('img/button2.png')
axButton2 = plt.subplot2grid((30, 60), (2, 10), colspan=16, rowspan=3)
axButton2 = Button(ax=axButton2,
		  label='',
		  image=button2)
plt.axis('off')

# Transmitted image prediction
img = plt.imread('img/info.png')
ax3 = plt.subplot2grid((30, 60), (8, 2), colspan=17, rowspan=3)
p = plt.imshow(img)
plt.axis('off')

# Received image prediction
img = plt.imread('img/info.png')
ax3 = plt.subplot2grid((30, 60), (8, 20), colspan=17, rowspan=3)
p = plt.imshow(img)
plt.axis('off')

# Filtered image prediction
img = plt.imread('img/info.png')
ax3 = plt.subplot2grid((30, 60), (8, 38), colspan=17, rowspan=3)
p = plt.imshow(img)
plt.axis('off')

# AWGN
img = plt.imread('img/increase.png')
increase = plt.subplot2grid((30, 60), (2, 26), colspan=2, rowspan=3)
increase = Button(ax=increase,
		  label='',
		  image=img)
plt.axis('off')

img = plt.imread('img/decrease.png')
decrease = plt.subplot2grid((30, 60), (2, 28), colspan=2, rowspan=3)
decrease = Button(ax=decrease,
		  label='',
		  image=img)
plt.axis('off')

# AWGN title
ax3 = plt.subplot2grid((30, 60), (4, 30), colspan=2, rowspan=3)
ax3.text(0.17, 0.97, 'AWGN:', verticalalignment='bottom', 
	horizontalalignment='left', color='#eee', fontsize=font_size)
plt.axis('off')

# AWGN value
ax3 = plt.subplot2grid((30, 60), (4, 32), colspan=2, rowspan=3)
ax3.text(1.42, 0.97, noise_level, verticalalignment='bottom', 
	horizontalalignment='left', color='#eee', fontsize=font_size)
plt.axis('off')

# Additional elements
img = plt.imread('img/bar.png')
ax4 = plt.subplot2grid((30, 60), (28, 2), colspan=17, rowspan=1)
p = plt.imshow(img)
plt.axis('off')
img = plt.imread('img/bar.png')
ax5 = plt.subplot2grid((30, 60), (28, 20), colspan=17, rowspan=1)
p = plt.imshow(img)
plt.axis('off')
img = plt.imread('img/bar.png')
ax5 = plt.subplot2grid((30, 60), (28, 38), colspan=17, rowspan=1)
p = plt.imshow(img)
plt.axis('off')

axButton1.on_clicked(transmit)
axButton2.on_clicked(chooseImage)
increase.on_clicked(increaseNoise)
decrease.on_clicked(decreaseNoise)

plt.show()
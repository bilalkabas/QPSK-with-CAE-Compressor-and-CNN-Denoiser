# Optimization of QPSK-Based Digital Communication Systems Using CAE Compression and CNN Denoising

[![LabVIEW Version](https://img.shields.io/badge/LabVIEW-Comms%202.0-%234285F4?style=flat-square)](https://www.ni.com/en-tr/support/downloads/software-products/download.labview-communications-system-design-suite.html#306816)
[![license](https://img.shields.io/badge/license-AGPL%203.0-%23F65314?style=flat-square)](LICENSE)

This repository contains implementation of a QPSK-based telecommunication system optimized using deep learning based image compression and denoising in LabVIEW Communications environment using Python and Keras. For more information, please refer to the [paper]().

> ***Abstract*** - In this research, deep learning methods are used for optimization of QPSK-based communication systems. We propose a convolutional autoencoder based lossy image compression technique to increase data rate. We also propose a ResNet classifier to analyze the performance of the CAE compressor. This research also presents a CNN denoiser architecture for removing additive white Gaussian noise (AWGN) from images. In this research, we conducted performance analysis of filters by using PSNR (peak-signal-to-noise-ratio) and SSIM (structural similarity index measure) evaluation metrics. Experimental results demonstrate that our CNN denoiser model outperforms classical image filtering techniques which are Gaussian and Wiener filtering based on SSIM values. Our QPSK implementation includes root-raised-cosine filtering for pulse shaping and matched filtering to improve bit error ratio as well as to minimize intersymbol interference (ISI). It also has an adaptive equalizer filter to eliminate the ISI caused by the filtering effect of the communication channel.

## Getting Started

LabVIEW Communications System Design Suite is where you can connect to USRP-2900, a software defined radio, and use it as transmitter and receiver. In this application, we basically send an image and receive it back. In this project, there are two main aims.

- Increase the data rate using image compression
- Eliminate AWGN (additive white Gaussian noise) using denoising

**What you can do with this repository**

- If you have LabVIEW Communications installed, you can run all the system.
- If you do not have LabVIEW Communications installed, you can run CAE compressor and CNN denoiser anyway.

### CAE Compression

To compress images, we designed a simple convolutional autoencoder (CAE) model. This is a lossy image compression technique and can be implemented only for gray scale handwritten digits since we used the MNIST handwritten digits data set [1] in training. The compressed form of images is called **bottleneck**. Original images are 28x28 and bottleneck is 4x4x8. We added an AWGN with random variances to training images to add a noise elimination feature to the CAE compressor. 

### CNN Denoiser

The denoiser is supposed to eliminate AWGN from received images since AWGN is one of the most common forms of distortion in communication systems. The CNN denoiser consist of consecutive convolutional layers. AWGN with random variances is added to the training set and mean squared error loss function was used in training.

## How to run codes

### Recommended requirements

If you have Python installed and have the libraries specified in `requirements.txt`, you can run directly run [`compressor.py`](app) and [`denoiser.py`](app). However, we recommend you to have [Anaconda](https://www.anaconda.com/products/individual) installed on your computer to create a virtual environment.


**If you have**

## Authors

- [Samet Öğüten](https://github.com/sametoguten)
- [Bilal Kabaş](https://github.com/bilalkabas)

## License

This project is licensed under the [GNU Affero General Public License](LICENSE).

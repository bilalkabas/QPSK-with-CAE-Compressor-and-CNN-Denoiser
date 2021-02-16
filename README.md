# Optimization of QPSK-Based Digital Communication Systems Using CAE Compression and CNN Denoising

[![LabVIEW Version](https://img.shields.io/badge/LabVIEW-Comms%202.0-%234285F4?style=flat-square)](https://www.ni.com/en-tr/support/downloads/software-products/download.labview-communications-system-design-suite.html#306816)
[![license](https://img.shields.io/badge/license-AGPL%203.0-%23F65314?style=flat-square)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.13140/RG.2.2.33830.65602-%239aed00?style=flat-square)](https://www.researchgate.net/publication/349346715_Optimization_of_QPSK-Based_Digital_Communication_Systems_Using_CAE_Compression_and_CNN_Denoising)

This repository contains implementation of a QPSK-based telecommunication system optimized using deep learning based image compression and denoising in LabVIEW Communications environment using Python and Keras. **To run the CAE compressor and CNN denoiser, you do not have to have LabVIEW installed**. For more information about the research, please refer to the [paper]().

> ***Abstract*** - In this research, deep learning methods are used for optimization of QPSK-based communication systems. We propose a convolutional autoencoder based lossy image compression technique to increase data rate. We also propose a ResNet classifier to analyze the performance of the CAE compressor. This research also presents a CNN denoiser architecture for removing additive white Gaussian noise (AWGN) from images. In this research, we conducted performance analysis of filters by using PSNR (peak-signal-to-noise-ratio) and SSIM (structural similarity index measure) evaluation metrics. Experimental results demonstrate that our CNN denoiser model outperforms classical image filtering techniques which are Gaussian and Wiener filtering based on SSIM values. Our QPSK implementation includes root-raised-cosine filtering for pulse shaping and matched filtering to improve bit error ratio as well as to minimize intersymbol interference (ISI). It also has an adaptive equalizer filter to eliminate the ISI caused by the filtering effect of the communication channel.

## Getting Started

<img align="right" width="400" src="https://user-images.githubusercontent.com/53112883/107996910-9a835880-6ff2-11eb-8140-e100847c6d0e.png">

LabVIEW Communications System Design Suite is where you can connect to USRP-2900, a software defined radio, and use it as transmitter and receiver. In this application, we basically send an image and receive it back. In this project, there are two main aims.

- Increase the data rate using image compression
- Eliminate AWGN (additive white Gaussian noise) using denoising

**What you can do with this repository**

- If you have LabVIEW Communications installed, you can run all the communication system.
- If you do not have LabVIEW Communications installed, you can run CAE compressor and CNN denoiser anyway.

### CAE Compression

To compress images, we designed a simple convolutional autoencoder (CAE) model. This is a lossy image compression technique and can be implemented only for gray scale handwritten digits since we used the MNIST handwritten digits data set [1] in training. The compressed form of images is called **bottleneck**. Original images are 28x28 and bottleneck is 4x4x8. We added an AWGN with random variances to training images to add a noise elimination feature to the CAE compressor. 

### CNN Denoiser

The denoiser is supposed to eliminate AWGN from received images since AWGN is one of the most common forms of distortion in communication systems. The CNN denoiser consist of consecutive convolutional layers. AWGN with random variances is added to the training set and mean squared error loss function was used in training.

## How to run codes

### Recommended requirements

If you have Python installed and have the libraries specified in `requirements.txt`, you can run directly run [`compressor.py`](app) and [`denoiser.py`](app). However, we recommend you to have [Anaconda](https://www.anaconda.com/products/individual) installed on your computer to create a virtual environment.

#### 1. Clone the repository

Open the `Anaconda Command Prompt` and clone this repository.

```
git clone https://github.com/bilalkabas/QPSK-with-CAE-Compressor-and-CNN-Denoiser
```

#### 2. Create a virtual environment in Anaconda

Create the environment `comm-env`

```
conda create -n comm-env python==3.8
```

Activate the environment


```
conda activate comm-env
```

#### 3. Install required Python libraries

```
cd QPSK-with-CAE-Compressor-and-CNN-Denoiser

pip install -r requirements.txt
```

#### 4. Load data sets (optional)

At this point, you can use the existing handwritten digits in `app/test-set` directory or you may go and download [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/) and save them in `app/test-set/cifar10` and `app/test-set/mnist` directories respectively.

#### 5. Run the CAE compressor

```
cd app

python compressor.py
```

#### 6. Run the CNN denoiser

Make sure that you are in the `app` directory.

```
python denoiser.py
```

#### 7. Transmit and receive data in LabVIEW using USRP (optional)

For this, you need to have [LabVIEW Communications System Design Suite](https://www.ni.com/en-tr/support/downloads/software-products/download.labview-communications-system-design-suite.html#306816) installed and also an USRP connected to your computer. In this research, we used USRP-2900 with lookback cable connected.

Go to main directory of this project and open `QPSK_Main.lvproject`. On the left, you should see two project files: `QPSK_TX.gvi` and `QPSK_RX.gvi`. Open and run `QPSK_TX.gvi`. It will open up the CAE compressor or CNN denoiser GUI based on your selection. Click the 'LabVIEW' button on the top-right corner so it turns to green. Select the image that you want to process and transmit then click 'Start'. Open and run `QPSK_RX.gvi`. Now, you should see the received image on the GUI.

## References

[1] Y. LeCun, C. Cortes, and C. Burges, “Mnist handwritten digit database,” ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, vol. 2, 2010

## Authors

- [Samet Öğüten](https://github.com/sametoguten)
- [Bilal Kabaş](https://github.com/bilalkabas)

## License

This project is licensed under the [GNU Affero General Public License](LICENSE).

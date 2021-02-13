# Optimization of QPSK-Based Digital Communication Systems Using CAE Compression and CNN Denoising

[![LabVIEW Version](https://img.shields.io/badge/LabVIEW-Comms%202.0-%234285F4?style=flat-square)](https://www.ni.com/en-tr/support/downloads/software-products/download.labview-communications-system-design-suite.html#306816)
[![license](https://img.shields.io/badge/license-AGPL%203.0-%23F65314?style=flat-square)](LICENSE)

> ***Abstract*** - In this research, deep learning methods are used for optimization of QPSK-based communication systems. We propose a convolutional autoencoder based lossy image compression technique to increase data rate. We also propose a ResNet classifier to analyze the performance of the CAE compressor. This research also presents a CNN denoiser architecture for removing additive white Gaussian noise (AWGN) from images. In this research, we conducted performance analysis of filters by using PSNR (peak-signal-to-noise-ratio) and SSIM (structural similarity index measure) evaluation metrics. Experimental results demonstrate that our CNN denoiser model outperforms classical image filtering techniques which are Gaussian and Wiener filtering based on SSIM values. Our QPSK implementation includes root-raised-cosine filtering for pulse shaping and matched filtering to improve bit error ratio as well as to minimize intersymbol interference (ISI). It also has an adaptive equalizer filter to eliminate the ISI caused by the filtering effect of the communication channel.

## Getting Started

## Authors

- [Samet Öğüten](https://github.com/sametoguten)
- [Bilal Kabaş](https://github.com/bilalkabas)

## License

This project is licensed under the [GNU Affero General Public License](LICENSE).

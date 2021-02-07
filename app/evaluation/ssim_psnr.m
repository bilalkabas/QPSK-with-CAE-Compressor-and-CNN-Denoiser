% Calculate mean SSIM and PSNR values on Berkeley's BSDS500 test images
clear;clc;

for i=0:38
    original = imread("figs/original" + i + ".jpg");
    noisy = imread("figs/noisy" + i + ".jpg");
    cnn = imread("figs/filtered_cnn" + i + ".jpg");
    gauss = imread("figs/filtered_gaussian" + i + ".jpg");
    wiener = imread("figs/filtered_wiener" + i + ".jpg");
    
    ssim_cnn(i+1) = ssim(cnn,original);
    ssim_gauss(i+1) = ssim(gauss,original);
    ssim_wiener(i+1) = ssim(wiener,original);
    ssim_noisy(i+1) = ssim(noisy,original);
    
    psnr_cnn(i+1) = psnr(cnn,original);
    psnr_gauss(i+1) = psnr(gauss,original);
    psnr_wiener(i+1) = psnr(wiener,original);
    psnr_noisy(i+1) = psnr(noisy,original);
end

mean_cnn = mean(ssim_cnn)
mean_gauss = mean(ssim_gauss)
mean_wiener = mean(ssim_wiener)
mean_noisy = mean(ssim_noisy)

mean_psnr_cnn = mean(psnr_cnn)
mean_psnr_gauss = mean(psnr_gauss)
mean_psnr_wiener = mean(psnr_wiener)
mean_psnr_noisy = mean(psnr_noisy)
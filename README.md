
To everyone interesting in our work [CFANet](https://openaccess.thecvf.com/content/WACV2021/html/Rong_Coarse-_and_Fine-Grained_Attention_Network_With_Background-Aware_Loss_for_Crowd_WACV_2021_paper.html) in WACV2021:

MARUNet in this repo is identical to the the CFANet without Density level estimator, that means only density map estimator and crowd region recognizer are used. The second row `w. CRR` means the MARUNet in Table 7 of our paper. The name MARUNet is unchanged since we wrote another manuscript before and upgrade it to CFANet and submit it to WACV2021. I have graduated last summer, so just use MARUNet is ok, which is also a good baseline, since it can get 56.9 MAE on SHA. 

# MARUNet

Multi-level Attention Refined UNet for crowd counting.

## Data preparation

Images and groundtruth are read into dataset via .json file which is specified in json directory. Preprocessed Shanghaitech and UCF-QNRF datasets can be downloaded from this [link](https://pan.baidu.com/s/1S3dstjZ6JyxceQ4mccj77w)(Extraction Code: xvd2). Modifying the path in .json file and data can be read.

## Training

```python train_generic.py --model MARNet --epochs 100 --dataset qnrf --train_json json/qnrf_train.json --val_json json/qnrf_val.json --loss 3avg-ms-ssim --lazy_val 0```

You can replace `MARNet` with `U_VGG`. Note that `MARNet` is identical to `MARUNet` and `U_VGG` is identical to `MSUNet` in our paper.

## Testing

Use test_one_image.py to test a given image. You need to set `divide` to 50 and `ds(downsample)` to 1 in `img_test()` to get correct results. Some unused functions are not removed so you need to remove them to run it. If you want to test a model on a dataset, you need to modify it.

## Pretrained Models

Download links:

||MARUNet(MARNet)||MSUNet(U_VGG)||
|-|-|-|-|-|
|SHA|[Google Drive](https://drive.google.com/file/d/12CKLhSkNPwCpSu0WwfQa-WGHMd4RXhlb/view?usp=sharing)|[Baidu Disk](https://pan.baidu.com/s/1ovKkAayigImwiIMmMYquLw), Extraction Code: hg9y|[Google Drive](https://drive.google.com/file/d/1S6wqC8si1l67tbnFxWGMjvZqkSs-zxn-/view?usp=sharing)|[Baidu Disk](https://pan.baidu.com/s/1ziUYS2E1epkmOAXvHXg3NQ) Extraction Code: ib2g|
|SHB||[Baidu Disk](https://pan.baidu.com/s/1ApbLPYsA1bKq3DaJczkBeQ) Extraction Code: 21x7||[Baidu Disk](https://pan.baidu.com/s/17vzda2tEm1Q1SPjQE6gbbw) Extraction Code: 0baw|
|QNRF||[Baidu Disk](https://pan.baidu.com/s/1SZIkroUG9Wr0Jo09bqf2dw) Extraction Code: 5ns9||[Baidu Disk](https://pan.baidu.com/s/1gsErvJOcyPFx3ycOT-VHMQ) Extraction Code: yjmr|

## Performance

Shanghaitech PartA

|Method|MAE|RMSE|SSIM|PSNR|
|--|--|--|--|--|
|MSUNet|57.9|94.5|0.86|29.79|
|MARUNet|56.9|91.8|0.86|29.90|

Shanghaitech PartB

|Method|MAE|RMSE|SSIM|PSNR|
|--|--|--|--|--|
|MSUNet|6.9|12.5|0.96|30.72|
|MARUNet|6.6|10.6|0.96|31.04|

UCF_CC_50

|Method|MAE|RMSE|SSIM|PSNR|
|--|--|--|--|--|
|MSUNet|316.1|442.3|0.61|19.33|
|MARUNet|233.3|313.8|0.63|19.82|

UCF-QNRF

|Method|MAE|RMSE|SSIM|PSNR|
|--|--|--|--|--|
|MSUNet|90.9|158.8|0.89|32.60|
|MARUNet|90.8|155.1|0.90|32.79|

## Other Retrained Models with MSL

We retrain existing models on SHA dataset with our Multi-scale Structural Loss(MSL). Compared to original MSE loss, better performance is achieved.

||Link|MAE(MSE/MSL)|RMSE(MSE/MSL)|
|-|-|-|-|
|MCNN|[Baidu Disk](https://pan.baidu.com/s/1qk69OX3OIRgOqVaQ9QWICA) Extraction Code: ubx5|110.2/**89.1**|173.2/**142.9**|
|CSRNet|[Baidu Disk](https://pan.baidu.com/s/1K38a3suPZlJNMoio7_s-qg) Extraction Code: iqme|68.2/**63.4**|115.0/**103.1**|
|CAN|[Baidu Disk](https://pan.baidu.com/s/1CjMQnC7371dT1_zhOYG3Qg) Extraction Code: s93r|62.3/**59.1**|100.0/**90.5**|

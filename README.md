
To everyone interested in our work [CFANet](https://openaccess.thecvf.com/content/WACV2021/html/Rong_Coarse-_and_Fine-Grained_Attention_Network_With_Background-Aware_Loss_for_Crowd_WACV_2021_paper.html) in WACV2021:

MARUNet in this repo is identical to the the CFANet without Density level estimator, that means only density map estimator and crowd region recognizer are used. The second row `w. CRR` means the MARUNet in Table 7 of our paper. The name MARUNet is unchanged since we wrote another manuscript before and upgrade it to CFANet and submit it to WACV2021. I have graduated last summer, so just use MARUNet is ok, which is also a good baseline, since it can get 56.9 MAE on SHA. 

# MARUNet

Multi-level Attention Refined UNet for crowd counting.

## Data preparation

Images and groundtruth are read into dataset via .json file which is specified in json directory. Preprocessed Shanghaitech and UCF-QNRF datasets can be downloaded from:

(1) **Baidu disk**: [link](https://pan.baidu.com/s/1S3dstjZ6JyxceQ4mccj77w)(Extraction Code: xvd2). 

(2) **Google drive**: [UCF-QNRF](https://drive.google.com/file/d/1lju4G1Da61ddXR-FQw-zOIe6cCsGg0OL/view?usp=sharing)(For images with width or height larger than 1024, we resize the larger side to 1024, e.g., 2048\*1024 -> 1024\*512), [ShanghaiTech](https://drive.google.com/file/d/1k1BOb-0wGO8PYt6_GVsof_Ne3udft0bG/view?usp=sharing)

Modifying the path in .json file and data can be read.

## Training

```python train_generic.py --model MARNet --epochs 100 --dataset qnrf --train_json json/qnrf_train.json --val_json json/qnrf_val.json --loss 3avg-ms-ssim --lazy_val 0```

(`MARNet` is identical to `MARUNet`.)

## Testing

Use test_one_image.py to test a given image. You need to set `divide` to 50 and `ds(downsample)` to 1 in `img_test()` to get correct results. Some unused functions are not removed so you need to remove them to run it. If you want to test a model on a dataset, you need to modify it.

## Pretrained Models

Download links:

||MARUNet(MARNet)||
|-|-|-|
|SHA|[Google Drive](https://drive.google.com/file/d/12CKLhSkNPwCpSu0WwfQa-WGHMd4RXhlb/view?usp=sharing)|[Baidu Disk](https://pan.baidu.com/s/1ovKkAayigImwiIMmMYquLw), Extraction Code: hg9y|
|SHB|[Google Drive](https://drive.google.com/file/d/1O7Yk3bbXPXUkTKPBP73j5q9v-ZCCHTIe/view?usp=sharing)|[Baidu Disk](https://pan.baidu.com/s/1ApbLPYsA1bKq3DaJczkBeQ) Extraction Code: 21x7|
|QNRF||[Baidu Disk](https://pan.baidu.com/s/1SZIkroUG9Wr0Jo09bqf2dw) Extraction Code: 5ns9|

## Performance

Shanghaitech PartA

|Method|MAE|RMSE|SSIM|PSNR|
|--|--|--|--|--|
|MARUNet|56.9|91.8|0.86|29.90|

Shanghaitech PartB

|Method|MAE|RMSE|SSIM|PSNR|
|--|--|--|--|--|
|MARUNet|6.6|10.6|0.96|31.04|

UCF_CC_50

|Method|MAE|RMSE|SSIM|PSNR|
|--|--|--|--|--|
|MARUNet|233.3|313.8|0.63|19.82|

UCF-QNRF

|Method|MAE|RMSE|SSIM|PSNR|
|--|--|--|--|--|
|MARUNet|90.8|155.1|0.90|32.79|

## Other Retrained Models with MSL

We retrain existing models on SHA dataset with our Multi-scale Structural Loss(MSL). Compared to original MSE loss, better performance is achieved.

||Link|MAE(MSE/MSL)|RMSE(MSE/MSL)|
|-|-|-|-|
|MCNN|[Baidu Disk](https://pan.baidu.com/s/1qk69OX3OIRgOqVaQ9QWICA) Extraction Code: ubx5|110.2/**89.1**|173.2/**142.9**|
|CSRNet|[Baidu Disk](https://pan.baidu.com/s/1K38a3suPZlJNMoio7_s-qg) Extraction Code: iqme|68.2/**63.4**|115.0/**103.1**|
|CAN|[Baidu Disk](https://pan.baidu.com/s/1CjMQnC7371dT1_zhOYG3Qg) Extraction Code: s93r|62.3/**59.1**|100.0/**90.5**|

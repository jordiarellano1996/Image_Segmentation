# UW-Madison GI Tract Image Segmentation.
## Introduction
Track healthy organs in medical scans to improve cancer treatment.
## Technologies
<ul>
  <li>Python</li>
  <li>Tensorflow</li>
  <li>Keras</li>
  <li>sklearn</li>
  <li>skimage</li>
  <li>Numpy</li>
  <li>Matplotlib</li>
</ul>

## ⚽ Methodlogy
<ul>
  <li>Train unet model using Tensorflow and Keras.</li>
  <li>As there are overlaps between Stomach, Large Bowel & Small Bowel classes, this is a MultiLabel Segmentation task,
  so final activaion should be sigmoid instead of softmax.</li>
  <li>For data split I'll be using StratifiedGroupFold to avoid data leakage due to case and to stratify empty
  and non-empty mask cases.</li>
</ul>

## 🚀Wandb report
Check results on: https://wandb.ai/jordiarellano1996/ImageSegmentation/reports/-AW-Madison-EDA-UNET-Train--VmlldzoyMzM1MjA3?accessToken=n56f735rghx5d20kos3vlp435ek8tq9qntdw5qjoenvkjfpg0djlm2f6ytqf8nzk





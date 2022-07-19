#!/bin/bash

apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
pip3 install pandas
pip3 install opencv-python
pip3 install -U scikit-learn
pip3 install pillow
pip3 install wandb


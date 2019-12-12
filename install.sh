#!/bin/bash
conda create -n 236 --python=3.7
conda actiavte 236
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch tqdm -y
# MoreMori

## Introduction

We perform a faceswap from anyone's face to Greg Mori. Greg Mori is a professor at SFU.

A variety of previous work exists regarding the use of [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 
for style transfers. CycleGAN has wide-ranging applications, and is not specific to human faces. 
[DeepFake](https://github.com/deepfakes/faceswap) builds on CycleGAN by mapping face-specific features to assist in 
alignment of features during transfer. 

We used a data-centric approach, leaving the networks of CycleGAN unmodified. This allowed us to explore the effects of 
data inputs on training and testing results with a known-good network.

The corresponding paper and poster for this project is available in the folder [docs](docs).

## Information

This repository is currently under construction to be more user friendly. 

The project will run on your computer's camera with<br/>
`main.py -n path/to/model.pth`

### Prequisites
cmake<br/>
[condas](https://conda.io/miniconda.html)

### Installation

```
conda env create -f more_mori_env.yml
source activate more_mori_env
pip install -r requirements.txt
```

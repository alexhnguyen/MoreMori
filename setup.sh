#!/usr/bin/env bash
# Cloning CycleGAN repository
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Adding augment files
cp augment/cyclegan_augment/augment_background.py pytorch-CycleGAN-and-pix2pix/data/augment_background.py
cp augment/cyclegan_augment/base_dataset.py pytorch-CycleGAN-and-pix2pix/data/base_dataset.py

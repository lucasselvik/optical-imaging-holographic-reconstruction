# Optical-Imaging-Holographic-Reconstruction

A MATLAB-based optical imaging system that models ray propagation using ray-transfer matrices and reconstructs holographic light-field data through clustering and thin-lens focusing.

Project Report

Read the full report here:
[Optical Imaging System for Holographic Reconstruction (PDF)
](https://drive.google.com/file/d/1HtWvbMBRDAds5VIGEWOiM0jaA1nplgyx/view?usp=drive_link)

Overview

This project simulates an optical imaging pipeline using linear system theory to analyze and reconstruct images encoded within a holographic light-field dataset. Millions of rays are propagated using free-space and thin-lens ray-transfer matrices, then segmented via K-means clustering to isolate distinct ray bundles. A two-stage focal length search is performed to achieve optimal image focus for each cluster, enabling the recovery of three embedded images.

Features

- Ray propagation using 4×4 ray-transfer matrices
- Free-space and thin-lens modeling
- K-means clustering of (θx, θy) to isolate image groups
- Automatic focal length optimization
- Reconstruction of three holographic images
- Sensor simulation using rays2img

Authors

- Lucas Selvik
- Jonas Kim

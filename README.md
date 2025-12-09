optical-imaging-holographic-reconstruction

A MATLAB-based optical imaging system that models ray propagation using ray-transfer matrices and reconstructs holographic light-field data through clustering and thin-lens focusing.

Project Report

Read the full report here:
[Optical Imaging System for Holographic Reconstruction (PDF)
](https://chatgpt.com/c/69388363-ea6c-8325-ae22-31075337a1ed#:~:text=full%20report%20here%3A-,Optical%20Imaging%20System%20for%20Holographic%20Reconstruction%20(PDF),-%F0%9F%94%8D%20Overview)

Overview

This project simulates an optical imaging pipeline using linear system theory to analyze and reconstruct images encoded within a holographic light-field dataset. Millions of rays are propagated using free-space and thin-lens ray-transfer matrices, then segmented via K-means clustering to isolate distinct ray bundles. A two-stage focal length search is performed to achieve optimal image focus for each cluster, enabling the recovery of three embedded images.

Features

• Ray propagation using 4×4 ray-transfer matrices
• Free-space and thin-lens modeling
• K-means clustering of (θx, θy) to isolate image groups
• Automatic focal length optimization
• Reconstruction of three holographic images
• Sensor simulation using rays2img

How to Run

1. Clone the repository:
  git clone https://github.com/YOUR-USERNAME/optical-imaging-holographic-reconstruction
2. Open MATLAB and add the repository to your path.
3. Load the rays dataset:
  load('rays.mat')
4. Run the main script to propagate, cluster, focus, and reconstruct the images.

Authors

• Lucas Selvik
• Jonas Kim

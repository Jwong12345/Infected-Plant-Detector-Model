Infected Plant Detector 

A deep learning-based plant disease detection system that classifies infected plant leaves and highlights infected regions using computer vision techniques.

This project combines ResNet50 for plant disease classification and U-Net for image segmentation to detect and visualize infected areas on plant leaves. Built using PyTorch and OpenCV.

Features
 Plant leaf disease classification
 Deep learning powered by ResNet50
 Infected region segmentation using U-Net
 Visualization of infected areas with OpenCV
 Built with PyTorch for efficient model training and inference
Model Architecture
1. ResNet50 - Disease Classification

The classification stage uses a pretrained ResNet50 model to identify whether a plant leaf is healthy or infected and determine the disease category.

Responsibilities:

Feature extraction
Disease classification
Transfer learning support
2. U-Net - Infection Segmentation

The segmentation stage uses U-Net to generate pixel-level masks that highlight infected regions on the leaf image.

Responsibilities:

Semantic segmentation
Region highlighting
Infection localization
3. OpenCV - Visualization

OpenCV is used to:

Process input images
Overlay segmentation masks
Highlight infected regions for visualization
Technologies Used
Python
PyTorch
OpenCV
NumPy
torchvision
matplotlib

Dataset

This project can be trained using publicly available plant disease datasets such as:

PlantVillage Dataset
License

This project is licensed under the MIT License.

Author

Developed using PyTorch, ResNet50, U-Net, and OpenCV for intelligent plant disease detection and visualization.

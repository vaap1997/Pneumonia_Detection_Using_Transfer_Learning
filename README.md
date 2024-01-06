# Pneumonia Detection Using Transfer Learning

### *[Medium Article](https://medium.com/@manideeptelukuntla/pneumonia-detection-using-transfer-learning-27275ce15148)*

<div align="center">
  <img src="https://github.com/vaap1997/Pneumonia_Detection_Using_Transfer_Learning/blob/main/Images/HeaderPicture.png" width="800" height="500" alt="Stock Portfolio Optimization">
</div>

## Table of Contents
1. [Introduction/Overview](#1-introductionoverview)
2. [Objective](#2-objective)
3. [Methodology/Approach](#3-methodologyapproach)
   - Why CNNs?
   - Models Employed
   - Libraries and Tools
   - Data Pre-Processing
   - Learning/Modeling
   - Interpretation of Models
4. [File Descriptions](#4-file-descriptions)
5. [Installation/Requirements](#5-installationrequirements)
6. [Data Collection and Sources](#6-data-collection-and-sources)
7. [Results/Conclusion](#7-resultsconclusion)
8. [Contributors/Team](#8-contributorsteam)
9. [References/Citations](#9-referencescitations)
10. [License](#10-license)

## 1. Introduction/Overview
In the realm of medical diagnostics, accurately predicting Pneumonia, a leading cause of mortality according to WHO's 2023 report, is crucial. Traditional diagnosis relies on chest X-ray analysis, a method fraught with human error and fatigue. Recent advancements in machine learning, specifically in image analysis through Convolutional Neural Networks (CNNs), present a promising alternative. This project utilizes various CNN architectures, leveraging transfer learning to enhance pneumonia detection in pediatric chest X-rays.

## 2. Objective
Our aim is to develop and compare different machine learning models for more accurate and reliable pneumonia detection, focusing on pediatric chest X-rays. This is particularly vital in areas with limited healthcare access or where radiologists face high workloads.

## 3. Methodology/Approach
### Why CNNs?
CNNs are chosen for their superiority in medical image classification, owing to hierarchical feature learning, spatial invariance, and efficiency in parameter sharing.

### Models Employed
We explored six CNN models: AlexNet, ResNet, DenseNet, VGG16, MobileNet, and ViT. Each offers unique advantages in pattern recognition and efficiency.

### Libraries and Tools
The implementation involved PyTorch and Keras/TensorFlow, with various functions and classes utilized for model training and evaluation.

### Data Pre-Processing
Using Keras's ImageDataGenerator, we normalized and resized images to fit model requirements, ensuring compatibility and optimal training conditions.

### Learning/Modeling
Each model underwent training with fine-tuning and additional layers where necessary. The process involved adjusting parameters like learning rate, batch size, and number of epochs to optimize performance.

### Interpretation of Models
Focusing on our best-performing model, AlexNet, we applied techniques like activation maps, salience maps, and integrated gradients to interpret the model's decision-making process. This transparency is crucial in healthcare applications.

## 4. File Descriptions
- **Notebooks**: Separate `.ipynb` files for each model
  - `Pneumonia-Detection-Using-AlexNet.ipynb`: this include the base transfeer learning and adding additional layers
  - `Pneumonia-Detection-Using-ResNet.ipynb`
  - `Pneumonia-Detection-Using-DenseNet.ipynb`
  - `Pneumonia-Detection-Using-VGG16.ipynb`
  - `Pneumonia-Detection-Using-MobileNet.ipynb`
  - `Pneumonia-Detection-Using-ViT.ipynb`
    
and one for interpretation techniques (`Interpretations-Using-AlexNet.ipynb`).
- **Images Folder**: Contains architecture diagrams, pre-processing figures, learning curves, etc.
- **Dataset Link**: Provided for easy access to the dataset used.

## 5. Installation/Requirements
- [Python](https://www.python.org/downloads/)
- [Jupyter Notebook](https://jupyter.org/install)
#### Additional Libraries Required
- TensorFlow
- PyTorch

For detailed information on what libraries have been used please go through the notebooks.

## 6. Data Collection and Sources
We utilized a dataset of 5,863 pediatric chest X-rays from Guangzhou Women and Children's Medical Centre, categorized into 'Pneumonia' and 'Normal'. Rigorous quality control and expert verification were employed. The dataset is available at [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2).

## 7. Results/Conclusion
This project demonstrates the potential of CNNs in improving pneumonia detection, especially in challenging scenarios like pediatric cases. Our exploration and comparisons of different models pave the way for more reliable and efficient diagnostic tools, potentially revolutionizing healthcare in resource-limited settings.

## 8. Contributors/Team
- Manideep Telukuntla
- Sankalp Kulkarni
- Vaishnavi Ganesh
- Vanesa Alcantara

## 9. References/Citations
  - https://arxiv.org/abs/1711.05225
  - https://link.springer.com/article/10.1007/s11042-023-16419-1
  - https://pubmed.ncbi.nlm.nih.gov/22009855/
  - https://content.iospress.com/articles/journal-of-x-ray-science-and-technology/xst00304
  - https://www.mdpi.com/2076-3417/10/2/559
  - https://www.sciencedirect.com/science/article/pii/S2666603022000069
  - https://www.cell.com/cell/pdf/S0092-8674(18)30154-5.pdf
  - https://proceedings.mlr.press/v97/tan19a.html?ref=jina-ai-gmbh.ghost.io
  - https://ieeexplore.ieee.org/abstract/document/7935507?casa_token=X5KeNFKctucAAAAA:iQal7BP4Yip3SeRl56t0kgmRgmgrBGhZkhEqCkN6TBRZ6pFDbUIIjOcYxS29vh-nbmkPXaM0BA
  - https://captum.ai/tutorials/TorchVision_Interpret
  - https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4

## 10. License
Licensed under [MIT License](https://github.com/ManideepTelukuntla/InvestigateTMDBMovieData/blob/master/LICENSE)


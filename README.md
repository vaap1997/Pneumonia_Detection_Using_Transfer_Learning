# Pneumonia Detection Using Transfer Learning: Project Overview

## Introduction
In the realm of medical diagnostics, accurately predicting Pneumonia, a leading cause of mortality according to WHO's 2023 report, is crucial. Traditional diagnosis relies on chest X-ray analysis, a method fraught with human error and fatigue. Recent advancements in machine learning, specifically in image analysis through Convolutional Neural Networks (CNNs), present a promising alternative. This project utilizes various CNN architectures, leveraging transfer learning to enhance pneumonia detection in pediatric chest X-rays.

## Project Objective
Our aim is to develop and compare different machine learning models for more accurate and reliable pneumonia detection, focusing on pediatric chest X-rays. This is particularly vital in areas with limited healthcare access or where radiologists face high workloads.

## Data Collection
We utilized a dataset of 5,863 pediatric chest X-rays from Guangzhou Women and Children's Medical Centre, categorized into 'Pneumonia' and 'Normal'. Rigorous quality control and expert verification were employed. The dataset is available at [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2).

## Methodology
### Why CNNs?
CNNs are chosen for their superiority in medical image classification, owing to hierarchical feature learning, spatial invariance, and efficiency in parameter sharing.

### Models Employed
We explored six CNN models: AlexNet, ResNet, DenseNet, VGG16, MobileNet, and ViT. Each offers unique advantages in pattern recognition and efficiency.

### Libraries and Tools
The implementation involved PyTorch and Keras/TensorFlow, with various functions and classes utilized for model training and evaluation.

### Data Pre-Processing
Using Keras's ImageDataGenerator, we normalized and resized images to fit model requirements, ensuring compatibility and optimal training conditions.

## Learning/Modeling
Each model underwent training with fine-tuning and additional layers where necessary. The process involved adjusting parameters like learning rate, batch size, and number of epochs to optimize performance.

## Interpretation of Models
Focusing on our best-performing model, AlexNet, we applied techniques like activation maps, salience maps, and integrated gradients to interpret the model's decision-making process. This transparency is crucial in healthcare applications.

## Repository Structure
- **Notebooks**: Separate `.ipynb` files for each model
  - `PneumoniaDetectionUsingAlexNet.ipynb`
  - `PneumoniaDetectionUsingResNet.ipynb`
  - `PneumoniaDetectionUsingDenseNet.ipynb`
  - `PneumoniaDetectionUsingVGG16.ipynb`
  - `PneumoniaDetectionUsingMobileNet.ipynb`
  - `PneumoniaDetectionUsingViT.ipynb`
    
and one for interpretation techniques (`InterpretationsUsingAlexNet.ipynb`).
- **Images Folder**: Contains architecture diagrams, pre-processing figures, learning curves, etc.
- **Dataset Link**: Provided for easy access to the dataset used.

## Conclusion
This project demonstrates the potential of CNNs in improving pneumonia detection, especially in challenging scenarios like pediatric cases. Our exploration and comparisons of different models pave the way for more reliable and efficient diagnostic tools, potentially revolutionizing healthcare in resource-limited settings.

---


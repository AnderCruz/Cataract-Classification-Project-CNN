# Cataract Classification Project

A deep learning project for classifying cataract images as "mature" or "immature" using TensorFlow/Keras.

## Project Overview

This project implements a convolutional neural network (CNN) to automatically classify cataract images into two categories:
- **Mature** cataracts
- **Immature** cataracts

The model is trained on a dataset of 410 eye images and achieves high accuracy in distinguishing between these two cataract types.

## Dataset

- **Total images**: 410
- **Classes**: 2 (mature, immature)
- **Training set**: 328 images (80% split)
- **Validation set**: 82 images (20% split)
- **Image dimensions**: 416 × 416 pixels

## Model Architecture

The project explores two different neural network architectures:

### 1. Simple Dense Network
- Input layer (416×416×3)
- Rescaling layer (normalization)
- Flatten layer
- Dense layer (64 units, ReLU activation)
- Output layer (1 unit, sigmoid activation)

### 2. Convolutional Neural Network
- Input layer (416×416×3)
- Rescaling layer (normalization)
- Conv2D layer (16 filters, 3×3 kernel, ReLU activation)
- MaxPooling2D layer (2×2)
- Conv2D layer (16 filters, 3×3 kernel, ReLU activation)
- MaxPooling2D layer (2×2)
- Flatten layer
- Dense layer (64 units, ReLU activation)
- Output layer (1 unit, sigmoid activation)

## Training Configuration

- **Batch size**: 16
- **Epochs**: 10
- **Optimizer**: Adam
- **Loss function**: Binary crossentropy
- **Metrics**: Accuracy

## Results

The CNN model demonstrated superior performance with:
- Training accuracy approaching 100%
- Validation accuracy reaching 92.68%
- Smooth convergence with minimal overfitting

## Project Structure

The notebook is organized into three main sections:

1. **Data Preparation** - Loading and preprocessing the cataract image dataset
2. **Simple Model Implementation** - Baseline model with dense layers
3. **CNN Implementation** - Advanced model with convolutional layers for improved feature extraction

## Requirements

- TensorFlow 2.x
- pathlib
- matplotlib

## Usage

1. Ensure the cataract dataset is available in the project directory
2. Run the Jupyter notebook cells sequentially
3. The model will automatically:
   - Load and preprocess images
   - Split data into training/validation sets
   - Train the neural network
   - Display training progress and results
   - Plot accuracy and loss curves

## Applications

This classification system can assist ophthalmologists in:
- Early detection of cataract maturity
- Treatment planning decisions
- Automated screening processes
- Medical education and training

## Future Improvements

- Data augmentation to improve generalization
- Transfer learning with pre-trained models
- Hyperparameter tuning for optimal performance
- Integration with clinical imaging systems


*Note: This project is for educational/research purposes. Always consult medical professionals for clinical diagnosis.*

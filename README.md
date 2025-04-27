# **Image Classification using CNN in PyTorch**

This repository contains a Jupyter Notebook demonstrating image classification using a Convolutional Neural Network (CNN) implemented in PyTorch. The project trains a CNN model to classify images from the CIFAR-10 dataset.

## **Table of Contents**

* [Project Description](#bookmark=id.qmdp0cwdf5p4)  
* [Dataset](#bookmark=id.fp9zr1lj0dc7)  
* [Model Architecture](#bookmark=id.jh1mjbb9l36g)  
* [Getting Started](#bookmark=id.qo7i8f63vrfr)  
  * [Prerequisites](#bookmark=id.kp3m0ij87sqa)  
  * [Installation](#bookmark=id.qw680p7c4pzh)  
  * [Running the Code](#bookmark=id.irls20njovf5)  
* [Training Process](#bookmark=id.zfvjldn8ypmz)  
* [Results](#bookmark=id.cibcisxnsyz4)  
* [Future Improvements](#bookmark=id.reszkntwskar)  
* [License](#bookmark=id.26jmjani9q0a)

## **Project Description**

This project implements a simple Convolutional Neural Network (CNN) in PyTorch for the task of image classification. The goal is to build and train a model capable of accurately classifying images into one of the 10 categories present in the CIFAR-10 dataset. The notebook walks through the steps of data loading, preprocessing, defining the model architecture, training, and evaluating the model.

## **Dataset**

The project uses the **CIFAR-10 dataset**.

* It consists of 60,000 32x32 colour images in 10 classes, with 6,000 images per class.  
* There are 50,000 training images and 10,000 test images.  
* The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

The dataset is automatically downloaded using torchvision.datasets.CIFAR10.

## **Model Architecture**

The CNN model is defined using torch.nn.Module. The architecture includes:

* Convolutional layers (nn.Conv2d) to extract features from the images.  
* Max Pooling layers (nn.MaxPool2d) to reduce spatial dimensions.  
* Fully connected layers (nn.Linear) for classification.  
* ReLU activation functions (F.relu) after convolutional and some linear layers.  
* Dropout (nn.Dropout) for regularization.

The specific layers are:

1. conv1: 3 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1\.  
2. conv2: 16 input channels, 32 output channels, 3x3 kernel, stride 1, padding 1\.  
3. pool: 2x2 max pooling with stride 2\.  
4. fc1: Linear layer mapping from the flattened output of convolutional layers to 512 nodes. The input size 32 \* 8 \* 8 is based on the image dimensions after pooling.  
5. fc2: Linear layer mapping from 512 nodes to the number of output classes (10 for CIFAR-10).

The model utilizes a forward method to define the data flow through the layers.

## **Getting Started**

These instructions will help you set up the project and run the code on your local machine or a cloud environment like Google Colab.

### **Prerequisites**

* Python 3.x  
* PyTorch  
* torchvision  
* numpy  
* matplotlib

### **Installation**

You can install the necessary libraries using pip:

pip install torch torchvision numpy matplotlib

If you are using Google Colab, most of these libraries are pre-installed.

### **Running the Code**

1. Clone the repository (if applicable) or download the Image\_Classification\_using\_CNN\_in\_PyTorch.ipynb notebook.  
2. Open the notebook in a Jupyter environment (like Jupyter Notebook, JupyterLab, or Google Colab).  
3. Run the cells sequentially. The notebook will:  
   * Import necessary libraries.  
   * Define data transformations and parameters.  
   * Download and load the CIFAR-10 dataset.  
   * Create data loaders for training, validation, and testing.  
   * Define and initialize the CNN model.  
   * Move the model to GPU if available.  
   * Define the loss function (Cross Entropy Loss) and optimizer (Adam or SGD).  
   * Train the model.  
   * Save the trained model.

## **Training Process**

* **Loss Function:** nn.CrossEntropyLoss()  
* **Optimizer:** optim.Adam() or optim.SGD()  
* **Learning Rate:** 0.01 (for Adam) or a specified learning\_rate (for SGD)  
* **Batch Size:** 20  
* **Validation Split:** 20% of the training data is used for validation.  
* **Epochs:** The notebook trains for a specified number of epochs (n\_epochs).  
* **Early Stopping:** The model state dictionary is saved whenever the validation loss decreases, effectively saving the best model based on validation performance.

## **Results**

During training, the notebook prints the training and validation loss for each epoch. An example output snippet shows the validation loss decreasing over epochs:

Epoch: 1 	Training Loss: 1.491867 	Validation Loss: 0.312859  
Validation loss decreased (inf \--\> 0.312859). Saving model...  
Epoch: 2 	Training Loss: 1.153366 	Validation Loss: 0.270589  
Validation loss decreased (0.312859 \--\> 0.270589). Saving model...  
Epoch: 3 	Training Loss: 1.017966 	Validation Loss: 0.243675  
Validation loss decreased (0.270589 \--\> 0.243675). Saving model...

The final performance on the test set can be evaluated by adding code to the notebook after the training loop.

## **Future Improvements**

* Implement a testing loop to evaluate the model's performance on the unseen test set.  
* Add visualizations for training and validation loss over epochs.  
* Experiment with different CNN architectures.  
* Tune hyperparameters (learning rate, batch size, optimizer, dropout rate).  
* Implement data augmentation techniques to improve generalization.  
* Explore transfer learning using pre-trained models.

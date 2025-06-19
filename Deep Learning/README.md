# CIFAR-10 Image Classification with TensorFlow/Keras

This project demonstrates how to build and train a Convolutional Neural Network (CNN)
for image classification using the CIFAR-10 dataset. The model is implemented
using TensorFlow and Keras.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes,
with 6,000 images per class. There are 50,000 training images and 10,000
test images.

The classes are:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Requirements

The necessary Python packages for this project are listed in the
`requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```
It's recommended to use a virtual environment.

## Usage

To train and evaluate the model, run the `cifar10_classifier.py` script:

```bash
python cifar10_classifier.py
```

## Script Overview (`cifar10_classifier.py`)

The script performs the following steps:
1.  **Loads Data**: Loads the CIFAR-10 dataset from `tensorflow.keras.datasets`.
2.  **Preprocesses Data**:
    *   Normalizes image pixel values to the range [0, 1].
    *   One-hot encodes the labels.
3.  **Defines Model**: Constructs a CNN model with convolutional, max-pooling, and dense layers.
4.  **Compiles Model**: Configures the model for training with the Adam optimizer and categorical cross-entropy loss.
5.  **Trains Model**: Trains the model on the training dataset and validates on the test set.
6.  **Evaluates Model**: Calculates and prints the loss and accuracy on the test set.

## Expected Output

Running the script will output the training progress (loss and accuracy for each epoch)
and finally, the test loss and test accuracy achieved by the trained model on the
CIFAR-10 test set. The exact accuracy may vary slightly due to the stochastic
nature of the training process.

Example output:
```
...
Epoch 15/15
...
Test loss: [some_value]
Test accuracy: [some_value_between_0_and_1]
```

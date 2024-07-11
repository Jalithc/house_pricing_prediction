# House Price Prediction Model

This repository contains a simple TensorFlow model to predict house prices based on the number of bedrooms. The model uses a single dense layer and is trained using mean squared error as the loss function and stochastic gradient descent as the optimizer.

## Model Overview

The `house_model` function defines and trains a neural network model using TensorFlow and Keras. The model aims to predict house prices given the number of bedrooms.

### Input and Output

- **Input (`xs`)**: An array representing the number of bedrooms (0 to 4).
- **Output (`ys`)**: An array representing the corresponding house prices (in some arbitrary units).

### Model Architecture

The model consists of:
- **1 Dense Layer**: A dense layer with 1 unit and an input shape of 1.

### Compilation

The model is compiled with:
- **Optimizer**: Stochastic Gradient Descent (`sgd`)
- **Loss Function**: Mean Squared Error (`mean_squared_error`)

### Training

The model is trained for 500 epochs on the provided input and output data.

## Usage

To use the model, simply run the script. It will define, compile, and train the model, and then make a prediction for a house with 7 bedrooms.

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras

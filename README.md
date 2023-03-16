
# Monkey vs Gorilla Neural Network Classification Model
This is a binary classification machine learning model, developed without the use of machine learning frameworks, that implements a neural network model that uses a series of interconnected layers to extract features from an input image and predict whether a given 256 x 256 pixel image is a monkey or a gorilla.

The input image is preprocessed by flattening it into a column vector and normalizing each pixel value by dividing them by 255. The model then passes the input through a series of hidden layers, each consisting of interconnected neurons, with each neuron performing a weighted sum of its inputs and passing the result through the rectified linear unit (ReLU) activation function.

The output layer consists of a single neuron with a sigmoid activation function. The model calculates the predicted value (yhat) for the input image by passing the weighted sum of the input through the sigmoid activation function. If yhat is greater than 0.5, the model predicts the input image to be a monkey. Otherwise, it predicts the input image to be a gorilla.

## Usage
It is recommended to run the model in a virtual environment avoid conflicts with other projects that may be using different versions of the same libraries.<br/><br/>
The learning rate, number of iterations, number of layers, and number of neurons in each layer can be adjusted in the model's hyperparameters to fine-tune the model's performance and achieve better test accuracy results.<br/><br/>
The model can be used for custom images by adding an image to the same directory and changing the "test_image" string to match the filename of the custom image.

## Example
###### Test accuracy: ~85.6% (rounded to the nearest hundredth) <br/>Hyperparameters: iterations = 2500, learning rate = 0.002, layer dimensions = [196608, 20, 7, 5, 1]
<img src="https://i.imgur.com/feGzzli.png" alt="gorilla prediction" width="400"/><img src="https://i.imgur.com/qwbskxH.png" alt="monkey prediction" width="400"/><img src="https://i.imgur.com/OPt62dM.png" alt="cost function curve" width="400"/>



## Install dependencies
```
pip install numpy
pip install pillow
pip install matplotlib
```

## Author
Nicholas Kann / [@Butter-My-Toast](https://github.com/Butter-My-Toast "Butter-My-Toast's github page")


## Credits
#### This project uses the following datasets:
- [Animal Species Classification - V3](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset) (Kaggle)
#### This project uses the following libraries:
- [numpy](https://github.com/numpy/numpy) (BSD-3-Clause License) - [License](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
- [Pillow](https://github.com/python-pillow/Pillow) (HPND License) - [License](https://github.com/python-pillow/Pillow/blob/main/LICENSE)
- [matplotlib](https://github.com/matplotlib/matplotlib) (MDT License) - [License](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE)

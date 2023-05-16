import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    classes = ["gorilla", "monkey"]
    # layers_dims = [12288, 20, 7, 5, 1]  # 4 layer model

    # y = 1 for "monkey"
    # y = 0 for "gorilla"
    train_x_orig, train_y_orig = load_dataset(os.path.join(path, "MonkeyVsGorillaDataset"), "train", classes)

    test_x_orig, test_y_orig = load_dataset(os.path.join(path, "MonkeyVsGorillaDataset"), "test", classes)

    # Shuffling the sets so that the model doesn't learn in a certain order and bias towards gorilla
    train_x_shuffle, train_y_shuffle = shuffle_set(train_x_orig, train_y_orig)
    test_x_shuffle, test_y_shuffle = shuffle_set(test_x_orig, test_y_orig)

    # Reshaping the label vectors to make each example be represented as a column vector
    # (m,) will be reshaped to (1, m)
    # This is done to ensure that the matrix operations between the input and label matrices are performed correctly
    train_y_shuffle = train_y_shuffle.reshape(1, train_y_shuffle.shape[0])
    test_y_shuffle = test_y_shuffle.reshape(1, test_y_shuffle.shape[0])

    # Flattens and transposes the input sets so that each column vector is a different example of the flattened image
    train_x_flatten = train_x_shuffle.reshape(train_x_shuffle.shape[0], -1).T
    test_x_flatten = test_x_shuffle.reshape(test_x_shuffle.shape[0], -1).T

    # Preprocessing the dataset

    # Standardizing the dataset
    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255

    parameters_list = []
    accuracy_list = []
    hyperparameter_list = []
    
    # Creating the model

    layer_dims = [196608, 20, 7, 5, 1]
    iterations = 2500
    learning_rate = 0.002

    parameters, cost = model(train_x, train_y_shuffle, layer_dims, learning_rate=learning_rate,
                             num_iterations=iterations, print_cost=True)
    _, train_accuracy = predict(train_x, train_y_shuffle, parameters, print_accuracy=True)
    _, test_accuracy = predict(test_x, test_y_shuffle, parameters, print_accuracy=True)


    # Creating a list of indexes of test examples to show
    indexes = [0, 1]
    
    show_prediction_example(model, test_x, indexes)
    
    # Plotting the cost function every hundred iterations
    costs = np.squeeze(cost)
    plt.plot(costs)
    plt.ylabel("Cost")
    plt.xlabel("Iterations (hundreds)")
    plt.title("Learning rate = " + str(learning_rate)
    plt.show()
    
    # Add your own test image to the same directory and change the test image to the file name
    # to see the model"s prediction of your image
    test_image = "none"
    test_true_label = [-1] # the true class of the test image (1 for monkey, 0 for gorilla)
    
    if test_image != "none":
        fname = os.path.join(path, test_image)
        image = np.array(Image.open(fname).resize((256, 256)))
        plt.imshow(image)
        image = image.reshape((256 * 256 * 3, 1))
        image = image / 255
        y_prediction = int(np.squeeze(predict(image, test_true_label, parameters)))
        class_prediction = "\"monkey\"" if y_prediction == 1 else "\"gorilla\""
        plt.title(f"y = {y_prediction}, the model predicted that it is a {class_prediction} picture.")
        plt.axis("off")
        plt.show()


def load_dataset(path, dataset_split, classes):
    x = []
    y = []
    # 0 for gorilla
    # 1 for monkey
    for label, cls in enumerate(classes):
        class_path = os.path.join(path, dataset_split, cls)
        for img_path in os.listdir(class_path):
            img = Image.open(os.path.join(class_path, img_path)).resize((256, 256))
            x.append(np.array(img))
            y.append(label)

    return np.array(x), np.array(y)


def shuffle_set(set_x, set_y):
    n_samples = set_x.shape[0]
    permutation = np.random.permutation(n_samples)
    return set_x[permutation], set_y[permutation]


def predict(X, Y, parameters, print_accuracy=True):
    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    probas, caches = forward_propagation(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    accuracy = np.sum((p == Y) / m)

    if print_accuracy:
        print("Accuracy: " + str(accuracy))

    return p, accuracy


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1. / m) * np.dot(dZ, A_prev.T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def initialize_parameters(layers_dims):
    parameters = {}
    for layer in range(1, len(layers_dims)):
        parameters["W" + str(layer)] = np.random.randn(layers_dims[layer], layers_dims[layer - 1]) * 0.01
        parameters["b" + str(layer)] = np.zeros((layers_dims[layer], 1))
    return parameters


def compute_activation(A_prev, W, b, activation):
    if activation == "relu":
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def forward_propagation(X, parameters):
    caches = []
    A = X
    layers = len(parameters) // 2
    for layer in range(1, layers):
        A_prev = A
        A, cache = compute_activation(A_prev, parameters["W" + str(layer)], parameters["b" + str(layer)], "relu")
        caches.append(cache)

    AL, cache = compute_activation(A, parameters["W" + str(layers)], parameters["b" + str(layers)], "sigmoid")
    caches.append(cache)

    return AL, caches


def compute_linear_derivatives(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    grads = {}
    layers = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[layers - 1]

    dA_prev_temp, dW_temp, db_temp = compute_linear_derivatives(dAL, current_cache, "sigmoid")

    grads["dA" + str(layers - 1)] = dA_prev_temp
    grads["dW" + str(layers)] = dW_temp
    grads["db" + str(layers)] = db_temp

    for layer in reversed(range(layers - 1)):
        current_cache = caches[layer]
        dA_prev_temp, dW_temp, db_temp = compute_linear_derivatives(grads["dA" + str(layer + 1)], current_cache, "relu")
        grads["dA" + str(layer)] = dA_prev_temp
        grads["dW" + str(layer + 1)] = dW_temp
        grads["db" + str(layer + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    layers = len(parameters) // 2

    for layer in range(layers):
        parameters["W" + str(layer + 1)] = parameters["W" + str(layer + 1)] - learning_rate * grads[
            "dW" + str(layer + 1)]
        parameters["b" + str(layer + 1)] = parameters["b" + str(layer + 1)] - learning_rate * grads[
            "db" + str(layer + 1)]

    return parameters


def model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=True):
    costs = []
    parameters = initialize_parameters(layers_dims)

    # gradient descent
    for i in range(num_iterations):
        # forward propagation
        AL, caches = forward_propagation(X, parameters)

        # cost computation
        cost = (1. / Y.shape[1]) * (-np.dot(Y, np.log(AL).T) - np.dot((1 - Y), np.log(1 - AL).T))
        cost = np.squeeze(cost)

        # backward propagation
        grads = backward_propagation(AL, Y, caches)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs


def show_prediction_example(model, test_x, index):
    for i in index:
        plt.imshow(test_x[:, i].reshape((256, 256, 3)))
        y_prediction = int(model["Y_prediction_test"][0, i])
        class_prediction = "\"monkey\"" if y_prediction == 1 else "\"gorilla""
        plt.title(f"y = {y_prediction}, the model predicted that it is a {class_prediction} picture.")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()

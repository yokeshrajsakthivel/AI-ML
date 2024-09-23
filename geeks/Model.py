import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    # Unroll nn_params back into Theta1 and Theta2
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))

    # Forward propagation
    m = X.shape[0]
    X = np.hstack([np.ones((m, 1)), X])  # Add bias unit to input layer
    
    a1 = X
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones((a2.shape[0], 1)), a2])  # Add bias unit to hidden layer
    
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)  # Output layer (hypothesis h_theta)

    # Recoding y labels into a matrix of binary values (one-hot encoding)
    y_vect = np.eye(num_labels)[y.astype(int)]

    # Cost function
    J = (-1 / m) * np.sum(np.sum(y_vect * np.log(a3) + (1 - y_vect) * np.log(1 - a3))) \
        + (lamb / (2 * m)) * (np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:])))

    # Backpropagation
    Delta3 = a3 - y_vect
    Delta2 = np.dot(Delta3, Theta2[:, 1:]) * sigmoid_gradient(z2)

    # Gradient calculation
    Theta1_grad = (1 / m) * np.dot(Delta2.T, a1)
    Theta2_grad = (1 / m) * np.dot(Delta3.T, a2)

    # Regularization for the gradients (excluding bias terms)
    Theta1_grad[:, 1:] += (lamb / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lamb / m) * Theta2[:, 1:]

    # Unroll gradients
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad

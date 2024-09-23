from scipy.io import loadmat
import numpy as np
import sys

# Append the directory to sys.path, not specific files
sys.path.append('D:\\AIML\\geeks')

# Corrected imports based on file names and functions
from Model import neural_network
from RandInitialise import initialise
from Prediction import predict
from scipy.optimize import minimize

# Loading the .mat file
# data = loadmat('mnist-original.mat')
data = loadmat('D:\\AIML\\geeks\\mnist-original.mat')


# Extracting and normalizing features
X = data['data'].T / 255  # Transpose and normalize the data

# Extracting labels
y = data['label'].flatten()

# Splitting data into training and test sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Neural network structure
input_layer_size = 784
hidden_layer_size = 100
num_labels = 10

# Random initialization of weights (Thetas)
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)

# Unrolling parameters into a single column vector
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))

# Optimization settings
maxiter = 100
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)

# Training the neural network using minimize
results = minimize(neural_network, x0=initial_nn_params, args=myargs, 
                   options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True)

# Extract trained weights from the result
nn_params = results["x"]
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], 
                    (hidden_layer_size, input_layer_size + 1))
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
                    (num_labels, hidden_layer_size + 1))

# Prediction accuracy on the test set
pred_test = predict(Theta1, Theta2, X_test)
print('Test Set Accuracy: {:f}'.format(np.mean(pred_test == y_test) * 100))

# Prediction accuracy on the training set
pred_train = predict(Theta1, Theta2, X_train)
print('Training Set Accuracy: {:f}'.format(np.mean(pred_train == y_train) * 100))

# Precision evaluation
true_positive = np.sum(pred_train == y_train)
false_positive = len(y_train) - true_positive
precision = true_positive / (true_positive + false_positive)
print('Precision =', precision)

# Saving trained Thetas to text files
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')

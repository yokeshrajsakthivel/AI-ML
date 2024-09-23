import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    
    # Adding bias unit to the input layer (a1)
    X = np.hstack([np.ones((m, 1)), X])  
    
    # Layer 2 (hidden layer) activations
    z2 = np.dot(X, Theta1.T)
    a2 = sigmoid(z2)
    
    # Adding bias unit to the hidden layer
    a2 = np.hstack([np.ones((a2.shape[0], 1)), a2])  
    
    # Layer 3 (output layer) activations
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)  # This is the hypothesis output

    # Predicting the class with the highest probability
    p = np.argmax(a3, axis=1)
    
    return p

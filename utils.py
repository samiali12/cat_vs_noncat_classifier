import numpy as np # for numerical computation
import matplotlib.pyplot as plt # for data visualization
import h5py

def load_data():
    
    train_data = h5py.File("datasets/train.h5", "r") # read h5py file
    train_x = np.array(train_data["train_set_x"][:]) # extract training examples
    train_y = np.array(train_data["train_set_y"][:]) # extract training labels
    
    test_data = h5py.File("datasets/test.h5", "r") # read h5py file
    test_x = np.array(test_data["test_set_x"][:])
    test_y = np.array(test_data["test_set_y"][:])
    
    classes = np.array(test_data["list_classes"][:]) # test_data classes ["cat", "not-cat"]
    
    train_y = train_y.reshape((1, train_y.shape[0])) # reshape into vector
    test_y = test_y.reshape((1, test_y.shape[0])) # reshape into vector
    
    return train_x, train_y, test_x, test_y, classes


def initialization_parameters(layer_dims):
    
    # layer_dims is vector of size (n,1)
    
    parameters = {} # store weights and bias
    
    np.random.seed(1)
    L = len(layer_dims)
    
    for l in range(1,L):
        
        # initialize weight for layers
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        
        # initialize bias for layers
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters



# GRADED FUNCTION: initialize_velocity

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(1, L + 1):
        # (approx. 2 lines)
        # v["dW" + str(l)] =
        # v["db" + str(l)] =
        # YOUR CODE STARTS HERE
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        
        # YOUR CODE ENDS HERE
        
    return v



    ''' implementation of forward propogation '''

# rectified linear activation function for hidden layers
def relu(z):   
    A = np.maximum(0,z)
    cache = z
    return A, cache

# sigmoid activation function
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

# implementation of forward propogation
def linear_forward(A, W, b):
    Z = W.dot(A) + b # multiply input with weight matrix
    cache = (A, W, b) 
    assert(Z.shape == (W.shape[0], A.shape[1]))
    return Z, cache

# implementation of forward propogation
def linear_forward_activation(A_prev, W,  b, activation):
    
    '''
    A_prev -- input of previous layer
    W --- is weights matrix of shape (hiddens units, size of prev_layer)
    b --- is bias vector of size (m,1)
    activation is "sigmoid or relu"
    '''
    
    if activation == "relu" or activation == "RELU":
        z, cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(z)
        
    elif activation == "sigmoid":
        z, cache = linear_forward(A_prev, W, b)
        A,activation_cache = sigmoid(z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache_ = (cache, activation_cache)
    
    return A, cache_

def forward_layer(X, parameters):
    
    '''this function take X input and parameters
    iterate and forward input into next layer '''
    
    caches = [] # keep track of cost
    A = X # make copy of input
    L = len(parameters)//2 # length of layers

    for l in range(1,L):
        
        A_prev = A # output of previous layer
        A, cache = linear_forward_activation(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation="relu")
        caches.append(cache)
        
    AL, cache = linear_forward_activation(A, parameters["W" + str(L)], parameters["b" + str(L)],activation="sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    
    return AL,caches

def compute_cost(AL, Y, parameters, lambd):
    
    m = Y.shape[1]
    L = len(parameters)//2
    parameters_sum = 0
    
    for l in range(0,L):
        parameters_sum += np.sum(np.square(parameters["W" + str(l+1)]))
        
    L2 = lambd * parameters_sum / (2*m)
    
    # Compute loss from aL and y.
    cost = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost += L2
    S
    total_cost = np.sum(cost)

    
    return total_cost


''' implementation of backward propogation '''
# derivative of sigmoid 
def d_sigmoid(z):
    d_sig = (1 + np.exp(-z)) ** -1
    return d_sig

def linear_backward(dZ, cache, lambd):
    
    A_prev, W, b = cache
    
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T) + (lambd * W) / m
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev,dW, db, 

# relu backward 
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

# sigmoid backward
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def linear_backward_activation(dA, cache, activation, lambd = 0.7):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
        
    return dA_prev, dW, db


## let's do backward propogation
def backward_layer(AL, Y, cache, lambd):
    
    grads = {} # store derivatives of weights
    L = len(cache) # number of layers
    m = AL.shape[1] # number of training examples
    Y = Y.reshape(AL.shape) 
    
    # initialize dervative 
    dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
    
    current_cache = cache[L-1]
    
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward_activation(dAL, current_cache,
                                                                                             activation = "sigmoid",
                                                                                            lambd = 0.7)
    
    for l in reversed(range(L-1)):
        current_cache = cache[l]
        dA_prev_temp, dW_temp, db_temp = linear_backward_activation(grads["dA" + str(l + 1)], 
                                                                    current_cache, activation = "relu",
                                                                    lambd = 0.7)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads


# GRADED FUNCTION: update_parameters_with_momentum

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(1, L + 1):
        
        # (approx. 4 lines)
        # compute velocities
        # v["dW" + str(l)] = ...
        # v["db" + str(l)] = ...
        # update parameters
        # parameters["W" + str(l)] = ...
        # parameters["b" + str(l)] = ...
        # YOUR CODE STARTS HERE
        v["dW" + str(l)] = (beta*v["dW" + str(l)]) + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = (beta*v["db" + str(l)]) + (1 - beta) * grads["db" + str(l)]
        
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v["db" + str(l)]
        # YOUR CODE ENDS HERE
        
    return parameters, v

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2

    for l in range(0,L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X,Y, parameters):
    
    m = X.shape[1]
    n = len(parameters)
    p = np.zeros((1,m))
    
    probs, cache = forward_layer(X, parameters)
    
    for i in range(0,probs.shape[1]):
        
        if(probs[0][i] > 0.5):
            p[0][i] = 1
        else:
            p[0][i] = 0
            
    print("Accuracy  = " + str(np.sum(((p == Y)/m)))) # calculate the accuracy of model
    
    return p
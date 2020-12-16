import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Linear_Regression(object):
    """ Linear Regression object
    """
    
    
    
    
    
    def __init__(self,alpha):
    """ Function initializing a linear regression object
    
    Args:
        alpha : learning rate
        
    Returns:
        Linear regression model
    """
        self.alpha = alpha
        self.theta = None
        self.history_loss = []
        self.history_J= []
        
    
    
    
    
    
    
    def add_ones(self,X):
    """ From a matrix of features, adds a column of ones corresponding to the intercept term.
    
    Args:
        X (numpy array of floats) : array of features
        
    Returns:
        X with a column of ones
        
    """
        return np.c_[np.ones(len(X)),X]
        
        
        
        
        
        
        
        
    def fit(self,X,Y,nb_epochs=10_000):
    """ Fits the model using data from X and Y.
    
    Args:
        X (numpy array of floats) : array of features
        Y (vector of floats) : vector of targets
        nb_epochs : Number of iterations, default value is 10_000
        
    Returns:
        Fitted model
        
    """
        X = self.add_ones(X)
        self.theta = self.initialize_theta(X)
        for epoch in tqdm(range(nb_epochs)):
            cost = self.get_cost(X,Y,self.theta)
            self.history_loss.append(cost)
            self.gradient_descent(X, Y)
        
        
        
        
        
        
        

    def initialize_theta(self, X):
    """ Function Initializing the model's parameters.
    
    Args:
        X (numpy array of floats) : array of features
        
    Returns:
        Vector theta of size = nb_features
        
    """
        return np.zeros(X.shape[1])
        
        
        
        
        
        
        
        
        
    def get_cost(self, X,Y,theta):
    """ Calculates the cost of the model.
    
    Args:
        X (numpy array of floats) : array of features
        Y (vector of floats) : vector of targets
        theta (vector of floats) : model's parameters
    Returns:
        cost (float) : cost of the model
        
    """
        return np.linalg.norm(np.dot(X,theta)-Y)
        
        
        
        
        
        
        
        
    def gradient_descent(self, X, Y):
    """ Performs gradient descent for a given iteration. Updates vector theta.
    
    Args:
        X (numpy array of floats) : array of features
        Y (vector of floats) : vector of targets
    Returns:
        None
        
    """
        temp = []
        for j in range(len(self.theta)):
            self.theta[j] -= self.alpha * np.sum(np.multiply(np.dot(X,self.theta)-Y,X[:,j]))/(2*len(X))
            temp.append(self.theta[j])
        self.history_J.append(temp)
        return








    def predict(self, X_test):
    """ Predicts using the trained model.
    
    Args:
        X_test (numpy array of floats) : array of features
    Returns:
        y_pred (vector of floats) : predictions for X_test
        
    """
        return np.dot(self.add_ones(X_test),self.theta)
        
        
        
        
        
        
        
        
        
        
    def contour_map(self, X, Y):
    """ Calculates the cost evolution using a mesh grid for gradient descent vizualisation.
    
    Args:
        X (numpy array of floats) : array of features
        Y (vector of floats) : vector of targets
    Returns:
        theta_0 (vector of floats) : Vector of values for the first parameter
        theta_1 (vector of floats) : Vector of values for the second parameter
        J_values (numpy 2D-array of floats) : Array of loss values for each pair of theta_0 and theta_1 values.
        
    """
        theta_0 = np.linspace(-10,10,100)
        theta_1 = np.linspace(-1,4,100)
        J_values = np.zeros((len(theta_0),len(theta_1)))
        for i in tqdm(range(J_values.shape[0])):
            for j in range(J_values.shape[1]):
                theta = np.array([theta_0[i],theta_1[j]])
                t = self.get_cost(self.add_ones(X),Y,theta)
                J_values[i,j]=t
        return theta_0, theta_1, J_values




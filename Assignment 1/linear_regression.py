import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Linear_Regression(object):

    def __init__(self,alpha):
        self.alpha = alpha
        self.theta = None
        self.history_loss = []
        self.history_J= []
        
    
    def add_ones(self,X):
        return np.c_[np.ones(len(X)),X]
        
        
    def fit(self,X,Y,nb_epochs=10_000):
        X = self.add_ones(X)
        self.theta = self.initialize_theta(X)
        for epoch in tqdm(range(nb_epochs)):
            cost = self.get_cost(X,Y,self.theta)
            self.history_loss.append(cost)
            self.gradient_descent(X, Y, cost)
        

    def initialize_theta(self, X):
        return np.zeros(X.shape[1])
        
        
    def get_cost(self, X,Y,theta):
        return np.linalg.norm(np.dot(X,theta)-Y)
        
        
    def gradient_descent(self, X, Y, cost):
        temp = []
        for j in range(len(self.theta)):
            self.theta[j] -= self.alpha * np.sum(np.multiply(np.dot(X,self.theta)-Y,X[:,j]))/(2*len(X))
            temp.append(self.theta[j])
        self.history_J.append(temp)
        return


    def predict(self, X_test):
        return np.dot(self.add_ones(X_test),self.theta)
        
        
    def contour_map(self, X, Y):
        theta_0 = np.linspace(-10,10,100)
        theta_1 = np.linspace(-1,4,100)
        J_values = np.zeros((len(theta_0),len(theta_1)))
        for i in tqdm(range(J_values.shape[0])):
            for j in range(J_values.shape[1]):
                theta = np.array([theta_0[i],theta_1[j]])
                t = self.get_cost(self.add_ones(X),Y,theta)
                J_values[i,j]=t
        return theta_0, theta_1, J_values

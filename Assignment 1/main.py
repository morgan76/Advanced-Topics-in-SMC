import click
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from linear_regression import *

@click.command()
@click.argument("training_data", type=str)
@click.option("--plot-data",type=bool,default=True)
@click.option("--n-epochs", type=int, default=1500)
@click.option("--learning-rate", type=float, default=0.01)
@click.option("--stratify", type=bool, default=False)

def main(
    training_data,
    plot_data,
    n_epochs,
    learning_rate,
    stratify
):
    
    # loading the data
    data = pd.read_csv('data/'+str(training_data)+'.csv')
    
    
######################## Plot of the data ########################################

    if plot_data:
    
        # Boxplot
        plt.figure()
        plt.title('Boxplot of the data')
        plt.boxplot(data['population'])
        plt.title('Distribution of the population')
        plt.show(block=False)
        
        # Scatter plot
        plt.figure()
        plt.title('Scatter plot of the data')
        plt.scatter(data['population'],data['profit'])
        plt.xlabel('Population')
        plt.ylabel('Profit')
        plt.title('Scatter Plot of the data')
        plt.show(block=False)
        
######################### Data preprocessing #####################################
    

    if stratify:
        
        # Discards outliers (profit >= 10 and population <=10)
        # Distributes equally instances between train and test set based on the profit value
        
        data['decile'] = pd.qcut(data['profit'], 5, labels=np.arange(5, 0, -1))


        indexNames = data[(data['profit'] >= 10) & (data['population'] <= 10)].index
        data.drop(indexNames , inplace=True)


        X_train, X_test, Y_train, Y_test = train_test_split(data['population'], data['profit'], test_size=0.4, random_state=0,stratify=data['decile'])

    else:
    
        X_train, X_test, Y_train, Y_test = train_test_split(data['population'], data['profit'], test_size=0.4, random_state=0)
    
    
    
    X_train = np.array(X_train).reshape(-1,1)
    X_test = np.array(X_test).reshape(-1,1)
    

######################## Models training #######################################
    
    # Initializes regression models
    Regression_scikit = LinearRegression()
    Regression = Linear_Regression(n_epochs, learning_rate)
    
    # Fits regression models
    Regression_scikit.fit(X_train,Y_train)
    Regression.fit(X_train,Y_train)
    
    
    
    print('Training set :')
    
    
    print('Scikit Learn Regressor :')
    print('Mean squared error:',mean_squared_error(Y_train, X_train*Regression_scikit.coef_+Regression_scikit.intercept_))
    print('Coefficient of determination:',r2_score(Y_train, X_train*Regression_scikit.coef_+Regression_scikit.intercept_))

    print('Our Regressor :')
    print('Mean squared error:',mean_squared_error(Y_train, X_train*Regression.theta[1]+Regression.theta[0]))
    print('Coefficient of determination:',r2_score(Y_train, X_train*Regression.theta[1]+Regression.theta[0]))
    
    print('##################################################')
    
   
######################## Models testing #######################################
    
    print('Test set :')

    # Calculates prediction with both regressors
    Y_pred_scikit = Regression_scikit.predict(X_test)
    Y_pred = Regression.predict(X_test)

######################## Results plotting #####################################
    
    # Displays MSE and R2 coefficient for scikit learn's regression model
    print('Scikit Learn Regressor results:')
    print('Mean squared error:',mean_squared_error(Y_test, Y_pred_scikit))
    print('Coefficient of determination:',r2_score(Y_test, Y_pred_scikit))

    # Displays MSE and R2 coefficient for our regression model
    print('Our Regressor results:')
    print('Mean squared error:',mean_squared_error(Y_test, Y_pred))
    print('Coefficient of determination:',r2_score(Y_test, Y_pred))

    # Plots both regression functions
    plt.figure()
    plt.scatter(X_test, Y_test,  color='black')
    plt.plot(X_test, Y_pred, color='blue', linewidth=1)
    plt.plot(X_test, Y_pred_scikit, color='red', linewidth=1)
    plt.xlabel('Population')
    plt.ylabel('Profit')
    plt.title('Scatter Plot of the data')
    plt.legend(['Our regressor','Scikit-learn regressor'])
    plt.show(block=False)


    # Plots regression parameters contour map
    theta_0, theta_1, J_values = Regression.contour_map(X_train, Y_train)

    plt.figure()
    plt.title('Gradient region')
    plt.contourf(theta_0, theta_1, J_values, 20, cmap='RdGy');
    plt.colorbar();
    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.show(block=False)

    print('Parameters of our regression :')
    print('Parameter 0 :',Regression.theta[0])
    print('Parameter 1 :',Regression.theta[1])

    print('------------------------------------')
    
    print('Parameters of scikit-learn regression :')
    print('Parameter 0 :',Regression_scikit.intercept_)
    print('Parameter 1 :',Regression_scikit.coef_[0])

    plt.figure()
    plt.title('Loss')
    plt.plot(Regression.history_loss[6:])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

if __name__ == "__main__":
    main()

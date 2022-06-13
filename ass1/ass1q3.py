from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

iris = load_iris()
X, y = iris['data'], iris['target']

N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]

Xtrain = pd.DataFrame(Xtrain,columns = ['sepal_length','sepal_width','petal_length','petal_width'])
ytrain = pd.DataFrame(ytrain,columns = ['class'])
Xtest = pd.DataFrame(Xtest,columns = ['sepal_length','sepal_width','petal_length','petal_width'])
ytest = pd.DataFrame(ytest,columns = ['class'])

class NBC():
  
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def fit(self, X , y):
    
        def get_mean_var_by_class(X , y):
            '''
            get mean, variance for each class
            ''' 
            mean = X.groupby(y['class']).apply(np.mean).to_numpy()
            var = X.groupby(y['class']).apply(np.var).to_numpy()
            return mean, var
    
        def get_prior(X , y):
            '''
            get prior probabilities
            '''
            prior = (X.groupby(y['class']).apply(lambda x: len(x)) / len(X)).to_numpy()
            return prior
    
        self.mean_arr, self.var_arr = get_mean_var_by_class(X , y)
        self.prior = get_prior(X , y)
        return
        
        
        
    def predict(self, X):
        
        def gaussian_density(mean_arr, var_arr, class_id, x):     
            '''   
            (1/√2pi*σ) * exp((-1/2)*((x-μ)^2)/(2*σ²))
            '''
            mean = mean_arr[class_id]
            var = var_arr[class_id]
            up = np.exp((-1/2)*((x-mean)**2) / (2 * var))
            down = np.sqrt(2 * np.pi * var)
            prob = up / down
            return prob
    
    
        def get_posterior(self, X):
            yhat = []
            for x in X.to_numpy():
                posteriors = []
                for i in range(self.num_classes):
                    conditional = np.prod(gaussian_density(self.mean_arr, self.var_arr, i, x))
                    posterior = self.prior[i]*conditional
                    posteriors.append(posterior)
                yhat.append(np.argmax(posteriors))
            return yhat
    
        return get_posterior(self, X)
        
    
    
nbc = NBC(num_classes=3)
nbc.fit(Xtrain,ytrain)
yhat = nbc.predict(Xtest)
test_accuracy = np.mean(yhat == ytest['class'])
print(f'The test_accuracy is : {test_accuracy:.4f}')
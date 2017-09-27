import numpy as np

#Training Data:
trainX = np.array(([3, 5], [6, 2], [10, 2], [6, 1.5], [8, 5], [10, 0], [0, 5], [0, 0], [7.5, 4.5], [10, 1], [11, 0]), dtype=float)
trainY = np.array(([55], [60], [60], [57], [100], [35], [40], [0], [95], [40], [20]), dtype=float)

#Testing Data:
testX = np.array(([4, 5.5], [4.5, 1], [9, 2.5], [6, 2]), dtype=float)
testY = np.array(([70], [89], [85], [75]), dtype=float)

#Normalize
trainX = trainX/np.amax(trainX, axis=0)
trainY = trainY/100.

testX = testX / np.amax(trainX, axis=0)
testY = testY / 100.


class Neural_Network(object):
    def __init__(self):
        #Parameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLAyerSize = 3
        self.Lambda = 0.0001
        
        #Weights
        self.W1 = np.random.randn(self.inputLayerSize,                                   self.hiddenLAyerSize)
        self.W2 = np.random.randn(self.hiddenLAyerSize,                                   self.outputLayerSize)
    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))
        
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    
    def sigmoidPrime(self, z):
        #derivative of Sigmoid func
        return np.exp(-z)/((1 + np.exp(-z))**2)
    
    def costFunction(self, X, y):
        return 0.5*np.sum((y - self.forward(X))**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
    
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W1 and W2
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3) + self.Lambda*self.W2
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2) + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    def getParams(self):
        #Get W1 and W2 Rolled into vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
        
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector
        W1_start = 0
        W1_end = self.hiddenLAyerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],                             (self.inputLayerSize, self.hiddenLAyerSize))
        W2_end = W1_end + self.hiddenLAyerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end],                             (self.hiddenLAyerSize, self.outputLayerSize))
    
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


from scipy import optimize

#Modified trainer for testing
class trainer(object):
    def __init__(self, N):
        self.N = N
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad
    
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))
    
    def train(self, trainX, trainY, testX, testY):
        #Make internal variable for callback function
        self.X = trainX
        self.y = trainY
        
        self.testX = testX
        self.testY = testY
        
        #Make empty list to store costs
        self.J = []
        self.testJ = []
        
        params0 = self.N.getParams()
        
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0,                                 jac=True, method='BFGS', args=(trainX, trainY),                                options=options, callback=self.callbackF)
        
        self.N.setParams(_res.x)
        self.optimizationResults = _res

NN = Neural_Network()


T = trainer(NN)

T.train(trainX, trainY, testX, testY)

import matplotlib.pyplot as plt


#See the optimization plot
plt.plot(T.J)
plt.plot(T.testJ)
plt.grid(1)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()


NN.forward(trainX)


hoursSleep = np.linspace(0, 10, 100)
hoursStudy = np.linspace(0, 5, 100)

hoursSleepNorm = hoursSleep/10.
hoursStudyNorm = hoursStudy/5.

a, b = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()

allOutputs = NN.forward(allInputs)

#Contour Plot
yy = np.dot(hoursStudy.reshape(100, 1), np.ones((1, 100)))
xx = np.dot(hoursSleep.reshape(100, 1), np.ones((1, 100))).T

CS = plt.contour(xx, yy, 100*allOutputs.reshape(100, 100))
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('Hours Sleep')
plt.ylabel('Hours Study')
plt.show()
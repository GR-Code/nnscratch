import numpy as np 
import random
from mnist_load import load_mnist_data

class Network:
    def __init__(self, layerSizes):
        self.layerSizes = layerSizes
        self.totalLayers = len(layerSizes)
        self.biases = [np.random.randn(k,1) for k in self.layerSizes[1:]] 
        self.weights = [np.random.randn(k,j) for j,k in zip(self.layerSizes[:-1],self.layerSizes[1:])]

    def feedforward(self, a):
        for b,w in zip(self.biases,self.weights):
            a = sigm(w@a + b)
        return a

    def gradient_descent(self, trainingData, epochs, eta, batchSize, testData = None):
        trainingData = list(trainingData)
        if testData:
            testData = list(testData)

        for i in range(epochs):
            random.shuffle(trainingData)
            batches = [trainingData[i:i+batchSize] for i in range(0,len(trainingData),batchSize)]
            for batch in batches:
                self.update(batch,eta)
            
            if testData:
                print(f"Epoch {i+1} : {self.evaluate(testData)}/{len(testData)}")

            else:
                print(f"Epoch {i+1} complete!")


    def update(self, batch, eta):
        delCbTotal = [np.zeros(b.shape) for b in self.biases]
        delCwTotal = [np.zeros(w.shape) for w in self.weights]

        for x,y in batch:
            delCb, delCw = self.backpropogation(x,y)
            delCbTotal = [cb + cbT for cb,cbT in zip(delCb, delCbTotal)]
            delCwTotal = [cw + cwT for cw,cwT in zip(delCw, delCwTotal)]

        self.biases = [b - (eta/len(batch))*delb for b, delb in zip(self.biases, delCbTotal)]
        self.weights = [w - (eta/len(batch))*delw for w, delw in zip(self.weights, delCwTotal)]

    def backpropogation(self, x, y):
        zs = []
        a = x
        activations = [x]
        for b,w in zip(self.biases, self.weights):
            z = w@a + b
            zs.append(z)
            a = sigm(z)
            activations.append(a)

        delta = self.costd(activations[-1], y) * sigmd(zs[-1]) # Final layer error, BP1
        delCb = [np.zeros(b.shape) for b in self.biases]
        delCw = [np.zeros(w.shape) for w in self.weights]
        delCb[-1] = delta
        delCw[-1] = delta @ (activations[-2].T) 

        for L in range(2,self.totalLayers):
            delta = ((self.weights[-L+1].T) @
             delta) * sigmd(zs[-L])
            delCb[-L] = delta
            delCw[-L] = delta @ (activations[-L-1].T)

        return delCb, delCw

    def evaluate(self,tests):
        results = [(np.argmax(self.feedforward(x)),y) for x,y in tests]
        return sum(int(x==y) for x,y in results)

    def costd(self, output, y):
        return (output-y)               # Cx = 0.5 || a^L - y ||^2 , therefore \partial Cx / \partial a = (a^L - y)


def sigm(x):
        return 1.0/(1.0 + np.exp(-x))   # Can swap this out for scipy.special.expit to remove RuntimeWarning

def sigmd(x):
        return sigm(x)*(1-sigm(x))

if __name__ == "__main__":
    test = Network((784,100,10))
    trainData = load_mnist_data("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
    testData = load_mnist_data("data/t10k-images-idx3-ubyte.gz","data/t10k-labels-idx1-ubyte.gz", True)
    test.gradient_descent(trainData,30,3,10,testData)
    
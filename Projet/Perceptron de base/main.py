import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_data():
    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_data()

class Perceptron:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def feedforward(self, x):
        z = np.dot(x, self.weights)
        return self.sigmoid(z)
    
    def train(self, train_images, train_labels, epochs, batch_size, learning_rate):
        errors = []
        for i in range(epochs):
            print("Epoch", i+1)
            for j in range(0, len(train_images), batch_size):
                x = train_images[j:j+batch_size]
                y = np.zeros((batch_size, 10))
                y[np.arange(batch_size), train_labels[j:j+batch_size]] = 1
                y_pred = self.feedforward(x)
                error = y_pred - y
                d_weights = np.dot(x.T, error)
                self.weights -= learning_rate * d_weights
                errors.append(np.mean(np.abs(error)))
        plt.plot(errors)
        plt.xlabel("Batch iterations")
        plt.ylabel("Error")
        plt.show()
    
    def evaluate(self, test_images, test_labels):
        y_pred = np.argmax(self.feedforward(test_images), axis=1)
        accuracy = np.mean(y_pred == test_labels)
        print("Accuracy:", accuracy)
        cm = confusion_matrix(test_labels, y_pred)
        print("Confusion matrix:\n", cm)

perceptron = Perceptron(784, 10)
perceptron.train(train_images, train_labels, epochs=30, batch_size=10, learning_rate=0.1)
perceptron.evaluate(test_images, test_labels)

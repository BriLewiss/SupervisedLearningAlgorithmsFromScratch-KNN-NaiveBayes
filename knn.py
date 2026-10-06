import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix 
import seaborn as sns
import matplotlib.pyplot as plt

class KNN:
    def __init__(self):
        pass

    def euclidean_distance(self, row1, row2):
        # calculate the Euclidean distance between two data points
        return np.sqrt(np.sum((row1 - row2) ** 2))

    def find_neighbors(self, X_train, X_test, y_train, k):
        # find the k nearest neighbors of a test data point and return the most common label
        distances = [self.euclidean_distance(X_test, x) for x in X_train]

        # sort the distances from smallest to largest and get the indices of the k nearest neighbors
        k_nearest = np.argsort(distances)[:k]

        # get the corresponding labels of the k nearest neighbors
        k_nearest_label = [y_train[i] for i in k_nearest]

        # select the most common label among the k nearest neighbors
        most_common = Counter(k_nearest_label).most_common(1)[0][0]

        return most_common

    def predict(self, X_train, X_test, y_train, k):
        # make predictions for each test data point
        predictions = [self.find_neighbors(X_train, x, y_train, k) for x in X_test]
        return np.array(predictions)

    def accuracy(self, y_test, predictions):
        # calculate the accuracy of the predictions
        return np.sum(y_test == predictions) / len(y_test)

    def plot_confusion_matrix(self, y_test, predictions, k):
        # create a confusion matrix to visualize the performance of the classifier
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for k = {k}')
        plt.show()
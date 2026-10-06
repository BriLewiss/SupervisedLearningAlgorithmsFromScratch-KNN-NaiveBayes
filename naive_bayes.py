import math
from sklearn.metrics import confusion_matrix 
import seaborn as sns
import matplotlib.pyplot as plt


class NaiveBayes:
    def __init__(self):
        pass

    def mean(self, numbers):
        # calculate the mean of a list of numbers
        return sum(numbers) / len(numbers)

    def stdev(self, numbers):
        # calculate the standard deviation of a list of numbers
        avg = self.mean(numbers)
        variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)

    def gaussian_probability(self, x, mean, stdev):
        # calculate the probability of a value occuring under a Gaussian distribution
        epsilon = 1e-10
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
        return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * exponent

    def class_probabilities(self, summaries, test_row):
        # calculate the probability that a test row belongs to each class
        probabilities = {}
        for class_value, class_summaries in summaries.items():
            # initialize the probability for this class
            probabilities[class_value] = 1

            # go through each attribute and calculate the probability of the test row belonging to this class
            for i in range(len(class_summaries)):
                # get the mean and stdev for this attribute in this class
                mean, stdev = class_summaries[i]
                # get the value of this attribute in the test row
                x = test_row[i]
                # calculate the Gaussian probability for this feature and multiply it with the current probability for this class
                probabilities[class_value] *= self.gaussian_probability(x, mean, stdev)

        # return the calculated probabilities for each class
        return probabilities

    def summarize_dataset(self, dataset):
        # summarize the dataset by calculating the mean and standard deviation for each attribute
        summaries = [(self.mean(column), self.stdev(column)) for column in zip(*dataset)]
        return summaries

    def predict(self, summaries, test_row):
        # calculate the probability of the test row belonging to each class and return the class with the highest probability
        probabilities = self.class_probabilities(summaries, test_row)
        best_label = max(probabilities, key=probabilities.get)
        return best_label

    def get_predictions(self, summaries, test_set):
        # make a prediction for each row in the test set
        predictions = [self.predict(summaries, instance) for instance in test_set]
        return predictions

    def accuracy(self, y_test, predictions):
        # calculate the accuracy of the predictions by comparing them to the true labels
        correct = sum(1 for i in range(len(y_test)) if y_test[i] == predictions[i])
        return (correct / float(len(y_test))) * 100.0

    def separate_by_class(self, X_train, y_train):
        # separate the training data by class
        separated = {}

        # go through every training example and add it to the appropriate class in the separated dictionary
        for i in range(len(X_train)):
            # get the class label for this training example
            class_value = y_train[i]
            if class_value not in separated:
                separated[class_value] = []
            # add the training example to the appropriate class in the separated dictionary
            separated[class_value].append(X_train[i])
        return separated


    def summarize_by_class(self, X_train, y_train):
        # separate the training data by class and summarize each class
        separated = self.separate_by_class(X_train, y_train)
        summaries = {}

        # go through each class and summarize the data for that class
        for class_value, rows in separated.items():
            # calculate the mean and stdev for each attribute in the class and store it in the summaries dictionary
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    def plot_confusion_matrix(self, y_test, predictions):
        # create a confusion matrix to visualize the performance of the classifier
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
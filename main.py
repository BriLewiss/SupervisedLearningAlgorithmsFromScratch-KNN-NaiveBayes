from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np

from knn import KNN
from naive_bayes import NaiveBayes


class Assignment2:
    def __init__(self):
        self.knn = KNN()
        self.naive_bayes = NaiveBayes()

    def digits_data(self):
        # load the digits dataset
        digits = load_digits()
        X = digits.data
        y = digits.target

        # split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # normalize the data using MixMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def breast_cancer_data(self):
        # load the breast cancer dataset
        cancer = load_breast_cancer()
        X = cancer.data
        y = cancer.target

        # split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # normalize the data using MixMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def run_knn(self, X_train_scaled, X_test_scaled, y_train, y_test):
        # run k nearest neighbors
        for k in [1,3,5,7]:
            predictions = self.knn.predict(X_train_scaled, X_test_scaled, y_train, k)
            # print the accuracy and confusion matrix for each k
            accuracy = self.knn.accuracy(y_test, predictions)
            self.knn.plot_confusion_matrix(y_test, predictions, k)
            print(f'Accuracy for k={k}: {accuracy:.4f}')

    def run_naive_bayes(self, X_train_scaled, X_test_scaled, y_train, y_test):
        # summarize the training data by class
        summaries = self.naive_bayes.summarize_by_class(X_train_scaled, y_train)

        # make predictions on the test data
        predictions = self.naive_bayes.get_predictions(summaries, X_test_scaled)

        # calculate accuracy
        accuracy = self.naive_bayes.accuracy(y_test, predictions)
        print(f'Naive Bayes Accuracy: {accuracy:.4f}')

        # print confusion matrix
        self.naive_bayes.plot_confusion_matrix(y_test, predictions)


    def digits_main(self):
        X_train_scaled, X_test_scaled, y_train, y_test = self.digits_data()
        # run knn
        self.run_knn(X_train_scaled, X_test_scaled, y_train, y_test)

        # run naive bayes
        nb = self.run_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test)

    def breast_cancer_main(self):
        X_train_scaled, X_test_scaled, y_train, y_test = self.breast_cancer_data()
        # run knn
        self.run_knn(X_train_scaled, X_test_scaled, y_train, y_test)

        # run naive bayes
        self.run_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test)

        

if __name__ == "__main__":
    assignment = Assignment2()
    #assignment.digits_main()
    assignment.breast_cancer_main()
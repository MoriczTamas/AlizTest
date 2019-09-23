from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas


class custom_data():
    def __init__(self):
        self.target = []
        self.data = []

class ThresholdBinarizer(BaseEstimator, TransformerMixin):
    """Optimises the threshold according to the Gini impurity"""
    
    def __init__(self, starting_threshold = 0.5):
        super().__init__()
        self.threshold = starting_threshold
        
    def binarize(self, prediction):
        """Uses the calculated threshold to turn a probability into a class label prediction"""
        
        if prediction >= self.threshold:
            return 1
        return 0
        
    def calculate_gini_impurity(self, data, probabilities):
        """Takes the calculated class probabilities and true class labels and calculated the Gini impurity"""
        
        class_0_correct = 0
        class_1_incorrect = 0
        class_0_incorrect = 0
        class_1_correct = 0
        
        for index in range(0, len(probabilities)):
            prediction = self.binarize(probabilities[index])
            if prediction == 1:
                if data.target[index] == 1:
                    class_1_correct += 1
                else:
                    class_1_incorrect += 1
            elif data.target[index] == 1:
                class_0_incorrect += 1
            else:
                class_0_correct += 1

        label_0_probability = class_0_correct / (class_0_correct + class_1_incorrect)
        label_1_probability = class_1_correct / (class_1_correct + class_0_incorrect)
        gini_impurity_0 = label_0_probability * (1-label_0_probability) * label_0_probability * (1-label_0_probability)
        gini_impurity_1 = label_1_probability * (1-label_1_probability) * label_1_probability * (1-label_1_probability)
        
        label_0_amount = class_0_correct + class_1_incorrect
        label_1_amount = class_0_incorrect + class_1_correct
        gini_impurity_total = gini_impurity_0 * (label_0_amount / (label_0_amount + label_1_amount)) + gini_impurity_1 * (label_1_amount / (label_0_amount + label_1_amount))
        return gini_impurity_total
        
    def optimise_threshold(self, data, model, max_iter = 20):
        """Uses a simple logarithmic algorithm to optimise the binary classification threshold"""
        
        change_amount = 0.1
        iter_amount = 0
        probabilities = []
        for datapoint in data.data:
            probabilities.append(model.predict_proba(datapoint.reshape(1, -1))[0][1])
            
        while iter_amount < max_iter:
            impurity = self.calculate_gini_impurity(data, probabilities)
            self.threshold += change_amount
            impurity_changed = self.calculate_gini_impurity(data, probabilities)
            if impurity < impurity_changed:
                self.threshold -= 2 * change_amount
                change_amount = change_amount / -2
            iter_amount += 1
    
class custom_estimator(BaseEstimator, TransformerMixin):
    """Handles data loading, logistic regression and binarizing the threshold"""
    def __init__(self, thresholderClass, dataset = custom_data()):
        super().__init__()
        self.data = dataset
        self.thresholdBinarizer = thresholderClass
        self.model = LogisticRegression(random_state=0, solver='liblinear')
        
    def fit(self, max_iter = 20):
        regression_result = self.model.fit(self.data.data, self.data.target)
        self.thresholdBinarizer.optimise_threshold(self.data, regression_result, max_iter)
        return True

    def predict(self, data):
        return self.thresholdBinarizer.binarize(self.model.predict_proba(data)[0][1])
    
    def load_data(self, filename):
        pre_data = pandas.read_csv(filename).values
        self.data.data = pre_data[:, :pre_data.shape[1]-1]
        self.data.target = pre_data[:, pre_data.shape[1]-1]
        
    def get_accuracy(self):
        total = 0
        correct = 0
        for index in range(0, self.data.data.shape[0]):
            prediction = self.predict(self.data.data[index].reshape(1, -1))
            if prediction == self.data.target[index]:
                correct += 1
            total += 1
        return correct / total
        
    

binarizer = ThresholdBinarizer()
estimator = custom_estimator(binarizer)
estimator.load_data("breast_cancer.csv")
estimator.fit()
print(estimator.get_accuracy())

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas


class ThresholdBinarizer(BaseEstimator, TransformerMixin):
    """Optimises the threshold according to the Gini impurity"""
    def __init__(self, starting_threshold = 0.5):
        super().__init__()
        self.threshold = starting_threshold
        
    def binarize(self, prediction):
        if prediction >= self.threshold:
            return 1
        return 0
        
    def calculate_gini_impurity(self, data, model):
        class_0_correct = 0
        class_1_incorrect = 0
        class_0_incorrect = 0
        class_1_correct = 0
        for datapoint in data.data:
            for label in data.target:
                prediction = self.binarize(model.predict_proba(datapoint.reshape(1, -1))[0][1])
                if prediction == 1:
                    if label == 1:
                        class_1_correct += 1
                    else:
                        class_1_incorrect += 1
                elif label == 1:
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
        change_amount = 0.1
        iter_amount = 0
        while iter_amount < max_iter:
            impurity = self.calculate_gini_impurity(data, model)
            self.threshold += change_amount
            impurity_changed = self.calculate_gini_impurity(data, model)
            if impurity > impurity_changed:
                self.threshold -= 2 * change_amount
                change_amount = change_amount / -2
            iter_amount += 1
    
class custom_estimator(BaseEstimator, TransformerMixin):
    """Handles data loading, logistic regression and binarizing the threshold"""
    def __init__(self, thresholderClass, dataset = None):
        super().__init__()
        self.data = dataset
        self.thresholdBinarizer = thresholderClass
        self.model = LogisticRegression(random_state=0, solver='liblinear')
        
    def fit(self, max_iter = 5):
        regression_result = self.model.fit(self.data.data, self.data.target)
        self.thresholdBinarizer.optimise_threshold(data, regression_result, max_iter)
        return True

   # def fit(self, x, y=None):
    #    return self

    #def transform(self, posts):
     #   return [{'length': len(text),
      #           'num_sentences': text.count('.')}
       #         for text in posts]
    
    def predict(self, data):
        return self.thresholdBinarizer.binarize(self.model.predict_proba(data)[0][1])
    
    def load_data(self, filename):
        self.data = pandas.read_csv(filename)
        
    def load_prebuilt_data(self):
        self.data = load_breast_cancer()
        
    def get_accuracy(self):
        total = 0
        correct = 0
        for datapoint in self.data.data:
            for label in self.data.target:
                prediction = self.thresholdBinarizer.binarize(self.model.predict_proba(datapoint.reshape(1, -1))[0][1])
                if prediction == label:
                    correct += 1
                total += 1
        return correct / total
        
    
data = load_breast_cancer()
binarizer = ThresholdBinarizer()
estimator = custom_estimator(binarizer)
estimator.load_prebuilt_data()
estimator.fit()
datapoint = 0
print(estimator.get_accuracy())

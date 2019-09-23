from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression


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
        
        #We're doing this here for efficiency
        probabilities = []
        for datapoint in data.data:
            probabilities.append(model.predict_proba(datapoint.reshape(1, -1))[0][1])
        
        #A simple logarithmic algorithm that keeps the threshold going in the right direction with decreasing step sizes
        while iter_amount < max_iter:
            impurity = self.calculate_gini_impurity(data, probabilities)
            self.threshold += change_amount
            impurity_changed = self.calculate_gini_impurity(data, probabilities)
            if impurity < impurity_changed:
                self.threshold -= 2 * change_amount
                change_amount = change_amount / -2
            iter_amount += 1
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
import pandas
from AlizTestPackage import ThresholdBinarizer, custom_data



class custom_estimator(BaseEstimator, TransformerMixin):
    """Handles data loading, logistic regression and binarizing the threshold"""
    
    def __init__(self, thresholderClass, dataset = custom_data()):
        super().__init__()
        self.data = dataset
        self.thresholdBinarizer = thresholderClass
        self.model = LogisticRegression(random_state=0, solver='liblinear')
        
    def fit(self, max_iter = 20):
        """Handles data fitting and threshold optimisation, max_iter is handed to the threshold optimiser"""
        
        regression_result = self.model.fit(self.data.data, self.data.target)
        self.thresholdBinarizer.optimise_threshold(self.data, regression_result, max_iter)
        return True

    def predict(self, data):
        """Runs inference on the classifier for a single data point"""
        
        return self.thresholdBinarizer.binarize(self.model.predict_proba(data)[0][1])
    
    def load_data(self, filename):
        """Loads a binary classification dataset from a csv file with no headers"""
        
        pre_data = pandas.read_csv(filename).values
        self.data.data = pre_data[:, :pre_data.shape[1]-1]
        self.data.target = pre_data[:, pre_data.shape[1]-1]
        
    def get_accuracy(self):
        """Calculates the accuracy score of the classifier, different from the model's own accuracy"""
        
        total = 0
        correct = 0
        for index in range(0, self.data.data.shape[0]):
            prediction = self.predict(self.data.data[index].reshape(1, -1))
            if prediction == self.data.target[index]:
                correct += 1
            total += 1
        return correct / total
        
    
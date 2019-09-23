# Question 2.1
A new classifier model identifies bad bonds in the financial market for a hedge fund. Bad bonds can have devastating effects and must be avoided in the portfolio. 0.01% of all bonds fall into this category and our model has an accuracy of 99.99%. Is this ML model doing a good job? Why?

# Answer
It is possible, but not certain. If the model predicts every bond to be good, it will be 99.99% accurate, yet it is completely useless. On the other hand, an accuracy of 99.99% can be achieved also if the model correctly identifies every bad bond, but then also misidentifies a few good bonds as bad. As such, the information provided is not enough to definitively answer the question, though obviously the more interesting answer is if the model is useless.

# Question 2.1.1
Fill in the empty confusion matrix below with a possible concrete outcome if there are 100,000 bonds in the market.

|           |         | Actual |         |   |
|-----------|---------|--------|---------|---|
|           |         | Bad    | Not bad |   |
| Predicted | Bad     | 0      | 0       |   |
|           | Not bad | 10     | 99 990  |   |


# Question 2.2
On a logistic regression model with binary outcome in {0,1} that is optimized with stochastic gradient descent you have to tune hyperparameters
●	learning rate
●	L2 regularization
●	batch size
●	threshold value: the predicted probability above which we assign 1

Choose 3  metrics that you can use to compare the trained model and decide which one is the best for this use case. Explain why.

# Answer
I would use accuracy with the final results, and the F1 score and log loss on the model’s direct output. The log loss will allow me to fine-tune the threshold value, since from two models with the same accuracy but a different log loss, the one with the higher threshold value is more likely to be good, assuming no other factors at play. The F1 score is good for tuning L2 regularization’s lambda parameter and batch size. Meanwhile, accuracy is a good, overall judge of a model’s performance.


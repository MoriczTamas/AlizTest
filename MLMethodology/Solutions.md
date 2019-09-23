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

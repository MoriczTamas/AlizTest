# Question 1.1
There is a confusion at the client about how to set up the machine learning task. What are the training examples (X) and what are the predicted outcomes (y) ? What kind of algorithm to use? What should be the evaluation metric of the model? Please share your suggestions with them for each question.

# Answer
A good way to think about this is that training examples are what we already know and have observed, while the predicted outcomes are the useful bits of info we’re trying to infer by using an ML solution. As such, in this case the training examples would be pairs of passenger info inputs and known bought pre-booked items outputs, while the predicted outcomes would be a vector of probabilities that the passenger in question would like each pre-booked item.

A very simple solution for such a multi-class classification problem is to use neural networks, they are capable of handling many-to-many transformations such as what is required here. Depending on the complexity of the dataset, this could very well be a simple MLP network.

The simplest evaluation metric is accuracy, which is merely the amount of correct inferences divided by the total amount of inferences, but it is a metric that can lead to very poor performances on certain datasets. Instead, we could use categorical crossentropy here, which is a metric designed for problems precisely such as this one, since it penalizes giving high probability values to non-booked items.

# Question 1.2
We know that the popularity of the products are varying. E.g. purchase rate for priority boarding is 20 % while for food is 2 %. How would this influence the recommendations? Do we need to handle it somehow?

# Answer
To illustrate my point, I’ll use a more extreme example. Let’s say we want to predict if someone will become the future president of the United States based on some amount of personal data. It is obvious that we will only have training data with a positive outcome, that is a US president for 45 people while we can have a negative database of thousands or millions of people. If our model then predicts that no matter what our input is, the person will not become president, it will be very accurate while also being completely useless.

As such, we need to weight our training examples inversely to their frequency with which they occur in the dataset. In the example in the question, the system would recommend priority boarding more than ten times as often as food, because recommending food is generally a risky thing for our ML system to do and it’s unlikely to learn to do it, unless the training examples are weighted.

# Question 1.3
We settled to use one year’s data of online pre-booked purchase behavior for model training, which we split into 70% training and 30% evaluation sets randomly. Our final model is ready and it performs well on both sets. The plan is to retrain the model (no hyperparameter-tuning, just re-run) every day at 1 am based on data of the previous 30 days.
A data scientist from the client’s team expresses concerns that the production system will not perform as well as indicated by our training setup. Is this concern valid? How would you address his concern? Write an email to him.

# Answer
Dear Sir/Madam,

I am writing in regards to the concerns you expressed about our ML model. We heard your feedback and went through the process one more time to make sure we did not miss anything. Allow me to do the same here in the hopes of assuaging your concerns.

You are completely correct that good performance on the training set is fairly meaningless. While it is true that with a problem as complex as this, even a good training set performance is hard to reach, it still provides us with no answers in regards to the actual real world usability of the model. But consider that we randomly chose 30% of the dataset you provided and used it only to evaluate our model, not train. There is an exceedingly low risk of our model overfitting on the training data when the evaluation set is so large.

Of course, it is still a risk and it is obvious that customer patterns change over time, which is why retraining is important. Our daily retraining of the model makes sure that the model cannot adapt too specifically to any irregularities in the data which would result in overfitting.

I hope my explanation was satisfactory. If you have any more questions, do not hesitate to contact me.

Best Regards,

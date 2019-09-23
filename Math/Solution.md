# Question 3.1
The matrix below shows the probability that you are in a state today given we know your state from yesterday. There are two states: each day either you read or you train.
When you read one day you are very likely to continue the book the following day.
When you train you decide with a coin flip whether to go out and train again the next day.

|         |         |    Today    |  |  |
|-----------------|---------------|---------------|----------------|---|
|         |         |    Reading    |    Training    |  |
|    Yesterday    |    Reading    |    0.9    |    0.1    |  |
|  | Training |    0.5    | 0.5 |  |

Which mathematics concept would you use to calculate  the probability of training at any given day? ( The probability of training after an infinite number of days?) You are not required to calculate it.

# Answer
Markov chains are precisely the what the question describes. They are a mathematical concept for a stochastic system whose state depends only on its previous state, exactly like the problem here.

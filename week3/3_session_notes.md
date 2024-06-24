## Supervised Learning : Classificaiton

Many things that apply to regression also apply to classification.

The general problems:
- Formulation
- types of error

Remember, you may have a very good model however it might not indicate any underlying causalities. 

Many times, we are making models of the phenomonon or what is going on.  

### Model-Based (**Gaussian** and **Bayesian**)
We are thinking from Y's to X's
Think of this like the world is made of Y's, then based on the Y's, nature somehow creates the X's.
When we create a predictor, we invert this model. 
Once you have a model where the Y's generate the X's, we need to then go backwards.
This is known as **discriminant analysis**

### Logistic Regression
This is the idea that we directly build a predictive model that goes from an X to a Y. 
This is almost exactly what we did for regression.  Once we have the predictor, we can then generate the prediction of the Y's.

### Nearest-Neighbors
This is classification with no modelling involved at all.

## The general Context
We recieve a group of data and our goal is to record the data into multiple categories. 
We train this model based on examples.

We recieve X and we have to determine Y.
Some example categories: 

   X Value		           Y Value
symptoms, test result		Cancer?
email message			Spam?
Image of a digit Y		Which digit is it? (m=10)
Image of an animal Y		Type of animal (cat, dog, cow..)

When X is an image, the model is actually viewing values comprising the individual pixels.
X is very high dimensional in cases where we are dealing with images.  


## Example
Our goal is to determine the likelyhood that a someone is going to default on a loan.
Our features of our dataset are Whether they are a student, the balance they have, and their yearly income. Our dataset has 10,000 samples, only 3.33% default.  
We can numerically code the "Yes" and "No" as numerical values.  
Our goal is to predict whether or not that customer will default on a loan based on our current information.  

We do this by making a prediciton of their actual Y.  This is a very simplified data, defaults after # of years, how soon after the loan began, overall incomve vs current income, income trends, credit score, any history, etc. 

## Building a Classifier
In regression, we are trying to build a predictor (x -> y).  
The metric that we judge the quality is the mean squared error (MSE)

In classification:
We create a value that inputs X's and outputs Y's.  Currently the only difference is that the classificaiton takes discrete variables. 
Our biggest difference is the metric that we are using to evaluate. 
We are concerned with the Probability that we are making a mistake. 
**P(mistake)**

One of the simplist classifiers works using a Linear Classifier.  We plot a line along a scatterplot and use this to determine whether or not a point is on one side of the line or not.  
This is known as a **Linear Classifier**, where we compare Theta^T @ X to a threshold. 
The goal here is to choose the best line.  This involves choosing the best Theta and choosign the best threshold. 

Another way to look at this is through **non-linear** examples.  Here we plot a function (h(x)) and evaluate in the same aspect.  The goal is to find a _good_ function h and a threshold.  

It is important to incorporate these wtihout overfiting. 

Remember you can still also incorporate additional features within! Maybe we add additional effeects.  If we plot this in 2D space, this will not be a straight line but will instead be a curved line.  
If you plot things as a 3 Dimensional space based on the engineered feature, this would no longer be a curved line but actually a linear plane.  

Theta1 * X1 + Theta2 * X2 + Theta12 * X1 * X2
This would normally be nonlinear, however, if we change this and add a dimension to our space based on the final term (Theta12 * X1 * X2), we could dfine this as still a linear function. 
The new dimensions would be X = (X1, X2, _X1X2_)

You can go nonlinear by including additional features.  These are typically based on understanding on a typical domain.  We can extract features within that image and then evaluate those features wihtin a linear way.  Or, you could coninue to go nonlinear (this is the path to **NEURAL NETWORKS**)

### Error Types and Confusion Matrix
If we take a classifier and change the classification, we will recieve different classificaitons.  This will produce different types of errors.

We can create a chart of True Labels vs Predicted Labels.
We are not just interested in the overall error rate but instaed the individual details of how our errors are occuring.  The final goal of the classifier is to have fewer errors within our new data records. 

We aim to have few errors on our training set.  

If we use a classifier that is smooth, we will always have mistakes on our training set.  
There is a tradeoff between the two error types.  If we take our classifier and just move it horizontally, we will incrase the number of one error rate while reducing another error rate. 

This is similar to an alarm system based on sensitivity.  We have the primary errors "False Alarm" and "Missed Detection". When we increase one, we decrease the other and vice versa.  There is a point where you will need to make the trade off and we will need to evaluate this based on our dataset.

This is known as a **confusion matrix**, and it is for any m-ary classification (mxm matrix). 

It is important to have a good understanding of the confusion matrix.  

Terms: Accuracy, precision, recall, sensitivity, specificity, false alarm

## Gaussian Model-Based Approach
Our goal is to build a classifier that makes a classification decision. 
First we are going to make a model that will run from Y -> X.  Then we will utilize Bayes' rule to reverse the probability and predict X -> Y hat.  

This is known as a Gaussian "generative" model
A generative model uses randomization to create the result. 
The model determines **how** to generate. 

Then we apply and exemplify Bayes' rule.  

This is known as **Bayesian Machine Learning** which is a seperate subfield within Machine Learning. 

We are pretending that the world works in a particular way.

Natures makes the decision if a person is either _healthy_ or _sick_.  Nature generates people within these categories using a fixed probability.  Say 90% healthy and 10% sick.  What doe these healthy people look like?

Say we have a feature that is associated with healthy people.  Those people have a probability distribuition for hte variable (x) we are looking for.  People that are unhealthy also hasve this feature, it is just represented based on a different probability distribution.

How do we make these classification decisions?  We get the probabilities based on the value for each category and then determine the probability that they are within a certain category.  

It is important that we utilize Bayes rule to determine our overall choice.  This is done by taking into account the initial probability from the first classes (This is the 90% or 10% we discussed at the beginning) on top of the probability distributions on our individual classed variables. 

We need to determine the Mean and the variance based on class1 and class2.  With this, we can then generalize and create a normal distribuition for each class. 

This also will occur on higher dimensional cases.  
**Generalization** 
- More classes
- higher-dimensional feature vectors **x**. 

Remember, std deviation and variance are interchangable.  Most textbooks utilize variance when describing a normal distribution.  

In a model, our first component is the prior probability.
P(Y = k) = pik (k=1, ..., m) 	"prior" probabilities.
K is the specific class we are concerend with.

Once we know the class of a person, those people are distributed within the class in a specific way.  
P(X | Y = k) ~ N(mewk, Ck)
- mewk : mean vector E[X|Y = k]
- Ck : covariance matrix [Cov(Xi, Xj)] for class k (This is the normal distribution over a high dimensional space essentially)

The covariance determines the shape of the distribution.  

Essentailly we have two different sets of topography, based on the ellipses that are generated incramentally by the multidimensional normal distribution.  (Look into **elliptical contours / level sets**)
We use these topology sets to evaluate whether or not a point is within one category or another. 
**I feel like there should be some official term for this try to find it**

You only get circles if the covariance matrix is a diagonal matrix.  Otherwise you will generate ellipses instead of circles.

Can we learn from our examples the parameters of the model itself? 
We need to determine 3 types of different data:

- P(Y=k) = pik 
We calculate this by determining the following:
pi hat k = # samples of class k / total # of samples. 

- mewk = E[X|Y=k]
We calculate this by determining the following:
Mew hat k = average value of X over all sapmles of class k

- Ck = E[(X-mewk)(X-mewk)^T | Y=k]
We calculate this by determining the following:
C Hat k = average values of (X - mew hat k) (X - mew hat k) ^ T over all samples of class k.

This is known as **Plugin Estimates** where we replace expectaitons with averages.

### Bayes Rule
Bayes rule is the probability before seeing the values.
P(Y = k | X) = (pik * P(X | Y = k)) / P(X)
The idea is that we are choosing based on the highest likelyhood
Given value X, choose k with the largest posterior.

The approach minimizes the probability of error. How do we apply this?

Minimize P(error)
pik * gammak * exp{0.5(X-mewk)^T * Ck^-1 * (X - mewk)}
gammak is a normalizing factor based on the determinent, it is a normalizing factor based on the evaluation that we have done.
_This is ugly should be LaTeX lol_

This will create classifiers that will establish spaced based on QDA or Quadratic discriminant analysis.

### Quadratic discriminant analysis vs Linear discriminant Analysis
Sometimes our classifier reacts in interesting ways when we have different means with the same covariances.

Other times we will have different means with different covarainces.  
- This is a quadratic divider that is described as a quadratic equation.

LDA gives straight lines while QDA gives curves instead. 

## Back to Example
Lets apply this to our example.

Recall that only 3.33% of the records correspond to defaults
Worst case scenario, we can evaluate as "no default" : Miss-classification rate is 3.33%
Let's try LDA/QDA
We can first try to make predictions using only one component of **X** ("Balance")
- LDA Predicts 2.81%
- QDA Predicts 2.73%

What if we use every value of X? 
- LDA Predicts 2.75%
- QDA Predicts 2.70%

This is still quite high.  What other methods can we try? 
These small improvmeents actually give quite a large amount of monitary return to a bank. 
We can use cross validatoin to establish the true error rate. 
Our error rate on the holdout set is 2.73% which shows that there is no overfitting occuring. 

Some error types are more expensive then others.  In this case, it is more moeny for a bank to evaluate someone as capable of paying and then defaulting then it is for them to mark the user as a potential default and then not give them a loan. 

cost default : cost if Y hat = good (give the loan) and Y = default
cost lost : cost if y hat = default and y = good (lost customer)

This is probably something that should be taken into account when evaluating the error rate. 
Minimize the probability of error versus the expected cost of the error:
compare cdef * P(Y = Default | X) to clost * P(Y = good | X)

This is interetsing, we would be increasing the error rate to recieve a lower cost to the actual business. 

## Learning/Regression Methods for Classifications

Lets look at an approach that does not work.  
WE could look for a linear classifier, however, there is no aglorithm that would find a classifier with the fewest errors on the training set.
Lets try to use regression.  Assuming a binary y value, we could attempt to use ordinary least squares to plot Y onto X, as if the Y's were continuous random variables.  
In general, this doesn't work well.  This is because the quadratic criteria does not make sense for distance measurement vs classification. 

Lets try to fit a straight line for a curve that ranges from 0 to 1 (Must be nonlinear)
We can then interpret an estimated probability of P(Y = 1 | X)
This is very useful because it can also tell us exactly how certain we are on a decision because the value is between 0 and 1 where 0 and 1 are the 100% probabilities. 

The catch here, is that we do not have an avalible data that can fit this function.  

### The Logistic Model
These functions are known as sigmoids.
e^x / (e^x + e^-x)
This is known as a logistic function.  

There are other curves of different types that we will also utilize.  We can generalize this class with some parameters that we will be able to change.

P(Y = 1 | x) = e^(theta1 * x) / (e^(theta0 * x) + e^(theta1 * x))
P(Y = 0 | x) = e^(theta0 * x) / (e^(theta0 * x) + e^(theta1 * x))
Note the primary difference in the numerator between these equations.

These are the types of curves that we will like to consider.  

**Generalizations**
- Remember we need to assume that we could have X as a vector.  With multiple classes we would just have a theta parameter for a single theta.
    - P(Y = k | X) ~ e^(Thetak * x)
    - This must be normalized
    - P(Y = k | X) = exp{Thetak^T @ X} / sum of s(exp{Thetas^T @ X})
Now we want to find the best choice for parameters.  We have one vector for each class.  The idea is that we choose K for which numerator value is the largest. 
Decision rule : choose class k with biggest (Theta Hat k ^ T @ X)
What do we use for an evaluation property? 

We do not have training data for these _probabilities_.  We need to do something else in order to properly evaluate them. 
The training is done using the Maximum Likelyhood Method
This is when you have a model with parameter Theta and we choose to use the k for the theta which is most likely to occur. 

The likelihood of the observed data is as follows: 
L(Data;Theta) = IIi(P(Yi | Xi))

This is messy but overall:
- Convex (Non-Linear)
- Gradients are easy to calculate
- efficient algorithms

In reality, this is all done within the program.

No closed form, this can only be done computationally. 
This is different from quadratic because we are fitting sigmoids and therefore we do not have examples using the falues for the function that we are trying to fit. 

Note, you can utilize bootstrap with this as well! You just need the confidence interval and the standard error for the original dataset.

### Comments and Caveats
- Caveats from Linear Regression also apply here
- Good predictors do not imply causality
- can use features exp{Thetak^T @ X} -> exp{Thetak^T @ Phi(X)}
- Can add **regularization terms**
  - LogL(Data;Theta) - gamma * ||Theta||^2
  - Gamma is choosen with trial and error
- Performance on training set is not enough, must use seperate validations set or k-folds, etc...
- You can utilize confidence intervals and testing etc. 

## Results
Using logistic regression, we actually perform worse! These methods are very complicated and are difficult to predict.  Sometimes you try different methods and some are worse then others.
If we run linear regression with the lasso type, this performed better then the other methods (2.66%) 
These results are on the training set, you should always cross validate.  
The result from cross validation were not very far from those on the training set which is a good indication of no overfitting.

What is special about this example is that we have unbalanced data sets and unbalanced costs.
LogL(Data; Theta) = Sum(logP(Y = 0 | Xi)) + Sum(logP(Y = 1 | Xi))
Because our dataset is unbalanced, we are focusing on the bad customers (the ones that are predicted good but actually default.  

How do we determine this? We have to add a weight in front of our sum category.
LogL(Data; Theta) = Sum(logP(Y = 0 | Xi)) + Weight * Sum(logP(Y = 1 | Xi))
If Weight = 2, this is the same as having twice as many defaulting terms.  This is a helpful way of adding false data.  There are other methods using synthetic methods to create customers that look similar using probabilistic distribution. 

## Nearest-Neighbor Classifiers
This doesn't involve any modelling or math.  
We plot all of the points on our dataset.  All we do is classify this as the closest point. 

We can get more accurate with this by using the K-NN classifier:
- Given new X
- Find k closest Xi in the dataset where k is a number of points
- prediction : majority vote of the k labels

This is simple, no training is applied
The downside is that it can take a while for very large datasets with many dimensions.

We classify this based on LOOCV and then evaluate the sum of errros.  Performance was best ~14 nearest neighbors (NN) and evaluate the performance.  You want to evaluate your choice based on the overall number rather then minor performance increase due to computatoinal time.  

This method will work very well with low dimensional datasets and can outperform many more difficult methods. 

If K is very large (The size of the entire dataset), this is the equivalent of just averaging, which would be the same thing as what we mentioned previously (Set everyone to 3.33%).  
Largest number of neighbors just defaults to majority vote.  

When a new person comes in, with logistical regression we just plug it into our model.  With nearest neighbors, everything must be recalculated.  

Scaling also matters a lot, if we scale one axis but not another the majority would change.  

equivalently use a weighted metric:
||X|| = sqrt(x1 ^ 2 + w * x2 ^ 2)

Nearest neighbor methods tend to be more sensitive to scaling later on. 



### Quick Tid Bits
The goal of regularization is to prevent overfitting and to improve the models ability to generate new unseen data.

L2 regularization term is the square of the magnitude of the coefficients which is most commonly used as the penalty term in Ridge Regression

Accuracy is used to measure classification model metrics.  Mean Absoulte Error, Root Mean Squared Erro, and Mean Squared error are typically used for regression tasks.

When you get results, remember, you should be able to explain the rational behind the machine learning. Nearest neighbors does not yield an interpretable model but it allows us to make predictions that are easily explained.   


# Session 2
### Quick Review
We have a structural model : Yi = (Theta Star)^T @ Xi + Wi
An Estimator : Theta Hat = (X^T @ X)^-1 @ X^T @ Y
Then we create a Predictor : Y Hat = Theta Hat^T @ X
Now we have a predictor, how do we evaluate?
Standard error sigmaj : standard deviation of Theta Hat j.
This standard error gives us a sense of the accuracy of our estimates.
With probability of 95%, we will capture the true value of the parameter.
Then we can apply a **Wald Test** or Rejecting (Theta Star j = 0) if 0 is not within the confidence interval.
### Software -> Standard Error -> Confidence Interval -> Hypothesis Testing


## What can go wrong? Multicollinearity
This happens when the feature vectors are confined into a lower dimensional set.
Say we have a two dimensional feature vector.  
If we plot two features of an observation and they happen to be positively correlated (Advertising agency increases radio advertising equally with TV advertising) it is difficult to seperate the two because they are both scaled with a similar correlation.  Because of this there is no way to establish a true model.  
This is caused to the features not being **rich** enough.  Both feature vectors are essentially within the same subspace.

If we plot multiple features, we are looking for no correlation.  

In a formal sense:
- Our Matrix does not have full rank (approximately)
  - One column is linearly dependent of another column. 
- Standard errors involve (X^T@X)^-1 : With this correllation we will have infinite or huge standard errors.

How do we fix this?
We remove some of the variables.  If two variables have a positive correlation we can just remove a variable as it is redundant. 
We could also do some feature engineering to attempt to combine or reduce the two variables to one.  
This is an example of how we could utilize PCA.

We haven't mentioned any cause/effect relations, lets take a look at this.
Say we are looking at a dataset with choclate consumption and the number of nobel prize winners. 
If we perform a hypothesis test or even just visual identificaiton, we can see a connection between choclate consumption and nobel prize winnings.  
The model is useful for making predictions but there are assumptions that cannot be made based on the mechanism. 
For instance we could view this as:
Choc -> Intelligence -> Nobel Prize
When in reality: 
Wealth -> Choclate consumption
Wealth -> Nobel Prize

There may be strong correlations but this doesn't tell you the true model in this situation. 
Making causal inferences is very difficult, the gold standard is to do **random standardized experiemnts**.

Suppose that for policy making, we want to assess the effect on Y, when we increase X by 1 unit.  

		Advertising (X)
	       /
Market Size (Z)
	       \
		Sales (Y)

If we change the way that the advertising department operates, this predictive model tells us that sales will increase (Because it is only evaluated by Market Size) when in reality the individual portion of the company may increase/decrease.
Regression finds that X predicts Y very well, but it cannot be used to predict the estimate of the effect on X on Y.

The only way to actually evaluate this type of difference is to perform a field assessment. 

## What can go wrong? Latent Variables
Latent Variable : A variable that hides behind the scences.  The market size (Z) is the latent variable here.

		Advertising (X)
	       /
Market Size (Z)
	       \
		Sales (Y)

Because we didn't model Z, the Market Size was a latent variable that we ignored which might cause significant problems within our world.

Suppose there are two types of towns/markets.
In one type of market, there are no competing store/dealers so advertising does not change the success of the business.
In another type of market, competing stores or dealers often advertise a lot but due to this, the sales will never be too high. 

Because we ignored the hidden structure (Types of Market) we are led to a wrong conclusion.
The classification of the type of market is a latent variable. 
This is known as **Simpson's Paradox**

We can still get an adequate predictor but we won't be able to correctly understand the structural model and we cannot answer "what if" questions.
This may also lead to some mathematical errors for std errors, etc.

If there are hiddel variables, how do we avoid these problems?
**Add more variables!**

Here we add a variable Z for our market size.

_Sales = Theta0 + Theta1 * (TV) + Theta2 * (Radio) + Theta3 * (NewsPaper) + Theta4 * (Z)_

We can then include another variable, the sales we expect if we move from one category (Market with competitors) to another category (Market without competitors)

_Sales = Theta0 + Theta1 * (TV) + Theta2 * (Radio) + Theta3 * (NewsPaper) + Theta4 * (Z) + Theta5 * (U)_

We can continue adding more variables, maybe the location of the store (V=0 : Rural, V=1 : Urban)

_Sales = Theta0 + Theta1 * (TV) + Theta2 * (Radio) + Theta3 * (NewsPaper) + Theta4 * (Z) + Theta5 * (U) + Theta6 * (V)_

Do note that some of these would be categorical variables.  There are no problems with categorical variables, linear regression is accpeting of discrete and continuous variables.  
The only thing that must be continous is the Y value that we will be predicting. 

Now we would be adding 4 catagories, can we encode this to a single variable?
Endode as C = 1, 2, 3, 4 and use + Theta7 * (C)
This is a bad idea.  Choosing a linear value gives a numerical order to the value which the model cannot interpret as we can due to the artificial nature of this order.
If 1, 2, 3, 4 corresponds to A, B, C, D then this would be acceptable not with a non associative entity.

Would the market size be continuous? In the real world money is continuous therefore we would take this as a continuous variable.

By including more variables, we increase the power of the model.
How do we figure out what variables to include?
Research.

Maybe you start this model by just evaluating a single style of advertisement (News Paper).  Then we find out they run other kinds of advertising.  We can include those as well.  
How could we describe a situation like this?
We would describe that we saw a correlation however after seeing the other variables (TV and Radio), we could see that newspaper doesn't provide any additional predictive power and therefore eliminate the Newspaper from the equation.
Is there causality here? 
This is a common thing to want to determine, however, with the data that we have we cannot create a direct inference based on this. 

## Using nonlinear features of the data
We can include more variables by taking the original variables and then transforming them.  
If X TV is a variable, it may make sense to create new variables that are just nonlinear functions of variables we are already working with.

X TV -> (X TV)^2
X TV -> log(X TV)

Original data vector: X = (1, x1, x2) -> Augmented data vector: Xaug = (1, x1, x2, logx2, x1x2)
The more important thing to realize is that we are taking linear combinations of things that we can incorporate into our spreadsheet.

You are trying to fit the actual preserved labels.

## **Deep Neural Networks has many layers and the beginning layers can be though of as "forming" these feature engineering aspects while the later ones recognize this feature.**
This is kinda an aha moment for me

This is basically the introduction to multivarity because we are now adding dependent but **not** colinear features.

## Overfitting and Regularization
The more variables we add, the "better" we fit the data. 
This gives us a problem however, we may run into **overfitting**. 
This may be an illusion though.  Overfitting occurs when we attempt to fit a polynomial to the data (A special case of linear regression). As we increase the power of our polynomial, we are more likely to run into a scenario where our model is fitting the training data perfectly.
Now, when you run this model, it will have huge errors within our test dataset.

How do we determine what sets we should keep vs remove? 
There are two approaches: 
- **Regularization** : Penalize Overfitting
- **Data-Driven Methods** that don't rely on formulas

### Regularization (Ridge Regression)
We incentivize parameters to stay small rather then fitting in the nosie.
min[ sum(yi-Theta^T@Xi)^2 + alpha * sum(Thetaj^2)]
Where on the left side we are summing the data records and on the right side we are summing the parameters.
What is alpha? This is a regularization hyperparameter. Think of this as a tuneable knob that allows you to choose which algorithm you are utilizing.  Choice of methods.  Think higher level.

In practise, we try different values of alpha and determine which works best.
This can actually be interpreted as a Bayesian formulation: zero-mean normal prior on Theta.
- Alpha reflects the inverse of the variance
- moves the parameter closer to 0

### Regularization (Lasso or Saprsity Enforcing)
Ideally if we add an insane amount of features, we need to incentivize the regression parameters to stay small rather then directly fitting the nosie. 
min[ sum(yi-Theta^T@Xi)^2 + alpha * sum(|Thetaj|)]
This is similar to Ridge Regression but it is changed to the absolute value. 
Because of the sharpness of this penalty, the optimal solution actually pushes many of the solutions for parameters to 0. 
A typical Theta for a instance like this is (Theta Hat = (Theta, 0, 0, 0, 0, Theta, 0, 0... etc)
This is a good way to find the important features that you can use.  Think of it like automatic feature selection.
If you have a prior believe, you can use this to reinforce your belief before continuing. 
Convex optimization problem : fast solvers.
Strong theoretical guarantees (can discover true sparsity structure)

_Try this with RD2L?_

If we try this with some marketing examples, we might Newspaper and Radio all occur to 0 while TV and TV x Radio both stay at a value.
If you value of alpha is too high, you might be left with no features! 
Nothing very concrete besides trial and error.  What is large? It depends on how your variables are scaling.

Neural networks are nonlinear in weights.

## Performance Assessment, testing and Validation
How well are we doing? 

This is a difficult question because many times there are too many 'optimal' methods or predictors to choose from. 
We need systematic ways to do the following: 
- Set hyperparameters
- choose which variables/features to include
- try less or more 'complex' models
- choose between different learning algorithms

In order to determine this, we need to remember what the primary goal of prediction is. 
**Goals**
- Take existing data and determine or explain the data
- The primary goal though is **generalization** or the ability to perform well on **new data** that is unseen to our model. Y Hat ~ Y
- Once we _choose a method_, we need to assess the standard errors. Are the parameter estimates close to the true value of those parameters? 

Before we do anything we first need to choose a model! 

### Assessing Predictor Performance: Validation
We have R^2, however, this only discusses the performance on the training data.  We are interested in how to evaluate performance on data we haven't seen before. 

First before anything, we divide data into **training** and **validation** (hold out) set.  Our model _only_ runs on the training data, then we run the model on the validation set to evaluate how well the model is fitting to all of the data. 

This will give us something like an R^2 or Mean Squared Error on data that we haven't seen before.

### Comparing different models
How do we determine how many features to use? 
We can plot how many features we utilize and then plot the overall prediction error based on the number of features.
_When we say prediction error, we mean something like MSE (Mean Squared Error)_
What happens when we assess data on the validation set? 

Initially we will have something know as **underfitting**.  This is a situation where we are not enough features and cannot get a good description of the dataset. 
On the other hand, if we add too many features, we will see that the performance of validation drastically drops.  This is due to **overfitting** which occurs when we are learning everything about a model, including the noise of a model. 

How do we split the data? The standard way is randomly, typically through sciplot
Usually this is between 20% - 40% of the data to be put into the validation set. 
It is random to ensure that there is a good distribution of the data within the dataset.

Suppose we try a very high number of algorithms.  Just by luck, we might have one algorithm that works extremely well on the validation set.  This is similar to hypothesis testing, where if we test for many different hypothesis, just by statistical chance we may run into a situation where a model wrongfully fits the data very well. 
To assess the choosen method, we need to test based on information that was not used on the data at all. 

**Test Set** is a third group of data where you evaluate a second time to ensure that there is no statistical probability of oversight occuring.

In practise this may look a little different.
Validation is used to assess and choose.  The Test set is used to evaluate how well the method is performing on things that the model has not seen before.

**Downsides of this split method**
There are some drawbacks with a validation process like this.  You are inherently losing some data and it is therefore "Wasted". 
This validation procedure is a little bit noisy.

### Leave-One-Out Cross-Validation (LOOCV)
In order to fix the problems with this method we will utilize **Leave-One-Out Cross-Validation (LOOCV)**
Say we have 100 data points.  We split them to 99:1
We train with the remaining data points and then evaluate how we predict.  
Repeat this for each value. 

0 : 1-99
1 : 0, 2-99
2 : 0, 1, 3-99

Then we take the results of each of these and calculate the Mean Squared Error over the entire process.

Pros
- No variability due to random choice of validation set
- Uses all data for training
Cons
- Have to train _n_ times. 
  - Think of this as the complexity, takes a long time to train and very computationally intense. 

### K-Fold Cross-Validation
This is a comprimise on the computation required for LOOCV.
Randomly divide the data into _k_ groups ("folds")
for i=1, ..., k:
- Keep ith fold as hold-out
- train based on remaining k-1 folds
- evaluate (mean square) error on the hold-out fold: Ei
 - Generate a summary score or E = 1/k * sum(Ei)
For k=n: k-folce CV = LOOCV
This is also somewhat computationally heavy however not even close to the required compautatoin of LOOCV

**LOOCV Is the golden standard.  K-Fold is the best we can do for very large datasets**.
When using K-Fold, we can rely on K=5 or 10.

#### Workflow Summary for K-Fold
We determine one ML method to utilize.  We will call this Method 1.  We will fold 4 times.
x---
-x--
--x-
---x
We train on the groups of '-' data, then we use the 'x' data to evaluate the performance. 
We repeat this for each point on the hold out group.
By taking the MSE (Mean Squared Error) of this group we get an estiamte using Method 1.  

Now we repeat this same method on another variation of Model to use.  
We repeat the process listed above to evaluate the overall performance of the model.

We then determine the best method to choose (What has the lowest MSE)
Once we have hte method of choice, what would be the parameter estimate to be used? 
We have 4 different estimates from each of the splits from each fold. 
Now we would train on the full dataset and use that as our complete parameter vector.

If you are working on a very important business decision (Life or Death) many times you will then have to evaluate the model based on an ENTIRELY different test set.  This would be something that your company wouldn't have seen at all.  

### Assessing Parameter Estimates : Bootstrap
After we have choosen a method, we want to know if our Theta Hat is a good fit or not. 
Is there a way of determining the Standard Error with no formulas and only the data that we have? 
Yes! This is called the Bootstrap method. 

Assume fixed phenomonen and Theta; fix data set size n, and a particular method of estimation.
The assumption implies that something is randomly making data based on a y variable that we do not know. 
Now we take our dataset and apply a statistical method which gives us a parameter estimate.
We would like to know or understand the distribution of the Theta Hat.  
DIfferent statisticians will be receiving different sampling distributions of multiple data sets.
The histogram of the data produced would help us learn the distribution of the Theta Hats. 
Unfortunately, we don't have different datasets. What can we do instead of this? We can generate artificial datasets.

How can we create datasets from our singular existing data? 
**Resampling** generate new data sets by sampling randomly from the original data! (same as _n_ )
The bootstrap method just samples our original dataset randomly to create a dataset of same size with the values within the original.  We are doing this using **sampling with replacement**, values can be choosen multiple times. We repeat this several times.
k or the number of times we repeat this (perhaps 100 times)

We can then evaluate the standard deviation of the histogram which nicely evaluates the estimation of the standard error within our dataset.

Sometimes, these synthetic datasets may be permuted but they could be hte exact same.  Many times though, they are different but only slightly. When K is a very large number, we have a very solid distribution of the information within our dataset.

This is a **simulation** method.
This Boostrap method is used very frequently! The Standard Errors defined here are much better then those calculated within the program.

What is the differenece between K-Folds and Boostrap? 
In both we reuse the data
- In K-Fold we use it to choose our method so that the predicted labels are close to the true labels
- Bootstrap assumes that we have choosen our method, the problem is for us to assess the standard error of our specific estimate. How confident are we on the estimated parameters of our model?



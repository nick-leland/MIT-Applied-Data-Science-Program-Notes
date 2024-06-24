# Machine Learning

## Overview of this Week
- Central Methods in Machine Learning
- Primarily focused on **supervised learning**.
  - Supervised learning indicates that we are learning based on labeled examples. 
- Our goal is to predict the value of an unobserved _y_
- How do we create the prediction line? With different ML techniques
  - Linear Regression 
    - The most basic machine learning fundamental 
- We will also look at classificaiton values. 

Classification determines the type of a datapoint, while regression determines the value of a datapoint. 
This is overall very basic to impliment.  
We will be assessing different aspects of the method utilization.
 - How good is our method?
 - How good is our model?
 - How well do we trust our prediction or classification? 
We will look into Testing and Validation. 
Because of how simple this is to impliment, this should be something you try first and then determine the effectiveness of later.

Lets look at a basic example.  Say we are evaluating a patient, and we are able to comprise this information as a value, x.  We then create a number that is a state of health, y.  For every patient we are able to create an x and y value.  When a new patient comes in, all we see are their medical record or their x.  Our goal is to **predict** their y or state of health.

This Y value could be either a classification problem or a regression problem. 
- Sick or Not sick (Binary) [classification]
- Life expectancy (Float) [regression]


Data -> training -> predictor -> prediction

Sometimes we want to go a bit further into detail and understand the model.  Here we can follow an indirect approach. Here we just care about a good predictor.

Data -> training -> math model -> model-based predictor

We are trying to discover the underlying factors utilizing a model.  Here we are worried about what is happening behind the scenes. 

Models can also be useful just as an intermediate step.  **Model** is an approximation of the real thing or fundamentals of what are going on.  

**"All models are wrong, some are useful"**

### Notation
 - predict _Y_ from **X** = (x1, ..., xm)
 - xi: covariates, independent variables, features
 - Y: response, dependent variable, target

 - Vectors are boldface while scalars are normal font.
 - **X**2 second data record
 - X2 second component of a vector X

With two vectors we can create the inner product or dot product of two vectors.
X^T@Y
 - Star stands for true quantity, eg, _0*_
 - Hat stands for estimates or Theta Hat 

Statistics has a very large overlap to Machine Learning.  We are discussing the topics of both fields. 
- Data Science: Extracting useful information from data
You can either be algorithmically oriented (Machine Learning) or prefer the theoretical aspects (Statisticians).
The best people within the field straddle both of the sides of this field. 

### Probability Notation
P(~) where ~ is something

## Example Advertising and Sales
We have data across 200 markets (Spendings over TV, radio, newspaper)
The markets are for individual places.

In these examples, the feautures (x's) are the spending on advertisement, what we are trying to predict is the sales (y's)
Is there any relation between advertising and sales?
Can we use this data to make predicitons on how much we **should** advertise?

### Real World 
In the real world we would need to know the time distribution of data.
What about seasonality effects?
Quarterly data, advertising in Q1 will affect Q2.
Advertising on one city might change sales in the neighboring city

We are ignoring these factors for the sake of simplicity

What would you do if this is your only amount of data?

1. Plot the Data (Ideally in four dimensions)
 - Because we cannot plot in 4D, we are plotting each advertising to sales price.
 - We can use this to visualize any relations
 - How do we analyze this data in a systematic way?? 

Our goal is to build a predictor which takes a number as an input and then outputs a different number.  Think of this as a subroutine or a function.
The output of this function is the predictor.
What is good? We need an error criteria in order to evaluate performance.

#### Objective Function
X is a vector with all the features for each individual
Here is the objective for a typical regression problem (risk)
E[(g(x) - Y)^2]
g(X) is our "mistake" that we are making. We compare this to the Expected value or the population average.

We do not have the population here, only the examples.  We can assume that these examples are representative of the population.  We can use this to create a **proxy** to determine how we are performing over our population.
Proxy = 1/n Sum (g(Xi) -Yi)^2
This is known as empirical risk minimization.
The problem with this is that we can do arbitary curves that perfectly align to each point on our graphed X,Y even though the curve doesn't actually predict the new data well.

This is known as **overfitting**.

The best way to avoid overfitting is to restrict to limited class of predictors.
Example:
Don't draw a curve just draw a straight line that best fits our current situation.
_I have never heard of this before might be linear regression specific I think_

Why use sum of square error vs absolute value difference? 
You get a more understandable value by using the squared errors which is why it is industry standard

Sometimes people do use the absolute value isntead of square error though.  
_Look into this for finding location differences for maze project_

We want to build a predictor that does very well for the dataset that we have while restricting it to prevent overfitting.

_Go back to 51:00 and do these notes in LaTeX_

Y Hat = Theta0 + Theta1 * x1 + ... + Thetam * xm
X = (1, X1, ..., Xm)
Theta = (Theta0, Theta1, ..., Thetam)

Y Hat = g(X) = Theta ^ T @ X

To design a good predictor, is to decide good values for Theta 0 and Theta 1, where in a 2D example, Theta 0 is the y intercept and Theta 1 is the slope. (Follows a simple linear formula)

We want to choose a line with the smallest possible sum of square errors
The residual or error corresponds to the difference between the original locatoin to the updated location.
This is called **Ordinary Least Squares**

How do we solve this optimization problem?
We have a function of the Thetas, the X's and Y's are fixed.  The function is a free parameter, we want to find when this function is minimized. 
We create a spreadsheet with all the points as a matrix, then we utilize a formula to generate the lowest value.

Theta = (X^T@X)^-1 @ X^T @ Y
We are minimizing a quadratic function by setting the derivative of the quadratic to 0

Because this is linear algebra it is solved very quickly.  Single line in python code.

Example: 
n = 200 
m+1 = 4

**Theta Hat Matrix**
[[ 2.94]
 [ .046]
 [ .19]
 [-.001]]

Sales = 2.94+.046 * (TV) + 0.19 * (Radio) - 0.001 * (NewsP)
The values within the Theta are the valuves we plug into our original equations.

Comparing this with simple linear regression we get two different conclusions
- Newspaper is useless wtih the Complete Linear Regression Example.
- In the individual regression (only two variables) we see a positive correlation

## Interpretation and Justification 
### Empirical risk minimization
Say we have a Large true population.  We have Features (x) and Labels (Y).
This could be represented as a true relation but it may be complex.  Think a curved line instead of a straight line. 

We are interested in the best linear predictor.  

We take a finite sample and our goal is to find the saple of best linear fit within the sample.  
Hopefully, the linear fit within our sample will be the best fit within the actual population.
As long as the datapoints are drawn representatively, there is a theory that as our sample size reaches infinity, we reach the population best linear fit.

Say we have a friend that crafts another finite sample.  We will get a different result based on the fact that they are sampling different populations of data. 

Some days this will be a good fit, other days, we might grab a sample that is full of noise! 
How do we know which line we have?

Lets look at another way to interpret this data

### Maximium Likelihood
The X's are not random, but instead somehow fixed; Y is then observed.
For any candidate Theta, how probable would it be to observe the Ys that were actually observed?

Likelihood: P(Y | X; Theta)
We consider different models, for any particular model, we ask the probability of observing the particular Y's that we actually observed.  How compatible are they to that specific model for a given Theta.
We then pick a model based on the **Maximum Likelhood method**.

The standard model:
Yi = Theta*0 + Theta*1 x Xi + Wi
Conditioned on all the Xi: all the Wi are independent and Normal(0, sigma^2)

This way of thinking leads to the same estimated Thetas as minimizing the empirical risk

**Maximizing the likelihood function = minimizing the empirical risk**

In what ways are they different? 
In one case we assume they are drawn from some distribution and then learn the best **linear predictor**.
In another, we make a strong assumption that the _world is linear_ and then we learn the **coefficients of the structural relation**.
- This is more similar to the traditional effect of creating a model
The fact that this leads to the same mathematical model is a very lucky example.

## Performance Assessment
Say we have a model 
x -> {} -> y Hat

And a predictor (Predictive Model)
Yhat = Theta^T @ x

We can assume that we have a true Theta^star by building our estimates.
Theta Hat is the best estimator of Theta^Star

If we don't know how accurate our predictions are, we really aren't predicting anything.
How do we perform a Performance Assessment? 

### R-Squared (R^2)
Say we have a dataset.  If someone asked to predict the Y, with no knowledge of X's, we would predict the Y by choosing the average Y. 
Prediction if no regression : Y = 1/n x Yi
This is known as the Total Sum of Squares
TSS = Sum (n, i=1) (Yi-yBar)^2
Then we create our linear line which we can compare utilizing:
Residual sum of Squares (RSS)
We are then trying to measure the unexplained variation in Y after taking into account the X

R^2 = 1- (RSS/TSS)
This is the fraction of variation in Y that **has been explained**

If RSS is 0, the line perfectly fits the data.
R^2 = 1

If RSS == TSS, the fraction we recieve is 0.
R^2 = 0 or Regression is Useless

Usually you will be somewhere in between, the higher r^2 is, the better fit you have on the dataset.
When X is one dimensional, R^2 has an interpretation as far as correlation, but there is no interpretation in higher dimensions

Think of this as an easy to understand coefficient to determine how well the data fit.

Higher r^2, the closer the lines fit to the data points
Lower r^2, the farther the lines are to the data points

With enough data, you can find the 'True' red line. 

If we perform this on the combined linear regression, we get a result of 0.897 which is a relatively high fit. 
If we perform this for the individual categories we find that R^2 for newspaper is .05 or the Newspaper budget explains very little.

With more variables R^2 can increase due to overfitting, we can therefore utilize **Adjusted R^2**
Adjusted R^2 = 1 - (RSS/(n-m-1))/(TSS/(n-1))

This is interpreting that the newspaper is relatively useless but we need a bit more of a defined statement that newspaper can be eliminated.

## Noise

How can we trust that the estimates of Theta Star are close to the true values?
Theta Hat ~?~ Theta Star
How can we measure the accuracy of the related coefficients?

Assume we have a structural model. If we don't have a structural model we need to resort to simulation/bootstrap (2nd session)
Structural meaning that we have some formula to represent our data (Linear is most common example).  We have a _simple way_ of describing the world. 
The estimate that we get is a randome variable depending on the random data.

(Theta Hatj - Theta Starj)^2 		(J is the jth component)
J could be newspaper where Theta Hatj is the estimated.

E[(Theta Hatj - Theta Starj)^2]
We are estimating the coefficients here.
This error can be composed into two pieces.
Systematic = (E[Theta Hatj] - Theta Starj)^2
Variance = var(Theta Hat j)

E[(Theta Hatj - Theta Starj)^2] = (E[Theta Hatj] - Theta Starj)^2 + var(Theta Hat j)

In linear regression there is no bias under the assumption that our model structure is correct.  Because of this, the first term dissapers.
We can therefore just look at the variance when we are looking at how noisy our variables are. 

E[Theta Hatj] = Theta Starj

When Theta Hat has a lot of variance, we have a higher likelyhood of having extremely different fitting lines between trials

Different statisticians collect different datasets.
If our estimates are biased, they tend to be above the true value.  
If they don't have bias, they wouldn't be systematically greater or less then the central point of our samples.

The least squares removes the bias, only concerned about the general variance.

Each estimate has a normal distribution.  Different statisticians will get different variables but it will obey the central limit theorem (therefore normal distribution)
We can then make a the assumption that the Wi's are normal therefore the Theta Hat's are on a normal distribution with Theta Star as the mean.  
Normal distributions are based on two values, **mean** and **variance**
We are only concerned about determining the variance. 
For a normal distribution, we know if we take the mean and go left or right from the mean by 2 standard deviations, we cover 95% of the distribution.
Knowing the variance or the standard deviation we can then determine the accuracy of our estimate.
Knowing how wide our distribution is proportionally shows how random our variance is

In machine learning this is known as the **Standard Error**.  

## Standard Error
How do we calculate this? There is a formula but software will implement this for you. 
It is important to understand this is the width.
This is in turn, the Covariance matrix of Theta Hat.

Theta Hat ~ N(Theta starj, sigma^2j)
With probability 95%: |error| = |Theta hatj - Theta starj| <= 2 std dev.j

Once we know our standard errors, we can go left or right two standard deviations which will define our 95% confidence interval.  

With probability 95%, the Theta Star j is true and therefore it does contain the True Value
P(Theta Starj is contained within CI) ~ 0.95

Our assumption was that the world is linear.  We are trying to estimate a true quantity based on noisy data.
Noisy data means that our estimate will be noisy and therefore our Theta Hat will also be noisy.  This noisy Theta Hat is what we use to determine the confidence interval.
Essentially 95% of statisticians will be lucky, 5% will be unlucky.
We just don't know if we are right or not based on that. 

Because datasets are random, it's always possible you recieve an incorrect dataset.

## Hypothesis Testing
Are the data compatible with the null hypothesis [Theta Star j =0]?
The null hypothesis is that there is no effect (Hence the 0)
With our confidence interval, if 0 is not contained within the confidence interval it suggests that we can reject the null hypothesis. 
If the confidence interval does include 0, as far as the data is concerned, the data is compatible with the 0 therefore we do not reject the null hypothesis.

Suppose that the null hypothesis is true. 
P(reject | Theta Starj = 0)
This happens if the confidence interval does not capture the value or if it misses 0.
P( the confidence interval 'misses' 0 | Theta star j=0) ~ 5%
This is known as the false discovery rate.

Two types of mistakes in hypothesis testing, 
- Theta star is 0 when it actually isn't
- Theta star isn't 0 when it actually is

You will also hear about p-value.
Under the null hypothesis the Theta Hat's have a particular distribution.
How far away is our Theta Hat from 0? We can count this by determining the probability of the location of our tail.  The probability of our result around 0 is the P value.  
We reject the p-value < 0.05.

When we look at our exmaple, we determine that the first three coefficients (TV, Intercept and Newspaper) are significant. For newspaper though, Theta Star news = 0 or it is removed from our model.
There is a big problem with statistical significance.  
There is a difference between we accept that Theta Star = 0 and "no effect".
We can never make a statement that we are sure about that within the statistical business. 
If we reject the null (We think there is an effect) 
We say that "With the data we have, it could be that Theta Star is 0 but it is unlikely to get such data where Theta star is 0."

Examples:
_no effect_ : Theta Starj is zero
_small effect_ : Theta Starj is so close to zero that the data cannot detect it
_too few data_ : Theta Starj may be nonzero but need more data to "see it"

## Prediction Errors
These can happen because the estimated coefficient is not the same as the true coefficient.
The Y has a lot of idiosyncratic knowledge. (I don't understand what this means)
We can utilize a **confidence band** to view the confidence interval visually around the Theta Star that we plot. 



# Time Series

What is a time series? 
Time series are everywhere, 
- Finance (Stocks interest rates, etc)
- Weather (Tempt today, tommorow...)

A time series is an observation of a specific phenomonon that is time stamped. 

If I think about the temperature today, I am thinking of a temperature in 10 minutes but it is based on the variable (the temperature previously)

If we think of traditional studying samples there is no direct dependency.
If we track the number of effects over a period of time this is a time series. 

We are interested in 
- Modeling the time series
- Predicting results wtihin the series
- Making a decision to recreate a different output

## Introduction to Time series
Lets say we are looking at a plot that depicts price over time.  The points on this plot represent a specific **trend**.
The trend is deterministic, the points will roughly follow the trend however there is a random distribution among the points and the trend that was determined. 

Exchange Rates Against the Dollar is something we could see a Time Series about. Tracking and Predicting the process. 

The key takeway is that we are plotting a value against time.

What you measure today, depends on yesterday
- Could be simple or domplicated (the dependence factor)
- Memory can be high (no limitation on the duration of dependency)
- Memory is typically unknown (you do not need the value)

The variations in teh data can be due to time-varying average : trends.  There are two types of trends
- Deterministic (Linear, Quadratic, etc.)
- Seasonal (Periodic, price is different in october vs january)

Transformation of the data may help.  

### Stationarity
Time series must have some sort of structure.  **Stationarity** is the fact that some variables are 'constant' over time.  
- Mean
- Autocovariance

What happens if such variables are changing? There is some sort of transformation occuring to a stationary series.  We can assume slow variation.

We have data {x0, x1, x2, x3, x4, ..., xn}
Why can we model this data? What property needs to hold?
**Stationarity** is the required property to be able to plot based on a time series.

If the data is not stationary, there has to be a trend to it.  

We only have the data once.  The idea is that if we re-ran this data, and collected the data multiple times, those sets of measurements would have the same mean and covariance.
**Mean** : Mewt = E(Xt) or what we estimate the value to be.
**Autocovariance** : Rx(t1, t2) = E((Xt1 - mewt1)(Xt2 - mewt2)) If we take two variables, what is the covariance betweeh the two random variables.
The mean is a function of time, the autocovariance is dependent on the variables t1 and t2.  

A process is considered stationary if the Mean of a process is constant over time.
E(Xt) = mew (around 0)

We do not have multiple time series, you can compute the mean of this process by computing the sample mean across the entire time.  This is a property of a stationary process. 

We can use the data to compute the sample mean of the data.

Sample Mean (Constant over Lambda)
Mew hat = 1/(n-lambda) * sum(Xi)
i = lambda

We should not see a large variation over lambda. 

How do we know that the sample version is stationary? We perform this evaluation again, but we shift over by one point and determine the mean of the smaller dataset.

This is our test to check if we can perform this. 

If you shift _too_ much, the overall amount will be too small to get a value off of.  

Autocovariance is a function of the time difference.  
Rx(t, t+r) = Rx(t)

Sample AUtocovariance 
R hat x (T) = 1/(N-lambda) * Sum((Xi-mew hat)(Xi+T - mew hat))

Its not that every time things are changing, its that the lag of the two are changing.  The autocovariance is a function of the lag difference.  You need to verify if this is true for **every** lag.  

For each lag (thao) we have a shift lambda.   
In order to check for thao = 1, we need to check the differences from (X1 -> X2, X2 -> X3, etc)
For each of these, we should have the same amount.  This repeats for Lag = 2, etc. 

This is the formula that has to be applied to the data to determine if we can apply a time series technique.  

This is really a correlation between the values.  This property has to hold for a description of a process.  

We can derrive a **De-trended** data by subtracting the mean from our graph.  

Removing Seasonality.  Say we have a model, and we have a seasonality model that is deterministic.  
We can estimate St using periodic regression.  Model the differences of the data. 
Xt = Xt-h = Yt - Yt-h 
Where h is the period of seasonality

Taking differences of the data is a very common way to remove the trends within data.

## Models of Time Series
### Time Series that are stationary
**White Noise** - Pure randomness.  At every step we sample at some random probability distribution.  At time t, we are sampling from an independent random variable.  
There is no information to extract because there are no dependencies.  
**Random Walk** Xt = Xt-1 + wt where wt is the white noise. Think of this as a drunk man walking, he is going along the straight path, however, the white noise is him stumbling around.

What we have now created is a time series.  The point from before, with a noise value to it.  What if we try to decide 

We will build a Auto-Regressive model (AR) or a Moving Average (MA) model. 

**Autoregressive Process** (AR(P)) What if we perform a regression over the randomness and then.
Xt = sum(ai * Xt-i + wt) 
One important interpretation of this model is the _memory_.  t is the size of the regression.  How far back are we combining variables to create a new variable? 
All of the data generated from an autoregressive model is a staitonary process.

How do you compute the exact autocovariance? 
Rx(t) = sum(ai * Rx(t - i) + sigma^2 * phi * (5))
This statisfies an interesting result.  A function that solves this **has** to be an exponential function. The autocovariance satisfies recursion and the only solution here is a decaying exponential.  

This means that we can compute this and therefore extract the form (verifying that it is an decaying exponential) 

This is interesting.  When we are training a model, the only thing that we have is the data.  We need to determine how to calculate these based off of just the sample informaiotn.  What can we say about the underlying data based on just the generated form? 

**Moving Average** (MA(q))
At any point, Xt is just the sum of the previous values with noise.
Xt = sum(bi * wt-i) 
X0 = W0
X1 = b0 * w1 + b1 * w0 

There is a correlation between X0 and X1 because they share values.  

We are _creating dependency_ by taking the combination of white noise signals.  Think of this like a moving average of the white noise.  

Autocovariance of MA
How do you compute the exact autocovariance? 
We use a finite suppot: 
Rx(t) = sigma^2 * sum(bj * bj-t

The autoregressive model thinks we are decaying exponentially, with moving average, they are exactly equal to 0 after a certain range.  

We are essentially using these terms to select what model we should be using based on the impyrical values.  

**Autoregressive Moving Average Process** (ARMA)
We can always use multiple and combine the two processes.  Utilize regression while also adding a moving average.  
Xt = sum(aj * Xt-i) + sum(bj * wt-j)
This is a superposition of different processes.  

## Summary 
We have multiple different mathematical models that can create time series, all using white noise as a buffer.  
How do we go from here to going further? j

## Learning Time Series
This is a very condensed subject! Usually takes much longer.
The object is to go from the **Data to a Model**

AR learning looks like a standard least squares.
What is the catch here? 

Just like any other model, we are splitting the data 80-20 for train vs test.

Learning AR(p)
We start with data {x0, x1, x2, .., xN}
We have a model known as P
Xt = sum(ai * Xt-i + wt)
We can then use the least squares to estimate (a1, a2, ..., ap)

If we know P, we could plug in the variables.  Xt - a1 * xt-1 + a2 * xt-2 
Since Xt is written as a combination of X= and X2, we could minimize this distance squared. 
( Xt - a1 * xt-1 + a2 * xt-2 ) ^ 2
This minimizes the distance between a1, a2.  
This is a simple least squares problem.  Say we determine p (p=2 for this guess)
The problem is that we haven't determined any truth about the data.  

Lets look at an example : 
Generate 100 data points according to 
xt = 0.7 * xt-1 + 0.1 * xt-2 + wt 
We will call this the _truth_ 

Our first reaction is to learn a model based on one parameter.  
Lets look at an autoregressive (AR(1)) model.  When we performed this, we got an answer of a = 0.8704.

What happens if we fit an AR(2) model? 
When we performed this, we got answer a1 = 0.7448 and a2 = 0.1742. 

What is our observation here? 

These are closer to the original values, as we increase this we converge to the actual result.

WE could transform this into matrix form based on every x location {x0, x1, x2} and the {xp-1, etc} we could think of this in a particular way. 

We are thinking about y - Ax.  The parameters within our matrices are correlated.  This ends up giving us a not very nice correlation of least squares.  If we guess the correct model structure, we get convergence.  We may get a good prediction, but no convergence.  What is convergence here? Is it important? 
We would like to recover the underlying system but realistically we do not need to know the exact.  
If we know the right size, we shoudl be able to cover the parts of the system that are unknown.  

The least squares is easy to compute, however, we need a good  estimate of P.  How do we get this?

### Order Estimation
We could run multiple different versions of P and then apply the results to the cross validations of P.  
Derive mutliple estimates and choose based on error on cross validation data. 

Instead of just minimizing between each, why don't you add a penalty as a function of P and N? 
Overfitting also occurs here, we don't want a very large P because we don't want to overfit on the noise.  

Instead of limiting P and restricting the test set, we could add a penalty funciton to P.  Eventually the penalty of P will dominate and our result will reflect the lower P value.  
This is known as **MDL** and **AID**, all different ways of adding penalties to the cost function.  How much do you penalize P? This is the primary problem with this method. You need to choose this very specifically.  

A quick and dirty method has to do with teh AUtocovariance fucntion and using it as guidance.  (ACF)  If you can compute the empirical of the autocovariance function, you can determine an estimate of P.  

**Partial Autocovariance Function** 
We transform the variable as a projection onto some parameters.  
Xt | Xt+1, Xt+2, ... Xt+k-1 | Xt + k
We can project Xt+1 and Xt+k on the variables between Pt and Pt+k

In the case of ARX, the new PACF is zero outside p.  

We create a new variable that has exactly 0 outspide P using this.  
Partial Autocovariance gives you a really good estiamte of what P should be.  Now you have a way to proceed with regression on that p value. 

We can do something similar with the Moving Average.  This is much easier though becasue we know that moving average has a finite support.  

With ARMA there really isn't an easy way to do this becasue both decay within ACF and PACF.  You cannot transform this to find a finite support so you need to essnetially use different properties to predict. 

There is a lot here, be sure to review time series more in depth.  




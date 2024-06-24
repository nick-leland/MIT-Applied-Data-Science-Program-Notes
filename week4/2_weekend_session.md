# Time Series

## Introduction
Time series is data taht is recorded on teh same subject at different points in time (regular intervals)

Some examples of this could be:
- Gross Domestic Product (GDP)
- Stock price over a period of time
- Heartbeat recorded at each second

**Time Series Forecasting** : Predicting Unknown Values

Components of Time Series
- Trend : An average of what the time series is doing 
  - Uptrend
  - Downtrend
  - Sideways Trend
  - Remember, this is dependent on the window of your viewpoint

- Seasonality : A repeating Trend
  - Easy to think of this like month/season spikes.  
  - Electricity Useage spikes during cold and warm months for climate control 

- Cyclical Variatiosn
  - Trends with **no** repetitions. 
  - Think of a bitcoin historical data plot, if we fully zoom out, there is no trend that is fitting for the information.  
  - There may be trends _within_ the data, but nothing that can encompass the entirety of the data.

- Irregular Variations 
  - These are random variations that are **random** and **erratic**.
  - We cannot predict something like volcanoes, wars, famines, etc.  
  - The time intervals are not the same here.  

### Stationary Time Series
What makes it stationary, constant statistical measures such as mean, variance, and co-variance must be constant over a period of time.

This is important because certain models require a stationary time series

How do we test this? We can use intuition (look at images) or a statistical approach (Augmented Dickey-Fuller test)

**If the time series is not stationary, we need to convert a time series into a stationary time series.**

### Time Series Forecasting Methods
- Classical 
Machine Learning
Deep learning : 

## Statistical Models
### AR
Auto Regression models are based on the idea that the current value can be explaiend as a linear combination of p, past values.  
Yt is stationary
wt and phi1, phi2, ..., phi
P value is the hyperparameter taht represents the length of "look back" in teh series. 

**White Noise** is the distribution of the points around the predicted.  It represents a random process with no discernible pattern or predictable structure. 

### MA (Moving Averages)
This algorithm is similar but focusess on teh error within the values.

### ARMA 
This is the AutoRegressive Moving Average Model.  The only downside to using this model is that it requieres a stationary model.  In many situations, time series can have trend (that makes the time series non-stationary)

## Autocorrelation
An autocorrelation plot is a pearson correlation coefficient between the time series with a lagged version of itself.
If we correlate two time series with itself, the correlation will be 1.  If we shift a time series one period with itself, we can view this correlation difference with 


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


## Models of Time Series

## Learning Time Series

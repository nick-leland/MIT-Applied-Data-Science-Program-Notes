# Recommendation Systems

## Outline
Module 1 Background
Why and What are they?
Example Datasets

Module 2 Problem Statement
Recommendation Systems : A prediction problem
Model : From caricature to extremely complex

Module 3 Simple Solutions
Solution 1 : Averaging
Solution 2 : Content-Based


## Background
Imagine we have a basic question.  What food should we eat today?
Why not just search something? We can Search online for a type of food, say Pizza, near me.

There are many options but what should be recommended? Which of these options should be advertised to the user?

The search narrows it down (Type of food) but there are still a wide variety of options.  

In the end, we are asking a question, evauating the answer, asking another question (to narrow down the previous), evaluating and continuing until we reach the response we are looking for.

Examples include Food, Online Dating, Content on Youtube or Spotify, Professional Connections, advertisements, etc.

Maybe _data_ can help us, what data do we need?
We need data not necessarily from a food critic, but from people like the user!  We could use something like Yelp data! 
There is a dataset that is publicly avalible containing yelp data.
What type of information does this dataset contain?
The Address, attributes, the business id, categories, what city it is located in, the exact lat/long location, the name, the review count, the overall stars the business has, etc.

There are 2 million users that have participated in this data.  That is ~10 times the amount of businesses. The users also have data.
The amount of time they have been yelping, the yelper "elites", friends in yelp, if people find their reviews funny, the number of reviews, number that are useful, etc.

This is essentially a social network within the app. 

There are 8 million reviews in the dataset.  Each review gives the business, the time it was reviewed, the number of stars, etc.
There are also check ins, which is essentially people just noting that they were eating at a location.  This data could be feature engineered to another evaluation.

Leaving off at 42:19

## Content based recommendation systems
**If a User likes an item, then they will also like a similar item**
- Works based on past interaction of a given user, doesn't take other users into consideration
- Uses information of all products and preferences of just one user

Pros
- Don't require a lot of user data
- Does not suffer from cold start
- Less expensive to build and maintain
Cons
- Unavalibility of features which explains more information about a user
- Recommendations will likely be direct substitutes instead of 

## Clustering  based recommendation systems
**Based on K-Means Clustering**
1. Compute similarity between each pair of users
2. Obtain representation of each user in low dimensional space
3. Perform the K-means clustering algorithm and find the number of clusters, i.e., K

## Hybrid Method 
Combines different methods, generally the content and collaborative filtering methods
- Can be implimented in a few different ways
  - Content based and collaborative based predictions seperately and then combine them
  - Calculate then average them
  - Calculate and weight a process for them.
Link for more information : https://medium.com/analytics-vidhya/7-types-of-hybrid-recommendation-system-3e4f78266ad8

# Case STudy
## Regular Expressions
Regular expressions are used to find patterns in other strings.
- Find all web links in a document
- Parse email addresses
- Remove/replace unwanted characters
import with 're'

In order to train a content based system, we need to be able to work with the content! This is where regular expression comes in.


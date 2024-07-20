# Weekend Session Week 6, Recomendation Systems

All the customer wants is to view products that they are likely to buy, or are relevant to what they are looking for.  
This is the main objective for a recomendation system. 

## Popularity based recommendation system
Pro
- No cold start problem
- No need for the user's historical  daata

Cons
- Not personalized to users

## Collaborative Filtering
If a users likes/dislikes are similar, then their tastes are considered similar! 
RMSE (Root Mean Square Error) = sqrt(Sum( (yi hat - yi)^2 / n))
RMSE punishes large differences much harder then other evaluations.  

Two types, **User-User** and **Item-Item**
**User-User** is based on the search of similar users from the user-item interaction matrix
**Item-Item** is based on the search of similar items from the user-item interaction matrix

Note, we are evaluating the user-item interaction matrix for both.
How do we measure similarities? 
One way to perform this is to evaluate the **Cos Similarity**.  Many times when using Cos Similarity, we will deal with very sparse data.  
In order to overcome this, you can utilize Matrix Factorization (SVD).  This decomposes the original sparse matrix to a low-dimensional matrices with latent features and less sparsity.  
Think of this like moving from individual movie ratings, to instead the genre ratings.  We can perform this operation by performing some sort of dimensionality reduction (PCA or TSNE)
This is known as Matrix Factorization because we generate a matrix A from different factors of other matrices. 
You can also utilize Stochastic Gradient Descent in order to find the optimal values for the two matrices.  

The idea is to fill the potential rating of the user, and then recomend them the movies that the algorithm believes they would enjoy the most.  
It is **important** to evaluate the performance, if we recommend something and the user interacts with the result, it is important to properly score this.  

Why use Timestamp over Date Time in python?
Timestamp takes up less memory when compared to Date Time.  You can then also convert it for ease of reading. 


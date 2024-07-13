# Introduction to Deep Learning

## Introduction to Neural Networks
- How do neural networks represent data?
- How can we train a neural network?

Why are Neural Networks important? We want to make predictions from all sorts of data, including all sorts of different types of predictions.  
How do we encode different types of data? This is where neural networks really shine.  

### General Strategy
- Encode data as useful, informative feature vectors
We have image, text, or a graph and we then need to transfer these to vectors.  This is known as encoding.  
Feature Encoding : Remember we evaluated Linear classifiers vs Nonlinear classifiers.  The key point to remember here is that **Feature-based linear classifier** is where we compare Theta (Coefficients of a linear function) to phi(X) (Where X is a combination of all variables).
We need to **re-encode** the data into a nonlinear classifier by creating a new feature vector. 

What is good encoding (representation) 
With neural networks, we can just learn directly from the data without having to make predictions by ourselves.  

Neural networks learn everything together, the function and the final classification points are all generated simultaneously.  

There are different specialized architecthures
CNN for images
Transformer for text
GNN/Transformer for graphs

### Deep Learning reasons for success
- Lots of data in modern formats
  - Many problems can be solved at scale
- Now due to the advancements with computational resources (GPU) we can do this.
- Large Scaleable Systems 
  - Larger models are easier to train
- These neural networks are very flexible (think lego piece)
  - common representations give a diversity of architecture choice.

## How do neural networks represent data?
The classic neural network uses **Feedforward Neural Networks**
The input layer is usually a vector.  Hidden layers will perform different functions.  
**Hidden Networks** are hidden because _typically_ you don't monitor those layers.  
The output layer gives the final result.  
This is also knwon as a **MLP** or a **Fully Connected Neural Network**
We can think of the output layer as a normal linear classifier or a linear regression/logistic regression problem.  

The number of neurons in a layer is known as the **width**.  The number of layers is known as the **depth**.  

### A unit in a Neural Network
1. takes the weighted sum of many different inputs. This is known as the **preactivation score**
_this is an inner product_
z = sum(wj * xj) = **w** ^T * **x** + b
2. Compares to the threshold.  
f(z) = {1 : if z > 0, 0 : else} 
We can think of one unit in a neural network as a linear activator. It is taking in multiple different inputs and weights and then determing whether or not the node or neuron activates or not.  

Lets look at an example to determine whether or not a person has the flu.  
x1 = Rashes
x2 = Fever
x3 = Cough

We want to find a rule that works with this preactivation rule.  
w1 * x1 + w2 * x2 + w3 * x3 > 5 ?

Rashes aren't a symptom of the flu so maybe we set the weight of that very low.  Lets look at some examples.
w1 = -1
w2 = 3
w3 = 3

Normally these weights are not manually set, they are learned! 
Think of it like each input is a possible piece of evidence (for or against) the arguement.  

This is not very practical because currently we have a very sharp "activation function".  We want something that is much more linear.  
Examples of Activation Functions
- Sigmoid (e^z / (1 + e^z)
- tanh(z) (Hyperbolic tangent) (tanh(z))
- Rectifier : Rectified Linear Unit **ReLU** (Max(0, z))
  - If the number is below 0, it sets to 0.  If not it keeps the value.  

What do we choose?
Sigmoid is actually bad for neural networks because it flattens out. The sigmoid is mainly used within the output.  ReLU is mainly for computer vision, tanh for recurrent neural networks. 

Left 46:18





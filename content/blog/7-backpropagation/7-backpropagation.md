---
title: Backpropagation
date: 2026-05-14
tags: Deep Learning, Machine Learning
---

{% set linearRegressionPost = "" %}
{% set mlePost = "" %}
{% set nnPost = "" %}

{% for post in collections.posts %}
  {% if post.data.title == "Linear Regression" %}
    {% set linearRegressionPost = post.url %}
  {% endif %}
  {% if post.data.title == "Maximum Likelihood Estimation" %}
    {% set mlePost = post.url %}
  {% endif %}
  {% if post.data.title == "Neural Networks & Inference" %}
    {% set nnPost = post.url %}
  {% endif %}
{% endfor %}

In the <a href={{nnPost}}>previous post</a> we briefly discussed neural networks and showed how they can be used to solve non-linear problems like the XOR problem. While we covered a solution for the XOR problem, we didn't cover how the weights of such a network can be derived. 

## The network

Lets say we again have the neural network that we used to solve the XOR problem. This network had an input layer with two nodes, one hidden layer with two nodes combined with the ReLU activation function and finally an output layer with one node and no activation function.

<div class="img-container">
<img src="./nn_xor.svg" alt="">
</div>

The output $a^{(l)}$ of each layer $l$ is defined as follows:

$$ a^{(1)} = g(W^{(1)T}x + b^{(1)}) $$

$$ a^{(2)} = \hat{y} = W^{(2)}h^{(1)} + b^{(2)} $$

## Finding the optimal weights using the MLE

Similar to other Machine Learning models discussed, we can use the MLE and its gradient to find suitable weights.
We will treat this problem as a regression problem, and therefore we can use the MSE as our loss function.

$$ MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $$

The optimization algorithm we will use later on, will process the data samply by sample. 
Therefore we will use the MSE per sample as our loss function. We scale by $1/2$ to make it easier to work with.

$$ L = \frac{1}{2} (y - \hat{y})^2 $$

## The backpropagation algorithm

Our goal is to find the derivatives of $\frac{\partial L}{\partial W^{(l)}}$ and $\frac{\partial L}{\partial b^{(l)}}$ for each layer $l$. Since all of these weights and biases are part of a network of computations, we need to work backwards from the output layer to the input layer to find these derivatives. When doing so, we will make use of the chain rule.  This is also known as **backpropagation**.

Let us first look at the derivative of the loss with respect to the output.

$$ \frac{\partial L}{\partial \hat{y}} =  \frac{\partial}{\partial \hat{y}} \left[ \frac{1}{2} (y - \hat{y})^2 \right] = \hat{y} - y $$

We can now use the chain rule to find the next derivative: the derivative of the loss with respect to the output layers' weights.

$$ \frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial W^{(2)}} = \hat{y} - y \cdot W^{(2)} $$

And we can do the same for the bias.

$$ \frac{\partial L}{\partial b^{(2)}} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial b^{(2)}} = \hat{y} - y \cdot 1 $$
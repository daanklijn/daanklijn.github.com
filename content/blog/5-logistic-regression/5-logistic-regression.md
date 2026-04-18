---
title: Logistic Regression
date: 2026-04-18
tags: Deep Learning, Machine Learning
---

{% set linearRegressionPost = "" %}
{% set mlePost = "" %}
{% set ridgePost = "" %}

{% for post in collections.posts %}
  {% if post.data.title == "Linear Regression" %}
    {% set linearRegressionPost = post.url %}
  {% endif %}
  {% if post.data.title == "Maximum Likelihood Estimation" %}
    {% set mlePost = post.url %}
  {% endif %}
  {% if post.data.title == "Ridge Regression" %}
    {% set ridgePost = post.url %}
  {% endif %}
{% endfor %}

Due to certain limitations, Linear Regression cannot be directly used to solve classification problems. Most notably, it's output is not guaranteed to be within a certain interval (e.g. $[0,1]$).

We can however modify the Linear Regression model to make it suitable for (binary) classification problems. We do this by wrapping the linear regression model with a sigmoid function. As the sigmoid function has a range of [0,1], we guarantee that the model's outputs are as well. This model can then be used as the probability parameter of a Bernoulli distribution. The model we end up with is known as the **Logistic Regression** model.

$$ y_i \sim Bernoulli(p_i)  \quad \text{where} \quad p_i = \sigma(\theta^Tx_i) $$

Similar to other models, we can use the MLE to find suitable weights.

## Deriving the negative log likelihood

Again, we define the negative log-likelihood function as:


$$J(\theta) = -\sum_{i=1}^n \log P(y_i \mid x_i; \theta)$$
$$ = -\sum_{i=1}^n \log \left( p_i^{y_i} (1 - p_i)^{1 - y_i} \right)$$
$$ = -\sum_{i=1}^n \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

We now find the gradient of the (negative) log-likelihood with respect to $\theta$.

$$ \nabla_\theta J = \frac{\partial}{\partial \theta} -\sum_{i=1}^n \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] $$

We apply the chain rule where $p_i = \sigma(z_i)$ and $z_i = \theta^T x_i$.

$$\nabla_\theta J = \frac{\partial J}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial \theta}$$

$$\nabla_\theta J = -\left( \frac{y_i}{p_i} - \frac{1 - y_i}{1 - p_i} \right) \cdot \left( p_i(1 - p_i) \right) \cdot x_i$$
$$\nabla_\theta J = -\left( \frac{y_i(1 - p_i) - p_i(1 - y_i)}{\cancel{p_i(1 - p_i)}} \right) \cdot \cancel{p_i(1 - p_i)} \cdot x_i$$
$$\nabla_\theta J = -(y_i - y_i p_i - p_i + y_i p_i)x_i$$
$$\nabla_\theta J = -(y_i - p_i)x_i = (p_i - y_i)x_i$$
$$\nabla_\theta J = (\sigma(w^Tx_i) - y_i)x_i$$

## Optimizing the weights

In contrast to the linear regression model, we cannot algebraically derive the weights $\theta$ that maximize the likelihood since $\nabla_\theta \ell = 0$ is not solvable due to the sigmoid term.

Instead, we will take a numerical approach and use gradient descent to find the optimal weights.

We define the gradient descent like this:

$$\theta^{(next)} = \theta^{(old)} - \alpha \nabla_\theta \ell(\theta)$$

Where $\alpha$ is the learning rate that determines how fast the weights are updated.

## Rough Python implementation


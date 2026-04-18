---
title: Bayesian Linear Regression
date: 2026-04-17
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

Previously, we discussed <a href={{linearRegressionPost}}>linear regression</a>, how we find its weights and the relation to <a href={{mlePost}}>Maximum Likelihood Estimation</a>. Both discussed approaches are more or less frequentist in nature. This article instead discusses a Bayesian approach to the same problem.

Lets again look at the linear regression model.

$$ y_i = w^Tx_i + \epsilon \quad \text{where} \quad \epsilon \sim N(0, \sigma^2) $$
$$ y_i \sim N(w^Tx_i, \sigma^2) $$

While frequentist approaches assume that the parameters of the distribution are fixed, Bayesian approaches instead assume that the parameters are random variables. In addition to that, Bayesians also take into account prior knowledge about the parameters. To incorporate prior knowledge, Bayesians use a different framework than MLE, combining a prior distribution with the likelihood to produce a posterior distribution.

For our specific case we will let the prior distribution be a normal distribution with mean $0$ and variance $\tau^2$.

$$P(w_i) \sim N(0, \tau^2)$$

Now that we have defined the prior distribution and know the likelihood, we can use **Bayes' theorem** to describe the posterior distribution.

$$P(w \mid y, X) = \frac{P(y \mid X, w) P(w)}{P(y \mid X)}$$

## Maximizing the posterior


Similar to MLE, once we know the posterior distribution, we can use it to find the weights that maximize it.
This maximum is also called the Maximum A Posteriori (MAP) estimate. Since the denominator $P(y \mid X)$ (the evidence) does not depend on $w$, we focus on the numerator:

$$w_{MAP} = \arg\max_{w}  P(y \mid X, w) P(w) $$

And similar to MLE, maximizing the log of the posterior is usually easier to do.

$$w_{MAP} = \arg\max_{w} \log  P(y \mid X, w) P(w) $$
$$ = \arg\max_{w} \log  P(y \mid X, w) + \log P(w) $$

We already know $\log P(y \mid X, w)$, as we previously derived this.

$$ \log P(y \mid X, w) = -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - w^T x_i)^2$$

The other term $\log P(w)$ can be defined based on the prior distribution.

$$\log P(w) = -\frac{n}{2} \log(2\pi\tau^2) - \frac{1}{2\tau^2} \sum_{i=1}^n w_i^2$$

Combining the two terms, and removing the terms that don't depend on $w$, we get the following expression.

$$w_{MAP} = \arg\max_{w} - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - w^T x_i)^2 - \frac{1}{2\tau^2} \sum_{i=1}^n w_i^2$$

When written in matrix form, this becomes:

$$w_{MAP} = \arg\max_{w} -\frac{1}{2\sigma^2} \|y - Xw\|^2_2 - \frac{1}{2\tau^2} \|w\|^2_2$$

If we flipr the objective function, we end up with:

$$w_{MAP} = \arg\min_{w} -\frac{1}{2\sigma^2} \|y - Xw\|^2_2 - \frac{1}{2\tau^2} \|w\|^2_2$$

We now find that this is is equivalent to the optimization problem of <a href={{ridgePost}}>Ridge Regression</a> where $\lambda =  \frac{1}{2\tau^2}$. Both Ridge Regression and Bayesian Linear Regression solve the same problem, but their underlying philosophy is different.

Both the regularization term and the prior distribution enforce a constraint on the weights that forces them to be small:
-  a large $\lambda$ forces the weights to be smaller
-  a small $\tau$ forces the prior distribution to be thinner and therefore also forces the weights to be smaller.

When the prior distribution is changed, the resulting regularization term changes as well. When a Laplace prior is used, we end up with Lasso regression. 

Bayesian Linear Regression is therefore just another way to add regularization to linear regression. 

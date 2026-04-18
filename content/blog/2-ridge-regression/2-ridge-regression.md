---
title: Ridge Regression
description: 
date: 2026-04-04
tags: Deep Learning, Machine Learning
---


{% set linearRegressionPost = "" %}

{% for post in collections.posts %}
  {% if post.data.title == "Linear Regression" %}
    {% set linearRegressionPost = post.url %}
  {% endif %}
{% endfor %}

In the previous post, we discussed <a href="{{ linearRegressionPost }}">Linear Regression</a> and implemented a simple variant of it in Python. When we increase the number of polynomials, the resulting model becomes increasingly complex, even when the true underlying function is simple. Below you can see data coming from the function $y = 2x + 5$ . The optimal weights however result in a very complex model that clearly overfits the training data.

<div class="img-container">
<img src="./graph_2.png" alt="Polynomial regression" >
</div>

When inspecting the weights of the model, we find that the optimal weights that were found by the linear regression algorithms are very large, especially compared to the function $y = 2x + 5$ that created the data.

```python
>>> weights
array([ 1.56753280e+01,  3.85287994e+01, -8.31689769e+01,  7.06624600e+01,
       -3.00112062e+01,  7.08036835e+00, -9.56975847e-01,  7.20111167e-02,
       -2.68660948e-03,  3.48744165e-05])
```

## Regularization

To counter this effect, we can add a regularization term to the loss function $J$. This term will punish the model for having large weights. There are many different regularization techniques, but for now we will focus on L2 regularization. The combination of L2 regularization and linear regression is known as **Ridge regression**.

As can be seen in the formula below, we combine the MSE with the L2 norm of the weights together with a small constant $\lambda$. As $\lambda$ increases, the model will be penalized more for having large weights.

$$J(w) = MSE + \lambda \| w \|_2^2$$

We can rewrite the loss function as follows:

$$J(w) = \|y - Xw\|^2 + \lambda \|w\|^2$$
$$ = (y - Xw)^T(y - Xw) + \lambda w^T w$$
$$ = y^T y - 2w^T X^T y + w^T X^T X w + \lambda w^T w$$

Similar to regular linear regression, we can find the optimal weights by finding the derivative of the loss function with respect to w and set it to zero.

$$\nabla_w J(w) = -2X^T y + 2X^T X w + 2\lambda w = 0$$
$$w = (X^T X + \lambda I)^{-1} X^T y$$

Now lets implement this in Python and see if it improves the model.

## Rough Python implementation







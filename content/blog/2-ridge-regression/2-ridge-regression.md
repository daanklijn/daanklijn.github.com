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
<img src="./ridge.png" alt="Polynomial regression" >
</div>
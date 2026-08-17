---
title: Convolutional Neural Networks
date: 2026-08-17
tags: Deep Learning, Machine Learning, Computer Vision
id: 13
---

So far, we have only discussed networks that are trained on data that is 1-dimensional and independent of order.
Meaning that is does not matter if we swap the first and second feature, the model will still learn the same weights.

Image data on the other hand is 2-dimensional (or even three-dimensional if we have channel to represent color). This data is also ordered, meaning that the position of the feature is important. Swapping the position of features will mess up the image. 

Regular neural networks are (or were) not well suited for this kind of data:
- A reasonably sized image can easily contain millions of features (pixels), if we were to add a fully connected layer on top of this, we would easily have a model with hundreds of millions of parameters. Especially in early days of neural networks, this was a major limitation.
- Regular neural network don't have a notion of spatial relationships between features. The weights in a fully connected layer don't have spatial information by default, and therefore can't be used to learn spatial relationships.
- In many image recognition tasks, we don't care where a certain object is in the image, we just care about it's presence. This raises the question if it makes sense to learn separate weights for each pixel in the image. We would need to learn very similar features for each pixel, which would for some part be a waste of computational resources.

Convolutional Neural Networks aimed to solve these problems by incorporating a layer of so-called Convolutions into the network. These convolutions are 2D (or higher dimensional) sets of weights (kernels) that are applied to the image in a window-like fashion. As they are not flat, they can be used to learn spatial relationships between features. Similarly their sliding behaviour allows them to share weights between different parts of the image, therefore becoming invariant to position. And finally, since we reuse weights across many different pixels, we can also greatly reduce the number of parameters in the network.

A single-channel convolution is defined as follows:

$$ Y_{i,j} = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} X_{(i \cdot S + m),\,(j \cdot S + n)} \cdot W_{m,n} + b $$

In this formula, $Y_{i,j}$ is the output value at position $(i,j)$ in the feature map; $X$ is the input image, indexed by row and column position. $W$ is the kernel, a matrix of $K \times K$ learnable weights. $W$ slides across the input for each $i$ and $j$. On top of that there is $S$, which is the stride, allowing us how far the kernel should slide across the input in every step. And finally $b$ is a bias term that is added to the output.

This sliding behaviour is visually illustrated in the animation below.

{% include "conv_animation.html" %}

Multiple convolutions are usually combined into a single layer, allowing us to learn multiple features at once. In addition to that it is also possible to stack multiple convolutioal layers on top of each other, allowing us to learn more complex/higher-level features. An example of a simple convolutional neural network is shown below.

{% include "cnn.html" %}

Now that we understand the basics of convolutional neural networks, lets go build our own!

## Building a simple convolutional neural network

To build a simple convolutional neural network, we will extend the Deep Learning library we have built in a previous post.
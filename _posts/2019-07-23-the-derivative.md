---
layout:     post
title:      The Derivative
date:       2019-07-23 12:22:18
summary:    Learn about derivatives and gradients.
categories: machinelearning basics
---

> "Whatcha doing George?" "Oh nothing Lenny, just working out some gradients." "On paper? I'm not sure if you'll be able to call `loss.backward()` there.   

---

Machine learning, especially deep learning, is built almost entirely on differentiation. In this post, we will briefly describe differentiation, derivatives, and gradients. From there, we will continue on to discuss their connection to the underlying core idea behind many popular deep learning frameworks: automatic differentiation.

<a href="https://colab.research.google.com/drive/1pD4Lnr1Gs8S3KiUT86D1zZzpWlHOBWRQ" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Derivatives

While derivatives can be explained and generalized to quite [a](#TODO) [few](#TODO) [domains](#TODO), we will start with the most elementary formulation: in the case of a function of a single, real-valued variable.

Concretely, a derivative in this setting measures sensitivity of change of the output with respect to its input: graphically in one dimension, this is the slope of the tangent line at $$f(x)$$; in physics, we often hear that the rate of change in position (output) with respect to time (input) is _velocity_, and similarly for velocity and acceleration.

![](/images/the-derivative/sincos.gif)

Derivatives can be generalized to both higher orders and higher dimensions. Higher orders of derivatives are just differentiating with respect to the input once again; for example, the second order derivative of position with respect to time brings us back to acceleration.

To understand higher dimensional derivatives, we can introduce _partial derivatives_. When given vector-values input $$\vec{x}$$, a partial derivative is just a derivative of the output with respect to a particular _component_ of $$\vec{x}$$. When dealing with scalar-valued functions (i.e the output is a scalar, or notationally, $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$), the vector of first order partial derivatives is called the _gradient_. If the output itself is vector-valued, the matrix of first-order partial derivatives is called the _Jacobian_; in fact, the Jacobian is just a generalization of the gradient.

Just as the derivative of a real-valued function is the slope of the tangent line, the gradient's direction is the _direction of greatest increase_, where the rate of that increase is given by the magnitude. Often, we will use the notation $$\nabla f(\vec{x})$$ to denote the gradient notationally.

In machine learning, we are generally concerned with scalar-valued functions, so we focus our attention mostly on the gradient. While higher-order derivatives of scalar-valued functions are used in machine learning (the matrix of second-order derivatives of a vector-valued function is called the _Hessian_, and when the derivatives are taken with respect to parameters instead of the inputs,the matrix is called the _Fisher Information Matrix_), they deserve to be covered in their own post, and are not central to discussion here.

## Machine Learning and Gradients

In machine learning, we care about _objective functions_ and how to optimize them, whether we are working with supervised, deep, reinforcement, ... learning. To optimize a function, we look for _optima_, or where $$\nabla f(\vec{x}) = 0$$ and the second-derivative test passes with the condition we're looking for. 

While generally, we can't make the assumption that our objective function is convex (which means that we can't guarantee that any local optima are globally optimal), we can still use gradients to find local minima (or maxima, depending on the problem we're trying to solve).

This gives rise to algorithms like _gradient descent_ (or more popularly, the more efficient _stochastic_ version); since we know that our gradient gives us the direction of fastest increase, we (to minimize an objective) can follow it in the ecact opposite direction.

While just following gradients can get us into trouble as well (saddles also have $$\nabla f(\vec{x}) = 0$$ just like optima, and unfortunately, it seems that [saddles are ubiquitous in nonconvex optimization](#TODO); SGD is generally considered to be effective at escaping these saddles, but the dynamics of gradient descent algorithms is still an [active area of research](#TODO)), we defer those discussions to a later post.

## Calculating Gradients, Numerically

To understand the numerical calculation of gradients, we must refer back to the definition of derivatives, in terms of limits:

$$ f'(a) = \lim_{h\rightarrow 0} \frac{f(a + h) - f(a)}{h} $$

However, since limits are a theoretical tool, they offer us no help in the calculation of gradients in practice.

Numerically, we can replace the infinitesimal $$h$$ with an arbitrarily small one, leading us to approximate the derivative as:

$$ f'(a) \approx \lim_{h\rightarrow 0} \frac{f(a + h) - f(a)}{h} $$

known as a _first-order divided difference_.

Numerical differentiation (known as _differential quadrature_) are still the leading methods to solve partial differential equations (and are used to calculate derivatives in your favorite graphing calculator as well!), they incur errors on an order of magnitude of $$h$$; first-order divided differences incurs errors on the order of $$O(h)$$, and while improved methods such as symmetric divided difference can lower this, differential quadrature methods can be prone to floating point approximation errors and implementation efficiency issues.

## Calculating Gradients, Automagically

### From Symbolic to Automatic

### The Chain Rule

### Sum of the Parts are Greater than the Whole

A section on how simple arithmetic operations and simple functions can be composed to efficiently calculate derivatives.

### Forward Mode and Backward Mode
 
 
### Linear Regression from an AD Perspective
 
Recall the matrix equation for linear regression, where $$\mathbf{X}: \mathbb{R}^{m \times n}$$ and $$\mathbf{\Theta}: \mathbb{R}^{n \times 1}$$:
 
$$\mathbf{\hat f}(\mathbf{X}; \mathbf{\Theta}) = \mathbf{X}\mathbf{\Theta}$$
 
Imagine we are given a training set $$\mathbf{X} = \{\mathbf{x}^{(0)}, \ldots, \mathbf{x}^{(n)}\}, \mathbf{Y} = \{\mathbf{y}^{(0)}, \ldots, \mathbf{y}^{(n)}\}$$. Our goal in OLS linear regression is to solve the following optimization problem:

$$\mathbf{\Theta}^* = \underset{\mathbf{\Theta}}{\operatorname{argmin}}||\mathbf{Y} - \mathbf{\hat f}(\mathbf{X}; \mathbf{\Theta})||^2$$

Assuming $$\mathbf{X}^\intercal\mathbf{X}$$ is invertible, this can be solved directly, using the normal equation:

$$\mathbf{\Theta}^* = (\mathbf{X}^\intercal\mathbf{X})^{-1}\mathbf{X}^\intercal\mathbf{Y}$$

However this requires computing $$(\mathbf{X}^\intercal\mathbf{X})^{-1}$$ which is $$\mathcal{O}(m^{2.373})$$ to the best of our knowledge. Another solution is to use the gradient:

$$\nabla_\mathbf{\Theta} ||\mathbf{Y} - \mathbf{X}\mathbf{\Theta}||^2$$

First, let us consider the scalar case, where $$f(x; \theta_0, \theta_1) = \theta_0 x + \theta_1$$:

$$\mathcal{L}(\mathbf{\Theta}) = \mathcal{L}(\theta_0, \theta_1) = \frac{1}{n}\sum_{i=0}^n(y_i - (\theta_0 x_i + \theta_1))^2$$

We would like to find $$\nabla_\mathbf{\Theta}\mathcal{L} = \lbrack \frac{\partial\mathcal{L}}{\partial \theta_0}, \frac{\partial\mathcal{L}}{\partial \theta_1}\rbrack$$. There are two ways to do this, using the finite difference method, and using automatic differentiation. Let's see the finite difference method with centered differences:

$$\frac{\partial\mathcal{L}}{\partial \theta_0} = \frac{\frac{1}{n}\sum_{i=0}^n(y_i - ((\theta_0 + h) x_i + \theta_1))^2 - \frac{1}{n}\sum_{i=0}^n(y_i - ((\theta_0 - h) x_i + \theta_1))^2}{2h}$$

$$\frac{\partial\mathcal{L}}{\partial \theta_1} = \frac{\frac{1}{n}\sum_{i=0}^n(y_i - (\theta_0 x_i + \theta_1 + h))^2 - \frac{1}{n}\sum_{i=0}^n(y_i - (\theta_0 x_i + \theta_1 - h))^2}{2h}$$

Alternatively, we can calculate the partials analytically, using the chain rule:

$$\begin{align}\frac{\partial\mathcal{L}}{\partial \theta_0} & = \frac{1}{n}\sum_{i=0}^n(y_i - (\theta_0 x_i + \theta_1))^2 \\\\ & = \frac{2}{n}\sum_{i=0}^n(y_i - (\theta_0 x_i + \theta_1))(-x_i)\end{align}$$

$$\begin{align}\frac{\partial\mathcal{L}}{\partial \theta_1} & = \frac{1}{n}\sum_{i=0}^n(y_i - (\theta_0 x_i + \theta_1))^2 \\\\ & = \frac{2}{n}\sum_{i=0}^n(y_i - (\theta_0 x_i + \theta_1))(-1)\end{align}$$

TODO: Expand on AD graph
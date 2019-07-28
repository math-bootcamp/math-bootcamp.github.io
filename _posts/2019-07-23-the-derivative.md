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
 
Imagine we are given the following dataset:

$$
\mathbf{X} = 
\begin{bmatrix}
\mathbf{x}_1 \\
\vdots \\
\mathbf{x}_m
\end{bmatrix} =
\begin{pmatrix}
1 & \ldots & x_{0n} \\
\vdots & \ddots & \vdots \\
1 & \ldots & x_{mn}
\end{pmatrix}, 
\mathbf{Y} = 
\begin{bmatrix}
y_1 \\
\vdots \\
y_m
\end{bmatrix}$$

Our goal in ordinary least squares (OLS) linear regression is to minimize the loss, or error between the data and the model's prediction:

$$\mathcal{L}(\mathbf{X}, \mathbf{Y}; \mathbf{\Theta}) = ||\mathbf{Y} - \mathbf{\hat f}(\mathbf{X}; \mathbf{\Theta})||^2$$

$$\mathbf{\Theta}^* = \underset{\mathbf{\Theta}}{\operatorname{argmin}}\mathcal{L}(\mathbf{X}, \mathbf{Y}; \mathbf{\Theta})$$

#### Finite Difference Method

First, we consider the scalar case, where $$\mathbf{\hat f}(\mathbf{X}; \mathbf{\Theta}) = \hat f(x; \theta_1, \theta_0) = \theta_1 x + \theta_0$$. Since $$\mathbf{X}, \mathbf{Y}$$ are considered to be fixed, we can rewrite $$\mathcal{L}(\mathbf{X}, \mathbf{Y}; \mathbf{\Theta})$$ as simply:

$$\mathcal{L}(\mathbf{\Theta}) = \mathcal{L}(\theta_1, \theta_0) = \frac{1}{m}\sum_{i=0}^m(y_i - (\theta_1 x_i + \theta_0))^2$$

To find the minimizer of $$\mathcal{L}(\mathbf{\Theta})$$, we need $$\nabla_\mathbf{\Theta}\mathcal{L} = \lbrack \frac{\partial\mathcal{L}}{\partial \theta_1}, \frac{\partial\mathcal{L}}{\partial \theta_0}\rbrack$$. There are various ways to compute this, of which we consider two: (1) [the finite difference method](https://en.wikipedia.org/wiki/Finite_difference) (FDM), and (2) symbolic differentiation. First, let's see FDM with centered differences:

$$\begin{align}
\frac{\partial\mathcal{L}}{\partial \theta_0} & = \underset{h \rightarrow 0}{\operatorname{lim}} \frac{\sum_{i=0}^m\left(y_i - \left(\theta_1 x_i + \theta_0 + h\right)\right)^2 - \sum_{i=0}^m\left(y_i - \left(\theta_1 x_i + \theta_0 - h\right)\right)^2}{2hm} \\\\ & = \underset{h \rightarrow 0}{\operatorname{lim}} \frac{1}{2hm}\sum_{i=0}^m\left(y_i - \left(\theta_1 x_i + \theta_0 + h\right)\right)^2 - \left(y_i - \left(\theta_1 x_i + \theta_0 - h\right)\right)^2 \\\\ 
\frac{\partial\mathcal{L}}{\partial \theta_1} & = \underset{h \rightarrow 0}{\operatorname{lim}} \frac{\sum_{i=0}^m\left(y_i - \left((\theta_1 + h) x_i + \theta_0\right)\right)^2 - \sum_{i=0}^m\left(y_i - \left(\left(\theta_1 - h\right) x_i + \theta_0\right)\right)^2}{2hm} \\\\ & = \underset{h \rightarrow 0}{\operatorname{lim}} \frac{1}{2hm}\sum_{i=0}^m\left(y_i - \left(\left(\theta_1 + h\right) x_i + \theta_0\right)\right)^2 - \left(y_i - \left(\left(\theta_1 - h\right) x_i + \theta_0\right)\right)^2
\end{align}$$

Using [computer](https://www.wolframalpha.com/input/?i=(y_i-((%CE%B8_1%2Bh)x_i%2B%CE%B8_0))%5E2-(y_i-((%CE%B8_1-h)x_i%2B%CE%B8_0))%5E2) [algebra](https://www.wolframalpha.com/input/?i=(y_i-(%CE%B8_1*x_i%2B%CE%B8_0%2Bh))%5E2%E2%88%92(y_i-(%CE%B8_1*x_i%2B%CE%B8_0-h))%5E2), these equations can be simplified considerably:

$$\begin{align}
\frac{\partial\mathcal{L}}{\partial \theta_0} & = \underset{h \rightarrow 0}{\operatorname{lim}} \frac{1}{2hm}\sum_{i=0}^m\left(4h ( \theta_0 +  \theta_1 x_i - y_i)\right) \\\\ & = \boxed{\frac{2}{m}\sum_{i=0}^m\left(\theta_0 +  \theta_1 x_i - y_i\right)} \\\\
\frac{\partial\mathcal{L}}{\partial \theta_1} & = \underset{h \rightarrow 0}{\operatorname{lim}} \frac{1}{2hm}\sum_{i=0}^m\left(4hx_i (\theta_1 x_i + \theta_0 - y_i)\right) \\\\ & = \boxed{\frac{2}{m}\sum_{i=0}^m(x_i)(\theta_1 x_i + \theta_0 - y_i)}
\end{align}$$

#### Partial Differentiation

Alternatively, we can calculate the partials analytically, by applying the chain rule:

$$\begin{align}\frac{\partial\mathcal{L}}{\partial \theta_0} & = \frac{\partial}{\partial \theta_0}\frac{1}{m}\sum_{i=0}^m(y_i - (\theta_1 x_i + \theta_0))^2 \\\\ & = \frac{1}{m}\sum_{i=0}^m 2 (y_i - (\theta_1 x_i + \theta_0))\frac{\partial}{\partial \theta_0}(y_i - (\theta_1 x_i + \theta_0)) \\\\ & = \frac{2}{m}\sum_{i=0}^m(y_i - (\theta_1 x_i + \theta_0))(-1) \\\\ & = \boxed{\frac{2}{m}\sum_{i=0}^m(\theta_1 x_i + \theta_0 - y_i)}\end{align}$$

$$\begin{align}\frac{\partial\mathcal{L}}{\partial \theta_1} & = \frac{\partial}{\partial \theta_1}\frac{1}{m}\sum_{i=0}^m(y_i - (\theta_1 x_i + \theta_0))^2 \\\\ & = \frac{1}{m}\sum_{i=0}^m 2(y_i - (\theta_1 x_i + \theta_0)) \frac{\partial}{\partial \theta_1}(y_i - (\theta_1 x_i + \theta_0)) \\\\ & = \frac{2}{m}\sum_{i=0}^m(y_i - (\theta_1 x_i + \theta_0))(-x_i) \\\\ & = \boxed{\frac{2}{m}\sum_{i=0}^m(x_i)(\theta_1 x_i + \theta_0 - y_i)}\end{align}$$

Notice how analytical differentiation gives us the same answer as the finite difference method (this is not by accident), with much less algebra. We can rewrite these two solutions in gradient form, i.e. as a column vector of partial derivatives:
 
$$
\nabla_\mathbf{\Theta}\mathcal{L} =
\begin{bmatrix}
     \frac{\partial\mathcal{L}}{\partial \theta_0} \\
     \frac{\partial\mathcal{L}}{\partial \theta_1}
 \end{bmatrix} = \frac{2}{m} 
\begin{bmatrix}
\sum_{i=0}^m(\theta_1 x_i + \theta_0 - y_i) \\\\ \sum_{i=0}^m(x_i)(\theta_1 x_i + \theta_0 - y_i)
\end{bmatrix}
$$
 
#### Matrix solutions

Having reviewed the scalar procedure for linear regression, let us now return to the general form of $$\mathcal L(\mathbf{\Theta})$$. Matrix notation allows us to simplify the loss considerably:

$$\begin{align}\mathcal L(\mathbf{\Theta}) & = \frac{1}{m} (\mathbf Y - \mathbf X \mathbf \Theta)^\intercal(\mathbf Y - \mathbf X \mathbf \Theta) \\\\ &= \frac{1}{m} (\mathbf Y^\intercal \mathbf Y - \mathbf Y^\intercal \mathbf X \mathbf \Theta - \mathbf \Theta^\intercal \mathbf X^\intercal \mathbf Y + \mathbf \Theta^\intercal \mathbf X^\intercal \mathbf X \mathbf \Theta) \\\\ &= \frac{1}{m} (\mathbf Y^\intercal \mathbf Y - 2 \mathbf \Theta^\intercal \mathbf X^\intercal \mathbf Y + \mathbf \Theta^\intercal \mathbf X^\intercal \mathbf X \mathbf \Theta)\end{align}$$

Matrix notation allows us to derive the gradient and requires far less algebra:

$$\begin{align}\nabla_{\mathbf{\Theta}}\mathcal L(\mathbf{\Theta}) & = \frac{1}{m} (\nabla_{\mathbf{\Theta}}\mathbf Y^\intercal \mathbf Y - 2 \nabla_{\mathbf{\Theta}} \mathbf \Theta^\intercal \mathbf X^\intercal \mathbf Y + \nabla_{\mathbf{\Theta}}\mathbf \Theta^\intercal \mathbf X^\intercal \mathbf X \mathbf \Theta) \\\\ & = \frac{1}{m} ( 0 - 2\mathbf{X}^\intercal \mathbf Y + 2 \mathbf{X}^\intercal \mathbf X \mathbf \Theta ) \\\\ & = \boxed{\frac{2}{m} (\mathbf{X}^\intercal \mathbf X \mathbf \Theta - \mathbf{X}^\intercal \mathbf Y)}\end{align}$$

For completeness, and to convince ourselves the matrix solution is indeed the same:

$$
\begin{align}
& = \frac{2}{m}\left(
 \underbrace{\begin{bmatrix}
    1 & \ldots & 1 \\
    x_0 & \ldots & x_m \\
 \end{bmatrix}}_{\mathbf{X}^\intercal}
 \underbrace{\begin{bmatrix}
    1 & x_0 \\
    \vdots & \vdots \\
    1 & x_m
 \end{bmatrix}}_{\mathbf{X}}
 \underbrace{\begin{bmatrix}
     \theta_0 \\
     \theta_1
 \end{bmatrix}}_{\mathbf{\Theta}} - 
 \underbrace{\begin{bmatrix}
    1 & \ldots & 1 \\
    x_0 & \ldots & x_m \\
 \end{bmatrix}}_{\mathbf{X}^\intercal}
  \underbrace{\begin{bmatrix}
      y_0 \\
      \vdots \\
      y_m
   \end{bmatrix}}_{\mathbf{Y}}\right) \\\\
& = \frac{2}{m}\left(
 \underbrace{\begin{bmatrix}
    m & \sum_{i=0}^{m}x_i \\
    \sum_{i=0}^{m}x_i & \sum_{i=0}^{m}x_i^2 \\
 \end{bmatrix}}_{\mathbf{X}^\intercal\mathbf{X}}
 \underbrace{\begin{bmatrix}
     \theta_0 \\
     \theta_1
 \end{bmatrix}}_{\mathbf{\Theta}} - 
 \underbrace{\begin{bmatrix}
      \sum_{i=0}^{m}y_i \\
      \sum_{i=0}^{m}x_iy_i
   \end{bmatrix}}_{\mathbf{X}^\intercal\mathbf{Y}}\right) \\\\
& = \frac{2}{m}\left(
 \underbrace{\begin{bmatrix}
     m \theta_0 + \sum_{i=0}^{m}\theta_1x_i \\
     \sum_{i=0}^{m}\theta_0x_i + \sum_{i=0}^{m}\theta_1x_i^2
 \end{bmatrix}}_{\mathbf{X}^\intercal\mathbf{X}\mathbf{\Theta}} - 
 \underbrace{\begin{bmatrix}
      \sum_{i=0}^{m}y_i \\
      \sum_{i=0}^{m}x_iy_i
   \end{bmatrix}}_{\mathbf{X}^\intercal\mathbf{Y}}\right) \\\\
& = \boxed{\frac{2}{m}
 \underbrace{\begin{bmatrix}
     \sum_{i=0}^{m}\theta_1x_i + \theta_0 - y_i \\
     \sum_{i=0}^{m}(x_i)(\theta_1x_i + \theta_0 - y_i)
 \end{bmatrix}}_{\mathbf{X}^\intercal\mathbf{X}\mathbf{\Theta} - \mathbf{X}^\intercal\mathbf{Y}} = 
\begin{bmatrix}
     \frac{\partial\mathcal{L}}{\partial \theta_0} \\
     \frac{\partial\mathcal{L}}{\partial \theta_1}
 \end{bmatrix} = \nabla_{\mathbf{\Theta}}\mathcal{L}(\mathbf{\Theta})}
\end{align}
$$

Notice how we recover the same solution obtained from partial differentiation and finite difference approximation, albeit in a more compact form. For a good introduction to matrix calculus, the textbook by Magnus & Neudecker (2007) [^4] is an excellent guide, of which Petersen & Pedersen (2012) [^2] and Parr & Howard (2018) [^3] offer a review of important identities. If $$\mathbf X^\intercal \mathbf X$$ is invertible, i.e. full-rank, implies a unique solution $$\mathbf \Theta^*$$, which we can solve for directly by setting $$\nabla_{\mathbf{\Theta}}\mathcal{L} = \mathbf{0}$$:

$$\begin{align}0 & = \mathbf X^\intercal \mathbf X \mathbf \Theta - \mathbf X ^ \intercal \mathbf Y \\\\ \mathbf \Theta &= (\mathbf X^\intercal \mathbf X)^{-1}\mathbf X^\intercal\mathbf Y \end{align}$$

However this requires computing $$(\mathbf{X}^\intercal\mathbf{X})^{-1}$$ which is at least $$\mathcal{O}(n^{2.373})$$[^1] to the best of our knowledge, i.e. quadratic with respect to the number of input dimensions. Another way to find $$\mathbf \Theta^*$$ is by initializing $$\mathbf{\Theta} \leftarrow \mathbf{0}$$ and repeating the following procedure until convergence:

$$\mathbf{\Theta}' \leftarrow \mathbf{\Theta} - \alpha \nabla_{\mathbf{\Theta}}\mathcal L(\mathbf{\Theta})$$

Typically, $$\alpha \in [0.001, 0.1]$$. Although hyperparameter tuning is required to find a suitable $$\alpha$$ (various improvements like [Nesterov momentum](http://mitliagkas.github.io/ift6085/ift-6085-lecture-6-notes.pdf) and [quasi-Newton methods](https://en.wikipedia.org/wiki/Quasi-Newton_method) also help to accelerate convergence), this procedure is guaranteed to be computationally more efficient than matrix inversion for sufficiently large $$m$$ and $$n$$. In practice, the normal equation is seldom used unless $$m$$ is very small.

[^1]: [Coppersmith–Winograd algorithm](https://en.wikipedia.org/wiki/Coppersmith%E2%80%93Winograd_algorithm)
[^2]: [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
[^3]: [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/index.html)
[^4]: [Matrix Differential Calculus with Applications in Statistics and Econometrics](https://www.amazon.ca/Differential-Calculus-Applications-Statistics-Econometrics/dp/1119541204)
# Wave-propagation
## 1. Physics-Informed Neural Networks (1D Wave) 
### Problem Statement
**The 1D wave equation is given by:**

$$
\frac{\partial^2 u(x,t)}{\partial t^2} = c^2 \frac{\partial^2 u(x,t)}{\partial x^2} \text{ for } 0 < x < 1 \text{ and }  t > 0
$$

where \( u(x,t) \) represents the wave function, and \( c \) is the wave speed.

**The boundary conditions for the problem are:**

$$
u(0,t) = u(1,t) = 0 \text{ for } t > 0
$$

**The initial conditions are:**

$$
u(x,0) = x(1 - x) \text{ for } 0 < x < 1
$$

$$
\frac{\partial u(x,0)}{\partial t} = 0 \text{ for } t > 0
$$
### Solution Approach

**Separation of Variables**

Assume a solution of the form:

$$
u(x, t) = X(x) T(t)
$$

Substitute this into the PDE:

$$
X(x) \frac{d^2 T(t)}{dt^2} = c^2 T(t) \frac{d^2 X(x)}{dx^2}
$$

Dividing both sides by \(X(x) T(t)\):

$$
\frac{1}{T(t)} \frac{d^2 T(t)}{dt^2} = c^2 \frac{1}{X(x)} \frac{d^2 X(x)}{dx^2} = -\lambda
$$

where $\lambda is a separation constant.

This leads to two ordinary differential equations:

1. For \(T(t)\):

$$
\frac{d^2 T(t)}{dt^2} + \lambda T(t) = 0
$$

2. For \(X(x)\):

$$
\frac{d^2 X(x)}{dx^2} + \frac{\lambda}{c^2} X(x) = 0
$$

**Solve for \(X(x)\)**

For the boundary conditions \(X(0) = 0\) and \(X(1) = 0\), the solution for \(X(x)\) is:

$$
X(x) = \sin(n \pi x)
$$

where \(\lambda = (n \pi)^2\) and \(n\) is a positive integer. This choice satisfies the boundary conditions.

**Solve for \(T(t)\)**

Substitute \(\lambda = (n \pi)^2\):

$$
\frac{d^2 T(t)}{dt^2} + (n \pi c)^2 T(t) = 0
$$

The general solution for \(T(t)\) is:

$$
T(t) = A_n \cos(n \pi c t) + B_n \sin(n \pi c t)
$$

**Form the General Solution**

Combine \(X(x)\) and \(T(t)\):

$$
u(x, t) = \sum_{n=1}^{\infty} \left[ A_n \cos(n \pi c t) + B_n \sin(n \pi c t) \right] \sin(n \pi x)
$$

**Apply Initial Conditions**

Apply \(u(x, 0) = x (1 - x)\):

$$
u(x, 0) = \sum_{n=1}^{\infty} A_n \sin(n \pi x) = x (1 - x)
$$

To find \(A_n\), use the Fourier series expansion:

$$
A_n = \frac{2}{1 - 0} \int_{0}^{1} x (1 - x) \sin(n \pi x) \, dx
$$

Calculate this integral to get \(A_n\), but here you already know the provided solution form.

Apply \(\frac{\partial u}{\partial t} (x, 0) = 0\):

$$
\frac{\partial u}{\partial t} (x, 0) = \sum_{n=1}^{\infty} B_n (n \pi c) \sin(n \pi x) = 0
$$

Thus, \(B_n = 0\) for all \(n\).

Given \(c = 1\), the solution simplifies to:

$$
u(x, t) = \sum_{k=1, \text{ odd}}^{\infty} \frac{8}{k^3 \pi^3} \sin(k \pi x) \cos(k \pi t)
$$

## 2. FEniCS (2D wave propagation) 
## 3. ABAQUS( Lamb wave propagation)

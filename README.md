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
### Analytical solution

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

where \( $\lambda$ \) is a separation constant.

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

where \($\lambda$ = $(n \pi)^2$ \) and \(n\) is a positive integer. This choice satisfies the boundary conditions.

**Solve for \(T(t)\)**

Substitute  \($\lambda$ = $(n \pi)^2$ \):

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

To find \($A_n$ \), use the Fourier series expansion:

$$
A_n = \frac{2}{1 - 0} \int_{0}^{1} x (1 - x) \sin(n \pi x) \, dx
$$

Calculate this integral to get \(A_n\), but here you already know the provided solution form.

Apply \($\frac{\partial u}{\partial t} (x, 0) = 0$ \):

$$
\frac{\partial u}{\partial t} (x, 0) = \sum_{n=1}^{\infty} B_n (n \pi c) \sin(n \pi x) = 0
$$

Thus, \($B_n = 0$ \) for all \(n\).

Given \(c = 1\), the solution simplifies to:

$$
u(x, t) = \sum_{n=1, \text{ odd}}^{\infty} \frac{8}{n^3 \pi^3} \sin(n \pi x) \cos(n \pi t)
$$
### Results
![image](https://github.com/user-attachments/assets/0594fc5a-c8fd-437f-be24-ff0cd02d05d0)
**Fig 1. The comparison of PINNs results with the Analytical solution.**

## 2. FEniCS (2D wave propagation) 

The wave equation in two dimensions is defined as:

$$
\frac{\partial^2 u(x,y,t)}{\partial t^2} = c^2 \left( \frac{\partial^2 u(x,y,t)}{\partial x^2} + \frac{\partial^2 u(x,yt)}{\partial y^2} \right)
$$

where \( u(x,y,t) \) represents the wave function, and \( c \) is the wave speed.

**2.1 Time Discetization**

We first discretize the wave equation in time. Let \($\Delta t$ \) represent the time step. The second derivative with respect to time can be approximated using a finite difference scheme. For instance:

$$
\frac{\partial^2 u}{\partial t^2} \approx \frac{u^{n+1} - 2u^n + u^{n-1}}{\Delta t^2}
$$

Substituting this into the wave equation:

$$
\frac{u^{n+1} - 2u^n + u^{n-1}}{\Delta t^2} = c^2 \left( \frac{\partial^2 u^n}{\partial x^2} + \frac{\partial^2 u^n}{\partial y^2} \right)
$$

Multiplying both sides by \($\Delta t^2$ \):

$$
u^{n+1} - 2u^n + u^{n-1} = \Delta t^2 c^2 \left( \frac{\partial^2 u^n}{\partial x^2} + \frac{\partial^2 u^n}{\partial y^2} \right)
$$

**2.2 Weak Form Derivation**

Next, we multiply both sides by a test function \(v(x, y)\) and integrate over the spatial domain \($\Omega$ \) to derive the weak form.

$$
\int_\Omega v \left( u^{n+1} - 2u^n + u^{n-1} \right) \, d\Omega = \Delta t^2 c^2 \int_\Omega v \left( \frac{\partial^2 u^n}{\partial x^2} + \frac{\partial^2 u^n}{\partial y^2} \right) \, d\Omega
$$

**2.3 Applying Integration by Parts**

The term involving the spatial derivatives can be integrated by parts to shift the derivatives from \($u^n$ \) to the test function \(v\). Assuming Dirichlet boundary conditions:

$$
\int_\Omega v \left( u^{n+1} - 2u^n + u^{n-1} \right) \, d\Omega = -\Delta t^2 c^2 \int_\Omega \nabla v \cdot \nabla u^n \, d\Omega
$$

This equation represents the weak form of the time-discretized 2D wave equation.

Final Weak Form

In the notation used for finite element implementations:

- **Bilinear Form** \(a(u_h^n, v_h)\):

$$
a(u_h^n, v_h) = \int_\Omega v_h u_h^{n+1} \, d\Omega + \Delta t^2 c^2 \int_\Omega \nabla v_h \cdot \nabla u_h^n \, d\Omega
$$

- **Linear Form** \(L(v_h)\):

$$
L(v_h) = \int_\Omega v_h \left( 2u_h^n - u_h^{n-1} \right) \, d\Omega
$$

The initial conditions are given as:

$$
u(x, 0) = x(1-x) \quad \text{for all } 0 < x < 1
$$

$$
\frac{\partial u(x, 0)}{\partial t} = 0 \quad \text{for all } 0 < x < 1
$$

\subsection{Boundary Conditions}

The boundary conditions are defined as follows:

$$
u(0, t) = 0 \quad \text{and} \quad u(1, t) = 0 \quad \text{for all } t > 0
$$

Additionally, for a source point on the left edge (\( x = 0 \)), the boundary condition is:

$$
u(0, t) = c \cdot \sin(10 \cdot t)
$$

\end{document}

![t-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/fe962ebc-ea4a-44d2-a70f-c44e7998822a)

## 3. ABAQUS( Lamb wave propagation)

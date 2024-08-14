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
\frac{\partial^2 u(x,t)}{\partial t^2} = c^2 \left( \frac{\partial^2 u(x,t)}{\partial x^2} + \frac{\partial^2 u(x,t)}{\partial y^2} \right)
$$

where \( u(x,t) \) represents the wave function, and \( c \) is the wave speed.

### Deriving the Weak Form
The weak form is derived as follows:

1. Multiply by a test function \( v \):
   
$$
v \left( \frac{\partial^2 u}{\partial t^2} - c^2 \nabla^2 u \right) = 0
$$

2. Integrate over the domain \( $\Omega$ \):

$$
\int_\Omega v \frac{\partial^2 u}{\partial t^2} \, d\Omega - c^2 \int_\Omega v \nabla^2 u \, d\Omega = 0
$$

3. Apply integration by parts to the Laplacian term:
   
$$

\int_\Omega v \nabla^2 u \, d\Omega 
$$

= \int_\Omega \nabla v \cdot \nabla u \, d\Omega - \int_{\partial\Omega} v \nabla u \cdot n \, d\Gamma

   Assuming homogeneous Neumann boundary conditions (\( \nabla u \cdot n = 0 \) on \( \partial\Omega \)), the boundary integral vanishes.

6. The final weak form is:
   \[
   \int_\Omega v \frac{\partial^2 u}{\partial t^2} \, d\Omega + c^2 \int_\Omega \nabla v \cdot \nabla u \, d\Omega = 0
   \]

\end{document}

1. **Multiply by a Test Function**: Multiply the wave equation by a test function \( v(x, y) \) and integrate over the domain \( \Omega \):

    $$
    \int_\Omega v \frac{\partial^2 u}{\partial t^2} \, d\Omega = c^2 \int_\Omega v \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) \, d\Omega
    $$

2. **Apply Integration by Parts**: Use integration by parts to move the derivatives off the solution \( u \) and onto the test function \( v \). This is done to reduce the order of the derivatives on \( u \):

    $$
    \int_\Omega v \frac{\partial^2 u}{\partial t^2} \, d\Omega = -c^2 \int_\Omega \nabla u \cdot \nabla v \, d\Omega + c^2 \int_{\partial \Omega} v \frac{\partial u}{\partial n} \, d\Gamma
    $$

    Here, \( \nabla u \cdot \nabla v \) is the dot product of the gradients, and \( \frac{\partial u}{\partial n} \) represents the normal derivative on the boundary \( \partial \Omega \).

3. **Enforce Boundary Conditions**: The boundary integral term \( \int_{\partial \Omega} v \frac{\partial u}{\partial n} \, d\Gamma \) will vanish for Dirichlet boundary conditions where \( u \) is fixed on \( \partial \Omega \). Thus, the weak form simplifies to:

    $$
    \int_\Omega v \frac{\partial^2 u}{\partial t^2} \, d\Omega = -c^2 \int_\Omega \nabla u \cdot \nabla v \, d\Omega
    $$

4. **Final Weak Form**: The final weak form of the wave equation becomes:

    $$
    \text{Find } u \in H^1(\Omega) \text{ such that for all } v \in H^1(\Omega),
    $$

    $$
    \int_\Omega v \frac{\partial^2 u}{\partial t^2} \, d\Omega + c^2 \int_\Omega \nabla u \cdot \nabla v \, d\Omega = 0
    $$

\section{Initial and Boundary Conditions}

\subsection{Initial Conditions}

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

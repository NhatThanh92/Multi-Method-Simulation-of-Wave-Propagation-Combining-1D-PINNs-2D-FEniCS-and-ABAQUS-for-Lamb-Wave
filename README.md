# Wave-propagation
## 1. Physics-Informed Neural Networks (1D Wave) 
### Problem Statement
The 1D wave equation is given by:

$$
\frac{\partial^2 u(x,t)}{\partial t^2} = c^2 \frac{\partial^2 u(x,t)}{\partial x^2}
$$

where \( u(x,t) \) represents the wave function, and \( c \) is the wave speed.

The boundary conditions for the problem are:

$$
\begin{cases}
u(0,t) = 0 \\
u(1,t) = 0
\end{cases}
$$

The initial conditions are:

$$
u(x,0) = x(1 - x) \text{ for } 0 < x < 1 \\
\frac{\partial u(x,0)}{\partial t} = 0 \text{ for } t > 0
$$

## Domain

The problem is defined for:

- \( 0 < x < 1 \)
- \( t > 0 \)

The wave equation should be solved within these constraints.
## 2. FEniCS (2D wave propagation) 
## 3. ABAQUS( Lamb wave propagation)

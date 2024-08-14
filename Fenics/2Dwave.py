from dolfin import *
import numpy as np

# Parameters
c = 5000
mesh = RectangleMesh(Point(-1, -1), Point(1, 1), 100, 100)
V = FunctionSpace(mesh, "Lagrange", 1)

# Time variables
dt = 0.000004
t = 0
T = 0.004

# Previous and current solution
u1 = interpolate(Constant(0.0), V)
u0 = interpolate(Constant(0.0), V)

# Variational problem at each time
u = TrialFunction(V)
v = TestFunction(V)

a = u*v*dx + dt*dt*c*c*inner(grad(u), grad(v))*dx
L = 2*u1*v*dx - u0*v*dx

bc = DirichletBC(V, 0, "on_boundary")
A, b = assemble_system(a, L, bc)

u = Function(V)

# VTK file for saving the solution
vtkfile = File("wave_solution.pvd")

# Time-stepping
while t <= T:
    A, b = assemble_system(a, L, bc)
    delta = PointSource(V, Point(-1.0, 0), sin(c * 10 * t))
    delta.apply(b)
    solve(A, u.vector(), b)
    u0.assign(u1)
    u1.assign(u)
    t += dt

    # Reduce the range of the solution
    j = 0
    for i in u.vector():
        i = min(.01, i)
        i = max(-.01, i)
        u.vector()[j] = i
        j += 1

    # Save solution to VTK format
    vtkfile << (u, t)

    # Optionally plot the solution 
    plot(u, interactive=False)

plot(u, interactive=True)

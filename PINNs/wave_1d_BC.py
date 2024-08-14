import numpy as np
from sympy import Symbol, sin, pi, cos
import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver 
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym.node import Node
from wave_equation_main import WaveEquation1D

@modulus.sym.main(config_path="./", config_name= "config_main.yml")
def run(cfg: ModulusConfig) -> None:
    # Computing wave equation
    we = WaveEquation1D(c = 1.0)
    
    # Creating a neuron network with inputs: t,x; output: u
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Creating Nodes and Domain
    nodes = we.make_nodes() + [wave_net.make_node(name="wave_network")]   
    domain = Domain()
    
    # Geometry and Constraints
    x_symbol, t_symbol = Symbol("x"), Symbol("t")
    L = 1
    geo = Line1D(0, L)
    time_range = {t_symbol: (0, 1)}
    
    # I.Cs
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        parameterization={t_symbol: 0.0},
        outvar={"u": x_symbol*(1-x_symbol), "u__t": 0},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 1.0, "u__t": 1.0},
    )
    domain.add_constraint(IC, "IC")
    
    # B.Cs
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")
    
    # Interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"wave_equation": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )
    domain.add_constraint(interior, "interior")
    
    # Analytical solution calculation using the sum
    def analytical_solution(x, t, C=1.0, num_terms=1000):
        u_sum = np.zeros_like(x)
        for k in range(1, num_terms*2, 2):  # Sum over odd k values
            u_sum += (8 / (k**3 * np.pi**3)) * np.sin(k * np.pi * x) * np.cos(C * k * np.pi * t)
        return u_sum

    deltaT = 0.1
    deltaX = 0.1
    x = np.arange(0, L, deltaX)
    t = np.arange(0, L, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)

    u = analytical_solution(X, T)

    invar_numpy = {"x": X, "t": T}
    outvar_numpy = {"u": u}
    validator = PointwiseValidator(
        nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy, batch_size=128
    )
    domain.add_validator(validator)
    
    slv = Solver(cfg, domain)
    
    slv.solve()
    
    # Exporting results to VTK file
    import pyvista as pv
    grid = pv.StructuredGrid(X, T, np.zeros_like(X))  # z = 0 for 2D
    grid["u"] = u.flatten()
    
    # Save the grid to a VTK file
    grid.save("wave_solution_analytic.vtk")


if __name__ == "__main__":
    run()

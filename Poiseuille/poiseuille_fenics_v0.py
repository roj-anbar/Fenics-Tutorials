"""
Created on Fri Jun 14 12:49:04 2024
@author: rojin

FEniCS tutorial demo program: Incompressible Navier-Stokes equations for channel flow (Poisseuille)
on the unit square using the Incremental Pressure Correction Scheme (IPCS).

# Naming conventions:
# u   : unknown velocity as the trial function in variational problem -> u^(n+1)
# u_  : most recent computed approximation of u^(n+1) as Function object (velocity outcome of step 1 of IPCS: u*)
# u_n : velocity at time-step n -> u^n
# U   : velocity at midpoint of time interval -> u^(n+1/2)

# p   : unknown pressure as the trial function in variational problem
# p_  : most recent computed approximation of p^(n+1) as Function object (pressure outcome of step 2 of IPCS)
# p_n : pressure at time-step n -> u^n

"""

from fenics import *
import numpy as np
from mpi4py import MPI
#import matplotlib.pyplot as plt

# Define problem constants
n_div = 10      #number of divisions in each direction
num_steps = 500 #total number of time steps
dt = 0.01       #[s] #time-step
mu = 1          #non-dimensional
rho = 1         #non-dimensional
H = 1           #non-dimensional #channel width

# Create a 2D unit square mesh 
mesh = UnitSquareMesh(n_div, n_div)

# Define function spaces (P2 for velocity, P1 for pressure)
V = VectorFunctionSpace(mesh, 'P', 2) #quadratic elements for velocity
Q = FunctionSpace(mesh, 'P', 1) #linear elements for pressure

# Define boundaries

#inflow  = 'near(x[0], 0)'
#outflow = 'near(x[0], 1)'
#walls   = 'near(x[1], 0) || near(x[1], 1)'

def inflow(x, on_boundary):
    return on_boundary and near(x[0], 0)

def outflow(x, on_boundary):
    return on_boundary and near(x[0], 1)

def walls(x, on_boundary):
    return on_boundary and not (near(x[1], 0)) or (near(x[1], 1))


# Define boundary conditions

bcu_walls   = DirichletBC(V, Constant((0, 0)), walls) #no-slip, both velocity components are zero
bcp_inflow  = DirichletBC(Q, Constant(8), inflow) #inlet pressure: calculated based on analytic solution
bcp_outflow = DirichletBC(Q, Constant(0), outflow) #outlet pressure zero


# Combine boundary conditions
bcu = [bcu_walls]
bcp = [bcp_inflow, bcp_outflow]


# Define trial and test functions (for both velocity and pressure)
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_  = Function(V)
u_n = Function(V)
p_  = Function(Q)
p_n = Function(Q)


# Define expressions used in variational forms
U   = 0.5*(u_n + u)       #u^(n+1/2)
n   = FacetNormal(mesh)
f   = Constant((0, 0))    #source term
k   = Constant(dt)
mu  = Constant(mu)       #viscosity
rho = Constant(rho)


# Define functions to calculate strain-rate and stress tensors

# Strain-rate tensor (epsilon)
def epsilon(u):
    return sym(nabla_grad(u))


# Stress tensor (sigma)
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))



# Define variational problem for each step of IPCS 

# Step 1: tentative velocity
# There is no u_ now but the solution of this problem gives u_
F1 =  rho * dot((u - u_n) / k, v)*dx \
    + rho * dot(dot(u_n, nabla_grad(u_n)), v)*dx \
    + inner(sigma(U, p_n), epsilon(v))*dx \
    + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
    - dot(f, v)*dx
# Extract linear and bilinear terms from the non-linear variational formulation automatically
a1 = lhs(F1)
L1 = rhs(F1)


# Step 2: update pressure
# The solution of this problem gives p_
a2 = dot(nabla_grad(p) , nabla_grad(q)) * dx  
L2 = dot(nabla_grad(p_n) , nabla_grad(q)) * dx  -  (rho/k) * div(u_) * q * dx

# Step 3: final velocity
# The solution of this problem gives u
a3 = dot(u, v) * dx
L3 = dot(u_,v) * dx - (k/rho) * dot(nabla_grad(p_ - p_n), v) * dx

# Assemble the coefficient matrix
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Initialize XDMF files for saving solutions
#xdmf_file_u = XDMFFile(MPI.comm_world, "./velocity.xdmf")
#xdmf_file_p = XDMFFile(MPI.comm_world, "./pressure.xdmf")


# March through time
t = 0
for n in range(num_steps):

    # update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)

    # Plot solution
    #plot(u_)

    # Compute error
    
    # Define exact solution (x, y components)
    u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
    u_e = interpolate(u_e, V)

    #replaced .array() in the original tutorial code with .get_local()
    error = np.abs(u_e.vector().get_local() - u_.vector().get_local()).max()
    print('t = %.2f: error = %.3g' % (t, error))
    print('max u:', u_.vector().get_local().max())

    # Save solutions to file at each time step
    #xdmf_file_u.write(u_, t)
    #xdmf_file_p.write(p_, t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)


# Hold plot
#interactive()


# Save solution to file in VTK format
# Initialize vtk files
vtkfile_u = File('./velocity.pvd')
vtkfile_p = File('./pressure.pvd')

# Write solutions to file
vtkfile_u << u_
vtkfile_p << p_

#xdmf_file_u.close()
#xdmf_file_p.close()

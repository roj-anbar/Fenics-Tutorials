"""
Created on Fri Jun 28 2024
@author: Rojin Anbarafshan
Contact: rojin.anbar@gmail.com

FEniCS tutorial demo program: Incompressible Navier-Stokes equations for channel flow with pulsatile inlet pressure (Womersley)
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
#for saving purposes
run_number = 8

from fenics import *
import numpy as np
import cmath
from mpi4py import MPI


# Define problem constants
n_div = 40      #number of divisions in each direction
num_steps = 200 #total number of time steps
dt = 0.01       #[s] #time-step

# Flow properties
#mu = 1      #non-dimensional
rho = 1      #density (non-dimensional)
H = 1        #channel width (non-dimensional)
freq = 1     #frequency
alpha = 10    #womersley number
K = 1        #strength of forcing pressure 

omega = 2.0 * pi * freq #angular frequency

#back calculate mu based on omega and alpha
mu = rho * (H**2) * omega / (4 * (alpha**2))
#nu = mu/rho

# Create a 2D unit square mesh 
#mesh = UnitSquareMesh(n_div, n_div)
mesh = RectangleMesh(Point(-0.5, -0.5), Point(0.5, 0.5), n_div, n_div)


# Define function spaces (P2 for velocity, P1 for pressure)
V = VectorFunctionSpace(mesh, 'P', 2) #quadratic elements for velocity
Q = FunctionSpace(mesh, 'P', 1) #linear elements for pressure


# Define boundaries
inflow  = "near(x[0], -0.5)"
outflow = "near(x[0], 0.5)"
walls   = "near(x[1], -0.5) || near(x[1], 0.5)"


# Define time-dependent inlet pressure
P0 = rho*K #pressure amplitude
p_inlet = Expression('P0*sin(omega*t)', P0=P0, omega=omega, t=0, degree=1)

# Define boundary conditions
bcu_walls   = DirichletBC(V, Constant((0, 0)), walls) #no-slip, both velocity components are zero
bcp_inflow  = DirichletBC(Q, p_inlet, inflow) #inlet pressure: calculated based on analytic solution
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
u_e = Function(V)  #exact solution
p_  = Function(Q)
p_n = Function(Q)


# Define expressions used in variational forms
U       = 0.5*(u_n + u)       #u^(n+1/2)
n       = FacetNormal(mesh)
f       = Constant((0, 0))    #source term
deltat  = Constant(dt)
mu_f    = Constant(mu)        #viscosity
rho_f   = Constant(rho)
#omega_f = Constant(omega)
#K_f       = Constant(K)
#H_f       = Constant(H)
#nu_f      = Constant(nu)

# Define functions to calculate strain-rate and stress tensors

# Strain-rate tensor (epsilon)
def epsilon(u):
    return sym(nabla_grad(u))


# Stress tensor (sigma)
def sigma(u, p):
    return 2*mu_f*epsilon(u) - p*Identity(len(u))


# Define exact solution (y component)
def exact_solution(K, omega, alpha, H, y, t):
    u_exact = np.imag((K/(1j*omega)) * (1- cmath.cosh(2*alpha*cmath.sqrt(1j)*y/H) / cmath.cosh(alpha*cmath.sqrt(1j))) * cmath.exp(1j*omega*t))
    return u_exact


# Define variational problem for each step of IPCS 

# Step 1: tentative velocity
# There is no u_ now but the solution of this problem gives u_
F1 =  rho_f * dot((u - u_n) / deltat, v)*dx \
    + rho_f * dot(dot(u_n, nabla_grad(u_n)), v)*dx \
    + inner(sigma(U, p_n), epsilon(v))*dx \
    + dot(p_n*n, v)*ds - dot(mu_f*nabla_grad(U)*n, v)*ds \
    - dot(f, v)*dx
# Extract linear and bilinear terms from the non-linear variational formulation automatically
a1 = lhs(F1)
L1 = rhs(F1)


# Step 2: update pressure
# The solution of this problem gives p_
a2 = dot(nabla_grad(p) , nabla_grad(q)) * dx  
L2 = dot(nabla_grad(p_n) , nabla_grad(q)) * dx  -  (rho_f/deltat) * div(u_) * q * dx

# Step 3: final velocity
# The solution of this problem gives u
a3 = dot(u, v) * dx
L3 = dot(u_,v) * dx - (deltat/rho_f) * dot(nabla_grad(p_ - p_n), v) * dx

# Assemble the coefficient matrix
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Create save folder name
save_folder = 'run'+str(run_number)+'_a=' + str(alpha) #+ '_dt=' + str(dt) + '_f=' + str(freq)

# Initialize XDMF files for saving solutions
xdmf_file_u   = XDMFFile('./results/'+save_folder+'/velocity.xdmf')
#xdmf_file_u_e = XDMFFile('./results/'+save_folder+'/velocity_exact.xdmf')
xdmf_file_p   = XDMFFile('./results/'+save_folder+'/pressure.xdmf')

# Initialize vtk files for saving solutions
#vtkfile_u = File('./' + save_folder + './results/velocity.pvd')
#vtkfile_p = File('./' + save_folder + './results/pressure.pvd')

# Get the coordinates of the mesh nodes
coords = mesh.coordinates()

# Construct the matrix to store exact solution
#u_e_array = np.zeros((coords.shape[0], 2))
u_e_array = np.zeros((n_div+1, num_steps))

# Get the local to global degree of freedom mapping
#dofs = V.dofmap().dofs()

# March through time
t = 0
for n in range(num_steps):

    # update current time
    t += dt

    # Update time in pressure expression
    p_inlet.t = t

    # Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]


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


    # Compute exact solution
    #for i, coord in enumerate(coords):
        #x = coord[0]
    #    y = coord[1]
    #    u_e_array[i,0] = exact_solution(K, omega, alpha, H, y, t)
        #print('x=%.2f, y=%.2f, ux = %.4g' % (x,y,u_e_array[i,0]))
    #    u_e_array[i,1] = 0


    for yind in range(0, n_div+1):
       y = -0.5 + yind/n_div
       #print(y)
       u_e_array[yind, n] = exact_solution(K, omega, alpha, H, y, t)

    #u_flat = u_e_array.flatten()
    #print(u_flat)
    # Assign the values to the function
    #u_e.vector().set_local(u_e_array.flatten())
    #u_e.vector().apply("insert")


    #replaced .array() in the original tutorial code with .get_local()
    #error = np.abs(u_e.vector().get_local().max() - u_.vector().get_local().max())
    error = np.abs(u_e.vector().get_local().max() - u_.vector().get_local().max())
    print('t = %.2f: error = %.3g' % (t, error))
    print('max u:', u_.vector().get_local().max())

    # Save solutions to file at each time step
    xdmf_file_u.write(u_, t)
    #xdmf_file_u_e.write(u_e, t)
    xdmf_file_p.write(p_, t)


    # Write solutions to file
    #vtkfile_u << (u_ , t)
    #vtkfile_p << (p_ , t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)


xdmf_file_u.close()
#xdmf_file_u_e.close()
xdmf_file_p.close()

# Save the array to a text file
np.savetxt('./results/'+save_folder+'/velocity_exact.csv', u_e_array, fmt='%0.4f',  delimiter=',')

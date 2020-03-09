import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
try:
  from adolc import *
except:
  if (MPI.COMM_WORLD.Get_rank() == 0):
    logger.warning("adolc not found, can't use adolc automatic differentiation")
## This script just checks to see if variables from the input deck exist.
# Eric, June 7 2018
if 'write_output' in globals():
  pass
else:
  write_output = 1

if 'procz' in globals():
  pass
else:
  procz = 1
  if (mpi_rank == 0):
    logger.warning("procz not found, setting procz=1")

if 'tau' in globals():
  pass
else:
  tau = dt 
  if (mpi_rank == 0):
    logger.warning("tau not found, setting tau = dt")


if 'n_blocks' in globals():
  pass
else:
  if (mpi_rank == 0): print('Assuming 1 Block Mesh, assigning n_blocks = 1' )
  n_blocks = 1
  BCs = [BCs]
  procx = [procx]
  procy = [procy]
  procz = [procz]
  starting_rank = [0]
  Nel_block = [Nel]
  x_block = [x]
  y_block = [y]
  z_block = [z]
  IC_function = [IC_function]

total_procs = 0
#for i in range(np.size(procx)):
total_procs = starting_rank[-1] +  procx[-1]*procy[-1]*procz[-1]

if (total_procs != num_processes):
   if (mpi_rank == 0): 
    print('==================================')
    print('Error in number of mpi_ranks. PyDG quitting')
    print('==================================')
   sys.exit()
 
if (len(BCs) != n_blocks):
  if (mpi_rank == 0): 
    print('==================================')
    print('Size of BCs in inputfile.py is not the same as the number of blocks. Add information for other blocks. PyDG quitting')
    print('==================================')
  sys.exit()

if (len(procx) != n_blocks):
  if (mpi_rank == 0): 
    print('==================================')
    print('Size of procx in inputfile.py is not the same as the number of blocks. Add information for other blocks. PyDG quitting')
    print('==================================')
  sys.exit()

if (len(procy) != n_blocks):
  if (mpi_rank == 0): 
    print('==================================')
    print('Size of procy in inputfile.py is not the same as the number of blocks. Add information for other blocks. PyDG quitting')
    print('==================================')
  sys.exit()

if (len(procz) != n_blocks):
  if (mpi_rank == 0): 
    print('==================================')
    print('Size of procz in inputfile.py is not the same as the number of blocks. Add information for other blocks. PyDG quitting')
    print('==================================')
  sys.exit()


if 'mol_str' in globals():
  pass
else:
  mol_str = 'False'



if 'linear_solver_str' in globals():
  pass
else:
  linear_solver_str = 'GMRes'
  if (mpi_rank == 0):
    print('Setting linear solver to GMRes by default. Ignore if you are using an explicit time scheme')

if 'nonlinear_solver_str' in globals():
  pass
else:
  nonlinear_solver_str = 'Newton'
  if (mpi_rank == 0):
    print('Setting nonlinear solver to Newton by default. Ignore if you are using an explicit time scheme')

if 'fsource' in globals():
  pass
else:
  fsource = False
  source_mag = []

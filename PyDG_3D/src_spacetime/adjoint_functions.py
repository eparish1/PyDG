import numpy as np
from block_classes import blockClass
from init_Classes import variables
from mpi4py import MPI
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
try:
  from adolc import *
except:
  if (MPI.COMM_WORLD.Get_rank() == 0):
    logger.warning("adolc not found, can't use adolc automatic differentiation")
try:
  import adolc
except:
  if (MPI.COMM_WORLD.Get_rank() == 0):
    logger.warning("adolc not found, can't use adolc automatic differentiation")

def getR(a,regionManager,eqns,turb_str_adjoint):
  regionManager2 = blockClass(regionManager.n_blocks,regionManager.starting_rank,regionManager.procx,regionManager.procy,regionManager.procz,regionManager.et,regionManager.dt,regionManager.save_freq,turb_str_adjoint,regionManager.Nel_block,regionManager.region[0].order,eqns,a.dtype)
  region_counter = 0
  for i in regionManager2.mpi_regions_owned:
    regionManager2.region.append( variables(regionManager2,region_counter,i,regionManager.Nel_block[i],regionManager.region[0].order,regionManager.region[0].quadpoints,eqns,regionManager.region[0].mu,regionManager.region[i].x,regionManager.region[i].y,regionManager.region[i].z,turb_str_adjoint,regionManager.procx[i],regionManager.procy[i],regionManager.procz[i],regionManager.starting_rank[i],regionManager.region[i].BCs,regionManager.region[i].fsource,regionManager.region[i].source_mag,False,False,regionManager.region[0].basis_args) )
  a0 = regionManager.a*1.
  regionManager2.a = a[:]
  try:
    regionManager2.V = regionManager.V
  except:
    pass
  for i in regionManager2.mpi_regions_owned:
    region = regionManager2.region[i]
    start_indx = regionManager2.solution_start_indx[region.region_counter]
    end_indx = regionManager2.solution_end_indx[region.region_counter]
    region.RHS = np.reshape(regionManager2.RHS[start_indx:end_indx],(eqns.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt))
    region.a.a = np.reshape(regionManager2.a[start_indx:end_indx],(eqns.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt))
  regionManager2.getRHS_REGION_OUTER(regionManager2,eqns)
  return regionManager2.RHS[:]


def adjoint_init(regionManager,eqns,turb_str_adjoint):
  if regionManager.iteration == 0:
    indx = int( (regionManager.et - regionManager.t) /regionManager.dt )
    sol = np.load('Solution_ROM/npsol_block0_' + str(indx) + '.npz')
    lam = regionManager.a*1.
    regionManager.region[0].a.a[:] = sol['a']
    #print('solnorm = ',np.linalg.norm(regionManager.a))
    N = np.size(regionManager.a)
    ax = adouble(regionManager.a.flatten())   
    a0 = regionManager.a[:]*1.
    trace_on(1)
    independent(ax)
    ay = getR(ax,regionManager,eqns,turb_str_adjoint)
    dependent(ay)
    trace_off()
    x = regionManager.a
    regionManager.vec_jac = vec_jac
    regionManager.jac_vec = jac_vec

    regionManager.a[:] = lam[:]
    print('===== FINISHED TAPING ADJOINT')

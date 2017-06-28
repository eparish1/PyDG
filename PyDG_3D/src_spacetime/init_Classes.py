import numpy as np
import sys
from mpi4py import MPI
#from petsc4py import PETSc
from MPI_functions import getRankConnectionsSlab
from legendreBasis import *
from fluxSchemes import inviscidFlux,centralFluxGeneral
from navier_stokes import *
from linear_advection import *
#from equationFluxes import *
from DG_functions import getFlux,getRHS
from turb_models import *
from viscousFluxesIP import *
from boundary_conditions import *
from equations_class import *
from gas import *
class variable:
  def __init__(self,nvars,order,quadpoints,Npx,Npy,Npz,Npt):
      self.nvars = nvars
      self.order = order
      self.quadpoints = quadpoints
      self.a =np.zeros((nvars,order[0],order[1],order[2],order[3],Npx,Npy,Npz,Npt))
      self.u =np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
      self.aR_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npy,Npz,Npt))
      self.aL_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npy,Npz,Npt))
      self.aU_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npx,Npz,Npt))
      self.aD_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npx,Npz,Npt))
      self.aF_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npx,Npy,Npt))
      self.aB_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npx,Npy,Npt))


      self.uR_edge = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npy,Npz,Npt))
      self.uL_edge = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npy,Npz,Npt))
      self.uU_edge = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npz,Npt))
      self.uD_edge = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npz,Npt))
      self.uF_edge = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npt))
      self.uB_edge = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npt))

      self.uR = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
      self.uL = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
      self.uU = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
      self.uD = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
      self.uF = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt))
      self.uB = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt))

      self.uFuture = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],Npx,Npy,Npz,Npt))
#      self.uPast = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],Npx,Npy,Npz,Npt))


      self.aR = np.zeros((nvars,order[1],order[2],order[3],Npx,Npy,Npz,Npt))
      self.aL = np.zeros((nvars,order[1],order[2],order[3],Npx,Npy,Npz,Npt))
      self.aU = np.zeros((nvars,order[0],order[2],order[3],Npx,Npy,Npz,Npt))
      self.aD = np.zeros((nvars,order[0],order[2],order[3],Npx,Npy,Npz,Npt))
      self.aF = np.zeros((nvars,order[0],order[1],order[3],Npx,Npy,Npz,Npt))
      self.aB = np.zeros((nvars,order[0],order[1],order[3],Npx,Npy,Npz,Npt))

      self.uLS = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
      self.uRS = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
      self.uUS = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
      self.uDS = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
      self.uBS = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt))
      self.uFS = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt))

class fluxvariable:
  def __init__(self,nvars,order,quadpoints,Npx,Npy,Npz,Npt):
    self.nvars = nvars
    self.order = order
    self.quadpoints = quadpoints
    self.fx = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fy = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fz = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))

    self.fL = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fR = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fU = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fD = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fF = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fB = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt))

    self.fLS = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fRS = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fUS = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fDS = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fFS = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fBS = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt))

    self.fLI = np.zeros((nvars,order[1],order[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fRI = np.zeros((nvars,order[1],order[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fUI = np.zeros((nvars,order[0],order[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fDI = np.zeros((nvars,order[0],order[2],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fFI = np.zeros((nvars,order[0],order[1],quadpoints[3],Npx,Npy,Npz,Npt))
    self.fBI = np.zeros((nvars,order[0],order[1],quadpoints[3],Npx,Npy,Npz,Npt))

    self.fR_edge = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npy,Npz,Npt))
    self.fL_edge = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npy,Npz,Npt))
    self.fU_edge = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npz,Npt))
    self.fD_edge = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npz,Npt))
    self.fF_edge = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npt))
    self.fB_edge = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npt))

class boundaryConditions:
  comm = MPI.COMM_WORLD
  mpi_rank = comm.Get_rank()
  def __init__(self,BC_type='periodic',BC_args=[]):
    check = 0
    if (BC_type == 'periodic'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = periodic_bc
      self.args = BC_args
    if (BC_type == 'isothermal_wall'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = isothermalwall_bc
      self.args = BC_args
    if (BC_type == 'adiabatic_wall'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = adiabaticwall_bc
      self.args = BC_args
    if (BC_type == 'dirichlet'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = dirichlet_bc
      self.args = BC_args

    if (check == 0):
      if (mpi_rank == 0): print('BC type ' + BC_type + ' not found. PyDG quitting')
      sys.exit()

    
    

class variables:
  def __init__(self,Nel,order,quadpoints,eqns,mu,xG,yG,zG,t,et,dt,iteration,save_freq,turb_str,procx,procy,BCs,source,source_mag,shock_capturing):
    ## DG scheme information
    self.Nel = Nel
    self.order = order
    self.quadpoints = quadpoints 
    self.t = t
    self.et = et
    self.dt = dt
    self.iteration = iteration
    self.save_freq = save_freq
    self.shock_capturing = shock_capturing
    ##============== MPI INFORMATION ===================
    self.procx = procx
    self.procy = procy
    self.comm = MPI.COMM_WORLD
    self.num_processes = self.comm.Get_size()
    self.mpi_rank = self.comm.Get_rank()
    if (procx*procy != self.num_processes):
      if (self.mpi_rank == 0):
        print('Error, correct x/y proc decomposition, now quitting!')
      sys.exit()
    self.Npy = int(float(Nel[1] / procy)) #number of points on each x plane. MUST BE UNIFORM BETWEEN PROCS
    self.Npx = int(float(Nel[0] / procx))
    self.Npz = int(Nel[2])
    self.Npt = Nel[-1]
    self.sy = slice(int(self.mpi_rank)/int(self.procx)*self.Npy,(int(self.mpi_rank)/int(self.procx) + 1)*self.Npy)  ##slicing in y direction
    self.sx = slice(int(self.mpi_rank%self.procx)*self.Npx,int(self.mpi_rank%self.procx + 1)*self.Npx)
    self.rank_connect,self.BC_rank = getRankConnectionsSlab(self.mpi_rank,self.num_processes,self.procx,self.procy)
    self.w,self.wp,self.wpedge,self.weights,self.zeta = gaussPoints(self.order[0],self.quadpoints[0])
    self.altarray = (-np.ones(self.order[0]))**(np.linspace(0,self.order[0]-1,self.order[0]))

    self.w0,self.wp0,self.wpedge0,self.weights0,self.zeta0 = gaussPoints(self.order[0],self.quadpoints[0])
    self.altarray0 = (-np.ones(self.order[0]))**(np.linspace(0,self.order[0]-1,self.order[0]))
    self.w1,self.wp1,self.wpedge1,self.weights1,self.zeta1 = gaussPoints(self.order[1],self.quadpoints[1])
    self.altarray1 = (-np.ones(self.order[1]))**(np.linspace(0,self.order[1]-1,self.order[1]))
    self.w2,self.wp2,self.wpedge2,self.weights2,self.zeta2 = gaussPoints(self.order[2],self.quadpoints[2])
    self.altarray2 = (-np.ones(self.order[2]))**(np.linspace(0,self.order[2]-1,self.order[2]))
    self.w3,self.wp3,self.wpedge3,self.weights3,self.zeta3 = gaussPoints(self.order[3],self.quadpoints[3])
    self.altarray3 = (-np.ones(self.order[3]))**(np.linspace(0,self.order[3]-1,self.order[3]))

    self.gas = gasClass() 
    self.reacting = False
    ## Initialize BCs
    self.BCs = BCs
    self.rightBC = boundaryConditions(BCs[0],BCs[1])
    self.topBC = boundaryConditions(BCs[2],BCs[3])
    self.leftBC = boundaryConditions(BCs[4],BCs[5])
    self.bottomBC = boundaryConditions(BCs[6],BCs[7])

    ## Sources
    self.source = source
    self.source_mag = source_mag
    ## Initialize arrays
    self.dx = xG[1] - xG[0]
    self.dy = yG[1] - yG[0]
    self.dz = zG[1] - zG[0]
    self.dx2 = np.diff(xG)[self.sx]
    self.dy2 = np.diff(yG)[self.sy]
    self.dz2 = np.diff(zG)

    #print(np.shape(xG))
    self.x = xG[self.sx]
    self.y = yG[self.sy] 
    self.z = zG
    self.xG = xG
    self.yG = yG
    self.zG = zG
    self.nvars = eqns.nvars

    self.a0 = np.zeros((eqns.nvars,self.order[0],self.order[1],self.order[2],self.order[3],self.Npx,self.Npy,self.Npz,self.Npt))
    self.a = variable(eqns.nvars,self.order,self.quadpoints,self.Npx,self.Npy,self.Npz,self.Npt)
    self.iFlux = fluxvariable(eqns.nvars,self.order,self.quadpoints,self.Npx,self.Npy,self.Npz,self.Npt)
    self.mus = mu
    self.mu = np.ones(np.append( eqns.nmus, np.shape( self.a.u[0])))*self.mus
    self.muR = np.ones(np.append( eqns.nmus, np.shape( self.a.uR[0])))*self.mus
    self.muL = np.ones(np.append( eqns.nmus, np.shape( self.a.uL[0])))*self.mus
    self.muU = np.ones(np.append( eqns.nmus, np.shape( self.a.uU[0])))*self.mus
    self.muD = np.ones(np.append( eqns.nmus, np.shape( self.a.uD[0])))*self.mus
    self.muF = np.ones(np.append( eqns.nmus, np.shape( self.a.uF[0])))*self.mus
    self.muB = np.ones(np.append( eqns.nmus, np.shape( self.a.uB[0])))*self.mus
    self.mu0 = np.ones(np.append( eqns.nmus, np.shape( self.a.u[0] )))*self.mus
    self.mu0R =np.ones(np.append( eqns.nmus, np.shape( self.a.uR[0])))*self.mus
    self.mu0L =np.ones(np.append( eqns.nmus, np.shape( self.a.uL[0])))*self.mus
    self.mu0U =np.ones(np.append( eqns.nmus, np.shape( self.a.uU[0])))*self.mus
    self.mu0D =np.ones(np.append( eqns.nmus, np.shape( self.a.uD[0])))*self.mus
    self.mu0F =np.ones(np.append( eqns.nmus, np.shape( self.a.uF[0])))*self.mus
    self.mu0B =np.ones(np.append( eqns.nmus, np.shape( self.a.uB[0])))*self.mus

    self.getFlux = getFlux
    self.RHS = np.zeros((eqns.nvars,self.order[0],self.order[1],self.order[2],self.order[3],self.Npx,self.Npy,self.Npz,self.Npt))
  
    ### Check turbulence models
    self.turb_str = turb_str
    check = 0
    if (turb_str == 'tau-model'):
      self.getRHS = tauModelLinearized
      check = 1
    if (turb_str == 'tau-modelFD'):
      self.getRHS = tauModelFD
      check = 1
    if (turb_str == 'FM1'):
      self.getRHS = FM1Linearized 
      check = 1
    if (turb_str == 'Smagorinsky'):
      self.getRHS = DNS
      if (self.mpi_rank == 0): print('Using Smagorinsky Model')
    if (check == 0):
      self.getRHS = DNS
#      if (self.mpi_rank == 0):
#         print('Error, turb model ' + turb_str + 'not found. Setting to DNS')
    else:
      if (self.mpi_rank == 0):
         print('Using turb model ' + turb_str)


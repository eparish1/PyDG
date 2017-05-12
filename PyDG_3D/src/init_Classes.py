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
from DG_functions import getFlux,getRHS_INVISCID,getRHS_IP
from turb_models import *
from viscousFluxesIP import *
class variable:
  def __init__(self,nvars,order,quadpoints,Npx,Npy,Npz):
      self.nvars = nvars
      self.order = order
      self.quadpoints = quadpoints
      self.a =np.zeros((nvars,order,order,order,Npx,Npy,Npz))
      self.u =np.zeros((nvars,quadpoints,quadpoints,quadpoints,Npx,Npy,Npz))

      self.aR_edge = np.zeros((nvars,order,order,order,Npy,Npz))
      self.aL_edge = np.zeros((nvars,order,order,order,Npy,Npz))
      self.aU_edge = np.zeros((nvars,order,order,order,Npx,Npz))
      self.aD_edge = np.zeros((nvars,order,order,order,Npx,Npz))
      self.aF_edge = np.zeros((nvars,order,order,order,Npx,Npy))
      self.aB_edge = np.zeros((nvars,order,order,order,Npx,Npy))


      self.uR_edge = np.zeros((nvars,quadpoints,quadpoints,Npy,Npz))
      self.uL_edge = np.zeros((nvars,quadpoints,quadpoints,Npy,Npz))
      self.uU_edge = np.zeros((nvars,quadpoints,quadpoints,Npx,Npz))
      self.uD_edge = np.zeros((nvars,quadpoints,quadpoints,Npx,Npz))
      self.uF_edge = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy))
      self.uB_edge = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy))

      self.uR = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
      self.uL = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
      self.uU = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
      self.uD = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
      self.uF = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
      self.uB = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))

      self.aU = np.zeros((nvars,order,order,Npx,Npy,Npz))
      self.aD = np.zeros((nvars,order,order,Npx,Npy,Npz))
      self.aR = np.zeros((nvars,order,order,Npx,Npy,Npz))
      self.aL = np.zeros((nvars,order,order,Npx,Npy,Npz))
      self.aF = np.zeros((nvars,order,order,Npx,Npy,Npz))
      self.aB = np.zeros((nvars,order,order,Npx,Npy,Npz))

      self.uUS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
      self.uDS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
      self.uLS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
      self.uRS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
      self.uBS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
      self.uFS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))

#      self.edge_tmpy = np.zeros((nvars,quadpoints,Npx)).flatten()
#      self.edge_tmpx = np.zeros((nvars,quadpoints,Npy)).flatten()

class fluxvariable:
  def __init__(self,nvars,order,quadpoints,Npx,Npy,Npz):
    self.nvars = nvars
    self.order = order
    self.quadpoints = quadpoints
    self.fx = np.zeros((nvars,quadpoints,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fy = np.zeros((nvars,quadpoints,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fz = np.zeros((nvars,quadpoints,quadpoints,quadpoints,Npx,Npy,Npz))

    self.fU = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fD = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fL = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fR = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fF = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fB = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))

    self.fUS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fDS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fLS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fRS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fFS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))
    self.fBS = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy,Npz))

    self.fUI = np.zeros((nvars,order,order,Npx,Npy,Npz))
    self.fDI = np.zeros((nvars,order,order,Npx,Npy,Npz))
    self.fLI = np.zeros((nvars,order,order,Npx,Npy,Npz))
    self.fRI = np.zeros((nvars,order,order,Npx,Npy,Npz))
    self.fFI = np.zeros((nvars,order,order,Npx,Npy,Npz))
    self.fBI = np.zeros((nvars,order,order,Npx,Npy,Npz))

    self.fR_edge = np.zeros((nvars,quadpoints,quadpoints,Npy,Npz))
    self.fL_edge = np.zeros((nvars,quadpoints,quadpoints,Npy,Npz))
    self.fU_edge = np.zeros((nvars,quadpoints,quadpoints,Npx,Npz))
    self.fD_edge = np.zeros((nvars,quadpoints,quadpoints,Npx,Npz))
    self.fF_edge = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy))
    self.fB_edge = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy))
  

class variables:
  def __init__(self,Nel,order,quadpoints,eqns,mu,xG,yG,zG,t,et,dt,iteration,save_freq,turb_str,procx,procy):
    ## DG scheme information
    self.Nel = Nel
    self.order = order
    self.quadpoints = quadpoints 
    self.t = t
    self.et = et
    self.dt = dt
    self.iteration = iteration
    self.save_freq = save_freq
    self.mu = mu
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

    self.sy = slice(int(self.mpi_rank)/int(self.procx)*self.Npy,(int(self.mpi_rank)/int(self.procx) + 1)*self.Npy)  ##slicing in y direction
    self.sx = slice(int(self.mpi_rank%self.procx)*self.Npx,int(self.mpi_rank%self.procx + 1)*self.Npx)
    self.rank_connect = getRankConnectionsSlab(self.mpi_rank,self.num_processes,self.procx,self.procy)
    self.w,self.wp,self.wpedge,self.weights,self.zeta = gaussPoints(self.order,self.quadpoints)
    self.altarray = (-np.ones(self.order))**(np.linspace(0,self.order-1,self.order))
    ## Initialize arrays
    self.dx = xG[1] - xG[0]
    self.dy = yG[1] - yG[0]
    self.dz = zG[1] - zG[0]
    self.x = xG[self.sx]
    self.y = yG[self.sy] 
    self.z = zG
    self.xG = xG
    self.yG = yG
    self.zG = zG
    self.nvars = eqns.nvars

    self.a0 = np.zeros((eqns.nvars,self.order,self.order,self.order,self.Npx,self.Npy,self.Npz))
    self.a = variable(eqns.nvars,self.order,self.quadpoints,self.Npx,self.Npy,self.Npz)
    self.iFlux = fluxvariable(eqns.nvars,self.order,self.quadpoints,self.Npx,self.Npy,self.Npz)
    if (eqns.vflux_type == 'BR1'):
      self.getRHS = getRHS_BR1
    if (eqns.vflux_type == 'IP'): 
      self.getRHS = getRHS_IP
    if (eqns.vflux_type == 'Inviscid'): 
      self.getRHS = getRHS_INVISCID

    self.getFlux = getFlux
    self.RHS = np.zeros((eqns.nvars,self.order,self.order,self.order,self.Npx,self.Npy,self.Npz))
    self.turb_str = turb_str
    if (turb_str == 'tau-model'):
      self.turb_model = tauModel
    if (turb_str == 'dynamic-tau model'):
      self.turb_model = DtauModel
    if (turb_str == 'DNS'):
      self.turb_model = DNS
    if (turb_str == 'orthogonal dynamics'):
      self.turb_model = orthogonalDynamics

class equations:
  def __init__(self,eq_str,schemes):
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    iflux_str = schemes[0]
    vflux_str = schemes[1]
    check_eq = 0
    if (eq_str == 'Navier-Stokes'):
      check_eq = 1
      self.nvars = 5
      self.nvisc_vars = 5
      self.evalFluxX = evalFluxXEuler 
      self.evalFluxY = evalFluxYEuler
      self.evalFluxZ = evalFluxZEuler
      ## select appopriate flux scheme
      checki = 0
      if (iflux_str == 'central'):
        self.inviscidFlux = eulercentralflux
        checki = 1
      if (iflux_str == 'roe'):
        self.inviscidFlux = kfid_roeflux
        checki = 1
      if (iflux_str == 'rusanov'):
        self.inviscidFlux = rusanovFlux
        checki = 1
      if (checki == 0):
        if (mpi_rank == 0): print('Error, inviscid flux scheme ' + iflux_str + ' not valid. Options are "central", "roe", "rusanov". PyDG quitting')
        sys.exit()
      checkv = 0 
      if (vflux_str == 'BR1'):
        print('Error, BR1 not completed for 3D. PyDG quitting')
        sys.exit()
        self.evalViscousFluxX = evalViscousFluxXNS_BR1
        self.evalViscousFluxY = evalViscousFluxYNS_BR1
        self.evalTauFluxX = evalTauFluxXNS_BR1
        self.evalTauFluxY = evalTauFluxYNS_BR1
        self.vflux_type = 'BR1'
        checkv = 1
        self.viscousFlux = centralFlux
      if (vflux_str == 'IP'):
        self.evalViscousFluxX = evalViscousFluxXNS_IP
        self.evalViscousFluxY = evalViscousFluxYNS_IP
        self.evalViscousFluxZ = evalViscousFluxZNS_IP
        self.getGs = getGsNS 
        self.getGsX = getGsNSX
        self.getGsY = getGsNSY 
        self.getGsZ = getGsNSZ 
        self.vflux_type = 'IP'
        checkv = 1
      if (vflux_str == 'Inviscid'):
        self.vflux_type = 'Inviscid'
        checkv = 1
      if (checkv == 0):
        if (mpi_rank == 0): print('Error, viscous flux scheme ' + vflux_str + ' not valid. Options are, "IP", "Inviscid". PyDG quitting')
        sys.exit() 
    if (eq_str == 'Linear-Advection'):
      check_eq = 1
      print('Linear-Advection is not yet complete for 3D, PyDG quitting')
      sys.exit()
      self.nvars = 1
      self.nvisc_vars = 2
      self.evalFluxX = evalFluxXLA
      self.evalFluxY = evalFluxYLA
      self.evalFluxZ = evalFluxZLA

      if (vflux_str == 'IP'):
        self.evalViscousFluxX = evalViscousFluxXLA_IP
        self.evalViscousFluxY = evalViscousFluxYLA_IP
        self.getGs = getGsLA
      if (vflux_str == 'BR1'):
        self.evalViscousFluxX = evalViscousFluxXLA_BR1
        self.evalViscousFluxY = evalViscousFluxYLA_BR1
        self.evalTauFluxX = evalTauFluxXLA_BR1
        self.evalTauFluxY = evalTauFluxYLA_BR1

    if (eq_str == 'Diffusion'):
      check_eq = 1
      print('Pure Diffusion is not yet complete for 3D, PyDG quitting')
      sys.exit()
      self.nvars = 1
      self.nvisc_vars = 2
      self.evalFluxX = evalFluxXD
      self.evalFluxY = evalFluxYD
      #self.getEigs = getEigsEuler
      if (vflux_str == 'IP'):
        self.evalViscousFluxX = evalViscousFluxXLA_IP
        self.evalViscousFluxY = evalViscousFluxYLA_IP
        self.getGs = getGsLA
      if (vflux_str == 'BR1'):
        self.evalViscousFluxX = evalViscousFluxXLA_BR1
        self.evalViscousFluxY = evalViscousFluxYLA_BR1
        self.evalTauFluxX = evalTauFluxXLA_BR1
        self.evalTauFluxY = evalTauFluxYLA_BR1
    if (check_eq == 0):
       if (mpi_rank == 0): print('Equation set ' + str(eq_str) + ' is not valid, PyDG quitting')
       sys.exit()


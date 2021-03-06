import numpy as np
from mpi4py import MPI
from MPI_functions import getRankConnections,sendEdges
from legendreBasis import *
from fluxSchemes import *
from equationFluxes import *
from DG_functions import getRHS,getFlux

class variable:
  def __init__(self,nvars,orderx,ordery,Npx,Npy):
      self.nvars = nvars
      self.order = orderx
      self.a =np.zeros((nvars,orderx,ordery,Npx,Npy))
      self.u =np.zeros((nvars,orderx,ordery,Npx,Npy))
      self.uU = np.zeros((nvars,orderx,Npx,Npy))
      self.uD = np.zeros((nvars,orderx,Npx,Npy))
      self.uU_edge = np.zeros((nvars,ordery,Npx))
      self.uD_edge = np.zeros((nvars,ordery,Npx))
      self.uR = np.zeros((nvars,ordery,Npx,Npy))
      self.uL = np.zeros((nvars,ordery,Npx,Npy))
      self.aU = np.zeros((nvars,orderx,Npx,Npy))
      self.aD = np.zeros((nvars,orderx,Npx,Npy))
      self.aR = np.zeros((nvars,orderx,Npx,Npy))
      self.aL = np.zeros((nvars,orderx,Npx,Npy))
      self.uUS = np.zeros((nvars,orderx,Npx,Npy))
      self.uDS = np.zeros((nvars,orderx,Npx,Npy))
      self.uLS = np.zeros((nvars,ordery,Npx,Npy))
      self.uRS = np.zeros((nvars,ordery,Npx,Npy))

      self.edge_tmp = np.zeros((nvars,orderx,Npx)).flatten()

class fluxvariable:
  def __init__(self,nvars,orderx,ordery,Npx,Npy):
    self.nvars = nvars
    self.order = orderx
    self.fx = np.zeros((nvars,orderx,ordery,Npx,Npy))
    self.fy = np.zeros((nvars,orderx,ordery,Npx,Npy))
    self.fU = np.zeros((nvars,orderx,Npx,Npy))
    self.fD = np.zeros((nvars,orderx,Npx,Npy))
    self.fL = np.zeros((nvars,ordery,Npx,Npy))
    self.fR = np.zeros((nvars,ordery,Npx,Npy))
    self.fUS = np.zeros((nvars,orderx,Npx,Npy))
    self.fDS = np.zeros((nvars,orderx,Npx,Npy))
    self.fLS = np.zeros((nvars,ordery,Npx,Npy))
    self.fRS = np.zeros((nvars,ordery,Npx,Npy))
    self.fUI = np.zeros((nvars,orderx,Npx,Npy))
    self.fDI = np.zeros((nvars,orderx,Npx,Npy))
    self.fLI = np.zeros((nvars,ordery,Npx,Npy))
    self.fRI = np.zeros((nvars,ordery,Npx,Npy))
    self.fU_edge = np.zeros((nvars,orderx,Npx))
    self.fD_edge = np.zeros((nvars,orderx,Npx))
   

class variables:
  def __init__(self,Nel,order,eqns,nu,xG,yG,t,et,dt,iteration,save_freq):
    ## DG scheme information
    self.Nel = Nel
    self.order = order 
    self.t = t
    self.et = et
    self.dt = dt
    self.iteration = iteration
    self.save_freq = save_freq
    self.nu = nu
    ##============== MPI INFORMATION ===================
    self.comm = MPI.COMM_WORLD
    self.num_processes = self.comm.Get_size()
    self.mpi_rank = self.comm.Get_rank()
    self.Npy = int(float(Nel[1] / self.num_processes)) #number of points on each x plane. MUST BE UNIFORM BETWEEN PROCS
    self.Npx = Nel[0]
    self.sy = slice(self.mpi_rank*self.Npy,(self.mpi_rank+1)*self.Npy)  ##slicing in y direction
    self.rank_connect = getRankConnections(self.mpi_rank,self.num_processes)
    self.w,self.wp,self.weights,self.zeta = gaussPoints(self.order)
    self.altarray = (-np.ones(self.order))**(np.linspace(0,self.order-1,self.order))
    ## Initialize arrays
    self.x = xG
    self.dx = self.x[1] - self.x[0]
    self.y = yG[self.sy] 
    self.dy = yG[1] - yG[0]
    self.a0 = np.zeros((eqns.nvars,self.order,order,self.Npx,self.Npy))
    self.a = variable(eqns.nvars,self.order,self.order,self.Npx,self.Npy)
    self.b = variable(eqns.nvisc_vars,self.order,self.order,self.Npx,self.Npy)
    self.iFlux = fluxvariable(eqns.nvars,self.order,self.order,self.Npx,self.Npy)
    self.vFlux = fluxvariable(eqns.nvisc_vars,self.order,self.order,self.Npx,self.Npy)
    self.vFlux2 = fluxvariable(eqns.nvars,self.order,self.order,self.Npx,self.Npy)
    self.getRHS = getRHS
    self.getFlux = getFlux

class fschemes:
  def __init__(self,iflux_str,vflux_str):
    if (iflux_str == 'central'):
      self.inviscidFlux = centralFlux
    if (iflux_str == 'rusanov'):
      self.inviscidFlux = rusanovFlux
    if (vflux_str == 'central'):
      self.viscousFlux = centralFlux

class equations:
  def __init__(self,eq_str):
    if (eq_str == 'Navier-Stokes'):
      self.nvars = 4
      self.nvisc_vars = 4
      self.evalFluxX = evalFluxXEuler 
      self.evalFluxY = evalFluxYEuler
      self.getEigs = getEigsEuler
      self.evalViscousFluxX = evalViscousFluxXNS
      self.evalViscousFluxY = evalViscousFluxYNS
      self.evalTauFluxX = evalTauFluxXNS
      self.evalTauFluxY = evalTauFluxYNS

    if (eq_str == 'Linear-Advection'):
      self.nvars = 1
      self.nvisc_vars = 2
      self.evalFluxX = evalFluxXLA
      self.evalFluxY = evalFluxYLA
      #self.getEigs = getEigsEuler
      self.evalViscousFluxX = evalViscousFluxXLA
      self.evalViscousFluxY = evalViscousFluxYLA
      self.evalTauFluxX = evalTauFluxXLA
      self.evalTauFluxY = evalTauFluxYLA

    if (eq_str == 'Diffusion'):
      self.nvars = 1
      self.nvisc_vars = 2
      self.evalFluxX = evalFluxXD
      self.evalFluxY = evalFluxYD
      #self.getEigs = getEigsEuler
      self.evalViscousFluxX = evalViscousFluxXD
      self.evalViscousFluxY = evalViscousFluxYD
      self.evalTauFluxX = evalTauFluxXD
      self.evalTauFluxY = evalTauFluxYD

import numpy as np
from mpi4py import MPI
from MPI_functions import getRankConnections,sendEdges
from legendreBasis import *
from fluxSchemes import *
from equationFluxes import *
from viscousFluxesBR1 import *
from viscousFluxesIP import *
from DG_functions import getRHS_IP,getRHS_BR1,getFlux
from turb_models import tauModel,DNS
class variable:
  def __init__(self,nvars,order,quadpoints,Npx,Npy):
      self.nvars = nvars
      self.order = order
      self.quadpoints = quadpoints
      self.a =np.zeros((nvars,order,order,Npx,Npy))
      self.u =np.zeros((nvars,quadpoints,quadpoints,Npx,Npy))
      self.uU = np.zeros((nvars,quadpoints,Npx,Npy))
      self.uD = np.zeros((nvars,quadpoints,Npx,Npy))
      self.uU_edge = np.zeros((nvars,quadpoints,Npx))
      self.uD_edge = np.zeros((nvars,quadpoints,Npx))
      self.uR = np.zeros((nvars,quadpoints,Npx,Npy))
      self.uL = np.zeros((nvars,quadpoints,Npx,Npy))
      self.aU = np.zeros((nvars,order,Npx,Npy))
      self.aD = np.zeros((nvars,order,Npx,Npy))
      self.aR = np.zeros((nvars,order,Npx,Npy))
      self.aL = np.zeros((nvars,order,Npx,Npy))
      self.uUS = np.zeros((nvars,quadpoints,Npx,Npy))
      self.uDS = np.zeros((nvars,quadpoints,Npx,Npy))
      self.uLS = np.zeros((nvars,quadpoints,Npx,Npy))
      self.uRS = np.zeros((nvars,quadpoints,Npx,Npy))
      self.edge_tmp = np.zeros((nvars,quadpoints,Npx)).flatten()

class fluxvariable:
  def __init__(self,nvars,order,quadpoints,Npx,Npy):
    self.nvars = nvars
    self.order = order
    self.quadpoints = quadpoints
    self.fx = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy))
    self.fy = np.zeros((nvars,quadpoints,quadpoints,Npx,Npy))
    self.fU = np.zeros((nvars,quadpoints,Npx,Npy))
    self.fD = np.zeros((nvars,quadpoints,Npx,Npy))
    self.fL = np.zeros((nvars,quadpoints,Npx,Npy))
    self.fR = np.zeros((nvars,quadpoints,Npx,Npy))
    self.fUS = np.zeros((nvars,quadpoints,Npx,Npy))
    self.fDS = np.zeros((nvars,quadpoints,Npx,Npy))
    self.fLS = np.zeros((nvars,quadpoints,Npx,Npy))
    self.fRS = np.zeros((nvars,quadpoints,Npx,Npy))
    self.fUI = np.zeros((nvars,order,Npx,Npy))
    self.fDI = np.zeros((nvars,order,Npx,Npy))
    self.fLI = np.zeros((nvars,order,Npx,Npy))
    self.fRI = np.zeros((nvars,order,Npx,Npy))
    self.fU_edge = np.zeros((nvars,quadpoints,Npx))
    self.fD_edge = np.zeros((nvars,quadpoints,Npx))
   

class variables:
  def __init__(self,Nel,order,quadpoints,eqns,mu,xG,yG,t,et,dt,iteration,save_freq,schemes,turb_str):
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
    self.comm = MPI.COMM_WORLD
    self.num_processes = self.comm.Get_size()
    self.mpi_rank = self.comm.Get_rank()
    self.Npy = int(float(Nel[1] / self.num_processes)) #number of points on each x plane. MUST BE UNIFORM BETWEEN PROCS
    self.Npx = Nel[0]
    self.sy = slice(self.mpi_rank*self.Npy,(self.mpi_rank+1)*self.Npy)  ##slicing in y direction
    self.rank_connect = getRankConnections(self.mpi_rank,self.num_processes)
    self.w,self.wp,self.wpedge,self.weights,self.zeta = gaussPoints(self.order,self.quadpoints)
    self.altarray = (-np.ones(self.order))**(np.linspace(0,self.order-1,self.order))
    ## Initialize arrays
    self.x = xG
    self.dx = self.x[1] - self.x[0]
    self.y = yG[self.sy] 
    self.dy = yG[1] - yG[0]
    self.nvars = eqns.nvars
    self.a0 = np.zeros((eqns.nvars,self.order,order,self.Npx,self.Npy))
    self.a = variable(eqns.nvars,self.order,self.quadpoints,self.Npx,self.Npy)
    self.b = variable(eqns.nvisc_vars,self.order,self.quadpoints,self.Npx,self.Npy)
    self.iFlux = fluxvariable(eqns.nvars,self.order,self.quadpoints,self.Npx,self.Npy)
    self.vFlux = fluxvariable(eqns.nvisc_vars,self.order,self.quadpoints,self.Npx,self.Npy)
    self.vFlux2 = fluxvariable(eqns.nvars,self.order,self.quadpoints,self.Npx,self.Npy)
    if (schemes.vflux_type == 'BR1'):
      self.getRHS = getRHS_BR1
    if (schemes.vflux_type == 'IP'): 
      self.getRHS = getRHS_IP
    self.getFlux = getFlux
    self.RHS = np.zeros((eqns.nvars,self.order,self.order,self.Npx,self.Npy))
    self.turb_str = turb_str
    if (turb_str == 'tau-model'):
      self.turb_model = tauModel
    if (turb_str == 'DNS'):
      self.turb_model = DNS

class fschemes:
  def __init__(self,iflux_str,vflux_str):
    if (iflux_str == 'central'):
      self.inviscidFlux = centralFlux
    if (iflux_str == 'rusanov'):
      self.inviscidFlux = rusanovFlux
    if (vflux_str == 'BR1'):
      self.vflux_type = 'BR1'
      self.viscousFlux = centralFlux
    if (vflux_str == 'IP'):
      self.vflux_type = 'IP'

class equations:
  def __init__(self,eq_str,schemes):
    if (eq_str == 'Navier-Stokes'):
      self.nvars = 4
      self.nvisc_vars = 4
      self.evalFluxX = evalFluxXEuler 
      self.evalFluxY = evalFluxYEuler
      self.getEigs = getEigsEuler
      if (schemes.vflux_type == 'IP'):
        self.evalViscousFluxX = evalViscousFluxXNS_IP
        self.evalViscousFluxY = evalViscousFluxYNS_IP
        self.getGs = getGsNS 
      if (schemes.vflux_type == 'BR1'):
        self.evalViscousFluxX = evalViscousFluxXNS_BR1
        self.evalViscousFluxY = evalViscousFluxYNS_BR1
        self.evalTauFluxX = evalTauFluxXNS_BR1
        self.evalTauFluxY = evalTauFluxYNS_BR1

    if (eq_str == 'Linear-Advection'):
      self.nvars = 1
      self.nvisc_vars = 2
      self.evalFluxX = evalFluxXLA
      self.evalFluxY = evalFluxYLA
      #self.getEigs = getEigsEuler
      if (schemes.vflux_type == 'IP'):
        self.evalViscousFluxX = evalViscousFluxXLA_IP
        self.evalViscousFluxY = evalViscousFluxYLA_IP
        self.getGs = getGsLA
      if (schemes.vflux_type == 'BR1'):
        self.evalViscousFluxX = evalViscousFluxXLA_BR1
        self.evalViscousFluxY = evalViscousFluxYLA_BR1
        self.evalTauFluxX = evalTauFluxXLA_BR1
        self.evalTauFluxY = evalTauFluxYLA_BR1

    if (eq_str == 'Diffusion'):
      self.nvars = 1
      self.nvisc_vars = 2
      self.evalFluxX = evalFluxXD
      self.evalFluxY = evalFluxYD
      #self.getEigs = getEigsEuler
      if (schemes.vflux_type == 'IP'):
        self.evalViscousFluxX = evalViscousFluxXLA_IP
        self.evalViscousFluxY = evalViscousFluxYLA_IP
        self.getGs = getGsLA
      if (schemes.vflux_type == 'BR1'):
        self.evalViscousFluxX = evalViscousFluxXLA_BR1
        self.evalViscousFluxY = evalViscousFluxYLA_BR1
        self.evalTauFluxX = evalTauFluxXLA_BR1
        self.evalTauFluxY = evalTauFluxYLA_BR1


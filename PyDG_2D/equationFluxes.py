import numpy as np

######  ====== Pure Diffusion ==== ###########
def evalFluxXD(u,f):
  f[0] = u[0]*0.

def evalFluxYD(u,f):
  f[0] = u[0]*0.

### viscous fluxes
def evalViscousFluxXD(u,fv):
  fv[0] = u[0]
  fv[1] = 0.
#
def evalViscousFluxYD(u,fv):
  fv[0] = 0.
  fv[1] = u[0] 

def evalTauFluxXD(tau,u,fvX):
  fvX[0] = tau[0]

def evalTauFluxYD(tau,u,fvY):
  fvY[0] = tau[1]


######  ====== Linear advection fluxes and eigen values ==== ###########
def evalFluxXLA(u,f):
  f[0] = u[0]

def evalFluxYLA(u,f):
  f[0] = u[0]

### viscous fluxes
def evalViscousFluxXLA(u,fv):
  fv[0] = u[0]
  fv[1] = 0.
#
def evalViscousFluxYLA(u,fv):
  fv[0] = 0.
  fv[1] = u[0] 

def evalTauFluxXLA(tau,u,fvX):
  fvX[0] = tau[0]

def evalTauFluxYLA(tau,u,fvY):
  fvY[0] = tau[1]
#######################################################################
###### ====== Euler Fluxes and Eigen Values ==== ############
def evalFluxXEuler(u,f):
  #f = np.zeros(np.shape(u))
  gamma = 1.4
  p = (gamma - 1.)*(u[3] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0])
  f[0] = u[1]
  f[1] = u[1]**2/u[0] + p
  f[2] = u[1]*u[2]/u[0]
  f[3] = (u[3] + p)*u[1]/u[0]

def evalFluxYEuler(u,f):
  #f = np.zeros(np.shape(u))
  gamma = 1.4
  p = (gamma - 1.)*(u[3] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0])
  f[0] = u[2]
  f[1] = u[1]*u[2]/u[0]
  f[2] = u[2]**2/u[0] + p
  f[3] = (u[3] + p)*u[2]/u[0]


def getEigsEuler(ustarLR,ustarUD):
  gamma = 1.4
  eigsLR = np.zeros(np.shape(ustarLR))
  eigsUD = np.zeros(np.shape(ustarUD))
  pLR = (gamma - 1.)*ustarLR[0]*(ustarLR[3]/ustarLR[0] - 0.5*ustarLR[1]**2/ustarLR[0]**2 - 0.5*ustarLR[2]**2/ustarLR[0]**2)
  aLR = np.sqrt(gamma*pLR/ustarLR[0])
  pUD = (gamma - 1.)*ustarUD[0]*(ustarUD[3]/ustarUD[0] - 0.5*ustarUD[1]**2/ustarUD[0]**2 - 0.5*ustarUD[2]**2/ustarUD[0]**2)
  aUD = np.sqrt(gamma*pUD/ustarUD[0])
  eigsLR[0] = np.maximum(abs(ustarLR[1]/ustarLR[0] + aLR),abs(ustarLR[1]/ustarLR[0] - aLR))
  eigsLR[1] = eigsLR[0]
  eigsLR[2] = eigsLR[0]
  eigsLR[3] = eigsLR[0]
  eigsUD[0] = np.maximum(abs(ustarUD[2]/ustarUD[0] + aUD),abs(ustarUD[2]/ustarUD[0] - aUD))
  eigsUD[1] = eigsUD[0]
  eigsUD[2] = eigsUD[0]
  eigsUD[3] = eigsUD[0]
  return eigsLR,eigsUD

### viscous fluxes
def evalViscousFluxXNS(u,fv):
  fv[0] = 4./3.*u[1]
  fv[1] = u[2]
  fv[2] = u[2]#u[2]
  fv[3] = -2./3.*u[1]
#
def evalViscousFluxYNS(u,fv):
  fv[0] = -2./3.*u[2]
  fv[1] = u[1]
  fv[2] = u[1]
  fv[3] = 4./3.*u[2]


def evalTauFluxXNS(tau,u,fvX):
  fvX[0] = 0.
  fvX[1] = tau[0]
  fvX[2] = tau[2]
  fvX[3] = tau[0]*u[1]/u[0] + tau[1]*u[2]/u[0]

def evalTauFluxYNS(tau,u,fvY):
  fvY[0] = 0.
  fvY[1] = tau[1]
  fvY[2] = tau[3]
  fvY[3] = tau[2]*u[1]/u[0] + tau[3]*u[2]/u[0]

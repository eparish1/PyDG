import numpy as np

######  ====== Pure Diffusion ==== ###########
def evalFluxXD(u,f):
  f[0] = u[0]*0.

def evalFluxYD(u,f):
  f[0] = u[0]*0.


######  ====== Linear advection fluxes and eigen values ==== ###########
def evalFluxXLA(u,f):
  f[0] = u[0]

def evalFluxYLA(u,f):
  f[0] = u[0]

#######################################################################
###### ====== Euler Fluxes and Eigen Values ==== ############
def evalFluxXEuler(u,f):
  #f = np.zeros(np.shape(u))
  es = 1.e-30
  gamma = 1.4
  p = (gamma - 1.)*(u[3] - 0.5*u[1]**2/(u[0]+es) - 0.5*u[2]**2/(u[0]+es))
  f[0] = u[1]
  f[1] = u[1]**2/(u[0]+es) + p
  f[2] = u[1]*u[2]/(u[0]+es)
  f[3] = (u[3] + p)*u[1]/(u[0]+es)

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
  es = 1.e-30
  eigsLR = np.zeros(np.shape(ustarLR))
  eigsUD = np.zeros(np.shape(ustarUD))
  pLR = (gamma - 1.)*ustarLR[0]*(ustarLR[3]/(ustarLR[0]+es) - 0.5*ustarLR[1]**2/(ustarLR[0]**2+es) - 0.5*ustarLR[2]**2/(ustarLR[0]**2+es))
  aLR = np.sqrt(gamma*pLR/(ustarLR[0]+es))
  pUD = (gamma - 1.)*ustarUD[0]*(ustarUD[3]/(ustarUD[0]+es) - 0.5*ustarUD[1]**2/(ustarUD[0]**2+es) - 0.5*ustarUD[2]**2/(ustarUD[0]**2+es))
  aUD = np.sqrt(gamma*pUD/(ustarUD[0]+es))
  eigsLR[0] = np.maximum(abs(ustarLR[1]/(ustarLR[0]+es) + aLR),abs(ustarLR[1]/(ustarLR[0]+es) - aLR))
  eigsLR[1] = eigsLR[0]
  eigsLR[2] = eigsLR[0]
  eigsLR[3] = eigsLR[0]
  eigsUD[0] = np.maximum(abs(ustarUD[2]/(ustarUD[0]+es) + aUD),abs(ustarUD[2]/(ustarUD[0]+es) - aUD))
  eigsUD[1] = eigsUD[0]
  eigsUD[2] = eigsUD[0]
  eigsUD[3] = eigsUD[0]
  return eigsLR,eigsUD

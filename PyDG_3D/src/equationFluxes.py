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
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  f[0] = u[1]
  f[1] = u[1]*u[1]/(u[0]+es) + p
  f[2] = u[1]*u[2]/(u[0]+es)
  f[3] = u[1]*u[3]/(u[0]+es)
  f[4] = (u[4] + p)*u[1]/(u[0]+es)

def evalFluxYEuler(u,f):
  #f = np.zeros(np.shape(u))
  gamma = 1.4
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])

  f[0] = u[2]
  f[1] = u[1]*u[2]/u[0]
  f[2] = u[2]*u[2]/u[0] + p
  f[3] = u[2]*u[3]/u[0] 
  f[4] = (u[4] + p)*u[2]/u[0]

def evalFluxZEuler(u,f):
  #f = np.zeros(np.shape(u))
  gamma = 1.4
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  f[0] = u[3]
  f[1] = u[1]*u[3]/u[0]
  f[2] = u[2]*u[3]/u[0] + p
  f[3] = u[3]*u[3]/u[0] 
  f[4] = (u[4] + p)*u[3]/u[0]



def getEigs(ustarLR,ustarUD,ustarFB):
  eigsLR = zeros(shape(ustarLR))
  eigsUD = zeros(shape(ustarLR))
  eigsFB = zeros(shape(ustarLR))
  pLR = (gamma - 1.)*ustarLR[0]*(ustarLR[4]/ustarLR[0] - 0.5*ustarLR[1]**2/ustarLR[0]**2 - 0.5*ustarLR[2]**2/ustarLR[0]**2 - 0.5*ustarLR[3]**2/ustarLR[0]**2)
  aLR = sqrt(gamma*pLR/ustarLR[0])
  pUD = (gamma - 1.)*ustarUD[0]*(ustarUD[4]/ustarUD[0] - 0.5*ustarUD[1]**2/ustarUD[0]**2 - 0.5*ustarUD[2]**2/ustarUD[0]**2 - 0.5*ustarUD[3]**2/ustarUD[0]**2)
  aUD = sqrt(gamma*pUD/ustarUD[0])
  pFB = (gamma - 1.)*ustarFB[0]*(ustarFB[4]/ustarFB[0] - 0.5*ustarFB[1]**2/ustarFB[0]**2 - 0.5*ustarFB[2]**2/ustarFB[0]**2 - 0.5*ustarFB[3]**2/ustarFB[0]**2)
  aFB = sqrt(gamma*pFB/ustarFB[0])
###
  eigsLR[0] = maximum(abs(ustarLR[1]/ustarLR[0] + aLR),abs(ustarLR[1]/ustarLR[0] - aLR))
  eigsLR[1:5] = eigsLR[0]
###
  eigsUD[0] = maximum(abs(ustarUD[2]/ustarUD[0] + aUD),abs(ustarUD[2]/ustarUD[0] - aUD))
  eigsUD[1:5] = eigsUD[0]
###
  eigsFB[0] = maximum(abs(ustarFB[3]/ustarFB[0] + aFB),abs(ustarFB[3]/ustarFB[0] - aFB))
  eigsFB[1:5] = eigsFB[0]
  return eigsLR,eigsUD,eigsFB


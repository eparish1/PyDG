import numpy as np

##### =========== Contains all the fluxes and physics neccesary to solve the Navier-Stokes equations within a DG framework #### ============

## the navier_stokes_entropy_eqn module has energy replaced with entropy

###### ====== Inviscid Fluxes Fluxes and Eigen Values (Eigenvalues currently not in use) ==== ############
def evalFluxXEulerEntropyEqn(main,u,f,args): 
  #f = np.zeros(np.shape(u))
  es = 1.e-30
  gamma = 1.4
  p = np.exp(u[4]/u[0]) * u[0]**gamma 
  
  f[0] = u[1]
  f[1] = u[1]*u[1]/u[0] + p
  f[2] = u[1]*u[2]/u[0]
  f[3] = u[1]*u[3]/u[0]
  f[4] = u[1]*u[4]/u[0]


def evalFluxYEulerEntropyEqn(main,u,f,args):
  #f = np.zeros(np.shape(u))
  gamma = 1.4
  p = np.exp(u[4]/u[0]) * u[0]**gamma 
  f[0] = u[2]
  f[1] = u[1]*u[2]/u[0]
  f[2] = u[2]*u[2]/u[0] + p
  f[3] = u[2]*u[3]/u[0] 
  f[4] = u[2]*u[4]/u[0]



def evalFluxZEulerEntropyEqn(main,u,f,args):
  #f = np.zeros(np.shape(u))
  gamma = 1.4
  p = np.exp(u[4]/u[0]) * u[0]**gamma 
  f[0] = u[3]
  f[1] = u[1]*u[3]/u[0]
  f[2] = u[2]*u[3]/u[0] 
  f[3] = u[3]*u[3]/u[0] + p 
  f[4] = u[3]*u[4]/u[0]

#==================== Numerical Fluxes for the Faces =====================
#== central flux

def eulerCentralFluxEntropyEqn(main,UL,UR,pL,pR,n,args=None):
# PURPOSE: This function calculates the flux for the Euler equations
# using the Roe flux function
#
# INPUTS:
#    UL: conservative state vector in left cell
#    UR: conservative state vector in right cell
#    n: normal pointing from the left cell to the right cell
#
# OUTPUTS:
#  F   : the flux out of the left cell (into the right cell)
#  smag: the maximum propagation speed of disturbance
#
  gamma = 1.4
  gmi = gamma-1.0
  #process left state
  rL = UL[0]
  uL = UL[1]/rL
  vL = UL[2]/rL
  wL = UL[3]/rL
  pL = np.exp(UL[4]/UL[0]) * UL[0]**gamma 
  unL = uL*n[0] + vL*n[1] + wL*n[2]

  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = UL[4]*unL

  # process right state
  rR = UR[0]
  uR = UR[1]/rR
  vR = UR[2]/rR
  wR = UR[3]/rR
  pR = np.exp(UR[4]/UR[0]) * UR[0]**gamma 
  unR = uR*n[0] + vR*n[1] + wR*n[2]

  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = UR[4]*unR
  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])#-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])#-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])#-0.5*smax*(UR[2] - UL[2])
  F[3]    = 0.5*(FL[3]+FR[3])#-0.5*smax*(UR[3] - UL[3])
  F[4]    = 0.5*(FL[4]+FR[4])#-0.5*smax*(UR[4] - UL[4])
  return F


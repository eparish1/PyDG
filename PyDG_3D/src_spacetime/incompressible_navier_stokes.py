import numpy as np

##### =========== Contains all the fluxes and physics neccesary to solve the incompressible Navier-Stokes equations within a DG framework #### ============


###### ====== Inviscid Fluxes Fluxes and Eigen Values (Eigenvalues currently not in use) ==== ############
def evalFluxXEulerIncomp(main,u,f,args): 
  #f = np.zeros(np.shape(u))
  f[0] = u[0]*u[0] + u[3]
  f[1] = u[0]*u[1]
  f[2] = u[0]*u[2]
  f[3] = 0.

def evalFluxYEulerIncomp(main,u,f,args):
  #f = np.zeros(np.shape(u))
  f[0] = u[1]*u[0]
  f[1] = u[1]*u[1] + u[3]
  f[2] = u[1]*u[2] 
  f[3] = 0.


def evalFluxZEulerIncomp(main,u,f,args):
  #f = np.zeros(np.shape(u))
  f[0] = u[2]*u[0]
  f[1] = u[2]*u[1] 
  f[2] = u[2]*u[2] + u[3]
  f[3] = 0.


#==================== Numerical Fluxes for the Faces =====================
#== central flux
#== rusanov flux
#== Roe flux

def eulerCentralFluxIncomp(main,UL,UR,pL,pR,n,args=None):
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
  #process left state
  uL = UL[0]
  vL = UL[1]
  wL = UL[2]
  unL = uL*n[0] + vL*n[1] + wL*n[2]
  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = UL[0]*unL + UL[3]*n[0]
  FL[1] = UL[1]*unL + UL[3]*n[1]
  FL[2] = UL[2]*unL + UL[3]*n[2]

  # process right state
  uR = UR[0]
  vR = UR[1]
  wR = UR[2]
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = UR[0]*unR + UR[3]*n[0]
  FR[1] = UR[1]*unR + UR[3]*n[1]
  FR[2] = UR[2]*unR + UR[3]*n[2]

  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])#-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])#-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])#-0.5*smax*(UR[2] - UL[2])
  F[3]    = 0.#-0.5*smax*(UR[3] - UL[3])
  return F

def LaxFriedrichsFluxIncomp(main,UL,UR,pL,pR,n,args=None):
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
  #process left state
  uL = UL[0]
  vL = UL[1]
  wL = UL[2]
  unL = uL*n[0] + vL*n[1] + wL*n[2]
  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = UL[0]*unL + UL[3]*n[0]
  FL[1] = UL[1]*unL + UL[3]*n[1]
  FL[2] = UL[2]*unL + UL[3]*n[2]

  # process right state
  uR = UR[0]
  vR = UR[1]
  wR = UR[2]
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = UR[0]*unR + UR[3]*n[0]
  FR[1] = UR[1]*unR + UR[3]*n[1]
  FR[2] = UR[2]*unR + UR[3]*n[2]
  smax = np.fmax( 2.*np.abs(unL) , 2.*np.abs(unR))*1.
  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*smax*(UR[2] - UL[2])
  F[3]    = 0.#-0.5*smax*(UR[3] - UL[3])
  return F

###============= Diffusion Fluxes =====================
#diffusion Viscous Fluxes
def evalViscousFluxXIncomp_BR1(main,u,fv,dum):
  fv[:] = 0.
  fv[0::3] = u[:]

#
def evalViscousFluxYIncomp_BR1(main,u,fv,dum):
  fv[:] = 0.
  fv[1::3] = u[:]


def evalViscousFluxZIncomp_BR1(main,u,fv,dum):
  fv[:] = 0.
  fv[2::3] = u[:]

def evalTauFluxXIncomp_BR1(main,tau,u,fvX,mu,dum):
  fvX[:] = mu*tau[::3]
  fvX[-1] = tau[-3]
def evalTauFluxYIncomp_BR1(main,tau,u,fvY,mu,dum):
  fvY[:] = mu*tau[1::3]
  fvY[-1] = tau[-2]

def evalTauFluxZIncomp_BR1(main,tau,u,fvZ,mu,dum):
  fvZ[:] = mu*tau[2::3]
  fvZ[-1] = tau[-1]


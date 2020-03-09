import numpy as np

##### =========== Contains all the fluxes and physics neccesary to solve the incompressible Navier-Stokes equations with a fractional step method within a DG framework #### ============
# In the fractional step method, pressure is not in the main variable set

###### ====== Inviscid Fluxes Fluxes and Eigen Values (Eigenvalues currently not in use) ==== ############
def evalFluxXEulerIncompFrac(main,u,f,args): 
  #f = np.zeros(np.shape(u))
  f[0] = u[0]*u[0] 
  f[1] = u[0]*u[1]
  f[2] = u[0]*u[2]

def evalFluxYEulerIncompFrac(main,u,f,args):
  #f = np.zeros(np.shape(u))
  f[0] = u[1]*u[0]
  f[1] = u[1]*u[1]
  f[2] = u[1]*u[2] 

def evalFluxZEulerIncompFrac(main,u,f,args):
  #f = np.zeros(np.shape(u))
  f[0] = u[2]*u[0]
  f[1] = u[2]*u[1] 
  f[2] = u[2]*u[2]


#==================== Numerical Fluxes for the Faces =====================
#== central flux
#== rusanov flux
#== Roe flux

def eulerCentralFluxIncompFrac(main,UL,UR,pL,pR,n,args=None):
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
  FL[0] = UL[0]*unL 
  FL[1] = UL[1]*unL 
  FL[2] = UL[2]*unL 

  # process right state
  uR = UR[0]
  vR = UR[1]
  wR = UR[2]
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = UR[0]*unR
  FR[1] = UR[1]*unR 
  FR[2] = UR[2]*unR 

  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])#-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])#-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])#-0.5*smax*(UR[2] - UL[2])
  return F

def LaxFriedrichsFluxIncompFrac(main,UL,UR,pL,pR,n,args=None):
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
  FL[0] = UL[0]*unL 
  FL[1] = UL[1]*unL 
  FL[2] = UL[2]*unL 

  # process right state
  uR = UR[0]
  vR = UR[1]
  wR = UR[2]
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = UR[0]*unR 
  FR[1] = UR[1]*unR 
  FR[2] = UR[2]*unR 
  smax = np.fmax( 2.*np.abs(unL) , 2.*np.abs(unR))*1.
  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*smax*(UR[2] - UL[2])
  return F

###============= Diffusion Fluxes =====================
#diffusion Viscous Fluxes
def evalViscousFluxXIncompFrac_BR1(main,u,fv,dum):
  fv[:] = 0.
  fv[0::3] = u[:]

#
def evalViscousFluxYIncompFrac_BR1(main,u,fv,dum):
  fv[:] = 0.
  fv[1::3] = u[:]


def evalViscousFluxZIncompFrac_BR1(main,u,fv,dum):
  fv[:] = 0.
  fv[2::3] = u[:]

def evalTauFluxXIncompFrac_BR1(main,tau,u,fvX,mu,dum):
  fvX[:] = mu*tau[::3]

def evalTauFluxYIncompFrac_BR1(main,tau,u,fvY,mu,dum):
  fvY[:] = mu*tau[1::3]

def evalTauFluxZIncompFrac_BR1(main,tau,u,fvZ,mu,dum):
  fvZ[:] = mu*tau[2::3]


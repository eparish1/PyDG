import numpy as np

### Warning, this is not yet finished for the 3D Solver
## Flux information for linear advection, advection diffusion, and pure diffusion

######  ====== Pure Diffusion ==== ###########
def evalFluxD(main,u,f,args):
  f[0] = u[0]*0.


### Pure Diffusion Viscous Fluxes
def evalViscousFluxD_BR1(F,main,uL,uR,n,args=None):
  nvars = 3
  F[0] = 0.5*(uL[0] + uR[0])*n[0]
  F[1] = 0.5*(uL[0] + uR[0])*n[1]
  F[2] = 0.5*(uL[0] + uR[0])*n[2]

def evalTauFluxD_BR1(fV,main,uL,uR,n,args):
  mu = main.mus
  tauL = args[0]
  tauR = args[1]
  fV[:] = 0.5* mu*( (tauL[0]*n[0] + tauL[1]*n[1] + tauL[2]*n[2]) + (tauR[0]*n[0] + tauR[1]*n[1] + tauR[2]*n[2]) )
#

### Pure Diffusion Viscous Fluxes
def evalViscousFluxXD_BR1(main,u,fv):
  fv[0] = u[0]
  fv[1] = 0.
  fv[2] = 0.
#
def evalViscousFluxYD_BR1(main,u,fv):
  fv[0] = 0.
  fv[1] = u[0]
  fv[2] = 0.

def evalViscousFluxZD_BR1(main,u,fv):
  fv[0] = 0.
  fv[1] = 0.
  fv[2] = u[0]


def evalTauFluxXD_BR1(main,tau,u,fvX,mu,dum):
  fvX[0] = mu*tau[0]

def evalTauFluxYD_BR1(main,tau,u,fvY,mu,dum):
  fvY[0] = mu*tau[1]

def evalTauFluxZD_BR1(main,tau,u,fvZ,mu,dum):
  fvZ[0] = mu*tau[2]



######  ====== Linear advection fluxes and eigen values ==== ###########
cx = 1.
cy = 3.
cz = 1.
def evalFluxXLA(main,u,f,args):
  f[0] = cx*u[0]

def evalFluxYLA(main,u,f,args):
  f[0] = cy*u[0]

def evalFluxZLA(main,u,f,args):
  f[0] = cz*u[0]



#### ================ Flux schemes for the faces ========= ###########
def linearAdvectionCentralFlux(F,main,UL,UR,n,args=None):
  F[0] = cx*0.5*(UR[0] + UL[0])*n[0] + cy*0.5*(UR[0] + UL[0])*n[1] +  cz*0.5*(UR[0] + UL[0])*n[2]

  return F

def linearAdvectionUpwindFlux(F,main,UL,UR,n,args=None):
  F[:] = 0.
  F[0] = cx*UL[0]*n[0] + cy*UL[0]*n[1] +  cz*UL[0]*n[2]

  return F


### diffusion
def getGsD_X(u,main,mu,V):
  nvars = np.shape(u)[0]
  fvG11 = np.zeros(np.shape(u))
  fvG21 = np.zeros(np.shape(u))
  fvG31 = np.zeros(np.shape(u))
  fvG11[0] = mu*V[0]
  return fvG11,fvG21,fvG31

def getGsD_Y(u,main,mu,V):
  nvars = np.shape(u)[0]
  fvG12 = np.zeros(np.shape(u))
  fvG22 = np.zeros(np.shape(u))
  fvG32 = np.zeros(np.shape(u))
  fvG22[0] = mu*V[0]
  return fvG12,fvG22,fvG32

def getGsD_Z(u,main,mu,V):
  nvars = np.shape(u)[0]
  fvG13 = np.zeros(np.shape(u))
  fvG23 = np.zeros(np.shape(u))
  fvG33 = np.zeros(np.shape(u))
  fvG33[0] = mu*V[0]
  return fvG13,fvG23,fvG33

def diffusionCentralFlux(f,main,UL,UR,n,args=None):
  f[:] = np.zeros(np.shape(UL))

def evalViscousFluxXLA_IP(main,u,Ux,Uy,Uz,mu):
  fx = np.zeros(np.shape(u))
  fx[0] = mu*Ux[0]
  return fx

def evalViscousFluxYLA_IP(main,u,Ux,Uy,Uz,mu):
  fy = np.zeros(np.shape(u))
  fy[0] = mu*Uy[0]
  return fy

def evalViscousFluxZLA_IP(main,u,Ux,Uy,Uz,mu):
  fz = np.zeros(np.shape(u))
  fz[0] = mu*Uz[0]
  return fz



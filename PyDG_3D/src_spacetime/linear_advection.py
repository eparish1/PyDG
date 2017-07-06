import numpy as np

### Warning, this is not yet finished for the 3D Solver
## Flux information for linear advection, advection diffusion, and pure diffusion

######  ====== Pure Diffusion ==== ###########
def evalFluxD(u,f,args):
  f[0] = u[0]*0.


### Pure Diffusion Viscous Fluxes
def evalViscousFluxXD_BR1(u,fv):
  fv[0] = u[0]
  fv[1] = 0.
  fv[2] = 0.
#
def evalViscousFluxYD_BR1(u,fv):
  fv[0] = 0.
  fv[1] = u[0]
  fv[2] = 0.

def evalViscousFluxZD_BR1(u,fv):
  fv[0] = 0.
  fv[1] = 0.
  fv[2] = u[0]


def evalTauFluxXD_BR1(tau,u,fvX):
  fvX[0] = tau[0]

def evalTauFluxYD_BR1(tau,u,fvY):
  fvY[0] = tau[1]

def evalTauFluxYD_BR1(tau,u,fvZ):
  fvZ[0] = tau[2]


### Pure Diffusion Viscous Fluxes
def evalViscousFluxXD_BR1(u,fv):
  fv[0] = u[0]
  fv[1] = 0.
  fv[2] = 0.
#
def evalViscousFluxYD_BR1(u,fv):
  fv[0] = 0.
  fv[1] = u[0]
  fv[2] = 0.

def evalViscousFluxZD_BR1(u,fv):
  fv[0] = 0.
  fv[1] = 0.
  fv[2] = u[0]


def evalTauFluxXD_BR1(tau,u,fvX):
  fvX[0] = tau[0]

def evalTauFluxYD_BR1(tau,u,fvY):
  fvY[0] = tau[1]

def evalTauFluxYD_BR1(tau,u,fvZ):
  fvZ[0] = tau[2]


######  ====== Linear advection fluxes and eigen values ==== ###########
def evalFluxXLA(u,f,args):
  f[0] = u[0]

def evalFluxYLA(u,f,args):
  f[0] = u[0]

def evalFluxZLA(u,f,args):
  f[0] = u[0]



#### ================ Flux schemes for the faces ========= ###########
def linearAdvectionCentralFlux(UL,UR,n,args=None):
  F = np.zeros(np.shape(UL))
  #F[0] = 0.5*(UR[0] + UL[0])
  F[0] = UL[0]

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

def diffusionCentralFlux(UL,UR,n,args=None):
  f = np.zeros(np.shape(UL))
  return f

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



import numpy as np

### Warning, this is not yet finished for the 3D Solver
## Flux information for linear advection, advection diffusion, and pure diffusion

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

def evalFluxZLA(u,f):
  f[0] = u[0]



#### ================ Flux schemes for the faces ========= ###########
def linearAdvectionCentralFlux(UL,UR,n):
  F = np.zeros(np.shape(UL))
  F[0] = 0.5*(UR[0] + UL[0])
  return F


### diffusion
def getGsLA(u,main):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G11 = np.zeros(ashape)
  G12 = np.zeros(ashape)
  G21 = np.zeros(ashape)
  G22 = np.zeros(ashape)
  G11[0] = 1.
  G22[0] = 1.
  G11 = G11*main.mu
  G12 = G12*main.mu
  G21 = G21*main.mu
  G22 = G22*main.mu
  return G11,G12,G21,G22

def evalViscousFluxXLA_IP(main,u,Ux,Uy):
  fx = np.zeros(np.shape(u))
  fx[0] = main.mu*Ux[0]
  return fx

def evalViscousFluxYLA_IP(main,u,Ux,Uy):
  fy = np.zeros(np.shape(u))
  fy[0] = main.mu*Uy[0]
  return fy


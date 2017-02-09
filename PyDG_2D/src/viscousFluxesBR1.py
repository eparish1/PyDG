import numpy as np

### Pure Diffusion Viscous Fluxes
def evalViscousFluxXD_BR1(u,fv):
  fv[0] = u[0]
  fv[1] = 0.
#
def evalViscousFluxYD_BR1(u,fv):
  fv[0] = 0.
  fv[1] = u[0]

def evalTauFluxXD_BR1(tau,u,fvX):
  fvX[0] = tau[0]

def evalTauFluxYD_BR1(tau,u,fvY):
  fvY[0] = tau[1]


### Linear Advection  Viscous Fluxes
def evalViscousFluxXLA_BR1(u,fv):
  fv[0] = u[0]
  fv[1] = 0.
#
def evalViscousFluxYLA_BR1(u,fv):
  fv[0] = 0.
  fv[1] = u[0]

def evalTauFluxXLA_BR1(tau,u,fvX):
  fvX[0] = tau[0]

def evalTauFluxYLA_BR1(tau,u,fvY):
  fvY[0] = tau[1]


### Navier Stokes Viscous Fluxes
def evalViscousFluxXNS_BR1(u,fv):
  fv[0] = 4./3.*u[1]
  fv[1] = u[2]
  fv[2] = u[2]#u[2]
  fv[3] = -2./3.*u[1]
#
def evalViscousFluxYNS_BR1(u,fv):
  fv[0] = -2./3.*u[2]
  fv[1] = u[1]
  fv[2] = u[1]
  fv[3] = 4./3.*u[2]


def evalTauFluxXNS_BR1(tau,u,fvX):
  fvX[0] = 0.
  fvX[1] = tau[0]
  fvX[2] = tau[2]
  fvX[3] = tau[0]*u[1]/u[0] + tau[1]*u[2]/u[0]

def evalTauFluxYNS_BR1(tau,u,fvY):
  fvY[0] = 0.
  fvY[1] = tau[1]
  fvY[2] = tau[3]
  fvY[3] = tau[2]*u[1]/u[0] + tau[3]*u[2]/u[0]


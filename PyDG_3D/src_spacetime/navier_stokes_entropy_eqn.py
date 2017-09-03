import numpy as np

##### =========== Contains all the fluxes and physics neccesary to solve the Navier-Stokes equations within a DG framework #### ============

## the navier_stokes_entropy_eqn module has energy replaced with entropy

###### ====== Inviscid Fluxes Fluxes and Eigen Values (Eigenvalues currently not in use) ==== ############
def evalFluxXEulerEntropy(main,u,f,args): 
  #f = np.zeros(np.shape(u))
  es = 1.e-30
  gamma = 1.4
  p = np.exp(u[4]/u[0]) * u[0]**gamma 
  
  f[0] = u[1]
  f[1] = u[1]*u[1]/u[0] + p
  f[2] = u[1]*u[2]/u[0]
  f[3] = u[1]*u[3]/u[0]
  f[4] = u[1]*u[4]/u[0]


def evalFluxYEulerEntropy(main,u,f,args):
  #f = np.zeros(np.shape(u))
  gamma = 1.4
  p = np.exp(u[4]/u[0]) * u[0]**gamma 
  f[0] = u[2]
  f[1] = u[1]*u[2]/u[0]
  f[2] = u[2]*u[2]/u[0] + p
  f[3] = u[2]*u[3]/u[0] 
  f[4] = u[2]*u[4]/u[0]



def evalFluxZEulerEntropy(main,u,f,args):
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

def eulerCentralFluxEntropy(main,UL,UR,pL,pR,n,args=None):
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



###============= Diffusion Fluxes =====================
def getGsNS(u,main):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G11 = np.zeros(ashape)
  G21 = np.zeros(ashape)
  G31 = np.zeros(ashape)
  G12 = np.zeros(ashape)
  G22 = np.zeros(ashape)
  G32 = np.zeros(ashape)
  G13 = np.zeros(ashape)
  G23 = np.zeros(ashape)
  G33 = np.zeros(ashape)

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2
  mu_by_rho = main.mu[0]/u[0]
  G11[1,0] = -4./3.*v1*mu_by_rho
  G11[1,1] = 4./3.*mu_by_rho
  G11[2,0] = -v2*mu_by_rho
  G11[2,2] = mu_by_rho
  G11[3,0] = -v3*mu_by_rho
  G11[3,3] = mu_by_rho
  G11[4,0] = -(4./3.*v1**2 + v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G11[4,1] = (4./3. - gamma/Pr)*v1*mu_by_rho
  G11[4,2] = (1. - gamma/Pr)*v2*mu_by_rho
  G11[4,3] = (1. - gamma/Pr)*v3*mu_by_rho
  G11[4,4] = gamma/Pr*mu_by_rho

  G21[1,0] = G11[2,0]#-v2*mu_by_rho
  G21[1,2] = mu_by_rho
  G21[2,0] = 2./3.*v1*mu_by_rho
  G21[2,1] = -2./3.*mu_by_rho
  G21[4,0] = -1./3.*v1*v2*mu_by_rho
  G21[4,1] = -2./3.*v2*mu_by_rho
  G21[4,2] = v1*mu_by_rho

  G31[1,0] = G11[3,0]#-v3*mu_by_rho
  G31[1,3] = mu_by_rho
  G31[3,0] = G21[2,0]#2./3.*v1*mu_by_rho
  G31[3,1] = G21[2,1]#-2./3.*mu_by_rho
  G31[4,0] = -1./3.*v1*v3*mu_by_rho
  G31[4,1] = 2./3.*G11[3,0]#-2./3.*v3*mu_by_rho
  G31[4,3] = G21[4,2]#v1*mu_by_rho

  G12[1,0] = -G21[4,1]#2./3.*v2*mu_by_rho
  G12[1,2] = G21[2,1]#-2./3.*mu_by_rho
  G12[2,0] = -G31[4,3]#-v1*mu_by_rho
  G12[2,1] = mu_by_rho
  G12[4,0] = G21[4,0]#-1./3.*v1*v2*mu_by_rho
  G12[4,1] = -G21[1,0]#v2*mu_by_rho
  G12[4,2] = -G31[3,0]#-2./3.*v1*mu_by_rho

  G22[1,0] = G12[2,0]#-v1*mu_by_rho
  G22[1,1] = mu_by_rho
  G22[2,0] = -4./3.*v2*mu_by_rho
  G22[2,2] = 4./3.*mu_by_rho
  G22[3,0] = G31[1,0]#-v3*mu_by_rho
  G22[3,3] = mu_by_rho
  G22[4,0] = -(v1**2 + 4./3.*v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G22[4,1] = (1. - gamma/Pr)*v1*mu_by_rho
  G22[4,2] = (4./3. - gamma/Pr)*v2*mu_by_rho
  G22[4,3] = G11[4,3]#(1. - gamma/Pr)*v3*mu_by_rho
  G22[4,4] = G11[4,4]#gamma/Pr*mu_by_rho

  G32[2,0] = G22[3,0]#-v3*mu_by_rho
  G32[2,3] = mu_by_rho
  G32[3,0] = G12[1,0]#2./3.*v2*mu_by_rho
  G32[3,2] = G12[1,2]#-2./3.*mu_by_rho
  G32[4,0] = -1./3.*v2*v3*mu_by_rho
  G32[4,2] = G31[4,1]#-2./3.*v3*mu_by_rho
  G32[4,3] = G12[4,1]#v2*mu_by_rho

  G13[1,0] = -G31[4,1]#2./3.*v3*mu_by_rho
  G13[1,3] = -2./3.*mu_by_rho
  G13[3,0] = G12[2,0]#-v1*mu_by_rho
  G13[3,1] = mu_by_rho
  G13[4,0] = G31[4,0]#-1./3.*v1*v3*mu_by_rho
  G13[4,1] = -G31[1,0]#v3*mu_by_rho
  G13[4,3] = G12[4,2]#-2./3.*v1*mu_by_rho

  G23[2,0] = G13[1,0]#2./3.*v3*mu_by_rho
  G23[2,3] = G12[1,2]#-2./3.*mu_by_rho
  G23[3,0] = G21[1,0]#-v2*mu_by_rho
  G23[3,2] = mu_by_rho
  G23[4,0] = -1./3.*v2*v3*mu_by_rho
  G23[4,2] = G13[4,1]#v3*mu_by_rho
  G23[4,3] = G21[4,1]#-2./3.*v2*mu_by_rho

  G33[1,0] = G12[2,0]#-v1*mu_by_rho
  G33[1,1] = mu_by_rho
  G33[2,0] = G21[1,0]#-v2*mu_by_rho
  G33[2,2] = mu_by_rho
  G33[3,0] = -4./3.*v3*mu_by_rho
  G33[3,3] = 4./3.*mu_by_rho
  G33[4,0] = -(v1**2 + v2**2 + 4./3.*v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G33[4,1] = G22[4,1]#(1. - gamma/Pr)*v1*mu_by_rho
  G33[4,2] = G11[4,2]#(1. - gamma/Pr)*v2*mu_by_rho
  G33[4,3] = (4./3. - gamma/Pr)*v3*mu_by_rho
  G33[4,4] = G11[4,4]#gamma/Pr*mu_by_rho

  return G11,G12,G13,G21,G22,G23,G31,G32,G33



def getGsNSX_FAST(u,main,mu,V):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  fvG11 = np.zeros(np.shape(u))
  fvG21 = np.zeros(np.shape(u))
  fvG31 = np.zeros(np.shape(u))

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  mu_by_rho = mu[0]/u[0]
  fvG11[1] = 4./3.*mu_by_rho*(V[1] - v1*V[0])
  fvG11[2] = mu_by_rho*(V[2] - v2*V[0])
  fvG11[3] = mu_by_rho*(V[3] - v3*V[0])
  fvG11[4] =  -(4./3.*v1**2 + v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*V[0] + \
            (4./3. - gamma/Pr)*v1*V[1] + (1. - gamma/Pr)*v2*V[2] + \
            (1. - gamma/Pr)*v3*V[3] + gamma/Pr*V[4]
  fvG11[4] *= mu_by_rho


  fvG21[1] = mu_by_rho*(V[2] - v2*V[0])
  fvG21[2] = 2./3.*mu_by_rho*(v1*V[0] - V[1])
  fvG21[3] = 0
  fvG21[4] = mu_by_rho*(v1*V[2] - 2./3.*v2*V[1] - 1./3.*v1*v2*V[0] )

  fvG31[1] = mu_by_rho*(V[3] - v3*V[0])
  fvG31[3] = 2./3.*mu_by_rho*(v1*V[0] - V[1])
  fvG31[4] = mu_by_rho*(v1*V[3] - 2./3.*v3*V[1] - 1./3.*v1*v3*V[0])
  return fvG11,fvG21,fvG31


def getGsNSY_FAST(u,main,mu,V):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  fvG12 = np.zeros(np.shape(u))
  fvG22 = np.zeros(np.shape(u))
  fvG32 = np.zeros(np.shape(u))

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  mu_by_rho = mu[0]/u[0]
  fvG12[1] = 2./3.*mu_by_rho*(v2*V[0] - V[2])
  fvG12[2] = mu_by_rho*(V[1] - v1*V[0])
  fvG12[4] = mu_by_rho*(-2./3.*v1*V[2] + v2*V[1] - 1./3.*v1*v2*V[0])


  fvG22[1] = mu_by_rho*(V[1] - v1*V[0])
  fvG22[2] = 4./3.*mu_by_rho*(V[2] - v2*V[0])
  fvG22[3] = mu_by_rho*(V[3] - v3*V[0])
  fvG22[4] = -(v1**2 + 4./3.*v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*V[0] + \
             (1. - gamma/Pr)*v1*V[1] + (4./3. - gamma/Pr)*v2*V[2] + \
             (1. - gamma/Pr)*v3*V[3] +  gamma/Pr*V[4]
  fvG22[4] *= mu_by_rho

  fvG32[2] = mu_by_rho*(V[3] - v3*V[0])
  fvG32[3] = 2./3.*mu_by_rho*(v2*V[0] - V[2])
  fvG32[4] = mu_by_rho*(v2*V[3] -2./3.*v3*V[2] - 1./3.*v2*v3*V[0])


  return fvG12,fvG22,fvG32

def getGsNSZ_FAST(u,main,mu,V):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  fvG13 = np.zeros(np.shape(u))
  fvG23 = np.zeros(np.shape(u))
  fvG33 = np.zeros(np.shape(u))

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  mu_by_rho = mu[0]/u[0]
  fvG13[1] = 2./3.*mu_by_rho*(v3*V[0] - V[3])
  fvG13[3] = mu_by_rho*(V[1] - v1*V[0])
  fvG13[4] = mu_by_rho*(-2./3.*v1*V[3] + v3*V[1] - 1./3.*v1*v3*V[0])

  fvG23[2] = 2./3.*mu_by_rho*(v3*V[0] - V[3])
  fvG23[3] = mu_by_rho*(V[2] - v2*V[0])
  fvG23[4] = mu_by_rho*(-2./3.*v2*V[3] + v3*V[2] - 1./3.*v2*v3*V[0])


  fvG33[1] = mu_by_rho*(V[1] - v1*V[0])
  fvG33[2] = mu_by_rho*(V[2] - v2*V[0])
  fvG33[3] = 4./3.*mu_by_rho*(V[3] - v3*V[0])
 
  fvG33[4] = -(v1**2 + v2**2 + 4./3.*v3**2 + gamma/Pr*(E - vsqr) )*V[0] + \
             (1. - gamma/Pr)*v1*V[1] + (1. - gamma/Pr)*v2*V[2] + \
             (4./3. - gamma/Pr)*v3*V[3] + gamma/Pr*V[4]
  fvG33[4] *= mu_by_rho

  return fvG13,fvG23,fvG33



def getGsNSX(u,main,mu):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G11 = np.zeros(ashape)
  G21 = np.zeros(ashape)
  G31 = np.zeros(ashape)

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  mu_by_rho = mu[0]/u[0]
  G11[1,0] = -4./3.*v1*mu_by_rho
  G11[1,1] = 4./3.*mu_by_rho
  G11[2,0] = -v2*mu_by_rho
  G11[2,2] = mu_by_rho
  G11[3,0] = -v3*mu_by_rho
  G11[3,3] = mu_by_rho
  G11[4,0] = -(4./3.*v1**2 + v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G11[4,1] = (4./3. - gamma/Pr)*v1*mu_by_rho
  G11[4,2] = (1. - gamma/Pr)*v2*mu_by_rho
  G11[4,3] = (1. - gamma/Pr)*v3*mu_by_rho
  G11[4,4] = gamma/Pr*mu_by_rho

  G21[1,0] = G11[2,0]#-v2*mu_by_rho
  G21[1,2] = mu_by_rho
  G21[2,0] = 2./3.*v1*mu_by_rho
  G21[2,1] = -2./3.*mu_by_rho
  G21[4,0] = -1./3.*v1*v2*mu_by_rho
  G21[4,1] = -2./3.*v2*mu_by_rho
  G21[4,2] = v1*mu_by_rho

  G31[1,0] = G11[3,0]#-v3*mu_by_rho
  G31[1,3] = mu_by_rho
  G31[3,0] = G21[2,0]#2./3.*v1*mu_by_rho
  G31[3,1] = G21[2,1]#-2./3.*mu_by_rho
  G31[4,0] = -1./3.*v1*v3*mu_by_rho
  G31[4,1] = 2./3.*G11[3,0]#-2./3.*v3*mu_by_rho
  G31[4,3] = G21[4,2]#v1*mu_by_rho
  return G11,G21,G31



def getGsNSY(u,main,mu):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G12 = np.zeros(ashape)
  G22 = np.zeros(ashape)
  G32 = np.zeros(ashape)

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  mu_by_rho = mu[0]/u[0]
  G12[1,0] = 2./3.*v2*mu_by_rho
  G12[1,2] = -2./3.*mu_by_rho
  G12[2,0] = -v1*mu_by_rho
  G12[2,1] = mu_by_rho
  G12[4,0] = -1./3.*v1*v2*mu_by_rho
  G12[4,1] = v2*mu_by_rho
  G12[4,2] = -2./3.*v1*mu_by_rho

  G22[1,0] = G12[2,0]#-v1*mu_by_rho
  G22[1,1] = mu_by_rho
  G22[2,0] = -4./3.*v2*mu_by_rho
  G22[2,2] = 4./3.*mu_by_rho
  G22[3,0] = -v3*mu_by_rho
  G22[3,3] = mu_by_rho
  G22[4,0] = -(v1**2 + 4./3.*v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G22[4,1] = (1. - gamma/Pr)*v1*mu_by_rho
  G22[4,2] = (4./3. - gamma/Pr)*v2*mu_by_rho
  G22[4,3] = (1. - gamma/Pr)*v3*mu_by_rho
  G22[4,4] = gamma/Pr*mu_by_rho

  G32[2,0] = G22[3,0]#-v3*mu_by_rho
  G32[2,3] = mu_by_rho
  G32[3,0] = G12[1,0]#2./3.*v2*mu_by_rho
  G32[3,2] = G12[1,2]#-2./3.*mu_by_rho
  G32[4,0] = -1./3.*v2*v3*mu_by_rho
  G32[4,2] = -2./3.*v3*mu_by_rho
  G32[4,3] = G12[4,1]#v2*mu_by_rho
  return G12,G22,G32

def getGsNSZ(u,main,mu):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G13 = np.zeros(ashape)
  G23 = np.zeros(ashape)
  G33 = np.zeros(ashape)

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2
  mu_by_rho = mu[0]/u[0]

  G13[1,0] = 2./3.*v3*mu_by_rho
  G13[1,3] = -2./3.*mu_by_rho
  G13[3,0] = -v1*mu_by_rho
  G13[3,1] = mu_by_rho
  G13[4,0] = -1./3.*v1*v3*mu_by_rho
  G13[4,1] = v3*mu_by_rho
  G13[4,3] = -2./3.*v1*mu_by_rho

  G23[2,0] = G13[1,0]#2./3.*v3*mu_by_rho
  G23[2,3] = -2./3.*mu_by_rho
  G23[3,0] = -v2*mu_by_rho
  G23[3,2] = mu_by_rho
  G23[4,0] = -1./3.*v2*v3*mu_by_rho
  G23[4,2] = G13[4,1]#v3*mu_by_rho
  G23[4,3] = -2./3.*v2*mu_by_rho

  G33[1,0] = -v1*mu_by_rho
  G33[1,1] = mu_by_rho
  G33[2,0] = -v2*mu_by_rho
  G33[2,2] = mu_by_rho
  G33[3,0] = -4./3.*v3*mu_by_rho
  G33[3,3] = 4./3.*mu_by_rho
  G33[4,0] = -(v1**2 + v2**2 + 4./3.*v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G33[4,1] = (1. - gamma/Pr)*v1*mu_by_rho
  G33[4,2] = (1. - gamma/Pr)*v2*mu_by_rho
  G33[4,3] = (4./3. - gamma/Pr)*v3*mu_by_rho
  G33[4,4] = gamma/Pr*mu_by_rho
  return G13,G23,G33


def evalViscousFluxXNS_IP(main,u,Ux,Uy,Uz,mu):
  gamma = 1.4
  Pr = 0.72
  ## ->  v_x = 1/rho d/dx(rho v) - rho v /rho^2 rho_x
  ux = 1./u[0]*(Ux[1] - u[1]/u[0]*Ux[0])
  vx = 1./u[0]*(Ux[2] - u[2]/u[0]*Ux[0])
  wx = 1./u[0]*(Ux[3] - u[3]/u[0]*Ux[0])
  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  uy = 1./u[0]*(Uy[1] - u[1]/u[0]*Uy[0])
  vy = 1./u[0]*(Uy[2] - u[2]/u[0]*Uy[0])
  ## ->  u_z = 1/rho d/dz(rho u) - rho u /rho^2 rho_z
  uz = 1./u[0]*(Uz[1] - u[1]/u[0]*Uz[0])
  wz = 1./u[0]*(Uz[3] - u[3]/u[0]*Uz[0])
  ## -> (kT)_x =d/dx[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dx - 0.5 d/dx(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dx - (v1 v1_x + v2 v2_x) ]
  ## ->  E_x = 1/rho d/x(rho E) - rho E /rho^2 rho_x
  kTx =( 1./u[0]*(Ux[4] - u[4]/u[0]*Ux[0] - (u[1]*ux + u[2]*vx + u[3]*wx)  ))*mu[0]*gamma/Pr

  fx = np.zeros(np.shape(u))
  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  fx[1] = 2./3.*mu[0]*(2.*ux - vy - wz) #tau11
  fx[2] = mu[0]*(uy + vx)  #tau11
  fx[3] = mu[0]*(uz + wx) #tau13
  fx[4] = fx[1]*v1 + fx[2]*v2 + fx[3]*v3 + kTx
  return fx


def evalViscousFluxYNS_IP(main,u,Ux,Uy,Uz,mu):
  gamma = 1.4
  Pr = 0.72
  ## ->  v_x = 1/rho d/dx(rho v) - rho v /rho^2 rho_x
  ux = 1./u[0]*(Ux[1] - u[1]/u[0]*Ux[0])
  vx = 1./u[0]*(Ux[2] - u[2]/u[0]*Ux[0])
  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  uy = 1./u[0]*(Uy[1] - u[1]/u[0]*Uy[0])
  vy = 1./u[0]*(Uy[2] - u[2]/u[0]*Uy[0])
  wy = 1./u[0]*(Uy[3] - u[3]/u[0]*Uy[0])
  ## ->  u_z = 1/rho d/dz(rho u) - rho u /rho^2 rho_z
  vz = 1./u[0]*(Uz[2] - u[2]/u[0]*Uz[0])
  wz = 1./u[0]*(Uz[3] - u[3]/u[0]*Uz[0])
  ## -> (kT)_x =d/dx[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dx - 0.5 d/dx(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dx - (v1 v1_x + v2 v2_x) ]
  ## ->  E_x = 1/rho d/x(rho E) - rho E /rho^2 rho_x
  kTy =( 1./u[0]*(Uy[4] - u[4]/u[0]*Uy[0] - (u[1]*uy + u[2]*vy + u[3]*wy)  ))*mu[0]*gamma/Pr

  fy = np.zeros(np.shape(u))
  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  fy[1] = mu[0]*(vx + uy)  #tau12
  fy[2] = 2./3.*mu[0]*(2.*vy - ux - wz) #tau22
  fy[3] = mu[0]*(vz + wy) #tau23
  fy[4] = fy[1]*v1 + fy[2]*v2 + fy[3]*v3 + kTy
  return fy

def evalViscousFluxZNS_IP(main,u,Ux,Uy,Uz,mu):
  gamma = 1.4
  Pr = 0.72
  ## ->  v_x = 1/rho d/dx(rho v) - rho v /rho^2 rho_x
  ux = 1./u[0]*(Ux[1] - u[1]/u[0]*Ux[0])
  wx = 1./u[0]*(Ux[3] - u[3]/u[0]*Ux[0])
  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  vy = 1./u[0]*(Uy[2] - u[2]/u[0]*Uy[0])
  wy = 1./u[0]*(Uy[3] - u[3]/u[0]*Uy[0])
  ## ->  u_z = 1/rho d/dz(rho u) - rho u /rho^2 rho_z
  uz = 1./u[0]*(Uz[1] - u[1]/u[0]*Uz[0])
  vz = 1./u[0]*(Uz[2] - u[2]/u[0]*Uz[0])
  wz = 1./u[0]*(Uz[3] - u[3]/u[0]*Uz[0])
  ## -> (kT)_x =d/dx[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dx - 0.5 d/dx(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dx - (v1 v1_x + v2 v2_x) ]
  ## ->  E_x = 1/rho d/x(rho E) - rho E /rho^2 rho_x
  kTz =( 1./u[0]*(Uz[4] - u[4]/u[0]*Uz[0] - (u[1]*uz + u[2]*vz + u[3]*wz) ) )*mu[0]*gamma/Pr

  fz = np.zeros(np.shape(u))
  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  fz[1] = mu[0]*(uz + wx)  #tau13
  fz[2] = mu[0]*(vz + wy)  #tau23
  fz[3] = 2./3.*mu[0]*(2.*wz - ux - vy)
  fz[4] = fz[1]*v1 + fz[2]*v2 + fz[3]*v3 + kTz
  return fz


### Diffusion fluxes for BR1

### viscous fluxes
def evalViscousFluxXNS_BR1(main,U,fv,cgas):
  u = U[1]/U[0]
  v = U[2]/U[0]
  w = U[3]/U[0]
  fv[0] =  4./3.*u  #tau11 = (du/dx + du/dx - 2/3 (du/dx + dv/dy + dw/dz) ) 
  fv[1] = -2./3.*u  #tau22 = (dv/dy + dv/dy - 2/3 (du/dx + dv/dy + dw/dZ) )
  fv[2] = -2./3.*u  #tau33 = (dw/dz + dw/dz - 2/3 (du/dx + dv/dy + dw/dz) )
  fv[3] = v         #tau12 = (du/dy + dv/dx)
  fv[4] = w         #tau13 = (du/dz + dw/dx)
  fv[5] = 0.           #tau23 = (dv/dz + dw/dy)
  T = (U[4]/U[0] - 0.5*( u**2 + v**2 + w**2 ) ) #kinda a psuedo tmp, should divide by Cv but it's constant so this is taken care of in the tauFlux with gamma
  fv[6] = T
  fv[7] = 0.
  fv[8] = 0.
#
def evalViscousFluxYNS_BR1(main,U,fv,cgas):
  u = U[1]/U[0]
  v = U[2]/U[0]
  w = U[3]/U[0]
  fv[0] = -2./3.*v  #tau11 = (du/dx + du/dx - 2/3 (du/dx + dv/dy + dw/dz) ) 
  fv[1] =  4./3.*v  #tau22 = (dv/dy + dv/dy - 2/3 (du/dx + dv/dy + dw/dZ) )
  fv[2] = -2./3.*v  #tau33 = (dw/dz + dw/dz - 2/3 (du/dx + dv/dy + dw/dz) )
  fv[3] = u        #tau12 = (du/dy + dv/dx)
  fv[4] = 0            #tau13 = (du/dz + dw/dx)
  fv[5] = w         #tau23 = (dv/dz + dw/dy)
  fv[6] = 0.
  T = (U[4]/U[0] - 0.5*( u**2 + v**2 + w**2 ) )
  fv[7] = T
  fv[8] = 0.

def evalViscousFluxZNS_BR1(main,U,fv,cgas):
  u = U[1]/U[0]
  v = U[2]/U[0]
  w = U[3]/U[0]
  fv[0] = -2./3.*w  #tau11 = (du/dx + du/dx - 2/3 (du/dx + dv/dy + dw/dz) ) 
  fv[1] = -2./3.*w  #tau22 = (dv/dy + dv/dy - 2/3 (du/dx + dv/dy + dw/dZ) )
  fv[2] =  4./3.*w  #tau33 = (dw/dz + dw/dz - 2/3 (du/dx + dv/dy + dw/dz) )
  fv[3] = 0.           #tau12 = (du/dy + dv/dx)
  fv[4] = u         #tau13 = (du/dz + dw/dx)
  fv[5] = v        #tau23 = (dv/dz + dw/dy)
  T = (U[4]/U[0] - 0.5*( u**2 + v**2 + w**2 ) )
  fv[6] = 0.
  fv[7] = 0.
  fv[8] = T

def evalTauFluxXNS_BR1(main,tau,u,fvX,mu,cgas):
  Pr = 0.72
  gamma = 1.4
  fvX[0] = 0.
  fvX[1] = mu*tau[0] #tau11
  fvX[2] = mu*tau[3] #tau21
  fvX[3] = mu*tau[4] #tau31
  fvX[4] = mu*(tau[0]*u[1]/u[0] + tau[3]*u[2]/u[0] + tau[4]*u[3]/u[0] + gamma/Pr*tau[6] )

def evalTauFluxYNS_BR1(main,tau,u,fvY,mu,cgas):
  Pr = 0.72
  gamma = 1.4
  fvY[0] = 0.
  fvY[1] = mu*tau[3] #tau21
  fvY[2] = mu*tau[1] #tau22
  fvY[3] = mu*tau[5] #tau23
  fvY[4] = mu*(tau[3]*u[1]/u[0] + tau[1]*u[2]/u[0] + tau[5]*u[3]/u[0] + gamma/Pr*tau[7])

def evalTauFluxZNS_BR1(main,tau,u,fvZ,mu,cgas):
  Pr = 0.72
  gamma = 1.4
  fvZ[0] = 0.
  fvZ[1] = mu*tau[4] #tau31
  fvZ[2] = mu*tau[5] #tau32
  fvZ[3] = mu*tau[2] #tau33
  fvZ[4] = mu*(tau[4]*u[1]/u[0] + tau[5]*u[2]/u[0] + tau[2]*u[3]/u[0] + gamma/Pr*tau[8])


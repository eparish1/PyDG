import numpy as np
from eos_functions import *
##### =========== Contains all the fluxes and physics neccesary to solve the Navier-Stokes equations within a DG framework #### ============


###### ====== Inviscid Fluxes Fluxes and Eigen Values (Eigenvalues currently not in use) ==== ############
def evalFluxXEuler_reacting(main,u,f,args): 
  #f = np.zeros(np.shape(u))
  es = 1.e-30
  gamma = 1.4
  #p = (main.a.gamma_star - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  #p,T = computePressure_and_Temperature_Cantera(main,u,main.cgas_field)
  p = main.a.p
  f[0] = u[1]
  f[1] = u[1]*u[1]/(u[0]) + p
  f[2] = u[1]*u[2]/(u[0])
  f[3] = u[1]*u[3]/(u[0])
  f[4] = (u[4] + p)*u[1]/(u[0])
  f[5::] = u[None,1]*u[5::]/u[None,0] 
# for i in range(0,np.shape(u)[0]-5):
#    f[5+i] = u[1]*u[5+i]/u[0]
 

def evalFluxYEuler_reacting(main,u,f,args):
  #f = np.zeros(np.shape(u))
  gamma = 1.4
  #p = (main.a.gamma_star - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
#  p,T = computePressure_and_Temperature_Cantera(main,u,main.cgas_field)
  p = main.a.p
  f[0] = u[2]
  f[1] = u[1]*u[2]/u[0]
  f[2] = u[2]*u[2]/u[0] + p
  f[3] = u[2]*u[3]/u[0] 
  f[4] = (u[4] + p)*u[2]/u[0]
  f[5::] = u[None,2]*u[5::]/u[None,0] 

#  for i in range(0,np.shape(u)[0]-5):
#    f[5+i] = u[2]*u[5+i]/u[0]



def evalFluxZEuler_reacting(main,u,f,args):
  #f = np.zeros(np.shape(u))
  gamma = 1.4
  #p = (main.a.gamma_star - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
#  p,T = computePressure_and_Temperature_Cantera(main,u,main.cgas_field)
  p = main.a.p
  f[0] = u[3]
  f[1] = u[1]*u[3]/u[0]
  f[2] = u[2]*u[3]/u[0] 
  f[3] = u[3]*u[3]/u[0] + p 
  f[4] = (u[4] + p)*u[3]/u[0]
  f[5::] = u[None,3]*u[5::]/u[None,0] 

#  for i in range(0,np.shape(u)[0]-5):
#    f[5] = u[3]*u[5+i]/u[0]



#==================== Numerical Fluxes for the Faces =====================
#== central flux
#== rusanov flux
#== Roe flux

def eulerCentralFlux_reacting(main,UL,UR,pL,pR,n,args=None):
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
  rL = UL[0]
  uL = UL[1]/rL
  vL = UL[2]/rL
  wL = UL[3]/rL

  unL = uL*n[0] + vL*n[1] + wL*n[2]

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  #pL,TL = computePressure_and_Temperature_Cantera(main,UL,cgas_field)

  #pL = (gamma-1)*(UL[4] - 0.5*rL*qL**2.)
  rHL = UL[4] + pL
  HL = rHL/rL
  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rL*unL
  FL[1] = UL[1]*unL #+ pL*n[0]
  FL[2] = UL[2]*unL #+ pL*n[1]
  FL[3] = UL[3]*unL #+ pL*n[2]
  FL[4] = rHL*unL

  # process right state
  rR = UR[0]
  uR = UR[1]/rR
  vR = UR[2]/rR
  wR = UR[3]/rR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  qR = np.sqrt(UR[1]*UR[1] + UR[2]*UR[2] + UR[3]*UR[3])/rR
  #pR,TR = computePressure_and_Temperature_Cantera(main,UR,cgas_field)

  #pR = (gamma-1)*(UR[4] - 0.5*rR*qR**2.)
  rHR = UR[4] + pR
  HR = rHR/rR
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])#-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])#-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])#-0.5*smax*(UR[2] - UL[2])
  F[3]    = 0.5*(FL[3]+FR[3])#-0.5*smax*(UR[3] - UL[3])
  F[4]    = 0.5*(FL[4]+FR[4])#-0.5*smax*(UR[4] - UL[4])
  ## Now add fluxes for the passive scalars
  for i in range(0,np.shape(UL)[0]-5):
    FL[5+i] = UL[5+i]*unL
    FR[5+i] = UR[5+i]*unR
    F[5+i]    = 0.5*(FL[5+i] + FR[5+i])

  return F


def rusanovFlux_reacting(main,UL,UR,pL,pR,n,args=None):
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

  unL = uL*n[0] + vL*n[1] + wL*n[2]

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  #pL = (gamma-1)*(UL[4] - 0.5*rL*qL**2.)
  #pL,TL = computePressure_and_Temperature_Cantera(main,UL,cgas_field)
  #pL,TL = computePressure_and_Temperature(main,UL)

  rHL = UL[4] + pL
  HL = rHL/rL
  cL = np.sqrt(gamma*pL/rL)
  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL

  # process right state
  rR = UR[0]
  uR = UR[1]/rR
  vR = UR[2]/rR
  wR = UR[3]/rR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  qR = np.sqrt(UR[1]*UR[1] + UR[2]*UR[2] + UR[3]*UR[3])/rR
  #pR = (gamma-1)*(UR[4] - 0.5*rR*qR**2.)
  #pR,TR = computePressure_and_Temperature_Cantera(main,UR,cgas_field)
  #pR,TR = computePressure_and_Temperature(main,UR)

  rHR = UR[4] + pR
  HR = rHR/rR
  cR = np.sqrt(gamma*pR/rR)
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR

  # difference in states
  du = UR - UL

  # Roe average
  di     = np.sqrt(rR/rL)
  d1     = 1.0/(1.0+di)

  ui     = (di*uR + uL)*d1
  vi     = (di*vR + vL)*d1
  wi     = (di*wR + wL)*d1
  Hi     = (di*HR + HL)*d1

  af     = 0.5*(ui*ui+vi*vi+wi*wi)
  ucp    = ui*n[0] + vi*n[1] + wi*n[2]
  c2     = gmi*(Hi - af)
  #ci     = np.sqrt(c2)
  #ci1    = 1.0/ci

  #% eigenvalues

  #sh = np.shape(ucp)
  #lsh = np.append(3,sh)
  #l = np.zeros(lsh)
  #l[0] = ucp+ci
  #l[1] = ucp-ci
  #l[2] = ucp
  #print(np.mean(cR),np.mean(ci))
  smax = np.abs(ucp) + np.abs(cR)
  #smax = np.maximum(np.abs(l[0]),np.abs(l[1]))
  # flux assembly
  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*smax*(UR[2] - UL[2])
  F[3]    = 0.5*(FL[3]+FR[3])-0.5*smax*(UR[3] - UL[3])
  F[4]    = 0.5*(FL[4]+FR[4])-0.5*smax*(UR[4] - UL[4])

#  F[0]    = 0.5*(FL[0]+FR[0])-0.5*main.dt/main.dx*(UR[0] - UL[0])*10000.
#  F[1]    = 0.5*(FL[1]+FR[1])-0.5*main.dt/main.dx*(UR[1] - UL[1])*10000.
#  F[2]    = 0.5*(FL[2]+FR[2])-0.5*main.dt/main.dx*(UR[2] - UL[2])*10000.
#  F[3]    = 0.5*(FL[3]+FR[3])-0.5*main.dt/main.dx*(UR[3] - UL[3])*10000.
#  F[4]    = 0.5*(FL[4]+FR[4])-0.5*main.dt/main.dx*(UR[4] - UL[4])*10000.
  ## Now add fluxes for the passive scalars
#  print(np.mean(smax),main.dt/main.dx*10000.)
  FL[5::] = UL[5::]*unL[None,:]
  FR[5::] = UR[5::]*unR[None,:]
  F[5::]    = 0.5*(FL[5::] + FR[5::]) - 0.5*smax[None,:]*(UR[5::] - UL[5::])
#  F[5::]    = 0.5*(FL[5::] + FR[5::]) - 0.5*main.dt/main.dx*(UR[5::] - UL[5::])*10000.

#  for i in range(0,np.shape(UL)[0]-5):
#    FL[5+i] = UL[5+i]*unL
#    FR[5+i] = UR[5+i]*unR
#    F[5+i]    = 0.5*(FL[5+i] + FR[5+i]) - 0.5*smax*(UR[5+i] - UL[5+i])
#  print(np.linalg.norm(F))
  return F
               


def HLLCFlux_reacting(main,UL,UR,pL,pR,n,args=None):

  #process left state
#  R = 8314.4621/1000.
# 
# 
#  Y_N2_R = 1. - np.sum(UR[5::]/UR[None,0],axis=0)
#  WinvR =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UR[5::]/UR[None,0]) + 1./main.W[-1]*Y_N2_R
#  CpR = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UR[5::]/UR[None,0]) + main.Cp[-1]*Y_N2_R
#  CvR = CpR - R*WinvR
#  gammaR = CpR/CvR
# 
#  Y_N2_L = 1. - np.sum(UL[5::]/UL[None,0],axis=0)
#  WinvL =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UL[5::]/UL[None,0]) + 1./main.W[-1]*Y_N2_L
#  CpL = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UL[5::]/UL[None,0]) + main.Cp[-1]*Y_N2_L
#  CvL = CpL - R*WinvL
#  gammaL = CpL/CvL
# Calculates HLLC for variable species navier-stokes equations
# implementation based on paper:
#   Discontinuous Galerkin method for multicomponent chemically reacting flows and combustion
#   Journal of Computational Physics 270 (2014) 105-137
  #process left state
  rhoL = UL[0]
  uL = UL[1]/rhoL
  vL = UL[2]/rhoL
  wL = UL[3]/rhoL
  rHL = UL[4] + pL
  HL = rHL/rhoL
  unL = uL*n[0] + vL*n[1] + wL*n[2]
  # process right state
  rhoR = UR[0]
  uR = UR[1]/rhoR
  vR = UR[2]/rhoR
  wR = UR[3]/rhoR
  rHR = UR[4] + pR
  HR = rHR/rhoR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  # compute speed of sounds 
  R = 8314.4621/1000.
  Y_N2_R = 1. - np.sum(UR[5::]/UR[None,0],axis=0)
  WinvR =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UR[5::]/UR[None,0]) + 1./main.W[-1]*Y_N2_R
  CpR = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UR[5::]/UR[None,0]) + main.Cp[-1]*Y_N2_R
  CvR = CpR - R*WinvR
  gammaR = CpR/CvR
  cR = np.sqrt(gammaR*pR/UR[0])

  Y_N2_L = 1. - np.sum(UL[5::]/UL[None,0],axis=0)
  WinvL =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UL[5::]/UL[None,0]) + 1./main.W[-1]*Y_N2_L
  CpL = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UL[5::]/UL[None,0]) + main.Cp[-1]*Y_N2_L
  CvL = CpL - R*WinvL
  gammaL = CpL/CvL

  cL = np.sqrt(gammaL*pL/UL[0])

  # make computations for HLLC
  rho_bar = 0.5*(UL[0] + UR[0])
  c_bar = 0.5*(cL + cR)

  p_pvrs = 0.5*(pL + pR) - 0.5*(unR - unL)*rho_bar*c_bar
  p_star = np.fmax(0,p_pvrs)

  qkR = np.ones(np.shape(pR))
  #qkR[p_star > pR] = ((1. + (gamma_star + 1.)/(2.*gamma_star) * (p_star/pR - 1.))[p_star > pR])**0.5 
  qkR[p_star > pR] = ((1. + (gammaR + 1.)/(2.*gammaR) * (p_star/pR - 1.))[p_star > pR])**0.5 
  qkL = np.ones(np.shape(pL))
  #qkL[p_star > pL] = ((1. + (gamma_star + 1.)/(2.*gamma_star) * (p_star/pL - 1.))[p_star > pL])**0.5 
  qkL[p_star > pL] = ((1. + (gammaL + 1.)/(2.*gammaL) * (p_star/pL - 1.))[p_star > pL])**0.5 

  SL = unL - cL*qkL
  SR = unR + cR*qkR
  S_star = ( pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR) ) / ( rhoL*(SL - unL) - rhoR*(SR - unR) ) 
  # Compute UStar state for HLLC
  #left state
  srat = (SL - unL)/(SL - S_star)
  UstarL = np.zeros(np.shape(UL))
  UstarL[0] = srat*( UL[0] )
  UstarL[1] = srat*( UL[1] + rhoL*(S_star - unL)*n[0] )
  UstarL[2] = srat*( UL[2] + rhoL*(S_star - unL)*n[1] )
  UstarL[3] = srat*( UL[3] + rhoL*(S_star - unL)*n[2] )
  UstarL[4] = srat*( UL[4] + (S_star - unL)*(rhoL*S_star + pL/(SL - unL) ) )
  UstarL[5::] = srat[None,:]*( UL[5::] ) 
  #right state
  srat[:] = (SR - unR)/(SR - S_star)
  UstarR = np.zeros(np.shape(UR))
  UstarR[0] = srat*( UR[0] )
  UstarR[1] = srat*( UR[1] + rhoR*(S_star - unR)*n[0] )
  UstarR[2] = srat*( UR[2] + rhoR*(S_star - unR)*n[1] )
  UstarR[3] = srat*( UR[3] + rhoR*(S_star - unR)*n[2] )
  UstarR[4] = srat*( UR[4] + (S_star - unR)*(rhoR*S_star + pR/(SR - unR) ) )
  UstarR[5::] = srat[None,:]*( UR[5::] )

  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rhoL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL
  FL[5::] = UL[5::]*unL[None,:]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rhoR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  FR[5::] = UR[5::]*unR[None,:]

  # Assemble final HLLC Flux
  indx0 = 0<SL
  indx1 = (SL <= 0) & (0 < S_star)
  indx2 = (S_star <= 0) & (0 < SR)
  indx3 = SR <= 0
  F = np.zeros(np.shape(UR))
  F[:,indx0] = FL[:,indx0]
  F[:,indx1] = (FL + SL[None,:]*(UstarL - UL) )[:,indx1]
  F[:,indx2] = (FR + SR[None,:]*(UstarR - UR) )[:,indx2]
  F[:,indx3] = FR[:,indx3]
  return F

def HLLCFlux_reacting_doubleflux_plus(main,UL,UR,pL2,pR2,gammaL,gammaR,n,args=None):
# Calculates HLLC for variable species navier-stokes equations
# implementation based on paper:
#   Discontinuous Galerkin method for multicomponent chemically reacting flows and combustion
#   Journal of Computational Physics 270 (2014) 105-137
  #process left state
  R = 8314.4621/1000.


  Y_N2_R = 1. - np.sum(UR[5::]/UR[None,0],axis=0)
  WinvR =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UR[5::]/UR[None,0]) + 1./main.W[-1]*Y_N2_R
  CpR = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UR[5::]/UR[None,0]) + main.Cp[-1]*Y_N2_R
  CvR = CpR - R*WinvR
  #gammaR = CpR/CvR

  Y_N2_L = 1. - np.sum(UL[5::]/UL[None,0],axis=0)
  WinvL =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UL[5::]/UL[None,0]) + 1./main.W[-1]*Y_N2_L
  CpL = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UL[5::]/UL[None,0]) + main.Cp[-1]*Y_N2_L
  CvL = CpL - R*WinvL
  #gammaL = CpL/CvL




  rhoL = UL[0]
  uL = UL[1]/rhoL
  vL = UL[2]/rhoL
  wL = UL[3]/rhoL
  q2L = uL**2 + vL**2 + wL**2
  hL = gammaR/rhoL*(UL[4] - 0.5*UL[0]*q2L)*(gammaL - 1.)/(gammaR - 1.) 
  pL = rhoL*hL*(gammaR - 1.)/gammaR
  cL = np.sqrt((gammaR - 1.)*hL)
  rHL =  rhoL*(hL + 0.5*q2L)
  #rHL = UL[4] + pL
  #print(np.mean(UL[4]),np.mean( rhoL*(hL + 0.5*uL**2) - pL) )
  unL = uL*n[0] + vL*n[1] + wL*n[2]
  # process right state
  rhoR = UR[0]
  uR = UR[1]/rhoR
  vR = UR[2]/rhoR
  wR = UR[3]/rhoR
  q2R = uR**2 + vR**2 + wR**2
  hR = gammaR/rhoR*(UR[4] - 0.5*UR[0]*q2R)
  pR = rhoR*hR*(gammaR - 1.)/gammaR
  cR = np.sqrt((gammaR - 1.)*hR)
  #rHR = UR[4] + pR
  rHR =  rhoR*(hR + 0.5*q2R)
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  # make computations for HLLC
  SL = np.fmin(unL - cL , unR - cR)
  SR = np.fmax(unL + cL , unR + cR)

  S_star = ( pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR) ) / ( rhoL*(SL - unL) - rhoR*(SR - unR) ) 

  # Compute UStar state for HLLC
  #left state
  srat = (SL - unL)/(SL - S_star)
  UstarL = np.zeros(np.shape(UL))
  UstarL[0] = srat*( UL[0] )
  UstarL[1] = srat*( UL[1] + rhoL*(S_star - unL)*n[0] )
  UstarL[2] = srat*( UL[2] + rhoL*(S_star - unL)*n[1] )
  UstarL[3] = srat*( UL[3] + rhoL*(S_star - unL)*n[2] )
  #UstarL[4] = srat*( rhoL*(hL + 0.5*q2L) - pL + (S_star - unL)*(rhoL*S_star + pL/(SL - unL) ) )
  UstarL[4] = srat*( UL[4] + (S_star - unL)*(rhoL*S_star + pL/(SL - unL) ) )
  UstarL[5::] = srat[None,:]*( UL[5::] ) 
  #right state
  srat[:] = (SR - unR)/(SR - S_star)
  UstarR = np.zeros(np.shape(UR))
  UstarR[0] = srat*( UR[0] )
  UstarR[1] = srat*( UR[1] + rhoR*(S_star - unR)*n[0] )
  UstarR[2] = srat*( UR[2] + rhoR*(S_star - unR)*n[1] )
  UstarR[3] = srat*( UR[3] + rhoR*(S_star - unR)*n[2] )
  #UstarR[4] = srat*( rhoR*(hR + 0.5*q2R) - pR + (S_star - unR)*(rhoR*S_star + pR/(SR - unR) ) )
  UstarR[4] = srat*( UR[4] + (S_star - unR)*(rhoR*S_star + pR/(SR - unR) ) )
  UstarR[5::] = srat[None,:]*( UR[5::] )

  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rhoL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL
  FL[5::] = UL[5::]*unL[None,:]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rhoR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  FR[5::] = UR[5::]*unR[None,:]

  # Assemble final HLLC Flux
  indx0 = 0<SL
  indx1 = (SL <= 0) & (0 < S_star)
  indx2 = (S_star <= 0) & (0 < SR)
  indx3 = SR <= 0
  F = np.zeros(np.shape(UR))
  F[:,indx0] = FL[:,indx0]
  F[:,indx1] = (FL + SL[None,:]*(UstarL - UL) )[:,indx1]
  F[:,indx2] = (FR + SR[None,:]*(UstarR - UR) )[:,indx2]
  F[:,indx3] = FR[:,indx3]
  return F



def HLLCFlux_reacting_doubleflux_minus(main,UL,UR,pL2,pR2,gammaL,gammaR,n,args=None):
# Calculates HLLC for variable species navier-stokes equations
# implementation based on paper:
#   Discontinuous Galerkin method for multicomponent chemically reacting flows and combustion
#   Journal of Computational Physics 270 (2014) 105-137
  #process left state
  R = 8314.4621/1000.


  Y_N2_R = 1. - np.sum(UR[5::]/UR[None,0],axis=0)
  WinvR =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UR[5::]/UR[None,0]) + 1./main.W[-1]*Y_N2_R
  CpR = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UR[5::]/UR[None,0]) + main.Cp[-1]*Y_N2_R
  CvR = CpR - R*WinvR
  #gammaR = CpR/CvR

  Y_N2_L = 1. - np.sum(UL[5::]/UL[None,0],axis=0)
  WinvL =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UL[5::]/UL[None,0]) + 1./main.W[-1]*Y_N2_L
  CpL = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UL[5::]/UL[None,0]) + main.Cp[-1]*Y_N2_L
  CvL = CpL - R*WinvL
  #gammaL = CpL/CvL




  rhoL = UL[0]
  uL = UL[1]/rhoL
  vL = UL[2]/rhoL
  wL = UL[3]/rhoL
  q2L = uL**2 + vL**2 + wL**2
  hL = gammaL/rhoL*(UL[4] - 0.5*UL[0]*q2L)
  pL = rhoL*hL*(gammaL - 1.)/gammaL
  cL = np.sqrt((gammaL - 1.)*hL)
  #rHL = UL[4] + pL
  rHL =  rhoL*(hL + 0.5*q2L)
  unL = uL*n[0] + vL*n[1] + wL*n[2]
  # process right state
  rhoR = UR[0]
  uR = UR[1]/rhoR
  vR = UR[2]/rhoR
  wR = UR[3]/rhoR
  q2R = uR**2 + vR**2 + wR**2
  hR = gammaL/rhoR*(UR[4] - 0.5*UR[0]*q2R)*(gammaR - 1.)/(gammaL - 1.) 
  pR = rhoR*hR*(gammaL - 1.)/gammaL
  cR = np.sqrt((gammaL - 1.)*hR)
  #rHR = UR[4] + pR
  rHR =  rhoR*(hR + 0.5*q2R) 
  #print(np.mean(rHR),np.mean(UR[4] + pR))
  unR = uR*n[0] + vR*n[1] + wR*n[2]

  # make computations for HLLC
  SL = np.fmin(unL - cL , unR - cR)
  SR = np.fmax(unL + cL , unR + cR)

  S_star = ( pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR) ) / ( rhoL*(SL - unL) - rhoR*(SR - unR) ) 

  # Compute UStar state for HLLC
  #left state
  srat = (SL - unL)/(SL - S_star)
  UstarL = np.zeros(np.shape(UL))
  UstarL[0] = srat*( UL[0] )
  UstarL[1] = srat*( UL[1] + rhoL*(S_star - unL)*n[0] )
  UstarL[2] = srat*( UL[2] + rhoL*(S_star - unL)*n[1] )
  UstarL[3] = srat*( UL[3] + rhoL*(S_star - unL)*n[2] )
  #UstarL[4] = srat*( rhoL*(hL + 0.5*q2L) - pL + (S_star - unL)*(rhoL*S_star + pL/(SL - unL) ) )
  UstarL[4] = srat*( UL[4] + (S_star - unL)*(rhoL*S_star + pL/(SL - unL) ) )
  UstarL[5::] = srat[None,:]*( UL[5::] ) 
  #right state
  srat[:] = (SR - unR)/(SR - S_star)
  UstarR = np.zeros(np.shape(UR))
  UstarR[0] = srat*( UR[0] )
  UstarR[1] = srat*( UR[1] + rhoR*(S_star - unR)*n[0] )
  UstarR[2] = srat*( UR[2] + rhoR*(S_star - unR)*n[1] )
  UstarR[3] = srat*( UR[3] + rhoR*(S_star - unR)*n[2] )
  #UstarR[4] = srat*( rhoR*(hR + 0.5*q2R) - pR + (S_star - unR)*(rhoR*S_star + pR/(SR - unR) ) )
  UstarR[4] = srat*( UR[4] + (S_star - unR)*(rhoR*S_star + pR/(SR - unR) ) )
  UstarR[5::] = srat[None,:]*( UR[5::] )

  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rhoL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL
  FL[5::] = UL[5::]*unL[None,:]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rhoR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  FR[5::] = UR[5::]*unR[None,:]

  # Assemble final HLLC Flux
  indx0 = 0<SL
  indx1 = (SL <= 0) & (0 < S_star)
  indx2 = (S_star <= 0) & (0 < SR)
  indx3 = SR <= 0
  F = np.zeros(np.shape(UR))
  F[:,indx0] = FL[:,indx0]
  F[:,indx1] = (FL + SL[None,:]*(UstarL - UL) )[:,indx1]
  F[:,indx2] = (FR + SR[None,:]*(UstarR - UR) )[:,indx2]
  F[:,indx3] = FR[:,indx3]
  return F

def HLLCFlux_reacting_doubleflux(main,UL,UR,pL,pR,rh0,gamma_star,n,args=None):
  R = 8314.4621/1000.

  Y_N2_R = 1. - np.sum(UR[5::]/UR[None,0],axis=0)
  WinvR =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UR[5::]/UR[None,0]) + 1./main.W[-1]*Y_N2_R
  CpR = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UR[5::]/UR[None,0]) + main.Cp[-1]*Y_N2_R
  CvR = CpR - R*WinvR
  gammaR = CpR/CvR
 
  Y_N2_L = 1. - np.sum(UL[5::]/UL[None,0],axis=0)
  WinvL =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UL[5::]/UL[None,0]) + 1./main.W[-1]*Y_N2_L
  CpL = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UL[5::]/UL[None,0]) + main.Cp[-1]*Y_N2_L
  CvL = CpL - R*WinvL
  gammaL = CpL/CvL

# Calculates HLLC for variable species navier-stokes equations
# implementation based on paper:
#   Discontinuous Galerkin method for multicomponent chemically reacting flows and combustion
#   Journal of Computational Physics 270 (2014) 105-137
  #process left state
  rhoL = UL[0]
  uL = UL[1]/rhoL
  vL = UL[2]/rhoL
  wL = UL[3]/rhoL
  q2L = uL**2 + vL**2 + wL**2
  #pL = (gamma_star - 1.)*(UL[4] - 0.5*q2L*UL[0])
  pL = (gammaL - 1.)*(UL[4] - 0.5*q2L*UL[0])

  rHL = UL[4] + pL
  unL = uL*n[0] + vL*n[1] + wL*n[2]
  # process right state
  rhoR = UR[0]
  uR = UR[1]/rhoR
  vR = UR[2]/rhoR
  wR = UR[3]/rhoR
  q2R = uR**2 + vR**2 + wR**2
  #pR = (gamma_star - 1.)*(UR[4] - 0.5*q2R*UR[0])
  pR = (gammaR - 1.)*(UR[4] - 0.5*q2R*UR[0])

  #print(np.mean(pR),np.mean(p2R),n)
  rHR = UR[4] + pR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  # compute speed of sounds 
  #cR = np.sqrt(gamma_star*pR/UR[0])
  #cL = np.sqrt(gamma_star*pL/UL[0])
  cR = np.sqrt(gammaR*pR/UR[0])
  cL = np.sqrt(gammaL*pL/UL[0])

  # make computations for HLLC
  SL = np.fmin(unL - cL , unR - cR)
  SR = np.fmax(unL + cL , unR + cR)

  S_star = ( pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR) ) / ( rhoL*(SL - unL) - rhoR*(SR - unR) ) 

  # Compute UStar state for HLLC
  #left state
  srat = (SL - unL)/(SL - S_star)
  UstarL = np.zeros(np.shape(UL))
  UstarL[0] = srat*( UL[0] )
  UstarL[1] = srat*( UL[1] + rhoL*(S_star - unL)*n[0] )
  UstarL[2] = srat*( UL[2] + rhoL*(S_star - unL)*n[1] )
  UstarL[3] = srat*( UL[3] + rhoL*(S_star - unL)*n[2] )
  UstarL[4] = srat*( UL[4] + (S_star - unL)*(rhoL*S_star + pL/(SL - unL) ) )
  UstarL[5::] = srat[None,:]*( UL[5::] ) 
  #right state
  srat[:] = (SR - unR)/(SR - S_star)
  UstarR = np.zeros(np.shape(UR))
  UstarR[0] = srat*( UR[0] )
  UstarR[1] = srat*( UR[1] + rhoR*(S_star - unR)*n[0] )
  UstarR[2] = srat*( UR[2] + rhoR*(S_star - unR)*n[1] )
  UstarR[3] = srat*( UR[3] + rhoR*(S_star - unR)*n[2] )
  UstarR[4] = srat*( UR[4] + (S_star - unR)*(rhoR*S_star + pR/(SR - unR) ) )
  UstarR[5::] = srat[None,:]*( UR[5::] )

  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rhoL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL
  FL[5::] = UL[5::]*unL[None,:]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rhoR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  FR[5::] = UR[5::]*unR[None,:]

  # Assemble final HLLC Flux
  indx0 = 0<SL
  indx1 = (SL <= 0) & (0 < S_star)
  indx2 = (S_star <= 0) & (0 < SR)
  indx3 = SR <= 0
  F = np.zeros(np.shape(UR))
  F[:,indx0] = FL[:,indx0]
  F[:,indx1] = (FL + SL[None,:]*(UstarL - UL) )[:,indx1]
  F[:,indx2] = (FR + SR[None,:]*(UstarR - UR) )[:,indx2]
  F[:,indx3] = FR[:,indx3]
  return F

def HLLCFlux_reacting_doublefluxL(main,UL,UR,pL,pR,rh0,gamma_star,n,args=None):
  R = 8314.4621/1000.

  Y_N2_R = 1. - np.sum(UR[5::]/UR[None,0],axis=0)
  WinvR =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UR[5::]/UR[None,0]) + 1./main.W[-1]*Y_N2_R
  CpR = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UR[5::]/UR[None,0]) + main.Cp[-1]*Y_N2_R
  CvR = CpR - R*WinvR
  gammaR = CpR/CvR
 
  Y_N2_L = 1. - np.sum(UL[5::]/UL[None,0],axis=0)
  WinvL =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UL[5::]/UL[None,0]) + 1./main.W[-1]*Y_N2_L
  CpL = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UL[5::]/UL[None,0]) + main.Cp[-1]*Y_N2_L
  CvL = CpL - R*WinvL
  gammaL = CpL/CvL

# Calculates HLLC for variable species navier-stokes equations
# implementation based on paper:
#   Discontinuous Galerkin method for multicomponent chemically reacting flows and combustion
#   Journal of Computational Physics 270 (2014) 105-137
  #process left state
  rhoL = UL[0]
  uL = UL[1]/rhoL
  vL = UL[2]/rhoL
  wL = UL[3]/rhoL
  q2L = uL**2 + vL**2 + wL**2
  #pL = (gamma_star - 1.)*(UL[4] - 0.5*q2L*UL[0])
  pL = (gammaR - 1.)*(UL[4] - 0.5*q2L*UL[0])

  rHL = UL[4] + pL
  unL = uL*n[0] + vL*n[1] + wL*n[2]
  # process right state
  rhoR = UR[0]
  uR = UR[1]/rhoR
  vR = UR[2]/rhoR
  wR = UR[3]/rhoR
  q2R = uR**2 + vR**2 + wR**2
  #pR = (gamma_star - 1.)*(UR[4] - 0.5*q2R*UR[0])
  pR = (gammaR - 1.)*(UR[4] - 0.5*q2R*UR[0])

  #print(np.mean(pR),np.mean(p2R),n)
  rHR = UR[4] + pR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  # compute speed of sounds 
  #cR = np.sqrt(gamma_star*pR/UR[0])
  #cL = np.sqrt(gamma_star*pL/UL[0])
  cR = np.sqrt(gammaR*pR/UR[0])
  cL = np.sqrt(gammaR*pL/UL[0])

  # make computations for HLLC
  SL = np.fmin(unL - cL , unR - cR)
  SR = np.fmax(unL + cL , unR + cR)

  S_star = ( pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR) ) / ( rhoL*(SL - unL) - rhoR*(SR - unR) ) 

  # Compute UStar state for HLLC
  #left state
  srat = (SL - unL)/(SL - S_star)
  UstarL = np.zeros(np.shape(UL))
  UstarL[0] = srat*( UL[0] )
  UstarL[1] = srat*( UL[1] + rhoL*(S_star - unL)*n[0] )
  UstarL[2] = srat*( UL[2] + rhoL*(S_star - unL)*n[1] )
  UstarL[3] = srat*( UL[3] + rhoL*(S_star - unL)*n[2] )
  UstarL[4] = srat*( UL[4] + (S_star - unL)*(rhoL*S_star + pL/(SL - unL) ) )
  UstarL[5::] = srat[None,:]*( UL[5::] ) 
  #right state
  srat[:] = (SR - unR)/(SR - S_star)
  UstarR = np.zeros(np.shape(UR))
  UstarR[0] = srat*( UR[0] )
  UstarR[1] = srat*( UR[1] + rhoR*(S_star - unR)*n[0] )
  UstarR[2] = srat*( UR[2] + rhoR*(S_star - unR)*n[1] )
  UstarR[3] = srat*( UR[3] + rhoR*(S_star - unR)*n[2] )
  UstarR[4] = srat*( UR[4] + (S_star - unR)*(rhoR*S_star + pR/(SR - unR) ) )
  UstarR[5::] = srat[None,:]*( UR[5::] )

  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rhoL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL
  FL[5::] = UL[5::]*unL[None,:]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rhoR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  FR[5::] = UR[5::]*unR[None,:]

  # Assemble final HLLC Flux
  indx0 = 0<SL
  indx1 = (SL <= 0) & (0 < S_star)
  indx2 = (S_star <= 0) & (0 < SR)
  indx3 = SR <= 0
  F = np.zeros(np.shape(UR))
  F[:,indx0] = FL[:,indx0]
  F[:,indx1] = (FL + SL[None,:]*(UstarL - UL) )[:,indx1]
  F[:,indx2] = (FR + SR[None,:]*(UstarR - UR) )[:,indx2]
  F[:,indx3] = FR[:,indx3]
  return F


def HLLCFlux_reacting_doublefluxR(main,UL,UR,pL,pR,rh0,gamma_star,n,args=None):
  R = 8314.4621/1000.

  Y_N2_R = 1. - np.sum(UR[5::]/UR[None,0],axis=0)
  WinvR =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UR[5::]/UR[None,0]) + 1./main.W[-1]*Y_N2_R
  CpR = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UR[5::]/UR[None,0]) + main.Cp[-1]*Y_N2_R
  CvR = CpR - R*WinvR
  gammaR = CpR/CvR
 
  Y_N2_L = 1. - np.sum(UL[5::]/UL[None,0],axis=0)
  WinvL =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UL[5::]/UL[None,0]) + 1./main.W[-1]*Y_N2_L
  CpL = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UL[5::]/UL[None,0]) + main.Cp[-1]*Y_N2_L
  CvL = CpL - R*WinvL
  gammaL = CpL/CvL

# Calculates HLLC for variable species navier-stokes equations
# implementation based on paper:
#   Discontinuous Galerkin method for multicomponent chemically reacting flows and combustion
#   Journal of Computational Physics 270 (2014) 105-137
  #process left state
  rhoL = UL[0]
  uL = UL[1]/rhoL
  vL = UL[2]/rhoL
  wL = UL[3]/rhoL
  q2L = uL**2 + vL**2 + wL**2
  #pL = (gamma_star - 1.)*(UL[4] - 0.5*q2L*UL[0])
  pL = (gammaL - 1.)*(UL[4] - 0.5*q2L*UL[0])

  rHL = UL[4] + pL
  unL = uL*n[0] + vL*n[1] + wL*n[2]
  # process right state
  rhoR = UR[0]
  uR = UR[1]/rhoR
  vR = UR[2]/rhoR
  wR = UR[3]/rhoR
  q2R = uR**2 + vR**2 + wR**2
  #pR = (gamma_star - 1.)*(UR[4] - 0.5*q2R*UR[0])
  pR = (gammaL - 1.)*(UR[4] - 0.5*q2R*UR[0])

  #print(np.mean(pR),np.mean(p2R),n)
  rHR = UR[4] + pR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  # compute speed of sounds 
  #cR = np.sqrt(gamma_star*pR/UR[0])
  #cL = np.sqrt(gamma_star*pL/UL[0])
  cR = np.sqrt(gammaL*pR/UR[0])
  cL = np.sqrt(gammaL*pL/UL[0])

  # make computations for HLLC
  SL = np.fmin(unL - cL , unR - cR)
  SR = np.fmax(unL + cL , unR + cR)

  S_star = ( pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR) ) / ( rhoL*(SL - unL) - rhoR*(SR - unR) ) 

  # Compute UStar state for HLLC
  #left state
  srat = (SL - unL)/(SL - S_star)
  UstarL = np.zeros(np.shape(UL))
  UstarL[0] = srat*( UL[0] )
  UstarL[1] = srat*( UL[1] + rhoL*(S_star - unL)*n[0] )
  UstarL[2] = srat*( UL[2] + rhoL*(S_star - unL)*n[1] )
  UstarL[3] = srat*( UL[3] + rhoL*(S_star - unL)*n[2] )
  UstarL[4] = srat*( UL[4] + (S_star - unL)*(rhoL*S_star + pL/(SL - unL) ) )
  UstarL[5::] = srat[None,:]*( UL[5::] ) 
  #right state
  srat[:] = (SR - unR)/(SR - S_star)
  UstarR = np.zeros(np.shape(UR))
  UstarR[0] = srat*( UR[0] )
  UstarR[1] = srat*( UR[1] + rhoR*(S_star - unR)*n[0] )
  UstarR[2] = srat*( UR[2] + rhoR*(S_star - unR)*n[1] )
  UstarR[3] = srat*( UR[3] + rhoR*(S_star - unR)*n[2] )
  UstarR[4] = srat*( UR[4] + (S_star - unR)*(rhoR*S_star + pR/(SR - unR) ) )
  UstarR[5::] = srat[None,:]*( UR[5::] )

  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rhoL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL
  FL[5::] = UL[5::]*unL[None,:]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rhoR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  FR[5::] = UR[5::]*unR[None,:]

  # Assemble final HLLC Flux
  indx0 = 0<SL
  indx1 = (SL <= 0) & (0 < S_star)
  indx2 = (S_star <= 0) & (0 < SR)
  indx3 = SR <= 0
  F = np.zeros(np.shape(UR))
  F[:,indx0] = FL[:,indx0]
  F[:,indx1] = (FL + SL[None,:]*(UstarL - UL) )[:,indx1]
  F[:,indx2] = (FR + SR[None,:]*(UstarR - UR) )[:,indx2]
  F[:,indx3] = FR[:,indx3]
  return F




def HLLCFlux_reacting_doubleflux2(main,UL,UR,pL,pR,rh0,gamma_star,n,args=None):
# Calculates HLLC for variable species navier-stokes equations
# implementation based on paper:
#   Discontinuous Galerkin method for multicomponent chemically reacting flows and combustion
#   Journal of Computational Physics 270 (2014) 105-137
  #process left state
  rhoL = UL[0]
  uL = UL[1]/rhoL
  vL = UL[2]/rhoL
  wL = UL[3]/rhoL
  q2L = uL**2 + vL**2 + wL**2
  pL = (gamma_star - 1.)*(UL[4] - 0.5*q2L*UL[0])
  rHL = UL[4] + pL
  unL = uL*n[0] + vL*n[1] + wL*n[2]
  # process right state
  rhoR = UR[0]
  uR = UR[1]/rhoR
  vR = UR[2]/rhoR
  wR = UR[3]/rhoR
  q2R = uR**2 + vR**2 + wR**2
  pR = (gamma_star - 1.)*(UR[4] - 0.5*q2R*UR[0])
  #print(np.mean(pR),np.mean(p2R),n)
  rHR = UR[4] + pR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  # compute speed of sounds 
  cR = np.sqrt(gamma_star*pR/UR[0])
  cL = np.sqrt(gamma_star*pL/UL[0])

  # make computations for HLLC
  rho_bar = 0.5*(UL[0] + UR[0])
  c_bar = 0.5*(cL + cR)

  p_pvrs = 0.5*(pL + pR) - 0.5*(unR - unL)*rho_bar*c_bar
  p_star = np.fmax(0,p_pvrs)

  qkR = np.ones(np.shape(pR))
  #qkR[p_star > pR] = ((1. + (gamma_star + 1.)/(2.*gamma_star) * (p_star/pR - 1.))[p_star > pR])**0.5 
  qkR[p_star > pR] = ((1. + (gamma_star + 1.)/(2.*gamma_star) * (p_star/pR - 1.))[p_star > pR])**0.5 
  qkL = np.ones(np.shape(pL))
  #qkL[p_star > pL] = ((1. + (gamma_star + 1.)/(2.*gamma_star) * (p_star/pL - 1.))[p_star > pL])**0.5 
  qkL[p_star > pL] = ((1. + (gamma_star + 1.)/(2.*gamma_star) * (p_star/pL - 1.))[p_star > pL])**0.5 

  SL = unL - cL*qkL
  SR = unR + cR*qkR
  S_star = ( pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR) ) / ( rhoL*(SL - unL) - rhoR*(SR - unR) ) 
  # Compute UStar state for HLLC
  #left state
  srat = (SL - unL)/(SL - S_star)
  UstarL = np.zeros(np.shape(UL))
  UstarL[0] = srat*( UL[0] )
  UstarL[1] = srat*( UL[1] + rhoL*(S_star - unL)*n[0] )
  UstarL[2] = srat*( UL[2] + rhoL*(S_star - unL)*n[1] )
  UstarL[3] = srat*( UL[3] + rhoL*(S_star - unL)*n[2] )
  UstarL[4] = srat*( UL[4] + (S_star - unL)*(rhoL*S_star + pL/(SL - unL) ) )
  UstarL[5::] = srat[None,:]*( UL[5::] ) 
  #right state
  srat[:] = (SR - unR)/(SR - S_star)
  UstarR = np.zeros(np.shape(UR))
  UstarR[0] = srat*( UR[0] )
  UstarR[1] = srat*( UR[1] + rhoR*(S_star - unR)*n[0] )
  UstarR[2] = srat*( UR[2] + rhoR*(S_star - unR)*n[1] )
  UstarR[3] = srat*( UR[3] + rhoR*(S_star - unR)*n[2] )
  UstarR[4] = srat*( UR[4] + (S_star - unR)*(rhoR*S_star + pR/(SR - unR) ) )
  UstarR[5::] = srat[None,:]*( UR[5::] )

  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rhoL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL
  FL[5::] = UL[5::]*unL[None,:]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rhoR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  FR[5::] = UR[5::]*unR[None,:]

  # Assemble final HLLC Flux
  indx0 = 0<SL
  indx1 = (SL <= 0) & (0 < S_star)
  indx2 = (S_star <= 0) & (0 < SR)
  indx3 = SR <= 0
  F = np.zeros(np.shape(UR))
  F[:,indx0] = FL[:,indx0]
  F[:,indx1] = (FL + SL[None,:]*(UstarL - UL) )[:,indx1]
  F[:,indx2] = (FR + SR[None,:]*(UstarR - UR) )[:,indx2]
  F[:,indx3] = FR[:,indx3]
  return F


def HLLEFlux_reacting(main,UL,UR,pL,pR,n,args=None):
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

  unL = uL*n[0] + vL*n[1] + wL*n[2]

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  #pL = (gamma-1)*(UL[4] - 0.5*rL*qL**2.)
  #pL,TL = computePressure_and_Temperature_Cantera(main,UL,cgas_field)
  #pL,TL = computePressure_and_Temperature(main,UL)

  rHL = UL[4] + pL
  HL = rHL/rL
  cL = np.sqrt(gamma*pL/rL)
  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL

  # process right state
  rR = UR[0]
  uR = UR[1]/rR
  vR = UR[2]/rR
  wR = UR[3]/rR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  qR = np.sqrt(UR[1]*UR[1] + UR[2]*UR[2] + UR[3]*UR[3])/rR
  #pR = (gamma-1)*(UR[4] - 0.5*rR*qR**2.)
  #pR,TR = computePressure_and_Temperature_Cantera(main,UR,cgas_field)
  #pR,TR = computePressure_and_Temperature(main,UR)

  rHR = UR[4] + pR
  HR = rHR/rR
  cR = np.sqrt(gamma*pR/rR)
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  FL[5::] = UL[5::]*unL[None,:]
  FR[5::] = UR[5::]*unR[None,:]

  #% eigenvalues
  Y_N2_R = 1. - np.sum(UR[5::]/UR[None,0],axis=0)
  gammaR = np.einsum('i...,ijk...->jk...',main.gamma[0:-1],UR[5::]/UR[None,0]) + main.gamma[-1]*Y_N2_R
  cR = np.sqrt(gammaR*pR/UR[0])

  Y_N2_L = 1. - np.sum(UL[5::]/UL[None,0],axis=0)
  gammaL = np.einsum('i...,ijk...->jk...',main.gamma[0:-1],UL[5::]/UL[None,0]) + main.gamma[-1]*Y_N2_L
  cL = np.sqrt(gammaR*pR/UR[0])
  #print(np.mean(cR),np.mean(cL))
  sL_min = np.fmin(0,unL - cL)
  sR_min = np.fmin(0,unR - cR)
  sL_max = np.fmax(0,unL + cL)
  sR_max = np.fmax(0,unR + cR)
  smin = np.fmin(sL_min,sR_min)
  smax = np.fmax(sL_max,sR_max)
  term1 = 0.5*(smax + smin)/(smax - smin)
  term2 = (smax*smin)/(smax - smin)
  #print(np.mean(smax)*main.dt*main.order[0]/main.dx/main.order[-1])
  #print((gammaL))#,np.amax(gammaL))
  # flux assembly
  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])-term1*(FR[0] - FL[0]) + term2*(UR[0] - UL[0])
  #Ystar = np.zeros(np.shape(F[5::]))
  #Ystar[:,F[0]<0] = UR[5::,F[0]<0]/UR[None,0,F[0]<0]
  #Ystar[:,F[0]>0] = UL[5::,F[0]>0]/UL[None,0,F[0]>0]
  F[1]    = 0.5*(FL[1]+FR[1])-term1*(FR[1] - FL[1]) + term2*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])-term1*(FR[2] - FL[2]) + term2*(UR[2] - UL[2])
  F[3]    = 0.5*(FL[3]+FR[3])-term1*(FR[3] - FL[3]) + term2*(UR[3] - UL[3])
  F[4]    = 0.5*(FL[4]+FR[4])-term1*(FR[4] - FL[4]) + term2*(UR[4] - UL[4])
  F[5::]    = 0.5*(FL[5::]+FR[5::])-term1[None,:]*(FR[5::] - FL[5::]) + term2[None,:]*(UR[5::] - UL[5::])
  #F[5::]    = F[None,0]*Ystar[:] 

#  F[5::]    = 0.5*(FL[5::] + FR[5::]) - 0.5*main.dt/main.dx*(UR[5::] - UL[5::])*10000.

#  for i in range(0,np.shape(UL)[0]-5):
#    FL[5+i] = UL[5+i]*unL
#    FR[5+i] = UR[5+i]*unR
#    F[5+i]    = 0.5*(FL[5+i] + FR[5+i]) - 0.5*smax*(UR[5+i] - UL[5+i])
#  print(np.linalg.norm(F))
  return F

  

def kfid_roeflux_reacting(main,UL,UR,n,args=None):
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

  unL = uL*n[0] + vL*n[1] + wL*n[2]

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  #pL = (gamma-1)*(UL[4] - 0.5*rL*qL*qL)
  pL,TL = computePressure_and_Temperature_Cantera(main,UL)

  rHL = UL[4] + pL
  HL = rHL/rL
  cL = np.sqrt(gamma*pL/rL) 
  print(np.mean(cL))
  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL

  # process right state
  rR = UR[0]
  uR = UR[1]/rR
  vR = UR[2]/rR
  wR = UR[3]/rR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  qR = np.sqrt(UR[1]*UR[1] + UR[2]*UR[2] + UR[3]*UR[3])/rR
  pR,TR = computePressure_and_Temperature_Cantera(main,UR)
  #pR = (gamma-1)*(UR[4] - 0.5*rR*qR*qR)
  rHR = UR[4] + pR
  HR = rHR/rR
  cR = np.sqrt(gamma*pR/rR)
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR

  # difference in states
  du = UR - UL

  # Roe average
  di     = np.sqrt(rR/rL)
  d1     = 1.0/(1.0+di)

  ui     = (di*uR + uL)*d1
  vi     = (di*vR + vL)*d1
  wi     = (di*wR + wL)*d1
  Hi     = (di*HR + HL)*d1

  af     = 0.5*(ui*ui+vi*vi+wi*wi)
  ucp    = ui*n[0] + vi*n[1] + wi*n[2]
  c2     = gmi*(Hi - af)
  ci     = np.sqrt(c2)
  ci1    = 1.0/ci



  #% eigenvalues

  sh = np.shape(ucp)
  lsh = np.append(3,sh)
  l = np.zeros(lsh)
  l[0] = ucp+ci
  l[1] = ucp-ci
  l[2] = ucp

  #% entropy fix
  epsilon = ci*.1
  #print(l,epsilon)
  labs = np.abs(l)
  for i in range(0,3):
    l[i,labs[i]<epsilon] =  (epsilon[labs[i]<epsilon] + l[i,labs[i]<epsilon]**2)/(2.*epsilon[labs[i]<epsilon])

  l = np.abs(l)
  l3 = l[2]
  # average and half-difference of 1st and 2nd eigs
  s1    = 0.5*(l[0] + l[1])
  s2    = 0.5*(l[0] - l[1])

  # left eigenvector product generators (see Theory guide)
  G1    = gmi*(af*du[0] - ui*du[1] - vi*du[2] -wi*du[3] + du[4])
  G2    = -ucp*du[0]+du[1]*n[0]+du[2]*n[1] + du[3]*n[2]

  # required functions of G1 and G2 (again, see Theory guide)
  C1    = G1*(s1-l3)*ci1*ci1 + G2*s2*ci1
  C2    = G1*s2*ci1          + G2*(s1-l3)

  # flux assembly
  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*(l3*du[0] + C1   )
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*(l3*du[1] + C1*ui + C2*n[0])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*(l3*du[2] + C1*vi + C2*n[1])
  F[3]    = 0.5*(FL[3]+FR[3])-0.5*(l3*du[3] + C1*wi + C2*n[2])
  F[4]    = 0.5*(FL[4]+FR[4])-0.5*(l3*du[4] + C1*Hi + C2*ucp  )

  ## Now add fluxes for the passive scalars
  for i in range(0,np.shape(UL)[0]-5):
    FL[5+i] = UL[5+i]*unL
    FR[5+i] = UR[5+i]*unL
    F[5+i]    = 0.5*(FL[5+i] + FR[5+i]) - 0.5*ucp*(UR[5+i] - UL[5+i])
  return F



###============= Diffusion Fluxes =====================
def getGsNSX_FAST_reacting(u,main,mu,V):
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
  fvG11[5::] = mu[1::]*V[5::] - mu[1::]*u[5::]*V[0]/u[0]


  fvG21[1] = mu_by_rho*(V[2] - v2*V[0])
  fvG21[2] = 2./3.*mu_by_rho*(v1*V[0] - V[1])
  fvG21[3] = 0
  fvG21[4] = mu_by_rho*(v1*V[2] - 2./3.*v2*V[1] - 1./3.*v1*v2*V[0] )

  fvG31[1] = mu_by_rho*(V[3] - v3*V[0])
  fvG31[3] = 2./3.*mu_by_rho*(v1*V[0] - V[1])
  fvG31[4] = mu_by_rho*(v1*V[3] - 2./3.*v3*V[1] - 1./3.*v1*v3*V[0])
  return fvG11,fvG21,fvG31


def getGsNSY_FAST_reacting(u,main,mu,V):
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
  fvG22[5::] = mu[1::]*V[5::] - mu[1::]*u[5::]*V[0]/u[0]

  fvG32[2] = mu_by_rho*(V[3] - v3*V[0])
  fvG32[3] = 2./3.*mu_by_rho*(v2*V[0] - V[2])
  fvG32[4] = mu_by_rho*(v2*V[3] -2./3.*v3*V[2] - 1./3.*v2*v3*V[0])


  return fvG12,fvG22,fvG32

def getGsNSZ_FAST_reacting(u,main,mu,V):
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
  fvG33[5::] = mu[1::]*V[5::] - mu[1::]*u[5::]*V[0]/u[0]
  return fvG13,fvG23,fvG33


def evalViscousFluxXNS_IP_reacting(main,u,Ux,Uy,Uz,mu):
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
 
  # rho D d/dx(Z) -> d/dx(rho Z) = rho d/dx Z + Z d/dx rho
  # -> d/dz(Z) = 1/rho*(d/dx rho Z - Z d/x rho) 
  # -> rho d/dx(Z) = d/dx(rho Z - Z d/dx rho)
  for i in range(0,main.nspecies):
    fx[5+i] = mu[i+1]*(Ux[5+i] - u[5+i]/u[0]*Ux[0])
  return fx


def evalViscousFluxYNS_IP_reacting(main,u,Ux,Uy,Uz,mu):
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
  for i in range(0,main.nspecies):
    fy[5+i] = mu[i+1]*(Uy[5+i] - u[5+i]/u[0]*Uy[0])
  return fy

def evalViscousFluxZNS_IP_reacting(main,u,Ux,Uy,Uz,mu):
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
  for i in range(0,main.nspecies):
    fz[5+i] = mu[i+1]*(Uz[5+i] - u[5+i]/u[0]*Uz[0])
  return fz


### Diffusion fluxes for BR1

### viscous fluxes
def evalViscousFluxXNS_BR1_reacting(main,U,fv,T):
  u = U[1]/U[0]
  v = U[2]/U[0]
  w = U[3]/U[0]
  fv[0] =  4./3.*u  #tau11 = (du/dx + du/dx - 2/3 (du/dx + dv/dy + dw/dz) ) 
  fv[1] = -2./3.*u  #tau22 = (dv/dy + dv/dy - 2/3 (du/dx + dv/dy + dw/dZ) )
  fv[2] = -2./3.*u  #tau33 = (dw/dz + dw/dz - 2/3 (du/dx + dv/dy + dw/dz) )
  fv[3] = v         #tau12 = (du/dy + dv/dx)
  fv[4] = w         #tau13 = (du/dz + dw/dx)
  fv[5] = 0.           #tau23 = (dv/dz + dw/dy)
  #p,T = computePressure_and_Temperature_Cantera(main,U,cgas_field)
  #T = (U[4]/U[0] - 0.5*( u**2 + v**2 + w**2 ) ) #kinda a psuedo tmp, should divide by Cv but it's constant so this is taken care of in the tauFlux with gamma
  fv[6] = T
  fv[7] = 0.
  fv[8] = 0.
  fv[9::3] = U[5::]/U[None,0] 
  fv[10::3] = 0.
  fv[11::3] = 0.

#
def evalViscousFluxYNS_BR1_reacting(main,U,fv,T):
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
  #p,T = computePressure_and_Temperature_Cantera(main,U,cgas_field)
  #T = (U[4]/U[0] - 0.5*( u**2 + v**2 + w**2 ) )
  fv[7] = T
  fv[8] = 0.
  fv[9::3] = 0. 
  fv[10::3] = U[5::]/U[None,0]
  fv[11::3] = 0.

def evalViscousFluxZNS_BR1_reacting(main,U,fv,T):
  u = U[1]/U[0]
  v = U[2]/U[0]
  w = U[3]/U[0]
  fv[0] = -2./3.*w  #tau11 = (du/dx + du/dx - 2/3 (du/dx + dv/dy + dw/dz) ) 
  fv[1] = -2./3.*w  #tau22 = (dv/dy + dv/dy - 2/3 (du/dx + dv/dy + dw/dZ) )
  fv[2] =  4./3.*w  #tau33 = (dw/dz + dw/dz - 2/3 (du/dx + dv/dy + dw/dz) )
  fv[3] = 0.           #tau12 = (du/dy + dv/dx)
  fv[4] = u         #tau13 = (du/dz + dw/dx)
  fv[5] = v        #tau23 = (dv/dz + dw/dy)
#  p,T = computePressure_and_Temperature_Cantera(main,U,cgas_field)
#  T = (U[4]/U[0] - 0.5*( u**2 + v**2 + w**2 ) )
  fv[6] = 0.
  fv[7] = 0.
  fv[8] = T
  fv[9::3] = 0. 
  fv[10::3] = 0.
  fv[11::3] = U[5::]/U[None,0]





def evalTauFluxXNS_BR1_reacting(main,tau,u,fvX,mu,cgas_field):
  Pr = 0.72
  gamma = 1.4
  kappa_by_mu = np.reshape(cgas_field.cp/Pr,np.shape(u[0]))
  D = 2.328448e-5/u[0]
  kappa = u[0]*main.cgas.cp*D
  fvX[0] = 0.
  fvX[1] = mu*tau[0] #tau11
  fvX[2] = mu*tau[3] #tau21
  fvX[3] = mu*tau[4] #tau31
  #diffusive heat flux
  sh = np.shape(u)[0] - 5 + 1
  sh = np.append(sh, np.shape(u[0]) )
  #partial_enthalpies = np.reshape(cgas_field.partial_molar_enthalpies*cgas_field.molecular_weights[None,:],sh)
  q =  kappa*tau[6] #+ D*u[0]*np.sum(partial_enthalpies[0:-1]*tau[9::3],axis=0)*0.
  fvX[4] = mu*(tau[0]*u[1]/u[0] + tau[3]*u[2]/u[0] + tau[4]*u[3]/u[0]) + q
  fvX[5::] = u[None,0]*D*tau[9::3]

def evalTauFluxYNS_BR1_reacting(main,tau,u,fvY,mu,cgas_field):
  Pr = 0.72
  gamma = 1.4
  #D = 1
  D = 2.328448e-5/u[0]
  kappa = u[0]*main.cgas.cp*D
  kappa_by_mu = np.reshape(cgas_field.cp/Pr,np.shape(u[0]))
  fvY[0] = 0.
  fvY[1] = mu*tau[3] #tau21
  fvY[2] = mu*tau[1] #tau22
  fvY[3] = mu*tau[5] #tau23
  sh = np.shape(u)[0] - 5 + 1
  sh = np.append(sh, np.shape(u[0]) )
  #partial_enthalpies = np.reshape(cgas_field.partial_molar_enthalpies*cgas_field.molecular_weights[None,:],sh)
  q =  kappa*tau[7] #+ D*u[0]*np.sum(partial_enthalpies[0:-1]*tau[10::3],axis=0)*0.
  fvY[4] = mu*(tau[3]*u[1]/u[0] + tau[1]*u[2]/u[0] + tau[5]*u[3]/u[0]) + q
  fvY[5::] = u[None,0]*D*tau[10::3]

def evalTauFluxZNS_BR1_reacting(main,tau,u,fvZ,mu,cgas_field):
  Pr = 0.72
  gamma = 1.4
  kappa_by_mu = np.reshape(cgas_field.cp/Pr,np.shape(u[0]))
  #D = 1
  D = 2.328448e-5/u[0]
  kappa = u[0]*main.cgas.cp*D
  fvZ[0] = 0.
  fvZ[1] = mu*tau[4] #tau31
  fvZ[2] = mu*tau[5] #tau32
  fvZ[3] = mu*tau[2] #tau33
  sh = np.shape(u)[0] - 5 + 1
  sh = np.append(sh, np.shape(u[0]) )
  #partial_enthalpies = np.reshape(cgas_field.partial_molar_enthalpies*cgas_field.molecular_weights[None,:],sh)
  q =  kappa*tau[8] #+ D*u[0]*np.sum(partial_enthalpies[0:-1]*tau[11::3],axis=0)*0.
  fvZ[4] = mu*(tau[4]*u[1]/u[0] + tau[5]*u[2]/u[0] + tau[2]*u[3]/u[0]) + q
  fvZ[5::] = u[None,0]*D*tau[11::3]


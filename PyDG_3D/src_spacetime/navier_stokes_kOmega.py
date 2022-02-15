import numpy as np
import numexpr as ne 

### 
def source_kOmega(eqns,region):
  betaStar = 0.09
  beta = 3./40.
  gammaTurb = 13./25.
  sigma_d = 0.
  Ux,Uy,Uz = region.basis.diffU(region.a.a,region)
  U = region.a.u
  # Compute turbulent stress tensor, tau_{ij}
  force = np.zeros(np.shape(region.iFlux.fx))
  ux = 1./U[0]*(Ux[1] - U[1]/U[0]*Ux[0])
  vx = 1./U[0]*(Ux[2] - U[2]/U[0]*Ux[0])
  wx = 1./U[0]*(Ux[3] - U[3]/U[0]*Ux[0])
  uy = 1./U[0]*(Uy[1] - U[1]/U[0]*Uy[0])
  vy = 1./U[0]*(Uy[2] - U[2]/U[0]*Uy[0])
  wy = 1./U[0]*(Uy[3] - U[3]/U[0]*Uy[0])
  uz = 1./U[0]*(Uz[1] - U[1]/U[0]*Uz[0])
  vz = 1./U[0]*(Uz[2] - U[2]/U[0]*Uz[0])
  wz = 1./U[0]*(Uz[3] - U[3]/U[0]*Uz[0]) 
  u_div = ux + vy + wz
  kx =     1./U[0]*(Ux[5] - U[5]/U[0]*Ux[0])
  omegax = 1./U[0]*(Ux[6] - U[6]/U[0]*Ux[0])
  ky =     1./U[0]*(Uy[5] - U[5]/U[0]*Uy[0])
  omegay = 1./U[0]*(Uy[6] - U[6]/U[0]*Uy[0])
  kz =     1./U[0]*(Uz[5] - U[5]/U[0]*Uz[0])
  omegaz = 1./U[0]*(Uz[6] - U[6]/U[0]*Uz[0])
  shp = np.shape(ux)
  shp = np.append( np.array([3,3]),shp)
  S = np.zeros(shp)
  tau = np.zeros(shp)
  mut = U[5]/U[6]*U[0]
  S[0,0] = ux 
  S[0,1] = 0.5*( uy + vx )
  S[0,2] = 0.5*( uz + wx )
  S[1,0] = 0.5*( vx + uy )
  S[1,1] = 0.5*( vy + vy )
  S[1,2] = 0.5*( vz + wy )
  S[2,0] = 0.5*( wx + uz )
  S[2,1] = 0.5*( wy + vz )
  S[2,2] = 0.5*( wz + wz )
  tau[0,0] = mut*(2.*S[0,0] - 2./3.*u_div ) - 2./3.*U[-2]
  tau[0,1] = mut*(2.*S[0,1]               )
  tau[0,2] = mut*(2.*S[0,2]               )
  tau[1,0] = mut*(2.*S[1,0]               )
  tau[1,1] = mut*(2.*S[1,1] - 2./3.*u_div ) - 2./3.*U[-2]
  tau[1,2] = mut*(2.*S[1,2]               )
  tau[2,0] = mut*(2.*S[2,0]               )
  tau[2,1] = mut*(2.*S[1,1]               )
  tau[2,2] = mut*(2.*S[2,2] - 2./3.*u_div ) - 2./3.*U[-2]
  P = tau[0,0]*ux + tau[0,1]*uy + tau[0,2]*uz + tau[1,0]*vx + tau[1,1]*vy + tau[1,2]*vz + tau[2,0]*wx + tau[2,1]*wy + tau[2,2]*wz
  force[-2] = P - betaStar*U[5]*U[6]/U[0]**2
  force[-1] = gammaTurb * U[-1]/U[-2] * P  - beta*U[-1]**2/U[0]
  final_term = U[0]**2 * sigma_d / U[6] * (kx*omegax + ky*omegay + kz*omegaz) 
  force[-1] += final_term
  return force


##### =========== Contains all the fluxes and physics neccesary to solve the Navier-Stokes equations within a DG framework #### ============


###### ====== Inviscid Fluxes Fluxes and Eigen Values (Eigenvalues currently not in use) ==== ############
def evalFluxXYZ_kOmega(eqns,region,u,fx,fy,fz,args):
  es = 1.e-30
  gamma = 1.4
  rho = u[0]
  rhoU = u[1]
  rhoV = u[2]
  rhoW = u[3]
  rhoE = u[4]
  rhoK = u[5]
  rhoOmega = u[6]
  p = (gamma - 1.)*(rhoE - 0.5*rhoU**2/rho - 0.5*rhoV**2/rho - 0.5*rhoW**2/rho)
  fx[0] = u[1]
  fx[1] = rhoU*rhoU/(rho) + p
  fx[2] = rhoU*rhoV/(rho) 
  fx[3] = rhoU*rhoW/(rho) 
  fx[4] = (rhoE + p)*rhoU/(rho) 
  fx[5] = rhoU*rhoK/(rho)
  fx[6] = rhoU*rhoOmega/(rho)

  fy[0] = u[2]
  fy[1] = rhoU*rhoV/(rho)
  fy[2] = rhoV*rhoV/(rho) + p 
  fy[3] = rhoV*rhoW/(rho) 
  fy[4] = (rhoE + p)*rhoV/(rho) 
  fy[5] = rhoV*rhoK/(rho)
  fy[6] = rhoV*rhoOmega/(rho)

  fz[0] = u[3]
  fz[1] = rhoU*rhoW/(rho)
  fz[2] = rhoV*rhoW/(rho) 
  fz[3] = rhoW*rhoW/(rho) + p 
  fz[4] = (rhoE + p)*rhoW/(rho) 
  fz[5] = rhoW*rhoK/(rho)
  fz[6] = rhoW*rhoOmega/(rho)




def rusanovFlux_kOmega(eqns,F,main,UL,UR,n,args=None):
  gamma = 1.4
  gmi = gamma-1.0
  #process left state
  rL = UL[0] + 1e-30
  uL = UL[1]/rL
  vL = UL[2]/rL
  wL = UL[3]/rL
  kL = UL[5]/rL
  omegaL = UL[6]/rL

  unL = uL*n[0] + vL*n[1] + wL*n[2]

  qL = (UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])**0.5/rL
  pL = (gamma-1)*(UL[4] - 0.5*rL*qL**2.)
  rHL = UL[4] + pL
  HL = rHL/rL
  cL =(gamma*pL/rL)**0.5
  # left flux
  FL = np.zeros(np.shape(UL),dtype=UL.dtype)
  FL[0] = rL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL
  FL[5] = UL[5]*unL
  FL[6] = UL[6]*unL

  # process right state
  rR = UR[0] + 1e-30
  uR = UR[1]/rR
  vR = UR[2]/rR
  wR = UR[3]/rR
  kR = UR[5]/rR
  omegaR = UR[6]/rR


  unR = uR*n[0] + vR*n[1] + wR*n[2]
  qR = (UR[1]*UR[1] + UR[2]*UR[2] + UR[3]*UR[3])**0.5/rR
  pR = (gamma-1)*(UR[4] - 0.5*rR*qR**2.)
  rHR = UR[4] + pR
  HR = rHR/rR
  cR = (gamma*pR/rR)**0.5
  # right flux
  FR = np.zeros(np.shape(UR),dtype=UR.dtype)
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  FR[5] = UR[5]*unR 
  FR[6] = UR[6]*unR 

  # difference in states
  du = UR - UL

  # Roe average
  di     = (rR/rL)**0.5
  d1     = 1.0/(1.0+di)

  ui     = (di*uR + uL)*d1
  vi     = (di*vR + vL)*d1
  wi     = (di*wR + wL)*d1
  Hi     = (di*HR + HL)*d1

  af     = 0.5*(ui*ui+vi*vi+wi*wi)
  ucp    = ui*n[0] + vi*n[1] + wi*n[2]
  c2     = gmi*(Hi - af)
  ci     = (c2)**0.5
  ci1    = 1.0/(ci + 1.e-30)

  #% eigenvalues

  sh = np.shape(ucp)
  lsh = np.append(3,sh)
#  l = np.zeros(lsh,dtype=lsh.dtype)
#  l[0] = ucp+ci
#  l[1] = ucp-ci
#  l[2] = ucp
  #print(np.shape(l))
  smax = np.abs(ucp) + np.abs(ci)
  #smax = np.maximum(np.abs(l[0]),np.abs(l[1]))
  # flux assembly
  #F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*smax*(UR[2] - UL[2])
  F[3]    = 0.5*(FL[3]+FR[3])-0.5*smax*(UR[3] - UL[3])
  F[4]    = 0.5*(FL[4]+FR[4])-0.5*smax*(UR[4] - UL[4])
  F[5]    = 0.5*(FL[5]+FR[5])-0.5*smax*(UR[5] - UL[5])
  F[6]    = 0.5*(FL[6]+FR[6])-0.5*smax*(UR[6] - UL[6])
  return F
               
  
def kfid_roeflux_kOmega(eqns,F,main,UL,UR,n,args=None):
  print('not implemented yet for kOmega')
  sys.exit()
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
  rL = UL[0] + 1.e-10
  rhoiL = 1./rL
  uL = UL[1]*rhoiL
  vL = UL[2]*rhoiL
  wL = UL[3]*rhoiL

  unL = uL*n[0] 
  unL += vL*n[1]
  unL += wL*n[2]

  qL = UL[1]**2
  qL += UL[2]**2 
  qL += UL[3]**2
  qL = (qL)**0.5
  qL *= rhoiL

  pL = (gamma-1)*(UL[4] - 0.5*rL*qL**2)
  rHL = UL[4] + pL
  HL = rHL*rhoiL
  cL = (np.abs(gamma*pL*rhoiL))**0.5
  # left flux
  FL = np.zeros(np.shape(UL),dtype=UL.dtype)
  FL[0] = rL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL

  # process right state
  rR = UR[0] + 1.e-50
  rhoiR = 1./rR
  uR = UR[1]*rhoiR
  vR = UR[2]*rhoiR

  wR = UR[3]*rhoiR

  unR =  uR*n[0]
  unR += vR*n[1]
  unR += wR*n[2]

  qR = UR[1]**2
  qR += UR[2]**2 
  qR += UR[3]**2
  qR = (qR)**0.5
  qR *= rhoiR

  pR = (gamma-1)*(UR[4] - 0.5*rR*qR*qR)
  rHR = UR[4] + pR
  HR = rHR*rhoiR
  cR = (np.abs(gamma*pR*rhoiR))**0.5
  # right flux
  FR = np.zeros(np.shape(UR),dtype=UR.dtype)
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR

  # difference in states
  du = UR - UL

  # Roe average
  di     = (np.abs(rR*rhoiL))**0.5
  d1     = 1.0/(1.0+di)

  ui     = (di*uR + uL)*d1
  vi     = (di*vR + vL)*d1
  wi     = (di*wR + wL)*d1
  Hi     = (di*HR + HL)*d1

  af     = 0.5*(ui**2+vi**2+wi**2)
  ucp    = ui*n[0] + vi*n[1] + wi*n[2]
  c2     = gmi*(Hi - af)
  ci     = (np.abs(c2))**0.5
  ci1    = 1.0/ci


  #% eigenvalues

  sh = np.shape(ucp)
  lsh = np.append(3,sh)
  l = np.zeros(lsh,dtype=UR.dtype)
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
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*(l3*du[0] + C1   )
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*(l3*du[1] + C1*ui + C2*n[0])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*(l3*du[2] + C1*vi + C2*n[1])
  F[3]    = 0.5*(FL[3]+FR[3])-0.5*(l3*du[3] + C1*wi + C2*n[2])
  F[4]    = 0.5*(FL[4]+FR[4])-0.5*(l3*du[4] + C1*Hi + C2*ucp  )
  return F



def evalViscousFluxNS_BR1_kOmega(eqns,fv,main,UL,UR,n,args=None):
  nvars = eqns.nvisc_vars
  sz = np.append(nvars,np.shape(UL[0]))
  fvL = np.zeros(sz,dtype=UL.dtype)
  fvR = np.zeros(sz,dtype=UR.dtype)
#  fvZL = np.zeros(sz)
#  fvXR = np.zeros(sz)
#  fvYR = np.zeros(sz)
#  fvZR = np.zeros(sz)

  rhoi = 1./(UL[0] + 1e-30)
  u = rhoi*UL[1]
  v = rhoi*UL[2]
  w = rhoi*UL[3]
  k = rhoi*UL[-2]
  omega = rhoi*UL[-1]

  T = (rhoi*UL[4] - 0.5*( u**2 + v**2 + w**2 ) ) #kinda a psuedo tmp, should divide by Cv but it's constant so this is taken care of in the tauFlux with gamma

  mtwo_thirds_un = u*n[0]
  mtwo_thirds_un += v*n[1]
  mtwo_thirds_un += w*n[2]
  mtwo_thirds_un *= -2./3.
  fvL[0] = mtwo_thirds_un
  fvL[0] += 2.*u*n[0]
  fvL[1] = mtwo_thirds_un
  fvL[1] += 2.*v*n[1]
  fvL[2] = mtwo_thirds_un
  fvL[2] += 2.*w*n[2]
  fvL[3] = v*n[0] + u*n[1]
  fvL[4] = w*n[0] + u*n[2] 
  fvL[5] = w*n[1] + v*n[2]
  fvL[6] = T*n[0]
  fvL[7] = T*n[1]
  fvL[8] = T*n[2]
  fvL[9] =  k*n[0]
  fvL[10] = k*n[1]
  fvL[11] = k*n[2]
  fvL[12] =  omega*n[0]
  fvL[13] = omega*n[1]
  fvL[14] = omega*n[2]



  rhoi = 1./(UR[0] + 1.e-30)
  u = rhoi*UR[1]
  v = rhoi*UR[2]
  w = rhoi*UR[3]
  k = rhoi*UR[-2]
  omega = rhoi*UR[-1]

  T = (rhoi*UR[4] - 0.5*( u**2 + v**2 + w**2 ) ) #kinda a psuedo tmp, should divide by Cv but it's constant so this is taken care of in the tauFlux with gamma
  mtwo_thirds_un = u*n[0]
  mtwo_thirds_un += v*n[1]
  mtwo_thirds_un += w*n[2]
  mtwo_thirds_un *= -2./3.
  fvR[0] = mtwo_thirds_un
  fvR[0] += 2.*u*n[0]
  fvR[1] = mtwo_thirds_un
  fvR[1] += 2.*v*n[1]
  fvR[2] = mtwo_thirds_un
  fvR[2] += 2.*w*n[2]
  fvR[3] = v*n[0] + u*n[1]
  fvR[4] = w*n[0] + u*n[2] 
  fvR[5] = w*n[1] + v*n[2]
  fvR[6] = T*n[0]
  fvR[7] = T*n[1]
  fvR[8] = T*n[2]
  fvR[9] =  k*n[0]
  fvR[10] = k*n[1]
  fvR[11] = k*n[2]
  fvR[12] = omega*n[0]
  fvR[13] = omega*n[1]
  fvR[14] = omega*n[2]
  fv[:] = 0.5*(fvL + fvR)

def evalViscousFluxXNS_BR1_kOmega(main,U,fv):
  u = U[1]/U[0]
  v = U[2]/U[0]
  w = U[3]/U[0]
  k = U[-2]/U[0]
  omega = U[-1]/U[0]

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
  fv[9] = k 
  fv[10] = 0.
  fv[11] = 0.
  fv[12] = omega
  fv[13] = 0.
  fv[14] = 0.


#
def evalViscousFluxYNS_BR1_kOmega(main,U,fv):
  u = U[1]/U[0]
  v = U[2]/U[0]
  w = U[3]/U[0]
  k = U[-2]/U[0]
  omega = U[-1]/U[0]
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
  fv[9] = 0. 
  fv[10] = k 
  fv[11] = 0.
  fv[12] = 0.
  fv[13] = omega 
  fv[14] = 0.

def evalViscousFluxZNS_BR1_kOmega(main,U,fv):
  u = U[1]/U[0]
  v = U[2]/U[0]
  w = U[3]/U[0]
  k = U[-2]/U[0]
  omega = U[-1]/U[0]
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
  fv[9] = 0. 
  fv[10] = 0. 
  fv[11] = k
  fv[12] = 0.
  fv[13] = 0. 
  fv[14] = omega

def evalTauFluxNS_BR1_kOmega(eqns,fv,main,uL,uR,n,args):
  sigmak = 0.5
  sigmaw = 0.5

  tauL = args[0]
  tauR = args[1]
  mu = main.mus
  muL,muR = mu,mu
  sz = np.shape(uL)
  fvL = np.zeros(sz,dtype=uL.dtype)
  fvR = np.zeros(sz,dtype=uR.dtype)

  Pr = 0.72
  Pri = 1./Pr
  gamma = 1.4

  rhoinv = 1./(uL[0] + 1.e-30)
  fvL[1] = tauL[0]*n[0]
  fvL[1] += tauL[3]*n[1]
  fvL[1] += tauL[4]*n[2]  #tau11*n1 + tau21*n2 + tau31*n3

  fvL[2] = tauL[3]*n[0]
  fvL[2] += tauL[1]*n[1]
  fvL[2] += tauL[5]*n[2]  #tau12*n1 + tau22*n2 + tau32*n3

  fvL[3] = tauL[4]*n[0]
  fvL[3] += tauL[5]*n[1]
  fvL[3] += tauL[2]*n[2]  #tau12*n1 + tau22*n2 + tau32*n3

  fvL[4] = fvL[1]*uL[1]
  fvL[4] += fvL[2]*uL[2]
  fvL[4] += fvL[3]*uL[3]
  fvL[4] *= rhoinv
  fvL[4] += gamma*Pri*(tauL[6]*n[0] + tauL[7]*n[1] + tauL[8]*n[2])

  fvL[5] = tauL[9]*n[0]
  fvL[5] += tauL[10]*n[1]
  fvL[5] += tauL[11]*n[2]  #k_x*n1 + k_y*n2 + k_z*n3

  fvL[6] = tauL[12]*n[0]
  fvL[6] += tauL[13]*n[1]
  fvL[6] += tauL[14]*n[2]  
  mut_L = (uL[-2]/uL[-1]*uL[0])
  fvL[0:5] *= (mu + mut_L)[None,:]
  fvL[5] *= (mu + mut_L *sigmak)
  fvL[6] *= (mu + mut_L *sigmaw)

  rhoinv = 1./(uR[0] + 1.e-30)
  fvR[1] = tauR[0]*n[0]
  fvR[1] += tauR[3]*n[1]
  fvR[1] += tauR[4]*n[2]  #tau11*n1 + tau21*n2 + tau31*n3

  fvR[2] = tauR[3]*n[0]
  fvR[2] += tauR[1]*n[1]
  fvR[2] += tauR[5]*n[2]  #tau12*n1 + tau22*n2 + tau32*n3

  fvR[3] = tauR[4]*n[0]
  fvR[3] += tauR[5]*n[1]
  fvR[3] += tauR[2]*n[2]  #tau12*n1 + tau22*n2 + tau32*n3

  fvR[4] = fvR[1]*uR[1]
  fvR[4] += fvR[2]*uR[2]
  fvR[4] += fvR[3]*uR[3]
  fvR[4] *= rhoinv
  fvR[4] += gamma*Pri*(tauR[6]*n[0] + tauR[7]*n[1] + tauR[8]*n[2])


  fvR[5] = tauR[9]*n[0]
  fvR[5] += tauR[10]*n[1]
  fvR[5] += tauR[11]*n[2]  #k_x*n1 + k_y*n2 + k_z*n3

  fvR[6] = tauR[12]*n[0]
  fvR[6] += tauR[13]*n[1]
  fvR[6] += tauR[14]*n[2]  

  mut_R = (uR[-2]/uR[-1]*uR[0])
  fvR[0:5] *= (mu + mut_R)[None,:]
  fvR[5] *= (mu + mut_R *sigmak)
  fvR[6] *= (mu + mut_R *sigmaw)
  fv[:] = 0.5*(fvL + fvR)
  return fv

def evalTauFluxXNS_BR1_kOmega(main,tau,u,fvX,mu,cgas):
  Pr = 0.72
  gamma = 1.4
  sigmak = 0.5
  sigmaw = 0.5
  mut = (u[-2]/u[-1]*u[0])
  fvX[0] = 0.
  fvX[1] = (mu+mut)*tau[0] #tau11
  fvX[2] = (mu+mut)*tau[3] #tau21
  fvX[3] = (mu+mut)*tau[4] #tau31
  fvX[4] = (mu+mut)*(tau[0]*u[1]/u[0] + tau[3]*u[2]/u[0] + tau[4]*u[3]/u[0] + gamma/Pr*tau[6] ) #+ (mu + mu_t/sigma_k)
  fvX[5] = (mu + u[-2]/u[-1]*u[0]*sigmak)*tau[9]
  fvX[6] = (mu + u[-2]/u[-1]*u[0]*sigmaw)*tau[12]


def evalTauFluxYNS_BR1_kOmega(main,tau,u,fvY,mu,cgas):
  sigmak = 0.5
  sigmaw = 0.5

  Pr = 0.72
  gamma = 1.4
  mut = (u[-2]/u[-1]*u[0])
  fvY[0] = 0.
  fvY[1] = (mu+mut)*tau[3] #tau21
  fvY[2] = (mu+mut)*tau[1] #tau22
  fvY[3] = (mu+mut)*tau[5] #tau23
  fvY[4] = (mu+mut)*(tau[3]*u[1]/u[0] + tau[1]*u[2]/u[0] + tau[5]*u[3]/u[0] + gamma/Pr*tau[7])
  fvY[5] = (mu + u[-2]/u[-1]*u[0]*sigmak)*tau[10]
  fvY[6] = (mu + u[-2]/u[-1]*u[0]*sigmaw)*tau[13]

def evalTauFluxZNS_BR1_kOmega(main,tau,u,fvZ,mu,cgas):
  sigmak = 0.5
  sigmaw = 0.5

  Pr = 0.72
  gamma = 1.4
  mut = (u[-2]/u[-1]*u[0])
  fvZ[0] = 0.
  fvZ[1] = (mu+mut)*tau[4] #tau31
  fvZ[2] = (mu+mut)*tau[5] #tau32
  fvZ[3] = (mu+mut)*tau[2] #tau33
  fvZ[4] = (mu+mut)*(tau[4]*u[1]/u[0] + tau[5]*u[2]/u[0] + tau[2]*u[3]/u[0] + gamma/Pr*tau[8])
  fvZ[5] = (mu + u[-2]/u[-1]*u[0]*sigmak)*tau[11]
  fvZ[6] = (mu + u[-2]/u[-1]*u[0]*sigmaw)*tau[14]


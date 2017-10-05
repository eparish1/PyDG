import numpy as np
from tensor_products import *
##### =========== Contains all the fluxes and physics neccesary to solve the Navier-Stokes equations using entropy variables within a DG framework #### ============
## mass matrix for entropy - i.e we have \int w dU(v)/dt d\Omega
## This function computes the matrix \int w du/dv w'
def getEntropyMassMatrix(main):
  def getInnerMassMatrix(main,g):
    f = main.w0[:,None,None,None,:,None,None,None]*main.w1[None,:,None,None,None,:,None,None]\
       *main.w2[None,None,:,None,None,None,:,None]*main.w3[None,None,None,:,None,None,None,:]
    norder = main.order[0]*main.order[1]*main.order[2]*main.order[3]
    M2 = np.zeros((5*norder,norder,\
                  main.Npx,main.Npy,main.Npz,1 ) )
    count = 0
    for i in range(0,main.order[0]):
      for j in range(0,main.order[1]):
        for k in range(0,main.order[2]):
          for l in range(0,main.order[3]):
            #M2[count] =np.reshape( volIntegrateGlob_einsum_2(main,(f*f[i,j,k,l])[None,:,:,:,:,:,:,:,:,None,None,None,None]*main.Jdet[None,None,None,None,None,:,:,:,None,:,:,:,None]) , np.shape(M2[0]))
            M2[:,count] =np.reshape( volIntegrateGlob_tensordot(main,g*f[i,j,k,l][None,:,:,:,:,None,None,None,None]*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.w3) , np.shape(M2[:,0]))
            count += 1
    return M2
  #=================
  dudv = mydUdV(main.a.u)
  norder = main.order[0]*main.order[1]*main.order[2]*main.order[3]
  M = np.zeros((norder*5,norder*5,main.Npx,main.Npy,main.Npz,main.Npt) )
  count = 0
  I = np.eye(5)
  #for i in range(0,5):
  for j in range(0,5):
      #M[i*norder:(i+1)*norder,j*norder:(j+1)*norder] = getInnerMassMatrix(main,dudv[i,j])
      M[:,j*norder:(j+1)*norder] = getInnerMassMatrix(main,dudv[:,j])

#  print(np.shape(M))
#  for i in range(0,main.Npx):
#    for j in range(0,main.Npy):
#      print(i,j,np.diag(M[:,:,i,j,0,0]))
#      Mi = np.linalg.inv(M[:,:,i,j,0,0])
  #M = np.reshape(M, (5*norder,5*norder,main.Npx,main.Npy,main.Npz,main.Npt))
  M = np.rollaxis( np.rollaxis(M,1,6),0,5)
  #print(np.shape(M))
  M = np.linalg.inv(M)
  M = np.rollaxis( np.rollaxis(M,4,0),5,1)
  #print(np.shape(M))
  return M
## Mappings between entropy variables (V) and conservative variables (U)
def dUdV(V):
  u = entropy_to_conservative(V)
  gamma = 1.4
  gamma_bar = gamma - 1.

  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  sz = np.shape(V)
  sz = np.append(5,sz)
  A0 = np.zeros(sz)
  k1 = 0.5*(V[1]**2 + V[2]**2 + V[3]**2)/V[4]
  k2 = k1 - gamma
  k3 = k1**2 - 2.*gamma*k1 + gamma
  k4 = k2 - gamma_bar
  k5 = k2**2 - gamma_bar*(k1 + k2)	
  c1 = gamma_bar*V[4] - V[1]**2
  c2 = gamma_bar*V[4] - V[2]**2
  c3 = gamma_bar*V[4] - V[3]**2
  d1 = -V[1]*V[2]
  d2 = -V[1]*V[3]
  d3 = -V[2]*V[3]
  e1 = V[1]*V[4]
  e2 = V[2]*V[4]
  e3 = V[3]*V[4]
  A0[0,0] = -V[4]**2
  A0[0,1] = e1
  A0[0,2] = e2
  A0[0,3] = e3
  A0[0,4] = V[4]*(1 - k1)
  A0[1,0] = e1
  A0[1,1] = c1
  A0[1,2] = d1
  A0[1,3] = d2
  A0[1,4] = V[1]*k2
  A0[2,0] = A0[0,2]
  A0[2,1] = A0[1,2]
  A0[2,2] = c2
  A0[2,3] = d3
  A0[2,4] = V[2]*k2
  A0[3,0] = A0[0,3]
  A0[3,1] = A0[1,3]
  A0[3,2] = A0[2,3]
  A0[3,3] = c3
  A0[3,4] = V[3]*k2
  A0[4,0] = A0[0,4]
  A0[4,1] = A0[1,4]
  A0[4,2] = A0[2,4]
  A0[4,3] = A0[3,4]
  A0[4,4] = -k3
  return A0 * p / ( gamma_bar**2 * V[4] )

def mydUdV(V):
  U = entropy_to_conservative(V)
  gamma = 1.4
  gamma_bar = gamma - 1.
  es = 1.e-30
  p = (gamma - 1.)*(U[4] - 0.5*U[1]**2/U[0] - 0.5*U[2]**2/U[0] - 0.5*U[3]**2/U[0])
  H = (U[4] + p) / U[0]
  asqr = gamma*p/U[0]
  sz = np.shape(V)
  sz = np.append(5,sz)
  A0 = np.zeros(sz)
  A0[0,:] = U[:]
  A0[1,0] = A0[0,1]
  A0[1,1] = U[1]**2/U[0] + p#U[1]*(1./(V[1]+es) - V[1]/V[4])
  A0[1,2] = -U[1]*V[2]/V[4]
  A0[1,3] = -U[1]*V[3]/V[4]
  A0[1,4] = -U[1]/V[4] - V[1]*U[4]/V[4]
  A0[2,0] = A0[0,2]
  A0[2,1] = A0[1,2]
  A0[2,2] = U[2]**2/U[0] + p#U[2]*(1./(V[2]+es) - V[2]/V[4])
  A0[2,3] = -U[2]*V[3]/V[4]
  A0[2,4] = -U[2]/V[4] - V[2]*U[4]/V[4]
  A0[3,0] = A0[0,3]
  A0[3,1] = A0[1,3]
  A0[3,2] = A0[2,3]
  A0[3,3] = U[3]**2/U[0] + p#U[3]*(1./(V[3]+es) - V[3]/V[4])
  A0[3,4] = -U[3]/V[4] - V[3]*U[4]/V[4]
  A0[4,0] = A0[0,4]
  A0[4,1] = A0[1,4]
  A0[4,2] = A0[2,4]
  A0[4,3] = A0[3,4]
  A0[4,4] = U[0]*H**2 - asqr*p/(gamma - 1.)
  return A0 



def dVdU(V):
  sz = np.shape(V[0])
  sz = np.append(5,sz)
  A0inv = np.zeros(sz)
  gamma = 1.4
  gamma_bar = gamma - 1.
  k1 = 0.5*(V[1]**2 + V[2]**2 + V[3]**2)/V[4]
  k2 = k1 - gamma
  k3 = k1**2 - 2.*gamma*k1 + gamma
  k4 = k2 - gamma_bar
  k5 = k2**2 - gamma_bar*(k1 + k2)	
  c1 = gamma_bar*V[4] - V[1]**2
  c2 = vamma_bar*V[4] - V[2]**2
  c3 = vamma_bar*V[4] - V[3]**2
  d1 = -V[1]*V[2]
  d2 = -V[1]*V[3]
  d3 = -V[2]*V[3]
  e1 = V[1]*V[4]
  e2 = V[2]*V[4]
  e3 = V[3]*V[4]

  A0inv[0,0] = k1**2 + gamma
  A0inv[0,1] = K1*V[1]
  A0inv[0,2] = k1*V[2]
  A0inv[0,3] = k1*V[3]
  A0inv[0,4] = (k1 + 1.)*V[4]
  A0inv[1,0] = A0inv[0,1]
  A0inv[1,1] = -V[1]**2 - V[4]
  A0inv[1,2] = -d1
  A0inv[1,3] = -d2
  A0inv[1,4] = e1
  A0inv[2,0] = A0inv[0,2]
  A0inv[2,1] = A0inv[1,2]
  A0inv[2,2] = -V[2]**2 - V[4]
  A0inv[2,3] = -d3
  A0inv[2,4] = e2
  A0inv[3,0] = A0inv[0,3]
  A0inv[3,1] = A0inv[1,3]
  A0inv[3,2] = A0inv[2,3]
  A0inv[3,3] = V[3]**2 - V[4]
  A0inv[3,4] = e3
  A0inv[4,0] = A0inv[0,4]
  A0inv[4,1] = A0inv[1,4]
  A0inv[4,2] = A0inv[2,4]
  A0inv[4,3] = A0inv[3,4]
  A0inv[4,4] = V[4]**2
  return - A0inv * gamma_bar/( p * V[4] )

def entropy_to_conservative(V):
  gamma = 1.4
  U = np.zeros(np.shape(V))
  gamma1 = gamma - 1.
  igamma1 = 1./gamma1
  gmogm1 = gamma*igamma1
  iu4 = 1./V[4]  #- p / rho
  u = -iu4*V[1]
  v = -iu4*V[2]
  w = -iu4*V[3]
  t0 = -0.5*iu4*(V[1]**2 + V[2]**2 + V[3]**2)
  t1 = V[0] - gmogm1 + t0
  t2 =np.exp(-igamma1*np.log(-V[4]) )
  t3 = np.exp(t1)
  U[0] = t2*t3
  H = -iu4*(gmogm1 + t0)
  E = (H + iu4)
  U[1] = U[0]*u 
  U[2] = U[0]*v
  U[3] = U[0]*w
  U[4] = U[0]*E
  return U


###### ====== Inviscid Fluxes Fluxes and Eigen Values (Eigenvalues currently not in use) ==== ############

def evalFluxXEulerEntropy(main,v,f,args):
  u = entropy_to_conservative(v) 
  es = 1.e-30
  gamma = 1.4
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  f[0] = u[1]
  f[1] = u[1]*u[1]/(u[0]) + p
  f[2] = u[1]*u[2]/(u[0])
  f[3] = u[1]*u[3]/(u[0])
  f[4] = (u[4] + p)*u[1]/(u[0])


def evalFluxYEulerEntropy(main,v,f,args):
  u = entropy_to_conservative(v) 
  gamma = 1.4
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  f[0] = u[2]
  f[1] = u[1]*u[2]/u[0]
  f[2] = u[2]*u[2]/u[0] + p
  f[3] = u[2]*u[3]/u[0] 
  f[4] = (u[4] + p)*u[2]/u[0]



def evalFluxZEulerEntropy(main,v,f,args):
  u = entropy_to_conservative(v) 
  gamma = 1.4
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  f[0] = u[3]
  f[1] = u[1]*u[3]/u[0]
  f[2] = u[2]*u[3]/u[0] 
  f[3] = u[3]*u[3]/u[0] + p 
  f[4] = (u[4] + p)*u[3]/u[0]


#==================== Numerical Fluxes for the Faces =====================
#== central flux
#== rusanov flux
#== Roe flux

def eulerCentralFluxEntropy(F,main,VL,VR,n,args=None):
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
  UL = entropy_to_conservative(VL) 
  UR = entropy_to_conservative(VR) 

  gamma = 1.4
  gmi = gamma-1.0
  #process left state
  rL = UL[0]
  uL = UL[1]/rL
  vL = UL[2]/rL
  wL = UL[3]/rL

  unL = uL*n[0] + vL*n[1] + wL*n[2]

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  pL = (gamma-1)*(UL[4] - 0.5*rL*qL**2.)
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
  pR = (gamma-1)*(UR[4] - 0.5*rR*qR**2.)
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
  F[0]    = 0.5*(FL[0]+FR[0])#-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])#-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])#-0.5*smax*(UR[2] - UL[2])
  F[3]    = 0.5*(FL[3]+FR[3])#-0.5*smax*(UR[3] - UL[3])
  F[4]    = 0.5*(FL[4]+FR[4])#-0.5*smax*(UR[4] - UL[4])
  return F


def ismailFluxEntropy(F,main,VL,VR,n,args=None):
# PURPOSE: This function calculates the flux for the Euler equations
# using the ismail flux function
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
  UL = entropy_to_conservative(VL) 
  UR = entropy_to_conservative(VR) 
  gamma = 1.4
  gmi = gamma-1.0
  #process left state
  rL = UL[0]
  uL = UL[1]/rL
  vL = UL[2]/rL
  wL = UL[3]/rL

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  pL = (gamma-1)*(UL[4] - 0.5*rL*qL**2.)

  z1L = np.sqrt(rL/pL)
  z2L = uL*z1L
  z3L = vL*z1L
  z4L = wL*z1L
  z5L = pL*z1L


  # process right state
  rR = UR[0]
  uR = UR[1]/rR
  vR = UR[2]/rR
  wR = UR[3]/rR
  qR = np.sqrt(UR[1]*UR[1] + UR[2]*UR[2] + UR[3]*UR[3])/rR
  pR = (gamma-1)*(UR[4] - 0.5*rR*qR**2.)

  z1R = np.sqrt(rR/pR)
  z2R = uR*z1R
  z3R = vR*z1R
  z4R = wR*z1R
  z5R = pR*z1R

  z1bar = 0.5*(z1L + z1R)
  z2bar = 0.5*(z2L + z2R)
  z3bar = 0.5*(z3L + z3R)
  z4bar = 0.5*(z4L + z4R)
  z5bar = 0.5*(z5L + z5R)
 
  zeta = z1L/z1R
  f =  (zeta - 1.)/(zeta + 1.)
  uf = f*f
  Fa = 1.0 + 1./3.*uf + 1./5.*uf**2 + 1./7.*uf**3 + 1./9.*uf**4
  eps = 1e-3
  Fa[uf>eps] = (0.5*np.log(zeta[uf>eps])/f[uf>eps])
  z1log = z1bar/Fa

  zeta = z5L/z5R
  f =  (zeta - 1.)/(zeta + 1.)
  uf = f*f
  Fa = 1.0 + 1./3.*uf + 1./5.*uf**2 + 1./7.*uf**3 + 1./9.*uf**4
  eps = 1e-3
  Fa[uf>eps] = (0.5*np.log(zeta[uf>eps])*1./f[uf>eps])
  z5log = z5bar/Fa

  rhohat = z1bar*z5log
  uhat = z2bar/z1bar
  vhat = z3bar/z1bar
  what = z4bar/z1bar
  V_sqr = uhat**2 + vhat**2 + what**2
  V_n = uhat*n[0] + vhat*n[1] + what*n[2]

  p1hat = z5bar/z1bar
  p2hat = 0.5*(gamma+1)/(gamma)*z5log/z1log + 0.5*(gamma - 1.)/gamma*z5bar/z1bar
  ahat_sqr = gamma*p2hat/rhohat
  Hhat = ahat_sqr/(gamma - 1.) + 0.5*V_sqr

  # right flux
  rhoV_n = rhohat*V_n
  F[0]    = rhoV_n 
  F[1]    = rhoV_n*uhat + p1hat*n[0]
  F[2]    = rhoV_n*vhat + p1hat*n[1]
  F[3]    = rhoV_n*what + p1hat*n[2]
  F[4]    = rhoV_n*Hhat 
  return F


def rusanovFluxEntropy(F,main,VL,VR,n,args=None):
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
  UL = entropy_to_conservative(VL) 
  UR = entropy_to_conservative(VR) 

  gamma = 1.4
  gmi = gamma-1.0
  #process left state
  rL = UL[0]
  uL = UL[1]/rL
  vL = UL[2]/rL
  wL = UL[3]/rL

  unL = uL*n[0] + vL*n[1] + wL*n[2]

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  pL = (gamma-1)*(UL[4] - 0.5*rL*qL**2.)
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
  pR = (gamma-1)*(UR[4] - 0.5*rR*qR**2.)
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
  #print(np.shape(l))
  smax = np.abs(ucp) + np.abs(ci)
  #smax = np.maximum(np.abs(l[0]),np.abs(l[1]))
  # flux assembly
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*smax*(UR[2] - UL[2])
  F[3]    = 0.5*(FL[3]+FR[3])-0.5*smax*(UR[3] - UL[3])
  F[4]    = 0.5*(FL[4]+FR[4])-0.5*smax*(UR[4] - UL[4])
  return F
               
  

def kfid_roefluxEntropy(F,main,VL,VR,n,args=None):
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
  UL = entropy_to_conservative(VL) 
  UR = entropy_to_conservative(VR) 

  gamma = 1.4
  gmi = gamma-1.0
  #process left state
  rL = UL[0]
  uL = UL[1]/rL
  vL = UL[2]/rL
  wL = UL[3]/rL

  unL = uL*n[0] + vL*n[1] + wL*n[2]

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  pL = (gamma-1)*(UL[4] - 0.5*rL*qL*qL)
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
  pR = (gamma-1)*(UR[4] - 0.5*rR*qR*qR)
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
  F[:] = 0.
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*(l3*du[0] + C1   )
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*(l3*du[1] + C1*ui + C2*n[0])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*(l3*du[2] + C1*vi + C2*n[1])
  F[3]    = 0.5*(FL[3]+FR[3])-0.5*(l3*du[3] + C1*wi + C2*n[2])
  F[4]    = 0.5*(FL[4]+FR[4])-0.5*(l3*du[4] + C1*Hi + C2*ucp  )
  return F



###============= Diffusion Fluxes =====================
def getGsNSEntropy(v,main):
  u = entropy_to_conservative(v) 

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



def getGsNSX_FASTEntropy(v,main,mu,V):
  u = entropy_to_conservative(v)
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


def getGsNSY_FASTEntropy(u,main,mu,V):
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

def getGsNSZ_FASTEntropy(v,main,mu,V):
  u = entropy_to_conservative(v)

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



def getGsNSXEntropy(v,main,mu):
  u = entropy_to_conservative(v)

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



def getGsNSYEntropy(v,main,mu):
  u = entropy_to_conservative(v)

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

def getGsNSZEntropy(v,main,mu):
  u = entropy_to_conservative(v)

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


def evalViscousFluxXNS_IPEntropy(main,v,Ux,Uy,Uz,mu):

  u = entropy_to_conservative(v)

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


def evalViscousFluxYNS_IPEntropy(main,v,Ux,Uy,Uz,mu):
  u = entropy_to_conservative(v)

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

def evalViscousFluxZNS_IPEntropy(main,v,Ux,Uy,Uz,mu):
  u = entropy_to_conservative(v)

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
def evalViscousFluxNS_BR1Entropy(fv,main,VL,VR,n,args=None):
  UL = entropy_to_conservative(VL)
  UR = entropy_to_conservative(VR)

  nvars = 9
  sz = np.append(nvars,np.shape(UL[0]))
  fvL = np.zeros(sz)
  fvR = np.zeros(sz)
#  fvZL = np.zeros(sz)
#  fvXR = np.zeros(sz)
#  fvYR = np.zeros(sz)
#  fvZR = np.zeros(sz)

  rhoi = 1./UL[0]
  u = rhoi*UL[1]
  v = rhoi*UL[2]
  w = rhoi*UL[3]
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

  rhoi = 1./UR[0]
  u = rhoi*UR[1]
  v = rhoi*UR[2]
  w = rhoi*UR[3]
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
  fv[:] = 0.5*(fvL + fvR)
  return fv


def evalViscousFluxXNS_BR1Entropy(main,V,fv):
  U = entropy_to_conservative(V)
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
def evalViscousFluxYNS_BR1Entropy(main,V,fv):
  U = entropy_to_conservative(V)
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

def evalViscousFluxZNS_BR1Entropy(main,V,fv):
  U = entropy_to_conservative(V)
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


def evalTauFluxNS_BR1Entropy(fv,main,VL,VR,n,args):
  uL = entropy_to_conservative(VL)
  uR = entropy_to_conservative(VR)

  tauL = args[0]
  tauR = args[1]
  mu = main.mus
  muL,muR = mu,mu
  sz = np.shape(uL)
  fvL = np.zeros(sz)
  fvR = np.zeros(sz)

  Pr = 0.72
  Pri = 1./Pr
  gamma = 1.4

  rhoinv = 1./uL[0]
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

  fvL *= mu

  rhoinv = 1./uR[0]
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

  fvR *= mu

  fv[:] = 0.5*(fvL + fvR)
 
  return fv

def evalTauFluxXNS_BR1Entropy(main,tau,v,fvX,mu,cgas):
  u = entropy_to_conservative(v)
  Pr = 0.72
  gamma = 1.4
  fvX[0] = 0.
  fvX[1] = mu*tau[0] #tau11
  fvX[2] = mu*tau[3] #tau21
  fvX[3] = mu*tau[4] #tau31
  fvX[4] = mu*(tau[0]*u[1]/u[0] + tau[3]*u[2]/u[0] + tau[4]*u[3]/u[0] + gamma/Pr*tau[6] )

def evalTauFluxYNS_BR1Entropy(main,tau,v,fvY,mu,cgas):
  u = entropy_to_conservative(v)
  Pr = 0.72
  gamma = 1.4
  fvY[0] = 0.
  fvY[1] = mu*tau[3] #tau21
  fvY[2] = mu*tau[1] #tau22
  fvY[3] = mu*tau[5] #tau23
  fvY[4] = mu*(tau[3]*u[1]/u[0] + tau[1]*u[2]/u[0] + tau[5]*u[3]/u[0] + gamma/Pr*tau[7])

def evalTauFluxZNS_BR1Entropy(main,tau,v,fvZ,mu,cgas):
  u = entropy_to_conservative(v)
  Pr = 0.72
  gamma = 1.4
  fvZ[0] = 0.
  fvZ[1] = mu*tau[4] #tau31
  fvZ[2] = mu*tau[5] #tau32
  fvZ[3] = mu*tau[2] #tau33
  fvZ[4] = mu*(tau[4]*u[1]/u[0] + tau[5]*u[2]/u[0] + tau[2]*u[3]/u[0] + gamma/Pr*tau[8])


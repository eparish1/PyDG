import numpy as np

def starState(var,main):
  uRLS = np.zeros((var.nvars,var.quadpoints,main.Npx+1,main.Npy))
  uUDS = np.zeros((var.nvars,var.quadpoints,main.Npx,main.Npy+1))
  uRLS[:,:,1:-1,:] = 0.5*(var.uR[:,:,0:-1,:] + var.uL[:,:,1::,:])
  uRLS[:,:,  -1,:] = 0.5*(var.uR[:,:,  -1,:] + var.uR_edge)
  uRLS[:,:,   0,:] = 0.5*(var.uL[:,:,0  ,:]  + var.uL_edge)
  uUDS[:,:,:,1:-1] = 0.5*(var.uU[:,:,:,0:-1] + var.uD[:,:,:,1::])
  uUDS[:,:,:,  -1] = 0.5*(var.uU[:,:,:,  -1] + var.uU_edge)
  uUDS[:,:,:,   0] = 0.5*(var.uD[:,:,:,   0] + var.uD_edge)
  return uRLS,uUDS

#def starState(var):
#  var.u.uRS[:,:,0:-1,:] = 0.5*(var.u.uR[:,:,0:-1,:] + var.u.uL[:,:,1::,:])
#  var.u.uRS[:,:,  -1,:] = 0.5*(var.u.uR[:,:,  -1,:] + var.u.uL[:,:,0  ,:])
#  var.u.uLS[:,:,1:: ,:] = var.u.uRS[:,:,0:-1,:]
#  var.u.uLS[:,:,0   ,:] = var.u.uRS[:,:,  -1,:]
#  var.u.uUS[:,:,:,0:-1] = 0.5*(var.u.uU[:,:,:,0:-1] + var.u.uD[:,:,:,1::])
#  var.u.uUS[:,:,:,  -1] = 0.5*(var.u.uU[:,:,:,  -1] + var.u.uU_edge)
#  var.u.uDS[:,:,:,1:: ] = var.fUS[:,:,:,0:-1]
#  var.u.uDS[:,:,:,0   ] = 0.5*(var.uD[:,:,:,   0] + var.uD_edge)

def centralFluxGeneral(fR,fL,fU,fD,fR_edge,fL_edge,fU_edge,fD_edge):
  fRS = np.zeros(np.shape(fR))
  fLS = np.zeros(np.shape(fL))
  fUS = np.zeros(np.shape(fU))
  fDS = np.zeros(np.shape(fD))
  fRS[:,:,0:-1,:] = 0.5*(fR[:,:,0:-1,:] + fL[:,:,1::,:])
  fRS[:,:,  -1,:] = 0.5*(fR[:,:,  -1,:] + fR_edge)
  fLS[:,:,1:: ,:] = fRS[:,:,0:-1,:]
  fLS[:,:,0   ,:] = 0.5*(fL[:,:,0,:]    + fL_edge)
  fUS[:,:,:,0:-1] = 0.5*(fU[:,:,:,0:-1] + fD[:,:,:,1::])
  fUS[:,:,:,  -1] = 0.5*(fU[:,:,:,  -1] + fU_edge)
  fDS[:,:,:,1:: ] = fUS[:,:,:,0:-1]
  fDS[:,:,:,0   ] = 0.5*(fD[:,:,:,   0] + fD_edge)
  return fRS,fLS,fUS,fDS


def centralFlux(main,eqns,schemes,fluxVar,var):
  fluxVar.fRS[:,:,0:-1,:] = 0.5*(fluxVar.fR[:,:,0:-1,:] + fluxVar.fL[:,:,1::,:])
  fluxVar.fRS[:,:,  -1,:] = 0.5*(fluxVar.fR[:,:,  -1,:] + fluxVar.fR_edge)
  fluxVar.fLS[:,:,1:: ,:] = fluxVar.fRS[:,:,0:-1,:]
  fluxVar.fLS[:,:,0   ,:] = 0.5*(fluxVar.fL[:,:,0,:]    + fluxVar.fL_edge)
  fluxVar.fUS[:,:,:,0:-1] = 0.5*(fluxVar.fU[:,:,:,0:-1] + fluxVar.fD[:,:,:,1::])
  fluxVar.fUS[:,:,:,  -1] = 0.5*(fluxVar.fU[:,:,:,  -1] + fluxVar.fU_edge)
  fluxVar.fDS[:,:,:,1:: ] = fluxVar.fUS[:,:,:,0:-1]
  fluxVar.fDS[:,:,:,0   ] = 0.5*(fluxVar.fD[:,:,:,   0] + fluxVar.fD_edge)


def rusanovFlux(main,eqns,schemes,fluxVar,var):
  uRLS,uUDS = starState(var,main)
  eigsRL,eigsUD = eqns.getEigs(uRLS,uUDS)
  fluxVar.fRS[:,:,0:-1,:] = 0.5*(fluxVar.fR[:,:,0:-1,:] + fluxVar.fL[:,:,1::,:]) - 0.5*eigsRL[:,:,1:-1,:]*(var.uL[:,:,1::,:] - var.uR[:,:,0:-1,:])
  fluxVar.fRS[:,:,  -1,:] = 0.5*(fluxVar.fR[:,:,  -1,:] + fluxVar.fR_edge) - 0.5*eigsRL[:,:,  -1,:]*(var.uR_edge - var.uR[:,:,  -1,:])
  fluxVar.fLS[:,:,1:: ,:] = fluxVar.fRS[:,:,0:-1,:]
  fluxVar.fLS[:,:,0   ,:] = 0.5*(fluxVar.fL_edge + fluxVar.fL[:,:,0,:] )         - 0.5*eigsRL[:,:,0,   :]*(var.uL[:,:,0,:  ] - var.uL_edge) 
  fluxVar.fUS[:,:,:,0:-1] = 0.5*(fluxVar.fU[:,:,:,0:-1] + fluxVar.fD[:,:,:,1::]) - 0.5*eigsUD[:,:,:,1:-1]*(var.uD[:,:,:,1::] - var.uU[:,:,:,0:-1])
  fluxVar.fUS[:,:,:,  -1] = 0.5*(fluxVar.fU[:,:,:,  -1] + fluxVar.fU_edge) -       0.5*eigsUD[:,:,:,  -1]*(var.uU_edge       - var.uU[:,:,:,  -1])
  fluxVar.fDS[:,:,:,1:: ] = 0.5*(fluxVar.fU[:,:,:,0:-1] + fluxVar.fD[:,:,:,1::]) - 0.5*eigsUD[:,:,:,1:-1]*(var.uD[:,:,:,1::] - var.uU[:,:,:,0:-1])
  fluxVar.fDS[:,:,:,0   ] = 0.5*(fluxVar.fD_edge + fluxVar.fD[:,:,:,0])          - 0.5*eigsUD[:,:,:,   0]*(var.uD[:,:,:,0  ] - var.uD_edge)


def roeFlux(main,eqns,schemes,fluxVar,var):
  nx = np.array([1,0])
  ny = np.array([0,1])
  fluxVar.fRS[:,:,0:-1,:] = kfid_roeflux(var.uR[:,:,0:-1,:],var.uL[:,:,1::,:],nx)
  fluxVar.fRS[:,:,  -1,:] = kfid_roeflux(var.uR[:,:,  -1,:],var.uR_edge,nx)
  fluxVar.fLS[:,:,1:: ,:] = fluxVar.fRS[:,:,0:-1,:]
  fluxVar.fLS[:,:,0   ,:] = kfid_roeflux(var.uL_edge,var.uL[:,:,0],nx)
  fluxVar.fUS[:,:,:,0:-1] = kfid_roeflux(var.uU[:,:,:,0:-1],var.uD[:,:,:,1::],ny )
  fluxVar.fUS[:,:,:,  -1] = kfid_roeflux(var.uU[:,:,:,  -1],var.uU_edge,ny)
  fluxVar.fDS[:,:,:,1:: ] = fluxVar.fUS[:,:,:,0:-1] 
  fluxVar.fDS[:,:,:,0   ] = kfid_roeflux(var.uD_edge,var.uD[:,:,:,0],ny)


#def roeFlux(main,eqns,schemes,fluxVar,var):
def kfid_roeflux(UL,UR,n):
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
  unL = uL*n[0] + vL*n[1]
  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2])/rL
  pL = (gamma-1)*(UL[3] - 0.5*rL*qL**2.)
  rHL = UL[3] + pL
  HL = rHL/rL
  cL = np.sqrt(gamma*pL/rL)
  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = rHL*unL

  # process right state
  rR = UR[0]
  uR = UR[1]/rR
  vR = UR[2]/rR
  unR = uR*n[0] + vR*n[1]
  qR = np.sqrt(UR[1]**2. + UR[2]**2.)/rR
  pR = (gamma-1)*(UR[3] - 0.5*rR*qR**2.)
  rHR = UR[3] + pR
  HR = rHR/rR
  cR = np.sqrt(gamma*pR/rR)
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = rHR*unR

  # difference in states
  du = UR - UL

  # Roe average
  di     = np.sqrt(rR/rL)
  d1     = 1.0/(1.0+di)

  ui     = (di*uR + uL)*d1
  vi     = (di*vR + vL)*d1
  Hi     = (di*HR + HL)*d1

  af     = 0.5*(ui*ui+vi*vi )
  ucp    = ui*n[0] + vi*n[1]
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
  G1    = gmi*(af*du[0] - ui*du[1] - vi*du[2] + du[3])
  G2    = -ucp*du[0]+du[1]*n[0]+du[2]*n[1]

  # required functions of G1 and G2 (again, see Theory guide)
  C1    = G1*(s1-l3)*ci1*ci1 + G2*s2*ci1
  C2    = G1*s2*ci1          + G2*(s1-l3)

  # flux assembly
  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*(l3*du[0] + C1   )
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*(l3*du[1] + C1*ui + C2*n[0])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*(l3*du[2] + C1*vi + C2*n[1])
  F[3]    = 0.5*(FL[3]+FR[3])-0.5*(l3*du[3] + C1*Hi + C2*ucp  )

  #% max wave speed
  smag = np.amax(l)
  return F
               
  

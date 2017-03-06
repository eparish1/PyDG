import numpy as np
from MPI_functions import sendEdgesGeneralSlab
from fluxSchemes import *
from scipy import weave
from scipy.weave import converters

def reconstructU(main,var):
  var.u[:] = 0.
  #var.u = np.einsum('lmpq,klmij->kpqij',main.w[:,None,:,None]*main.w[None,:,None,:],var.a) ## this is actually much slower than the two line code
  tmp =  np.einsum('rn,zpqrijk->zpqnijk',main.w,var.a)
  tmp = np.einsum('qm,zpqnijk->zpmnijk',main.w,tmp)
  var.u = np.einsum('pl,zpmnijk->zlmnijk',main.w,tmp)

def reconstructU2(main,var):
  var.u[:] = 0.
  var.u = np.einsum('pqrlmn,zpqrijk->zlmnijk',main.w[:,None,None,:,None,None]*main.w[None,:,None,None,:,None]*main.w[None,None,:,None,None,:],var.a) ## this is actually much slower than the two line code


#def reconstructU(w,u,a):
#  nvars,order,order,Nelx,Nely = np.shape(a)
#  u[:] = 0.
#  for l in range(0,order):
#    for m in range(0,order):
#      for n in range(0,order):
#        for z in range(0,nvars):
#          u[z,:,:,:,:] += w[l][:,None,None,None,None,None]*w[m][None,:,None,None,None,None]*w[n][None,None,:,None,None,None]*a[z,l,m,n,:,:,:]


def diffU(a,main):
  tmp =  np.einsum('rn,zpqrijk->zpqnijk',main.w,var.a) #reconstruct along third axis 
  tmp2 = np.einsum('qm,zpqnijk->zpmnijk',main.w,tmp) #reconstruct along second axis
  ux = np.einsum('pl,zpmnijk->zlmnijk',main.wp,tmp2) # get ux by differentiating along the first axis

  tmp2 = np.einsum('qm,zpqnijk->zpmnijk',main.wp,tmp) #diff tmp along second axis
  uy = np.einsum('pl,zpmnijk->zlmnijk',main.wp,tmp2) # get uy by reconstructing along the first axis

  tmp =  np.einsum('rn,zpqrijk->zpqnijk',main.wp,var.a) #diff along third axis 
  tmp = np.einsum('qm,zpqnijk->zpmnijk',main.w,tmp) #reconstruct along second axis
  uz = np.einsum('pl,zpmnijk->zlmnijk',main.w,tmp) # reconstruct along the first axis
  return ux,uy,uz


def diffCoeffs(a):
  atmp = np.zeros(np.shape(a))
  atmp[:] = a[:]
  nvars,order,order,order,Nelx,Nely,Nelz = np.shape(a)
  ax = np.zeros((nvars,order,order,order,Nelx,Nely,Nelz))
  ay = np.zeros((nvars,order,order,order,Nelx,Nely,Nelz))
  az = np.zeros((nvars,order,order,order,Nelx,Nely,Nelz))

  for j in range(order-1,2,-1):
    ax[:,j-1,:,:] = (2.*j-1)*atmp[:,j,:,:]
    atmp[:,j-2,:,:] = atmp[:,j-2,:,:] + atmp[:,j,:,:]

  if (order >= 3):
    ax[:,1,:,:] = 3.*atmp[:,2,:,:]
    ax[:,0,:,:] = atmp[:,1,:,:]
  if (order == 2):
    ax[:,1,:,:] = 0.
    ax[:,0,:,:] = atmp[:,1,:,:]
  if (order == 1):
    ax[:,0,:,:] = 0.

  atmp[:] = a[:]
  for j in range(order-1,2,-1):
    ay[:,:,j-1,:] = (2.*j-1)*atmp[:,:,j,:]
    atmp[:,:,j-2,:] = atmp[:,:,j-2,:] + atmp[:,:,j,:]

  if (order >= 3):
    ay[:,:,1,:] = 3.*atmp[:,:,2,:]
    ay[:,:,0,:] = atmp[:,:,1,:]
  if (order == 2):
    ay[:,:,1,:] = 0.
    ay[:,:,0,:] = atmp[:,:,1,:]
  if (order == 1):
    ay[:,:,0,:] = 0.


  atmp[:] = a[:]
  for j in range(order-1,2,-1):
    az[:,:,:,j-1] = (2.*j-1)*atmp[:,:,:,j]
    atmp[:,:,:,j-2] = atmp[:,:,:,j-2] + atmp[:,:,:,j]
  if (order >= 3):
    az[:,:,:,1] = 3.*atmp[:,:,:,2]
    az[:,:,:,0] = atmp[:,:,:,1]
  if (order == 2):
    az[:,:,:,1] = 0.
    az[:,:,:,0] = atmp[:,:,:,1]
  if (order == 1):
    az[:,:,:,0] = 0.


  return ax,ay,az

#def diffU2(a,main):
#  nvars = np.shape(a)[0]
#  order = np.shape(a)[1]
#  ux = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy))
#  uy = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy))
#
#  for l in range(0,order):
#    for m in range(0,order):
#      for k in range(0,nvars):
#        ux[k,:,:,:,:] += main.wp[l][:,None,None,None]*main.w[m][None,:,None,None]*a[k,l,m,:,:]
#        uy[k,:,:,:,:] += main.w[l][:,None,None,None]*main.wp[m][None,:,None,None]*a[k,l,m,:,:]
#  return ux,uy



def reconstructEdgesGeneral(a,main):
  nvars = np.shape(a)[0]
  aR = np.einsum('zpqrijk->zqrijk',a)
  aL = np.einsum('zpqrijk->zqrijk',a*main.altarray[None,:,None,None,None,None,None])

  aU = np.einsum('zpqrijk->zprijk',a)
  aD = np.einsum('zpqrijk->zprijk',a*main.altarray[None,None,:,None,None,None,None])

  aF = np.einsum('zpqrijk->zpqijk',a)
  aB = np.einsum('zpqrijk->zpqijk',a*main.altarray[None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqrijk->zqnijk',main.w,aR)
  uR  = np.einsum('qm,zqnijk->zmnijk',main.w,tmp)
  tmp = np.einsum('rn,zqrijk->zqnijk',main.w,aL)
  uL  = np.einsum('qm,zqnijk->zmnijk',main.w,tmp)

  tmp = np.einsum('rn,zprijk->zpnijk',main.w,aU)
  uU  = np.einsum('pl,zpnijk->zlnijk',main.w,tmp)
  tmp = np.einsum('rn,zprijk->zpnijk',main.w,aD)
  uD  = np.einsum('pl,zpnijk->zlnijk',main.w,tmp)

  tmp = np.einsum('qm,zpqijk->zpmijk',main.w,aF)
  uF  = np.einsum('pl,zpmijk->zlmijk',main.w,tmp)
  tmp = np.einsum('qm,zpqijk->zpmijk',main.w,aB)
  uB  = np.einsum('pl,zpmijk->zlmijk',main.w,tmp)

  return uR,uL,uU,uD,uF,uB



def volIntegrate(weights,f):
  return  np.einsum('zpqrijk->zijk',weights[None,:,None,None,None,None,None]*weights[None,None,:,None,None,None,None]*weights[None,None,None,:,None,None,None]*f[:,:,:,:,:])


def volIntegrate2(weights,f):
  return np.sum( weights[None,:,None,None,None,None,None]*weights[None,None,:,None,None,None,None]*weights[None,None,None,:,None,None,None]*f[:,:,:,:,:] ,axis=(1,2,3))
  #  return arr.sum((1,2,3))

def faceIntegrate(weights,f):
  return np.einsum('zpqijk->zijk',weights[None,:,None,None,None,None]*weights[None,None,:,None,None,None]*f)



def getFlux(main,eqns,schemes):
  # first reconstruct states
  main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB = reconstructEdgesGeneral(main.a.a,main)
  main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  inviscidFlux(main,eqns,schemes,main.iFlux,main.a)
  # now we need to integrate along the boundary 
  for i in range(0,main.order):
    for j in range(0,main.order):
      main.iFlux.fRI[:,i,j] = faceIntegrate(main.weights,main.w[i][None,:,None,None,None,None]*main.w[j][None,None,:,None,None,None]*main.iFlux.fRS)
      main.iFlux.fLI[:,i,j] = faceIntegrate(main.weights,main.w[i][None,:,None,None,None,None]*main.w[j][None,None,:,None,None,None]*main.iFlux.fLS)
      main.iFlux.fUI[:,i,j] = faceIntegrate(main.weights,main.w[i][None,:,None,None,None,None]*main.w[j][None,None,:,None,None,None]*main.iFlux.fUS)
      main.iFlux.fDI[:,i,j] = faceIntegrate(main.weights,main.w[i][None,:,None,None,None,None]*main.w[j][None,None,:,None,None,None]*main.iFlux.fDS)
      main.iFlux.fFI[:,i,j] = faceIntegrate(main.weights,main.w[i][None,:,None,None,None,None]*main.w[j][None,None,:,None,None,None]*main.iFlux.fFS)
      main.iFlux.fBI[:,i,j] = faceIntegrate(main.weights,main.w[i][None,:,None,None,None,None]*main.w[j][None,None,:,None,None,None]*main.iFlux.fBS)


def getRHS_INVISCID2(main,eqns,schemes):
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,eqns,schemes)
  ### Get interior vol terms
  eqns.evalFluxX(main.a.u,main.iFlux.fx)
  eqns.evalFluxY(main.a.u,main.iFlux.fy)
  eqns.evalFluxZ(main.a.u,main.iFlux.fz)
  # Now form RHS
  for i in range(0,main.order):
    for j in range(0,main.order):
      for k in range(0,main.order):
        main.RHS[:,i,j,k] = volIntegrate(main.weights,main.wp[i][None,:,None,None,None,None,None]*main.w[j][None,None,:,None,None,None,None]*main.w[k][None,None,None,:,None,None,None]*main.iFlux.fx)*2./main.dx
        main.RHS[:,i,j,k] +=volIntegrate(main.weights,main.w[i][None,:,None,None,None,None,None]*main.wp[j][None,None,:,None,None,None,None]*main.w[k][None,None,None,:,None,None,None]*main.iFlux.fy)*2./main.dx
        main.RHS[:,i,j,k] +=volIntegrate(main.weights,main.w[i][None,:,None,None,None,None,None]*main.w[j][None,None,:,None,None,None,None]*main.wp[k][None,None,None,:,None,None,None]*main.iFlux.fz)*2./main.dz 
        main.RHS[:,i,j,k] +=  (-main.iFlux.fRI[:,j,k] + main.iFlux.fLI[:,j,k]*main.altarray[i])*2./main.dx + \
                          (-main.iFlux.fUI[:,i,k] + main.iFlux.fDI[:,i,k]*main.altarray[j])*2./main.dy + \
                          (-main.iFlux.fFI[:,i,j] + main.iFlux.fBI[:,i,j]*main.altarray[k])*2./main.dz 
        main.RHS[:,i,j,k] = main.RHS[:,i,j,k]*(2.*i + 1.)*(2.*j + 1.)*(2.*k+1.)/8.


def getRHS_INVISCID(main,eqns,schemes):
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,eqns,schemes)
  ### Get interior vol terms
  eqns.evalFluxX(main.a.u,main.iFlux.fx)
  eqns.evalFluxY(main.a.u,main.iFlux.fy)
  eqns.evalFluxZ(main.a.u,main.iFlux.fz)
  # Now form RHS
  for i in range(0,main.order):
    for j in range(0,main.order):
      for k in range(0,main.order):
        main.RHS[:,i,j,k] = volIntegrate(main.weights,main.wp[i][None,:,None,None,None,None,None]*main.w[j][None,None,:,None,None,None,None]*main.w[k][None,None,None,:,None,None,None]*main.iFlux.fx)*2./main.dx + \
                          volIntegrate(main.weights,main.w[i][None,:,None,None,None,None,None]*main.wp[j][None,None,:,None,None,None,None]*main.w[k][None,None,None,:,None,None,None]*main.iFlux.fy)*2./main.dy + \
                          volIntegrate(main.weights,main.w[i][None,:,None,None,None,None,None]*main.w[j][None,None,:,None,None,None,None]*main.wp[k][None,None,None,:,None,None,None]*main.iFlux.fz)*2./main.dz + \
                          (-main.iFlux.fRI[:,j,k] + main.iFlux.fLI[:,j,k]*main.altarray[i])*2./main.dx + \
                          (-main.iFlux.fUI[:,i,k] + main.iFlux.fDI[:,i,k]*main.altarray[j])*2./main.dy + \
                          (-main.iFlux.fFI[:,i,j] + main.iFlux.fBI[:,i,j]*main.altarray[k])*2./main.dz 
        main.RHS[:,i,j,k] = main.RHS[:,i,j,k]*(2.*i + 1.)*(2.*j + 1.)*(2.*k+1.)/8.


def getViscousFlux(main,eqns,schemes):
  gamma = 1.4
  Pr = 0.72
  a = main.a.a
  nvars,order,order,order,Npx,Npy,Npz = np.shape(a)
  fvRG11 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvLG11 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvRG21 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvLG21 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvRG31 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvLG31 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))

  fvUG12 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvDG12 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvUG22 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvDG22 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvUG32 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvDG32 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))

  fvFG13 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvBG13 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvFG23 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvBG23 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvFG33 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvBG33 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))


  fvR2 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvL2 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvU2 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvD2 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))

  uhatR,uhatL,uhatU,uhatD,uhatF,uhatB = centralFluxGeneral(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
 
  G11R,G21R = eqns.getGsX(main.a.uR,main)
  G11L,G21L = eqns.getGsX(main.a.uL,main)
  G12U,G22U = eqns.getGsY(main.a.uU,main)
  G12D,G22D = eqns.getGsY(main.a.uD,main)
  G13U,G23U = eqns.getGsZ(main.a.uF,main)
  G13D,G23D = eqns.getGsZ(main.a.uB,main)

  for i in range(0,nvars):
    for j in range(0,nvars):
      fvRG11[i] += G11R[i,j]*(main.a.uR[j] - uhatR[j])
      fvLG11[i] += G11L[i,j]*(main.a.uL[j] - uhatL[j])
      fvRG21[i] += G21R[i,j]*(main.a.uR[j] - uhatR[j])
      fvLG21[i] += G21L[i,j]*(main.a.uL[j] - uhatL[j])
      fvRG31[i] += G31R[i,j]*(main.a.uR[j] - uhatR[j])
      fvLG31[i] += G31L[i,j]*(main.a.uL[j] - uhatL[j])


      fvUG12[i] += G12U[i,j]*(main.a.uU[j] - uhatU[j])
      fvDG12[i] += G12D[i,j]*(main.a.uD[j] - uhatD[j])
      fvUG22[i] += G22U[i,j]*(main.a.uU[j] - uhatU[j])
      fvDG22[i] += G22D[i,j]*(main.a.uD[j] - uhatD[j])
      fvUG32[i] += G32U[i,j]*(main.a.uU[j] - uhatU[j])
      fvDG32[i] += G32D[i,j]*(main.a.uD[j] - uhatD[j])

      fvFG13[i] += G13U[i,j]*(main.a.uF[j] - uhatF[j])
      fvBG13[i] += G13D[i,j]*(main.a.uB[j] - uhatB[j])
      fvFG23[i] += G23U[i,j]*(main.a.uF[j] - uhatF[j])
      fvBG23[i] += G23D[i,j]*(main.a.uB[j] - uhatB[j])
      fvFG33[i] += G33U[i,j]*(main.a.uF[j] - uhatF[j])
      fvBG33[i] += G33D[i,j]*(main.a.uB[j] - uhatB[j])


  apx,apy,apz = diffCoeffs(main.a.a)
  apx = apx*2./main.dx
  apy = apy*2./main.dy
  apy = apy*2./main.dz

  UxR,UxL,UxU,UxD,UxF,UxB = reconstructEdgesGeneral(apx,main)
  UyR,UyL,UyU,UyD,UyF,UyB = reconstructEdgesGeneral(apy,main)
  UzR,UzL,UzU,UzD,UzF,UzB = reconstructEdgesGeneral(apz,main)

  UxR_edge,UxL_edge,UxU_edge,UxD_edge,UxF_edge,UxB_edge = sendEdgesGeneralSlab(UxL,UxR,UxD,UxU,UxB,UxF,main)
  UyR_edge,UyL_edge,UyU_edge,UyD_edge,UyF_edge,UyB_edge = sendEdgesGeneralSlab(UyL,UyR,UyD,UyU,UyB,UyF,main)
  UzR_edge,UzL_edge,UzU_edge,UzD_edge,UzF_edge,UzB_edge = sendEdgesGeneralSlab(UzL,UzR,UzD,UzU,UzB,UzF,main)

  fvxR = eqns.evalViscousFluxX(main,main.a.uR,UxR,UyR,UzR)
  fvxL = eqns.evalViscousFluxX(main,main.a.uL,UxL,UyL,UzL)
  fvxR_edge = eqns.evalViscousFluxX(main,main.a.uR_edge,UxR_edge,UyR_edge,UzR_edge)
  fvxL_edge = eqns.evalViscousFluxX(main,main.a.uL_edge,UxL_edge,UyL_edge,UyL_edge)

  fvyU = eqns.evalViscousFluxY(main,main.a.uU,UxU,UyU,UzU)
  fvyD = eqns.evalViscousFluxY(main,main.a.uD,UxD,UyD,UzD)
  fvyU_edge = eqns.evalViscousFluxY(main,main.a.uU_edge,UxU_edge,UyU_edge,UzU_edge)
  fvyD_edge = eqns.evalViscousFluxY(main,main.a.uD_edge,UxD_edge,UyD_edge,UzD_edge)

  fvzF = eqns.evalViscousFluxZ(main,main.a.uF,UxF,UyF,UzF)
  fvzB = eqns.evalViscousFluxZ(main,main.a.uB,UxB,UyB,UzB)
  fvzF_edge = eqns.evalViscousFluxZ(main,main.a.uF_edge,UxF_edge,UyF_edge,UzF_edge)
  fvzB_edge = eqns.evalViscousFluxZ(main,main.a.uB_edge,UxB_edge,UyB_edge,UzB_edge)

  shatR,shatL,shatU,shatD,shatF,shatB = centralFluxGeneral(fvxR,fvxL,fvyU,fvyD,fvzF,fvzB,fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge)
  jumpR,jumpL,jumpU,jumpD,jumpF,jumpB = computeJump(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge)
  fvR2[:] = shatR[:] - 2.*main.mu*jumpR[:]*3**2/main.dx
  fvL2[:] = shatL[:] - 2.*main.mu*jumpL[:]*3**2/main.dx
  fvU2[:] = shatU[:] - 2.*main.mu*jumpU[:]*3**2/main.dy
  fvD2[:] = shatD[:] - 2.*main.mu*jumpD[:]*3**2/main.dy
  fvF2[:] = shatF[:] - 2.*main.mu*jumpF[:]*3**2/main.dy
  fvB2[:] = shatB[:] - 2.*main.mu*jumpB[:]*3**2/main.dy

 # now we need to integrate along the boundary 
  fvRIG11 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvLIG11 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))

  fvRIG21 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvLIG21 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))

  fvUIG12 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvDIG12 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))

  fvUIG22 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvDIG22 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))

  fvR2I = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvL2I = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvU2I = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvD2I = np.zeros((main.nvars,main.order,main.Npx,main.Npy))


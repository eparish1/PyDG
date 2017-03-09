import numpy as np
from MPI_functions import sendEdgesGeneralSlab
from fluxSchemes import *
from scipy import weave
from scipy.weave import converters
import time
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
  tmp =  np.einsum('rn,zpqrijk->zpqnijk',main.w,a) #reconstruct along third axis 
  tmp2 = np.einsum('qm,zpqnijk->zpmnijk',main.w,tmp) #reconstruct along second axis
  ux = np.einsum('pl,zpmnijk->zlmnijk',main.wp,tmp2) # get ux by differentiating along the first axis

  tmp2 = np.einsum('qm,zpqnijk->zpmnijk',main.wp,tmp) #diff tmp along second axis
  uy = np.einsum('pl,zpmnijk->zlmnijk',main.wp,tmp2) # get uy by reconstructing along the first axis

  tmp =  np.einsum('rn,zpqrijk->zpqnijk',main.wp,a) #diff along third axis 
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


## All about the same speed
def volIntegrate2(weights,f):
  return np.sum( weights[None,:,None,None,None,None,None]*weights[None,None,:,None,None,None,None]*weights[None,None,None,:,None,None,None]*f[:,:,:,:,:] ,axis=(1,2,3))

def volIntegrate3(weights,f):
  tmp = weights[:,None,None]*weights[None,:,None]*weights[None,None,:]
  tmp1 = np.sum(tmp[None,:,:,:,None,None,None]*f[:,:,:,:,:] ,axis=1)
  tmp2 = np.sum(tmp1,axis=1)
  return np.sum(tmp2,axis=1)


  #  return arr.sum((1,2,3))

def faceIntegrate(weights,f):
  return np.einsum('zpqijk->zijk',weights[None,:,None,None,None,None]*weights[None,None,:,None,None,None]*f)




def getFlux(main,eqns,schemes):
  # first reconstruct states
  main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB = reconstructEdgesGeneral(main.a.a,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
#  print(np.shape(main.a.uD_edge))
#  print(main.mpi_rank,np.linalg.norm(main.a.uD_edge))
  inviscidFlux(main,eqns,schemes,main.iFlux,main.a)
  # now we need to integrate along the boundary 
  for i in range(0,main.order):
    fRI_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*main.iFlux.fRS)
    fLI_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*main.iFlux.fLS)
    fUI_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*main.iFlux.fUS)
    fDI_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*main.iFlux.fDS)
    fFI_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*main.iFlux.fFS)
    fBI_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*main.iFlux.fBS)
    for j in range(0,main.order):
      main.iFlux.fRI[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fRI_i)
      main.iFlux.fLI[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fLI_i)
      main.iFlux.fUI[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fUI_i)
      main.iFlux.fDI[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fDI_i)
      main.iFlux.fFI[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fFI_i)
      main.iFlux.fBI[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fBI_i)


def getFlux2(main,eqns,schemes):
  # first reconstruct states
  main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB = reconstructEdgesGeneral(main.a.a,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
#  print(np.shape(main.a.uD_edge))
#  print(main.mpi_rank,np.linalg.norm(main.a.uD_edge))
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


def getRHS_IP(main,eqns,schemes):
  t0 = time.time()
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,eqns,schemes)
  ### Get interior vol terms
  eqns.evalFluxX(main.a.u,main.iFlux.fx)
  eqns.evalFluxY(main.a.u,main.iFlux.fy)
  eqns.evalFluxZ(main.a.u,main.iFlux.fz)

  # now get viscous flux
  fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I = getViscousFlux(main,eqns,schemes)
  upx,upy,upz = diffU(main.a.a,main)
  upx = upx*2./main.dx
  upy = upy*2./main.dy
  upz = upz*2./main.dz
  G11,G12,G13,G21,G22,G23,G31,G32,G33 = eqns.getGs(main.a.u,main)
  fvGX = np.einsum('ij...,j...->i...',G11,upx)
  fvGX += np.einsum('ij...,j...->i...',G12,upy)
  fvGX += np.einsum('ij...,j...->i...',G13,upz)
  fvGY = np.einsum('ij...,j...->i...',G21,upx)
  fvGY += np.einsum('ij...,j...->i...',G22,upy)
  fvGY += np.einsum('ij...,j...->i...',G23,upz)
  fvGZ = np.einsum('ij...,j...->i...',G31,upx)
  fvGZ += np.einsum('ij...,j...->i...',G32,upy)
  fvGZ += np.einsum('ij...,j...->i...',G33,upz)
  # Now form RHS
  for i in range(0,main.order):
    ## This is important. Do partial integrations in each direction to avoid doing for each ijk
    v1i = np.einsum('zpqrijk->zqrijk',main.weights[None,:,None,None,None,None,None]*main.wp[i][None,:,None,None,None,None,None]*main.iFlux.fx)
    v2i = np.einsum('zpqrijk->zqrijk',main.weights[None,:,None,None,None,None,None]*main.w[i][None,:,None,None,None,None,None]*main.iFlux.fy)
    v3i = np.einsum('zpqrijk->zqrijk',main.weights[None,:,None,None,None,None,None]*main.w[i][None,:,None,None,None,None,None]*main.iFlux.fz)
    for j in range(0,main.order):
      v1ij = np.einsum('zqrijk->zrijk',main.weights[None,:,None,None,None,None]*main.w[j][None,:,None,None,None,None]*v1i)
      v2ij = np.einsum('zqrijk->zrijk',main.weights[None,:,None,None,None,None]*main.wp[j][None,:,None,None,None,None]*v2i)
      v3ij = np.einsum('zqrijk->zrijk',main.weights[None,:,None,None,None,None]*main.w[j][None,:,None,None,None,None]*v3i)
      for k in range(0,main.order):
        scale =  (2.*i + 1.)*(2.*j + 1.)*(2.*k+1.)/8.
        dxi = 2./main.dx*scale
        dyi = 2./main.dy*scale
        dzi = 2./main.dz*scale
        v1ijk = np.einsum('zrijk->zijk',main.weights[None,:,None,None,None]*main.w[k][None,:,None,None,None]*v1ij)*dxi
        v2ijk = np.einsum('zrijk->zijk',main.weights[None,:,None,None,None]*main.w[k][None,:,None,None,None]*v2ij)*dyi
        v3ijk = np.einsum('zrijk->zijk',main.weights[None,:,None,None,None]*main.wp[k][None,:,None,None,None]*v3ij)*dzi
        #tmp = volIntegrate(main.weights,main.wp[i][None,:,None,None,None,None,None]*main.w[j][None,None,:,None,None,None,None]*main.w[k][None,None,None,:,None,None,None]*main.iFlux.fx)*dxi 
        #tmp +=  volIntegrate(main.weights,main.w[i][None,:,None,None,None,None,None]*main.wp[j][None,None,:,None,None,None,None]*main.w[k][None,None,None,:,None,None,None]*main.iFlux.fy)*dyi
        #tmp +=  volIntegrate(main.weights,main.w[i][None,:,None,None,None,None,None]*main.w[j][None,None,:,None,None,None,None]*main.wp[k][None,None,None,:,None,None,None]*main.iFlux.fz)*dzi
        tmp = v1ijk + v2ijk + v3ijk
        tmp +=  (-main.iFlux.fRI[:,j,k] + main.iFlux.fLI[:,j,k]*main.altarray[i])*dxi
        tmp +=  (-main.iFlux.fUI[:,i,k] + main.iFlux.fDI[:,i,k]*main.altarray[j])*dyi
        tmp +=  (-main.iFlux.fFI[:,i,j] + main.iFlux.fBI[:,i,j]*main.altarray[k])*dzi 
        tmp +=  (fvRIG11[:,j,k]*main.wpedge[i,1] + fvRIG21[:,j,k] + fvRIG31[:,j,k]  - (fvLIG11[:,j,k]*main.wpedge[i,0] + fvLIG21[:,j,k]*main.altarray[i] + fvLIG31[:,j,k]*main.altarray[i]) )*dxi
        tmp +=  (fvUIG12[:,i,k] + fvUIG22[:,i,k]*main.wpedge[j,1] + fvUIG32[:,i,k]  - (fvDIG12[:,i,k]*main.altarray[j] + fvDIG22[:,i,k]*main.wpedge[j,0] + fvDIG32[:,i,k]*main.altarray[j]) )*dyi
        tmp +=  (fvFIG13[:,i,j] + fvFIG23[:,i,j] + fvFIG33[:,i,j]*main.wpedge[k,1]  - (fvBIG13[:,i,j]*main.altarray[k] + fvBIG23[:,i,j]*main.altarray[k] + fvBIG33[:,i,j]*main.wpedge[k,0]) )*dzi 
        tmp +=  (fvR2I[:,j,k] - fvL2I[:,j,k]*main.altarray[i])*dxi + (fvU2I[:,i,k] - fvD2I[:,i,k]*main.altarray[j])*dyi + (fvF2I[:,i,j] - fvB2I[:,i,j]*main.altarray[k])*dzi
        main.RHS[:,i,j,k] = tmp[:]
  
def getRHS_IP2(main,eqns,schemes):
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,eqns,schemes)
  ### Get interior vol terms
  eqns.evalFluxX(main.a.u,main.iFlux.fx)
  eqns.evalFluxY(main.a.u,main.iFlux.fy)
  eqns.evalFluxZ(main.a.u,main.iFlux.fz)

  # now get viscous flux
  fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I = getViscousFlux(main,eqns,schemes)
  upx,upy,upz = diffU(main.a.a,main)
  upx = upx*2./main.dx
  upy = upy*2./main.dy
  upz = upz*2./main.dz
  fvGX = np.zeros(np.shape(main.a.u))
  fvGY = np.zeros(np.shape(main.a.u))
  fvGZ = np.zeros(np.shape(main.a.u))
  G11,G12,G13,G21,G22,G23,G31,G32,G33 = eqns.getGs(main.a.u,main)
  for i in range(0,eqns.nvars):
    for j in range(0,eqns.nvars):
      fvGX[i] = fvGX[i] + G11[i,j]*upx[j] + G12[i,j]*upy[j] + G13[i,j]*upz[j]
      fvGY[i] = fvGY[i] + G21[i,j]*upx[j] + G22[i,j]*upy[j] + G32[i,j]*upz[j]
      fvGZ[i] = fvGZ[i] + G31[i,j]*upx[j] + G32[i,j]*upy[j] + G33[i,j]*upz[j]
  # Now form RHS
  for i in range(0,main.order):
    for j in range(0,main.order):
      for k in range(0,main.order):
        main.RHS[:,i,j,k] = volIntegrate(main.weights,main.wp[i][None,:,None,None,None,None,None]*main.w[j][None,None,:,None,None,None,None]*main.w[k][None,None,None,:,None,None,None]*main.iFlux.fx)*2./main.dx + \
                          volIntegrate(main.weights,main.w[i][None,:,None,None,None,None,None]*main.wp[j][None,None,:,None,None,None,None]*main.w[k][None,None,None,:,None,None,None]*main.iFlux.fy)*2./main.dy + \
                          volIntegrate(main.weights,main.w[i][None,:,None,None,None,None,None]*main.w[j][None,None,:,None,None,None,None]*main.wp[k][None,None,None,:,None,None,None]*main.iFlux.fz)*2./main.dz + \
                          (-main.iFlux.fRI[:,j,k] + main.iFlux.fLI[:,j,k]*main.altarray[i])*2./main.dx + \
                          (-main.iFlux.fUI[:,i,k] + main.iFlux.fDI[:,i,k]*main.altarray[j])*2./main.dy + \
                          (-main.iFlux.fFI[:,i,j] + main.iFlux.fBI[:,i,j]*main.altarray[k])*2./main.dz + \
                          (fvRIG11[:,j,k]*main.wpedge[i,1] + fvRIG21[:,j,k] + fvRIG31[:,j,k]  - (fvLIG11[:,j,k]*main.wpedge[i,0] + fvLIG21[:,j,k]*main.altarray[i] + fvLIG31[:,j,k]*main.altarray[i]) )*2./main.dx + \
                          (fvUIG12[:,i,k] + fvUIG22[:,i,k]*main.wpedge[j,1] + fvUIG32[:,i,k]  - (fvDIG12[:,i,k]*main.altarray[j] + fvDIG22[:,i,k]*main.wpedge[j,0] + fvDIG32[:,i,k]*main.altarray[j]) )*2./main.dy + \
                          (fvFIG13[:,i,j] + fvFIG23[:,i,j] + fvFIG33[:,i,j]*main.wpedge[k,1]  - (fvBIG13[:,i,j]*main.altarray[k] + fvBIG23[:,i,j]*main.wpedge[k,0] + fvBIG33[:,i,j]*main.altarray[k]) )*2./main.dz + \
                        + (fvR2I[:,j,k] - fvL2I[:,j,k]*main.altarray[i])*2./main.dx + (fvU2I[:,i,k] - fvD2I[:,i,k]*main.altarray[j])*2./main.dy + (fvF2I[:,i,j] - fvB2I[:,i,j]*main.altarray[k])*2./main.dz
 
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

def getViscousFlux2(main,eqns,schemes):
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
  fvF2 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  fvB2 = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))

  uhatR,uhatL,uhatU,uhatD,uhatF,uhatB = centralFluxGeneral(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
 
  G11R,G21R,G31R = eqns.getGsX(main.a.uR,main)
  G11L,G21L,G31L = eqns.getGsX(main.a.uL,main)
  G12U,G22U,G32U = eqns.getGsY(main.a.uU,main)
  G12D,G22D,G32D = eqns.getGsY(main.a.uD,main)
  G13F,G23F,G33F = eqns.getGsZ(main.a.uF,main)
  G13B,G23B,G33B = eqns.getGsZ(main.a.uB,main)

 # test = np.einsum('ij...,j...->i...',G11R,main.a.uR - uhatR)
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

      fvFG13[i] += G13F[i,j]*(main.a.uF[j] - uhatF[j])
      fvBG13[i] += G13B[i,j]*(main.a.uB[j] - uhatB[j])
      fvFG23[i] += G23F[i,j]*(main.a.uF[j] - uhatF[j])
      fvBG23[i] += G23B[i,j]*(main.a.uB[j] - uhatB[j])
      fvFG33[i] += G33F[i,j]*(main.a.uF[j] - uhatF[j])
      fvBG33[i] += G33B[i,j]*(main.a.uB[j] - uhatB[j])

  
  apx,apy,apz = diffCoeffs(main.a.a)
  apx = apx*2./main.dx
  apy = apy*2./main.dy
  apz = apz*2./main.dz

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
  jumpR,jumpL,jumpU,jumpD,jumpF,jumpB = computeJump(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
  fvR2[:] = shatR[:] - 6.*main.mu*jumpR[:]*3**2/main.dx
  fvL2[:] = shatL[:] - 6.*main.mu*jumpL[:]*3**2/main.dx
  fvU2[:] = shatU[:] - 6.*main.mu*jumpU[:]*3**2/main.dy
  fvD2[:] = shatD[:] - 6.*main.mu*jumpD[:]*3**2/main.dy
  fvF2[:] = shatF[:] - 6.*main.mu*jumpF[:]*3**2/main.dz
  fvB2[:] = shatB[:] - 6.*main.mu*jumpB[:]*3**2/main.dz

 # now we need to integrate along the boundary 
  fvRIG11 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvLIG11 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvRIG21 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvLIG21 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvRIG31 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvLIG31 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))


  fvUIG12 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvDIG12 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvUIG22 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvDIG22 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvUIG32 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvDIG32 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))

  fvFIG13 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvBIG13 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvFIG23 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvBIG23 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvFIG33 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvBIG33 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))

  fvR2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvL2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvU2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvD2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvF2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvB2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))

  for i in range(0,main.order): 
    for j in range(0,main.order):
      fvRIG11[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvRG11)
      fvLIG11[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvLG11)
      fvRIG21[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvRG21)
      fvLIG21[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvLG21)
      fvRIG31[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvRG31)
      fvLIG31[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvLG31)


      fvUIG12[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvUG12)
      fvDIG12[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvDG12)
      fvUIG22[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvUG22)
      fvDIG22[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvDG22)
      fvUIG32[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvUG32)
      fvDIG32[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvDG32)

      fvFIG13[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvFG13)
      fvBIG13[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvBG13)
      fvFIG23[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvFG23)
      fvBIG23[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvBG23)
      fvFIG33[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvFG33)
      fvBIG33[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvBG33)


      fvR2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvR2)
      fvL2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvL2)
      fvU2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvU2)
      fvD2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvD2)
      fvF2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvF2)
      fvB2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvB2)

  return fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I

def computeJump(uR,uL,uU,uD,uF,uB,uR_edge,uL_edge,uU_edge,uD_edge,uF_edge,uB_edge):
  nvars,order,order,Npx,Npy,Npz = np.shape(uR)
  jumpR = np.zeros((nvars,order,order,Npx,Npy,Npz))
  jumpL = np.zeros((nvars,order,order,Npx,Npy,Npz))
  jumpU = np.zeros((nvars,order,order,Npx,Npy,Npz))
  jumpD = np.zeros((nvars,order,order,Npx,Npy,Npz))
  jumpF = np.zeros((nvars,order,order,Npx,Npy,Npz))
  jumpB = np.zeros((nvars,order,order,Npx,Npy,Npz))

  jumpR[:,:,:,0:-1,:,:] = uR[:,:,:,0:-1,:,:] - uL[:,:,:,1::,:,:]
  jumpR[:,:,:,-1   ,:,:] = uR[:,:,:,  -1,:,:] - uR_edge
  jumpL[:,:,:,1:: ,:,:] = jumpR[:,:,:,0:-1,:,:]
  jumpL[:,:,:,0   ,:,:] = uL_edge - uL[:,:,:,  0,:,:]
  jumpU[:,:,:,:,0:-1,:] = uU[:,:,:,:,0:-1,:] - uD[:,:,:,:,1::,:]
  jumpU[:,:,:,:,  -1,:] = uU[:,:,:,:,  -1,:] - uU_edge
  jumpD[:,:,:,:,1:: ,:] = jumpU[:,:,:,:,0:-1,:]
  jumpD[:,:,:,:,0   ,:] = uD_edge - uD[:,:,:,:,   0,:]
  jumpF[:,:,:,:,:,0:-1] = uF[:,:,:,:,:,0:-1] - uB[:,:,:,:,:,1::]
  jumpF[:,:,:,:,:,  -1] = uF[:,:,:,:,:,  -1] - uF_edge
  jumpB[:,:,:,:,:,1:: ] = jumpF[:,:,:,:,:,0:-1]
  jumpB[:,:,:,:,:,0   ] = uB_edge - uB[:,:,:,:,:,   0]

  return jumpR,jumpL,jumpU,jumpD,jumpF,jumpB



def getViscousFlux(main,eqns,schemes):
  gamma = 1.4
  Pr = 0.72
  a = main.a.a
  nvars,order,order,order,Npx,Npy,Npz = np.shape(a)


  uhatR,uhatL,uhatU,uhatD,uhatF,uhatB = centralFluxGeneral(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
 
  G11R,G21R,G31R = eqns.getGsX(main.a.uR,main)
  G11L,G21L,G31L = eqns.getGsX(main.a.uL,main)
  G12U,G22U,G32U = eqns.getGsY(main.a.uU,main)
  G12D,G22D,G32D = eqns.getGsY(main.a.uD,main)
  G13F,G23F,G33F = eqns.getGsZ(main.a.uF,main)
  G13B,G23B,G33B = eqns.getGsZ(main.a.uB,main)

  fvRG11 = np.einsum('ij...,j...->i...',G11R,main.a.uR - uhatR)
  fvLG11 = np.einsum('ij...,j...->i...',G11L,main.a.uL - uhatL)
  fvRG21 = np.einsum('ij...,j...->i...',G21R,main.a.uR - uhatR)
  fvLG21 = np.einsum('ij...,j...->i...',G21L,main.a.uL - uhatL)
  fvRG31 = np.einsum('ij...,j...->i...',G31R,main.a.uR - uhatR)
  fvLG31 = np.einsum('ij...,j...->i...',G31L,main.a.uL - uhatL)

  fvUG12 = np.einsum('ij...,j...->i...',G12U,main.a.uU - uhatU)
  fvDG12 = np.einsum('ij...,j...->i...',G12D,main.a.uD - uhatD)
  fvUG22 = np.einsum('ij...,j...->i...',G22U,main.a.uU - uhatU)
  fvDG22 = np.einsum('ij...,j...->i...',G22D,main.a.uD - uhatD)
  fvUG32 = np.einsum('ij...,j...->i...',G32U,main.a.uU - uhatU)
  fvDG32 = np.einsum('ij...,j...->i...',G32D,main.a.uD - uhatD)

  fvFG13 = np.einsum('ij...,j...->i...',G13F,main.a.uF - uhatF)
  fvBG13 = np.einsum('ij...,j...->i...',G13B,main.a.uB - uhatB)
  fvFG23 = np.einsum('ij...,j...->i...',G23F,main.a.uF - uhatF)
  fvBG23 = np.einsum('ij...,j...->i...',G23B,main.a.uB - uhatB)
  fvFG33 = np.einsum('ij...,j...->i...',G33F,main.a.uF - uhatF)
  fvBG33 = np.einsum('ij...,j...->i...',G33B,main.a.uB - uhatB)

  
  apx,apy,apz = diffCoeffs(main.a.a)
  apx *= 2./main.dx
  apy *= 2./main.dy
  apz *= 2./main.dz

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
  jumpR,jumpL,jumpU,jumpD,jumpF,jumpB = computeJump(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
  fvR2 = shatR - 6.*main.mu*jumpR*3**2/main.dx
  fvL2 = shatL - 6.*main.mu*jumpL*3**2/main.dx
  fvU2 = shatU - 6.*main.mu*jumpU*3**2/main.dy
  fvD2 = shatD - 6.*main.mu*jumpD*3**2/main.dy
  fvF2 = shatF - 6.*main.mu*jumpF*3**2/main.dz
  fvB2 = shatB - 6.*main.mu*jumpB*3**2/main.dz

 # now we need to integrate along the boundary 
  fvRIG11 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvLIG11 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvRIG21 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvLIG21 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvRIG31 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvLIG31 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))


  fvUIG12 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvDIG12 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvUIG22 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvDIG22 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvUIG32 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvDIG32 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))

  fvFIG13 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvBIG13 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvFIG23 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvBIG23 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvFIG33 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvBIG33 = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))

  fvR2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvL2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvU2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvD2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvF2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))
  fvB2I   = np.zeros((main.nvars,main.order,main.order,main.Npx,main.Npy,main.Npz))

  for i in range(0,main.order):
    fvRIG11_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvRG11)
    fvLIG11_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvLG11)
    fvRIG21_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.wp[i][None,:,None,None,None,None]*fvRG21)
    fvLIG21_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.wp[i][None,:,None,None,None,None]*fvLG21)
    fvRIG31_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvRG31)
    fvLIG31_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvLG31)

    fvUIG12_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.wp[i][None,:,None,None,None,None]*fvUG12)
    fvDIG12_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.wp[i][None,:,None,None,None,None]*fvDG12)
    fvUIG22_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvUG22)
    fvDIG22_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvDG22)
    fvUIG32_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvUG32)
    fvDIG32_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvDG32)

    fvFIG13_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.wp[i][None,:,None,None,None,None]*fvFG13)
    fvBIG13_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.wp[i][None,:,None,None,None,None]*fvBG13)
    fvFIG23_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvFG23)
    fvBIG23_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvBG23)
    fvFIG33_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvFG33)
    fvBIG33_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvBG33)

    fvR2I_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvR2)
    fvL2I_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvL2)
    fvU2I_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvU2)
    fvD2I_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvD2)
    fvF2I_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvF2)
    fvB2I_i = np.einsum('zpqijk->zqijk',main.weights[None,:,None,None,None,None]*main.w[i][None,:,None,None,None,None]*fvB2)

    for j in range(0,main.order):
      fvRIG11[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvRIG11_i)
      fvLIG11[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvLIG11_i)
      fvRIG21[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvRIG21_i)
      fvLIG21[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvLIG21_i)
      fvRIG31[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.wp[j][None,:,None,None,None]*fvRIG31_i)
      fvLIG31[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.wp[j][None,:,None,None,None]*fvLIG31_i)

      fvUIG12[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvUIG12_i)
      fvDIG12[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvDIG12_i)
      fvUIG22[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvUIG22_i)
      fvDIG22[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvDIG22_i)
      fvUIG32[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.wp[j][None,:,None,None,None]*fvUIG32_i)
      fvDIG32[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.wp[j][None,:,None,None,None]*fvDIG32_i)

      fvFIG13[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvFIG13_i)
      fvBIG13[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvBIG13_i)
      fvFIG23[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.wp[j][None,:,None,None,None]*fvFIG23_i)
      fvBIG23[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.wp[j][None,:,None,None,None]*fvBIG23_i)
      fvFIG33[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvFIG33_i)
      fvBIG33[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvBIG33_i)

      fvR2I[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvR2I_i)
      fvL2I[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvL2I_i)
      fvU2I[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvU2I_i)
      fvD2I[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvB2I_i)
      fvF2I[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvF2I_i)
      fvB2I[:,i,j] = np.einsum('zqijk->zijk',main.weights[None,:,None,None,None]*main.w[j][None,:,None,None,None]*fvD2I_i)

#      fvRIG11[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvRG11)
#      fvLIG11[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvLG11)
#      fvRIG21[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvRG21)
#      fvLIG21[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvLG21)
#      fvRIG31[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvRG31)
#      fvLIG31[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvLG31)


#      fvUIG12[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvUG12)
#      fvDIG12[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvDG12)
#      fvUIG22[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvUG22)
#      fvDIG22[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvDG22)
#      fvUIG32[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvUG32)
#      fvDIG32[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvDG32)

 #     fvFIG13[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvFG13)
 #     fvBIG13[:,i,j] = faceIntegrate(main.weights,(main.wp[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvBG13)
 #     fvFIG23[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvFG23)
 #     fvBIG23[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.wp[j][None,:])[:,:,None,None,None]*fvBG23)
 #     fvFIG33[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvFG33)
 #     fvBIG33[:,i,j] = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvBG33)

#      fvR2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvR2)
#      fvL2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvL2)
#      fvU2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvU2)
#      fvD2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvD2)
#      fvF2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvF2)
#      fvB2I[:,i,j]   = faceIntegrate(main.weights,(main.w[i][:,None]*main.w[j][None,:])[:,:,None,None,None]*fvB2)

  return fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I



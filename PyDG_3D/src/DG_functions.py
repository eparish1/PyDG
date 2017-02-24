import numpy as np
from MPI_functions import sendEdges,sendEdgesGeneral,sendEdgesSlab,sendEdgesGeneralSlab
from fluxSchemes import *
from scipy import weave
from scipy.weave import converters

def reconstructU(main,var):
  var.u[:] = 0.
  #var.u = np.einsum('lmpq,klmij->kpqij',main.w[:,None,:,None]*main.w[None,:,None,:],var.a) ## this is actually much slower than the two line code
  tmp =  np.einsum('rn,zpqrijk->zpqnijk',main.w,var.a)
  tmp = np.einsum('qm,zpqnijk->zpmnijk',main.w,tmp)
  var.u = np.einsum('pl,zpmnijk->zlmnijk',main.w,tmp)

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


#def reconstructEdges2(main,var):
#  var.aU[:] =  np.sum(var.a,axis=(2))
#  var.aD[:] =  np.sum(var.a*main.altarray[None,None,:,None,None],axis=(2))
#  var.aR[:] =  np.sum(var.a,axis=(1))
#  var.aL[:] =  np.sum(var.a*main.altarray[None,:,None,None,None],axis=(1))
#
#  var.uU[:] = 0. 
#  var.uD[:] = 0.
#  var.uL[:] = 0. 
#  var.uR[:] = 0.
#  for m in range(0,var.order):
#    #for k in range(0,var.nvars):
#       var.uU[:,:,:,:] += main.w[None,m,:,None,None]*var.aU[:,m,:,:]
#       var.uD[:,:,:,:] += main.w[None,m,:,None,None]*var.aD[:,m,:,:]
#       var.uR[:,:,:,:] += main.w[None,m,:,None,None]*var.aR[:,m,:,:]
#       var.uL[:,:,:,:] += main.w[None,m,:,None,None]*var.aL[:,m,:,:]



def volIntegrate(weights,f):
  return  np.einsum('zpqrijk->zijk',weights[None,:,None,None,None,None,None]*weights[None,None,:,None,None,None,None]*weights[None,None,None,:,None,None,None]*f[:,:,:,:,:])


def faceIntegrate(weights,f):
  return np.einsum('zpqijk->zijk',weights[None,:,None,None,None,None]*weights[None,None,:,None,None,None]*f)






def getFlux(main,eqns,schemes):
  # first reconstruct states
  reconstructEdges(main,main.a)
  sendEdgesSlab(main,main.a)
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



import numpy as np
from MPI_functions import sendEdgesGeneralSlab,sendaEdgesGeneralSlab
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

def reconstructUGeneral(main,a):
  tmp =  np.einsum('rn,zpqrijk->zpqnijk',main.w,a)
  tmp = np.einsum('qm,zpqnijk->zpmnijk',main.w,tmp)
  return np.einsum('pl,zpmnijk->zlmnijk',main.w,tmp)


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
  uy = np.einsum('pl,zpmnijk->zlmnijk',main.w,tmp2) # get uy by reconstructing along the first axis

  tmp =  np.einsum('rn,zpqrijk->zpqnijk',main.wp,a) #diff along third axis 
  tmp = np.einsum('qm,zpqnijk->zpmnijk',main.w,tmp) #reconstruct along second axis
  uz = np.einsum('pl,zpmnijk->zlmnijk',main.w,tmp) # reconstruct along the first axis
  return ux,uy,uz

def diffUXEdge_edge(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge*main.wpedge[:,1][None,:,None,None,None,None])
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.wpedge[:,0][None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray[None,None,None,:,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face

  tmp = np.einsum('rn,zqr...->zqn...',main.w,aR)  #reconstruct in y and z
  uxR  = np.einsum('qm,zqn...->zmn...',main.w,tmp)*2./main.dx 
  tmp = np.einsum('rn,zqr...->zqn...',main.w,aL)
  uxL  = np.einsum('qm,zqn...->zmn...',main.w,tmp)*2./main.dx

  tmp = np.einsum('rn,zpr...->zpn...',main.w,aU) #reconstruct in x and z 
  uxU  = np.einsum('pl,zpn...->zln...',main.wp,tmp)*2./main.dx
  tmp = np.einsum('rn,zpr...->zpn...',main.w,aD)
  uxD  = np.einsum('pl,zpn...->zln...',main.wp,tmp)*2./main.dx

  tmp = np.einsum('qm,zpq...->zpm...',main.w,aF) #reconstruct in x and y
  uxF  = np.einsum('pl,zpm...->zlm...',main.wp,tmp)*2./main.dx
  tmp = np.einsum('qm,zpq...->zpm...',main.w,aB)
  uxB  = np.einsum('pl,zpm...->zlm...',main.wp,tmp)*2./main.dx
  return uxR,uxL,uxU,uxD,uxF,uxB


def diffUX_edge(a,main):
  aR = np.einsum('zpqrijk->zqrijk',a*main.wpedge[:,1][None,:,None,None,None,None,None])
  aL = np.einsum('zpqrijk->zqrijk',a*main.wpedge[:,0] [None,:,None,None,None,None,None])

  aU = np.einsum('zpqrijk->zprijk',a)
  aD = np.einsum('zpqrijk->zprijk',a*main.altarray[None,None,:,None,None,None,None])

  aF = np.einsum('zpqrijk->zpqijk',a)
  aB = np.einsum('zpqrijk->zpqijk',a*main.altarray[None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqrijk->zqnijk',main.w,aR)  #reconstruct in y and z
  uxR  = np.einsum('qm,zqnijk->zmnijk',main.w,tmp)*2./main.dx 
  tmp = np.einsum('rn,zqrijk->zqnijk',main.w,aL)
  uxL  = np.einsum('qm,zqnijk->zmnijk',main.w,tmp)*2./main.dx

  tmp = np.einsum('rn,zprijk->zpnijk',main.w,aU) #reconstruct in x and z 
  uxU  = np.einsum('pl,zpnijk->zlnijk',main.wp,tmp)*2./main.dx
  tmp = np.einsum('rn,zprijk->zpnijk',main.w,aD)
  uxD  = np.einsum('pl,zpnijk->zlnijk',main.wp,tmp)*2./main.dx

  tmp = np.einsum('qm,zpqijk->zpmijk',main.w,aF) #reconstruct in x and y
  uxF  = np.einsum('pl,zpmijk->zlmijk',main.wp,tmp)*2./main.dx
  tmp = np.einsum('qm,zpqijk->zpmijk',main.w,aB)
  uxB  = np.einsum('pl,zpmijk->zlmijk',main.wp,tmp)*2./main.dx
  return uxR,uxL,uxU,uxD,uxF,uxB

def diffUYEdge_edge(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge*main.wpedge[:,1][None,None,:,None,None,None])
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.wpedge[:,0][None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray[None,None,None,:,None,None])
#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.w,aR)  #reconstruct in y and z
  uyR  = np.einsum('qm,zqn...->zmn...',main.wp,tmp)*2./main.dy
  tmp = np.einsum('rn,zqr...->zqn...',main.w,aL)
  uyL  = np.einsum('qm,zqn...->zmn...',main.wp,tmp)*2./main.dy

  tmp = np.einsum('rn,zpr...->zpn...',main.w,aU) #recounstruct in x and z
  uyU  = np.einsum('pl,zpn...->zln...',main.w,tmp)*2./main.dy
  tmp = np.einsum('rn,zpr...->zpn...',main.w,aD)
  uyD  = np.einsum('pl,zpn...->zln...',main.w,tmp)*2./main.dy

  tmp = np.einsum('qm,zpq...->zpm...',main.wp,aF) #reconstruct in x and y
  uyF  = np.einsum('pl,zpm...->zlm...',main.w,tmp)*2./main.dy
  tmp = np.einsum('qm,zpq...->zpm...',main.wp,aB)
  uyB  = np.einsum('pl,zpm...->zlm...',main.w,tmp)*2./main.dy
  return uyR,uyL,uyU,uyD,uyF,uyB

def diffUY_edge(a,main):
  aR = np.einsum('zpqrijk->zqrijk',a)
  aL = np.einsum('zpqrijk->zqrijk',a*main.altarray[None,:,None,None,None,None,None])

  aU = np.einsum('zpqrijk->zprijk',a*main.wpedge[:,1][None,None,:,None,None,None,None])
  aD = np.einsum('zpqrijk->zprijk',a*main.wpedge[:,0][None,None,:,None,None,None,None])

  aF = np.einsum('zpqrijk->zpqijk',a)
  aB = np.einsum('zpqrijk->zpqijk',a*main.altarray[None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqrijk->zqnijk',main.w,aR)  #reconstruct in y and z
  uyR  = np.einsum('qm,zqnijk->zmnijk',main.wp,tmp)*2./main.dy
  tmp = np.einsum('rn,zqrijk->zqnijk',main.w,aL)
  uyL  = np.einsum('qm,zqnijk->zmnijk',main.wp,tmp)*2./main.dy

  tmp = np.einsum('rn,zprijk->zpnijk',main.w,aU) #recounstruct in x and z
  uyU  = np.einsum('pl,zpnijk->zlnijk',main.w,tmp)*2./main.dy
  tmp = np.einsum('rn,zprijk->zpnijk',main.w,aD)
  uyD  = np.einsum('pl,zpnijk->zlnijk',main.w,tmp)*2./main.dy

  tmp = np.einsum('qm,zpqijk->zpmijk',main.wp,aF) #reconstruct in x and y
  uyF  = np.einsum('pl,zpmijk->zlmijk',main.w,tmp)*2./main.dy
  tmp = np.einsum('qm,zpqijk->zpmijk',main.wp,aB)
  uyB  = np.einsum('pl,zpmijk->zlmijk',main.w,tmp)*2./main.dy
  return uyR,uyL,uyU,uyD,uyF,uyB


def diffUZEdge_edge(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge*main.wpedge[:,1][None,None,None,:,None,None])
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.wpedge[:,0][None,None,None,:,None,None])

  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.wp,aR)  #reconstruct in y and z
  uzR  = np.einsum('qm,zqn...->zmn...',main.w,tmp)*2./main.dz 
  tmp = np.einsum('rn,zqr...->zqn...',main.wp,aL)
  uzL  = np.einsum('qm,zqn...->zmn...',main.w,tmp)*2./main.dz

  tmp = np.einsum('rn,zpr...->zpn...',main.wp,aU) #recounstruct in x and z
  uzU  = np.einsum('pl,zpn...->zln...',main.w,tmp)*2./main.dz
  tmp = np.einsum('rn,zpr...->zpn...',main.wp,aD)
  uzD  = np.einsum('pl,zpn...->zln...',main.w,tmp)*2./main.dz

  tmp = np.einsum('qm,zpq...->zpm...',main.w,aF) #reconstruct in x and y
  uzF  = np.einsum('pl,zpm...->zlm...',main.w,tmp)*2./main.dz
  tmp = np.einsum('qm,zpq...->zpm...',main.w,aB)
  uzB  = np.einsum('pl,zpm...->zlm...',main.w,tmp)*2./main.dz
  return uzR,uzL,uzU,uzD,uzF,uzB


def diffUZ_edge(a,main):
  aR = np.einsum('zpqrijk->zqrijk',a)
  aL = np.einsum('zpqrijk->zqrijk',a*main.altarray[None,:,None,None,None,None,None])

  aU = np.einsum('zpqrijk->zprijk',a)
  aD = np.einsum('zpqrijk->zprijk',a*main.altarray[None,None,:,None,None,None,None])

  aF = np.einsum('zpqrijk->zpqijk',a*main.wpedge[:,1][None,None,None,:,None,None,None])
  aB = np.einsum('zpqrijk->zpqijk',a*main.wpedge[:,0][None,None,None,:,None,None,None])

  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqrijk->zqnijk',main.wp,aR)  #reconstruct in y and z
  uzR  = np.einsum('qm,zqnijk->zmnijk',main.w,tmp)*2./main.dz 
  tmp = np.einsum('rn,zqrijk->zqnijk',main.wp,aL)
  uzL  = np.einsum('qm,zqnijk->zmnijk',main.w,tmp)*2./main.dz

  tmp = np.einsum('rn,zprijk->zpnijk',main.wp,aU) #recounstruct in x and z
  uzU  = np.einsum('pl,zpnijk->zlnijk',main.w,tmp)*2./main.dz
  tmp = np.einsum('rn,zprijk->zpnijk',main.wp,aD)
  uzD  = np.einsum('pl,zpnijk->zlnijk',main.w,tmp)*2./main.dz

  tmp = np.einsum('qm,zpqijk->zpmijk',main.w,aF) #reconstruct in x and y
  uzF  = np.einsum('pl,zpmijk->zlmijk',main.w,tmp)*2./main.dz
  tmp = np.einsum('qm,zpqijk->zpmijk',main.w,aB)
  uzB  = np.einsum('pl,zpmijk->zlmijk',main.w,tmp)*2./main.dz
  return uzR,uzL,uzU,uzD,uzF,uzB


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

def reconstructEdgeEdgesGeneral(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray[None,None,None,:,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.w,aR)
  uR  = np.einsum('qm,zqn...->zmn...',main.w,tmp)
  tmp = np.einsum('rn,zqr...->zqn...',main.w,aL)
  uL  = np.einsum('qm,zqn...->zmn...',main.w,tmp)

  tmp = np.einsum('rn,zpr...->zpn...',main.w,aU)
  uU  = np.einsum('pl,zpn...->zln...',main.w,tmp)
  tmp = np.einsum('rn,zpr...->zpn...',main.w,aD)
  uD  = np.einsum('pl,zpn...->zln...',main.w,tmp)

  tmp = np.einsum('qm,zpq...->zpm...',main.w,aF)
  uF  = np.einsum('pl,zpm...->zlm...',main.w,tmp)
  tmp = np.einsum('qm,zpq...->zpm...',main.w,aB)
  uB  = np.einsum('pl,zpm...->zlm...',main.w,tmp)
  return uR,uL,uU,uD,uF,uB


def volIntegrate(weights,f):
  return  np.einsum('zpqrijk->zijk',weights[None,:,None,None,None,None,None]*weights[None,None,:,None,None,None,None]*weights[None,None,None,:,None,None,None]*f[:,:,:,:,:])


def faceIntegrate(weights,f):
  return np.einsum('zpqijk->zijk',weights[None,:,None,None,None,None]*weights[None,None,:,None,None,None]*f)


def faceIntegrateGlob(main,f,w1,w2):
  tmp = np.einsum('nr,zqrijk->zqnijk',w2,main.weights[None,None,:,None,None,None]*f)
  return np.einsum('mq,zqnijk->zmnijk',w1,main.weights[None,:,None,None,None,None]*tmp)



def getFlux(main,eqns,schemes):
  # first reconstruct states
  main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB = reconstructEdgesGeneral(main.a.a,main)
#  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  main.a.aR_edge[:],main.a.aL_edge[:],main.a.aU_edge[:],main.a.aD_edge[:],main.a.aF_edge[:],main.a.aB_edge[:] = sendaEdgesGeneralSlab(main.a.a,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = reconstructEdgeEdgesGeneral(main)
  inviscidFlux(main,eqns,schemes,main.iFlux,main.a)
  # now we need to integrate along the boundary 
  main.iFlux.fRI = faceIntegrateGlob(main,main.iFlux.fRS,main.w,main.w)
  main.iFlux.fLI = faceIntegrateGlob(main,main.iFlux.fLS,main.w,main.w)
  main.iFlux.fUI = faceIntegrateGlob(main,main.iFlux.fUS,main.w,main.w)
  main.iFlux.fDI = faceIntegrateGlob(main,main.iFlux.fDS,main.w,main.w)
  main.iFlux.fFI = faceIntegrateGlob(main,main.iFlux.fFS,main.w,main.w)
  main.iFlux.fBI = faceIntegrateGlob(main,main.iFlux.fBS,main.w,main.w)


def volIntegrateGlob(main,f,w1,w2,w3):
  tmp = np.einsum('nr,zpqrijk->zpqnijk',w3,main.weights[None,None,None,:,None,None,None]*f)
  tmp = np.einsum('mq,zpqnijk->zpmnijk',w2,main.weights[None,None,:,None,None,None,None]*tmp)
  return np.einsum('lp,zpmnijk->zlmnijk',w1,main.weights[None,:,None,None,None,None,None]*tmp)


def getRHS_IP(main,eqns,schemes):
  t0 = time.time()
  reconstructU(main,main.a)
  order = np.shape(main.a.a)[1]
  # evaluate inviscid flux
  getFlux(main,eqns,schemes)
  ### Get interior vol terms
  eqns.evalFluxX(main.a.u,main.iFlux.fx)
  eqns.evalFluxY(main.a.u,main.iFlux.fy)
  eqns.evalFluxZ(main.a.u,main.iFlux.fz)
  # now get viscous flux
  fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I = getViscousFlux(main,eqns,schemes) ##takes roughly 20% of the time
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
  t1 = time.time()
  ## This is important. Do partial integrations in each direction to avoid doing for each ijk
  ord_arr= np.linspace(0,order-1,order)
  scale =  (2.*ord_arr[:,None,None] + 1.)*(2.*ord_arr[None,:,None] + 1.)*(2.*ord_arr[None,None,:]+1.)/8.
  dxi = 2./main.dx*scale
  dyi = 2./main.dy*scale
  dzi = 2./main.dz*scale
  v1ijk = volIntegrateGlob(main,main.iFlux.fx - fvGX,main.wp,main.w,main.w)*dxi[None,:,:,:,None,None,None]
  v2ijk = volIntegrateGlob(main,main.iFlux.fy - fvGY,main.w,main.wp,main.w)*dyi[None,:,:,:,None,None,None]
  v3ijk = volIntegrateGlob(main,main.iFlux.fz - fvGZ,main.w,main.w,main.wp)*dzi[None,:,:,:,None,None,None]

  tmp = v1ijk + v2ijk + v3ijk
  tmp +=  (-main.iFlux.fRI[:,None,:,:] + main.iFlux.fLI[:,None,:,:]*main.altarray[None,:,None,None,None,None,None])*dxi[None,:,:,:,None,None,None]
  tmp +=  (-main.iFlux.fUI[:,:,None,:] + main.iFlux.fDI[:,:,None,:]*main.altarray[None,None,:,None,None,None,None])*dyi[None,:,:,:,None,None,None]
  tmp +=  (-main.iFlux.fFI[:,:,:,None] + main.iFlux.fBI[:,:,:,None]*main.altarray[None,None,None,:,None,None,None])*dzi[None,:,:,:,None,None,None]
  tmp +=  (fvRIG11[:,None,:,:]*main.wpedge[None,:,None,None,1,None,None,None] + fvRIG21[:,None,:,:] + fvRIG31[:,None,:,:]  - (fvLIG11[:,None,:,:]*main.wpedge[None,:,None,None,0,None,None,None] + fvLIG21[:,None,:,:]*main.altarray[None,:,None,None,None,None,None] + fvLIG31[:,None,:,:]*main.altarray[None,:,None,None,None,None,None]) )*dxi[None,:,:,:,None,None,None]
  tmp +=  (fvUIG12[:,:,None,:] + fvUIG22[:,:,None,:]*main.wpedge[None,None,:,None,1,None,None,None] + fvUIG32[:,:,None,:]  - (fvDIG12[:,:,None,:]*main.altarray[None,None,:,None,None,None,None] + fvDIG22[:,:,None,:]*main.wpedge[None,None,:,None,0,None,None,None] + fvDIG32[:,:,None,:]*main.altarray[None,None,:,None,None,None,None]) )*dyi[None,:,:,:,None,None,None]
  tmp +=  (fvFIG13[:,:,:,None] + fvFIG23[:,:,:,None] + fvFIG33[:,:,:,None]*main.wpedge[None,None,None,:,1,None,None,None]  - (fvBIG13[:,:,:,None]*main.altarray[None,None,None,:,None,None,None] + fvBIG23[:,:,:,None]*main.altarray[None,None,None,:,None,None,None] + fvBIG33[:,:,:,None]*main.wpedge[None,None,None,:,0,None,None,None]) )*dzi[None,:,:,:,None,None,None] 
  tmp +=  (fvR2I[:,None,:,:] - fvL2I[:,None,:,:]*main.altarray[None,:,None,None,None,None,None])*dxi[None,:,:,:,None,None,None] + (fvU2I[:,:,None,:] - fvD2I[:,:,None,:]*main.altarray[None,None,:,None,None,None,None])*dyi[None,:,:,:,None,None,None] + (fvF2I[:,:,:,None] - fvB2I[:,:,:,None]*main.altarray[None,None,None,:,None,None,None])*dzi[None,:,:,:,None,None,None]
  main.RHS = tmp



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

  
#  apx,apy,apz = diffCoeffs(main.a.a)
#  apx *= 2./main.dx
#  apy *= 2./main.dy
#  apz *= 2./main.dz

#  UxR,UxL,UxU,UxD,UxF,UxB = reconstructEdgesGeneral(apx,main)
#  UyR,UyL,UyU,UyD,UyF,UyB = reconstructEdgesGeneral(apy,main)
#  UzR,UzL,UzU,UzD,UzF,UzB = reconstructEdgesGeneral(apz,main)

  UxR,UxL,UxU,UxD,UxF,UxB = diffUX_edge(main.a.a,main)
  UyR,UyL,UyU,UyD,UyF,UyB = diffUY_edge(main.a.a,main)
  UzR,UzL,UzU,UzD,UzF,UzB = diffUZ_edge(main.a.a,main)


#  UxR_edge,UxL_edge,UxU_edge,UxD_edge,UxF_edge,UxB_edge = sendEdgesGeneralSlab(UxL,UxR,UxD,UxU,UxB,UxF,main)
#  UyR_edge,UyL_edge,UyU_edge,UyD_edge,UyF_edge,UyB_edge = sendEdgesGeneralSlab(UyL,UyR,UyD,UyU,UyB,UyF,main)
#  UzR_edge,UzL_edge,UzU_edge,UzD_edge,UzF_edge,UzB_edge = sendEdgesGeneralSlab(UzL,UzR,UzD,UzU,UzB,UzF,main)

  UxR_edge,UxL_edge,UxU_edge,UxD_edge,UxF_edge,UxB_edge = diffUXEdge_edge(main)
  UyR_edge,UyL_edge,UyU_edge,UyD_edge,UyF_edge,UyB_edge = diffUYEdge_edge(main)
  UzR_edge,UzL_edge,UzU_edge,UzD_edge,UzF_edge,UzB_edge = diffUZEdge_edge(main)

#  print(np.linalg.norm(UzB_edge2 - UzB_edge))

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

  fvRIG11 = faceIntegrateGlob(main,fvRG11,main.w,main.w) 
  fvLIG11 = faceIntegrateGlob(main,fvLG11,main.w,main.w) 
  fvRIG21 = faceIntegrateGlob(main,fvRG21,main.wp,main.w)
  fvLIG21 = faceIntegrateGlob(main,fvLG21,main.wp,main.w)
  fvRIG31 = faceIntegrateGlob(main,fvRG31,main.w,main.wp)
  fvLIG31 = faceIntegrateGlob(main,fvLG31,main.w,main.wp)

  fvUIG12 = faceIntegrateGlob(main,fvUG12,main.wp,main.w) 
  fvDIG12 = faceIntegrateGlob(main,fvDG12,main.wp,main.w) 
  fvUIG22 = faceIntegrateGlob(main,fvUG22,main.w,main.w) 
  fvDIG22 = faceIntegrateGlob(main,fvDG22,main.w,main.w) 
  fvUIG32 = faceIntegrateGlob(main,fvUG32,main.w,main.wp) 
  fvDIG32 = faceIntegrateGlob(main,fvDG32,main.w,main.wp) 

  fvFIG13 = faceIntegrateGlob(main,fvFG13,main.wp,main.w)  
  fvBIG13 = faceIntegrateGlob(main,fvBG13,main.wp,main.w) 
  fvFIG23 = faceIntegrateGlob(main,fvFG23,main.w,main.wp) 
  fvBIG23 = faceIntegrateGlob(main,fvBG23,main.w,main.wp) 
  fvFIG33 = faceIntegrateGlob(main,fvFG33,main.w,main.w) 
  fvBIG33 = faceIntegrateGlob(main,fvBG33,main.w,main.w) 

  fvR2I = faceIntegrateGlob(main,fvR2,main.w,main.w)   
  fvL2I = faceIntegrateGlob(main,fvL2,main.w,main.w)  
  fvU2I = faceIntegrateGlob(main,fvU2,main.w,main.w)  
  fvD2I = faceIntegrateGlob(main,fvD2,main.w,main.w)  
  fvF2I = faceIntegrateGlob(main,fvF2,main.w,main.w)  
  fvB2I = faceIntegrateGlob(main,fvB2,main.w,main.w)  

  return fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I


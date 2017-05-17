from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs#,sendaEdgesGeneralSlab
from fluxSchemes import *
from navier_stokes import evalViscousFluxZNS_IP
from navier_stokes import evalViscousFluxYNS_IP
from navier_stokes import evalViscousFluxXNS_IP
from tensor_products import *
import time

def diffU(a,main):
  tmp = np.tensordot(a,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  #ux = np.swapaxes( np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5), 3, 6)
  ux = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
 
  tmpu = np.tensordot(a,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmpu,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  #uy = np.swapaxes( np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5), 3, 6)
  uy = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  #tmp = np.tensordot(a,main.w,axes=([1],[0]))
  tmp = np.tensordot(tmpu,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  #uz = np.swapaxes( np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5), 3, 6)
  uz = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  return ux,uy,uz

#def diffU(a,main):
#  tmp =  np.einsum('rn,zpqr...->zpqn...',main.w,a) #reconstruct along third axis 
#  tmp2 = np.einsum('qm,zpqn...->zpmn...',main.w,tmp) #reconstruct along second axis
#  ux = np.einsum('pl,zpmn...->zlmn...',main.wp,tmp2) # get ux by differentiating along the first axis
#
#  tmp2 = np.einsum('qm,zpqn...->zpmn...',main.wp,tmp) #diff tmp along second axis
#  uy = np.einsum('pl,zpmn...->zlmn...',main.w,tmp2) # get uy by reconstructing along the first axis
#
#  tmp =  np.einsum('rn,zpqr...->zpqn...',main.wp,a) #diff along third axis 
#  tmp = np.einsum('qm,zpqn...->zpmn...',main.w,tmp) #reconstruct along second axis
#  uz = np.einsum('pl,zpmn...->zlmn...',main.w,tmp) # reconstruct along the first axis
#  return ux,uy,uz



def diffUXEdge_edge2(main):
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


def diffUXEdge_edge(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge*main.wpedge0[:,1][None,:,None,None,None,None])
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.wpedge0[:,0][None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray1[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray2[None,None,None,:,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uxR = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uxL = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aU,main.wp0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uxU = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aD,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uxD = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aF,main.wp0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  uxF = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aB,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  uxB = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  return uxR,uxL,uxU,uxD,uxF,uxB


def diffUX_edge(a,main):
  aR = np.einsum('zpqr...->zqr...',a*main.wpedge0[:,1][None,:,None,None,None,None,None])
  aL = np.einsum('zpqr...->zqr...',a*main.wpedge0[:,0] [None,:,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray1[None,None,:,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray2[None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uxR = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uxL = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aU,main.wp0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uxU = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aD,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uxD = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aF,main.wp0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  uxF = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aB,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  uxB = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  return uxR,uxL,uxU,uxD,uxF,uxB

def diffUX_edge2(a,main):
  aR = np.einsum('zpqr...->zqr...',a*main.wpedge[:,1][None,:,None,None,None,None,None])
  aL = np.einsum('zpqr...->zqr...',a*main.wpedge[:,0] [None,:,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray[None,None,:,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray[None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.w2,aR)  #reconstruct in y and z
  uxR  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dx 
  tmp = np.einsum('rn,zqr...->zqn...',main.w2,aL)
  uxL  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dx

  tmp = np.einsum('rn,zpr...->zpn...',main.w2,aU) #reconstruct in x and z 
  uxU  = np.einsum('pl,zpn...->zln...',main.wp0,tmp)*2./main.dx
  tmp = np.einsum('rn,zpr...->zpn...',main.w2,aD)
  uxD  = np.einsum('pl,zpn...->zln...',main.wp0,tmp)*2./main.dx

  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aF) #reconstruct in x and y
  uxF  = np.einsum('pl,zpm...->zlm...',main.wp0,tmp)*2./main.dx
  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aB)
  uxB  = np.einsum('pl,zpm...->zlm...',main.wp0,tmp)*2./main.dx
  return uxR,uxL,uxU,uxD,uxF,uxB

def diffUYEdge_edge2(main):
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


def diffUYEdge_edge(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray0[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge*main.wpedge1[:,1][None,None,:,None,None,None])
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.wpedge1[:,0][None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray2[None,None,None,:,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.wp1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyR = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aL,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyL = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyU = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyD = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))*2./main.dx
  uyF = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))*2./main.dx
  uyB = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  return uyR,uyL,uyU,uyD,uyF,uyB


def diffUY_edge2(a,main):
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray[None,:,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a*main.wpedge[:,1][None,None,:,None,None,None,None])
  aD = np.einsum('zpqr...->zpr...',a*main.wpedge[:,0][None,None,:,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray[None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.w2,aR)  #reconstruct in y and z
  uyR  = np.einsum('qm,zqn...->zmn...',main.wp1,tmp)*2./main.dy
  tmp = np.einsum('rn,zqr...->zqn...',main.w2,aL)
  uyL  = np.einsum('qm,zqn...->zmn...',main.wp1,tmp)*2./main.dy

  tmp = np.einsum('rn,zpr...->zpn...',main.w2,aU) #recounstruct in x and z
  uyU  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dy
  tmp = np.einsum('rn,zpr...->zpn...',main.w2,aD)
  uyD  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dy

  tmp = np.einsum('qm,zpq...->zpm...',main.wp1,aF) #reconstruct in x and y
  uyF  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dy
  tmp = np.einsum('qm,zpq...->zpm...',main.wp1,aB)
  uyB  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dy
  return uyR,uyL,uyU,uyD,uyF,uyB

def diffUY_edge(a,main):
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray0[None,:,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a*main.wpedge1[:,1][None,None,:,None,None,None,None])
  aD = np.einsum('zpqr...->zpr...',a*main.wpedge1[:,0][None,None,:,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray2[None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.wp1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyR = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aL,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyL = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyU = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyD = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))*2./main.dx
  uyF = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))*2./main.dx
  uyB = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  return uyR,uyL,uyU,uyD,uyF,uyB


def diffUZEdge_edge2(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge*main.wpedge[:,1][None,None,None,:,None,None])
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.wpedge[:,0][None,None,None,:,None,None])

  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.wp2,aR)  #reconstruct in y and z
  uzR  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dz 
  tmp = np.einsum('rn,zqr...->zqn...',main.wp2,aL)
  uzL  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dz

  tmp = np.einsum('rn,zpr...->zpn...',main.wp2,aU) #recounstruct in x and z
  uzU  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dz
  tmp = np.einsum('rn,zpr...->zpn...',main.wp2,aD)
  uzD  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dz

  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aF) #reconstruct in x and y
  uzF  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dz
  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aB)
  uzB  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dz
  return uzR,uzL,uzU,uzD,uzF,uzB


def diffUZEdge_edge(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray0[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray1[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge*main.wpedge2[:,1][None,None,None,:,None,None])
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.wpedge2[:,0][None,None,None,:,None,None])


#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzR = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzL = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzU = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzD = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  uzF = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  uzB = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  return uzR,uzL,uzU,uzD,uzF,uzB


def diffUZ_edge2(a,main):
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray[None,:,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray[None,None,:,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a*main.wpedge[:,1][None,None,None,:,None,None,None])
  aB = np.einsum('zpqr...->zpq...',a*main.wpedge[:,0][None,None,None,:,None,None,None])

  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.wp2,aR)  #reconstruct in y and z
  uzR  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dz 
  tmp = np.einsum('rn,zqr...->zqn...',main.wp2,aL)
  uzL  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dz

  tmp = np.einsum('rn,zpr...->zpn...',main.wp2,aU) #recounstruct in x and z
  uzU  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dz
  tmp = np.einsum('rn,zpr...->zpn...',main.wp2,aD)
  uzD  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dz

  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aF) #reconstruct in x and y
  uzF  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dz
  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aB)
  uzB  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dz
  return uzR,uzL,uzU,uzD,uzF,uzB


def diffUZ_edge(a,main):
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray0[None,:,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray1[None,None,:,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a*main.wpedge2[:,1][None,None,None,:,None,None,None])
  aB = np.einsum('zpqr...->zpq...',a*main.wpedge2[:,0][None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzR = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzL = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzU = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzD = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  uzF = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  uzB = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  return uzR,uzL,uzU,uzD,uzF,uzB


def diffCoeffs(a):
  atmp = np.zeros(np.shape(a))
  atmp[:] = a[:]
  nvars,order,order,order,Nelx,Nely,Nelz = np.shape(a)
  ax = np.zeros((nvars,order[0],order[1],order[2],Nelx,Nely,Nelz))
  ay = np.zeros((nvars,order[0],order[1],order[2],Nelx,Nely,Nelz))
  az = np.zeros((nvars,order[0],order[1],order[2],Nelx,Nely,Nelz))

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





def volIntegrateGlob(main,f,w1,w2,w3):
  tmp = np.tensordot(main.weights0[None,:,None,None,None,None,None]*f  ,w1,axes=([1],[1]))
  tmp = np.tensordot(main.weights1[None,:,None,None,None,None,None]*tmp,w2,axes=([1],[1]))
  tmp = np.tensordot(main.weights2[None,:,None,None,None,None,None]*tmp,w3,axes=([1],[1]))
#  return np.swapaxes( np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5), 3, 6)
  return np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)


def faceIntegrate(weights,f):
  return np.einsum('zpqijk->zijk',weights[None,:,None,None,None,None]*weights[None,None,:,None,None,None]*f)

def faceIntegrateGlob(main,f,w1,w2,weights1,weights2):
  tmp = np.tensordot(weights1[None,:,None,None,None,None]*f,w1,axes=([1],[1]))
  tmp = np.tensordot(weights2[None,:,None,None,None,None]*tmp,w2,axes=([1],[1]))
  #return np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5)
  return np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)


def faceIntegrateGlob2(main,f,w1,w2,weights0,weights1):
  tmp = np.einsum('nr,zqrijk->zqnijk',w2,weights1[None,None,:,None,None,None]*f)
  return np.einsum('mq,zqnijk->zmnijk',w1,weights0[None,:,None,None,None,None]*tmp)



def getFlux(main,MZ,eqns,args):
  # first reconstruct states
  main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB = reconstructEdgesGeneral(main.a.a,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  #main.a.aR_edge[:],main.a.aL_edge[:],main.a.aU_edge[:],main.a.aD_edge[:],main.a.aF_edge[:],main.a.aB_edge[:] = sendaEdgesGeneralSlab(main.a.a,main)
  #main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = reconstructEdgeEdgesGeneral(main)
  inviscidFluxGen(main,eqns,main.iFlux,main.a,args)
  # now we need to integrate along the boundary 
  main.iFlux.fRI = faceIntegrateGlob(main,main.iFlux.fRS,MZ.w1,MZ.w2,MZ.weights1,MZ.weights2)
  main.iFlux.fLI = faceIntegrateGlob(main,main.iFlux.fLS,MZ.w1,MZ.w2,MZ.weights1,MZ.weights2)
  main.iFlux.fUI = faceIntegrateGlob(main,main.iFlux.fUS,MZ.w0,MZ.w2,MZ.weights0,MZ.weights2)
  main.iFlux.fDI = faceIntegrateGlob(main,main.iFlux.fDS,MZ.w0,MZ.w2,MZ.weights0,MZ.weights2)
  main.iFlux.fFI = faceIntegrateGlob(main,main.iFlux.fFS,MZ.w0,MZ.w1,MZ.weights0,MZ.weights1)
  main.iFlux.fBI = faceIntegrateGlob(main,main.iFlux.fBS,MZ.w0,MZ.w1,MZ.weights0,MZ.weights1)


#def volIntegrateGlob(main,f,w1,w2,w3):
#  tmp = np.einsum('nr,zpqrijk->zpqnijk',w3,main.weights[None,None,None,:,None,None,None]*f)
#  tmp = np.einsum('mq,zpqnijk->zpmnijk',w2,main.weights[None,None,:,None,None,None,None]*tmp)
#  return np.einsum('lp,zpmnijk->zlmnijk',w1,main.weights[None,:,None,None,None,None,None]*tmp)

def getRHS_FM1(main,MZ,eqns,args=[]):
  t0 = time.time()
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,MZ,eqns,args)
  ### Get interior vol terms
  eqns.evalFluxX(main.a.u,main.iFlux.fx,args)
  eqns.evalFluxY(main.a.u,main.iFlux.fy,args)
  eqns.evalFluxZ(main.a.u,main.iFlux.fz,args)

  # now get viscous flux
  fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I = getViscousFlux(main,eqns) ##takes roughly 20% of the time

  upx,upy,upz = diffU(main.a.a,main)
  upx = upx*2./main.dx
  upy = upy*2./main.dy
  upz = upz*2./main.dz

  fvGX = eqns.evalViscousFluxX(main,main.a.u,upx,upy,upz)
  fvGY = eqns.evalViscousFluxY(main,main.a.u,upx,upy,upz)
  fvGZ = eqns.evalViscousFluxZ(main,main.a.u,upx,upy,upz)
  #print(np.linalg.norm(fvGX2 - fvGX))
  # Now form RHS
  t1 = time.time()
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])

  scale =  (2.*ord_arr0[:,None,None] + 1.)*(2.*ord_arr1[None,:,None] + 1.)*(2.*ord_arr2[None,None,:]+1.)/8.
  dxi = 2./main.dx*scale
  dyi = 2./main.dy*scale
  dzi = 2./main.dz*scale
  v1ijk = volIntegrateGlob(main,main.iFlux.fx - fvGX,main.wp0,main.w1,main.w2)*dxi[None,:,:,:,None,None,None]
  v2ijk = volIntegrateGlob(main,main.iFlux.fy - fvGY,main.w0,main.wp1,main.w2)*dyi[None,:,:,:,None,None,None]
  v3ijk = volIntegrateGlob(main,main.iFlux.fz - fvGZ,main.w0,main.w1,main.wp2)*dzi[None,:,:,:,None,None,None]

  tmp = v1ijk + v2ijk + v3ijk
  tmp +=  (-main.iFlux.fRI[:,None,:,:] + main.iFlux.fLI[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None])*dxi[None,:,:,:,None,None,None]
  tmp +=  (-main.iFlux.fUI[:,:,None,:] + main.iFlux.fDI[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None])*dyi[None,:,:,:,None,None,None]
  tmp +=  (-main.iFlux.fFI[:,:,:,None] + main.iFlux.fBI[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None])*dzi[None,:,:,:,None,None,None]
  tmp +=  (fvRIG11[:,None,:,:]*main.wpedge0[None,:,None,None,1,None,None,None] + fvRIG21[:,None,:,:] + fvRIG31[:,None,:,:]  - (fvLIG11[:,None,:,:]*main.wpedge0[None,:,None,None,0,None,None,None] + fvLIG21[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None] + fvLIG31[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None]) )*dxi[None,:,:,:,None,None,None]
  tmp +=  (fvUIG12[:,:,None,:] + fvUIG22[:,:,None,:]*main.wpedge1[None,None,:,None,1,None,None,None] + fvUIG32[:,:,None,:]  - (fvDIG12[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None] + fvDIG22[:,:,None,:]*main.wpedge1[None,None,:,None,0,None,None,None] + fvDIG32[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None]) )*dyi[None,:,:,:,None,None,None]
  tmp +=  (fvFIG13[:,:,:,None] + fvFIG23[:,:,:,None] + fvFIG33[:,:,:,None]*main.wpedge2[None,None,None,:,1,None,None,None]  - (fvBIG13[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None] + fvBIG23[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None] + fvBIG33[:,:,:,None]*main.wpedge2[None,None,None,:,0,None,None,None]) )*dzi[None,:,:,:,None,None,None] 
  tmp +=  (fvR2I[:,None,:,:] - fvL2I[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None])*dxi[None,:,:,:,None,None,None] + (fvU2I[:,:,None,:] - fvD2I[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None])*dyi[None,:,:,:,None,None,None] + (fvF2I[:,:,:,None] - fvB2I[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None])*dzi[None,:,:,:,None,None,None]
 
  if (main.source):
    force = np.zeros(np.shape(fvGX))
    for i in range(0,main.nvars):
      force[i] = main.source_mag[i]
    tmp += volIntegrateGlob(main, force ,main.w0,main.w1,main.w2)*scale[None,:,:,:,None,None,None]

  main.RHS = tmp
  main.comm.Barrier()




def getRHS_IP(main,MZ,eqns,args=[]):
  t0 = time.time()
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,MZ,eqns,args)
  ### Get interior vol terms
  eqns.evalFluxX(main.a.u,main.iFlux.fx,args)
  eqns.evalFluxY(main.a.u,main.iFlux.fy,args)
  eqns.evalFluxZ(main.a.u,main.iFlux.fz,args)

  # now get viscous flux
  fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I = getViscousFlux(main,eqns) ##takes roughly 20% of the time

  upx,upy,upz = diffU(main.a.a,main)
  upx = upx*2./main.dx
  upy = upy*2./main.dy
  upz = upz*2./main.dz

  fvGX = eqns.evalViscousFluxX(main,main.a.u,upx,upy,upz)
  fvGY = eqns.evalViscousFluxY(main,main.a.u,upx,upy,upz)
  fvGZ = eqns.evalViscousFluxZ(main,main.a.u,upx,upy,upz)
  #print(np.linalg.norm(fvGX2 - fvGX))
  # Now form RHS
  t1 = time.time()
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])

  scale =  (2.*ord_arr0[:,None,None] + 1.)*(2.*ord_arr1[None,:,None] + 1.)*(2.*ord_arr2[None,None,:]+1.)/8.
  dxi = 2./main.dx*scale
  dyi = 2./main.dy*scale
  dzi = 2./main.dz*scale
  v1ijk = volIntegrateGlob(main,main.iFlux.fx - fvGX,main.wp0,main.w1,main.w2)*dxi[None,:,:,:,None,None,None]
  v2ijk = volIntegrateGlob(main,main.iFlux.fy - fvGY,main.w0,main.wp1,main.w2)*dyi[None,:,:,:,None,None,None]
  v3ijk = volIntegrateGlob(main,main.iFlux.fz - fvGZ,main.w0,main.w1,main.wp2)*dzi[None,:,:,:,None,None,None]

  tmp = v1ijk + v2ijk + v3ijk
  tmp +=  (-main.iFlux.fRI[:,None,:,:] + main.iFlux.fLI[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None])*dxi[None,:,:,:,None,None,None]
  tmp +=  (-main.iFlux.fUI[:,:,None,:] + main.iFlux.fDI[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None])*dyi[None,:,:,:,None,None,None]
  tmp +=  (-main.iFlux.fFI[:,:,:,None] + main.iFlux.fBI[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None])*dzi[None,:,:,:,None,None,None]
  tmp +=  (fvRIG11[:,None,:,:]*main.wpedge0[None,:,None,None,1,None,None,None] + fvRIG21[:,None,:,:] + fvRIG31[:,None,:,:]  - (fvLIG11[:,None,:,:]*main.wpedge0[None,:,None,None,0,None,None,None] + fvLIG21[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None] + fvLIG31[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None]) )*dxi[None,:,:,:,None,None,None]
  tmp +=  (fvUIG12[:,:,None,:] + fvUIG22[:,:,None,:]*main.wpedge1[None,None,:,None,1,None,None,None] + fvUIG32[:,:,None,:]  - (fvDIG12[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None] + fvDIG22[:,:,None,:]*main.wpedge1[None,None,:,None,0,None,None,None] + fvDIG32[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None]) )*dyi[None,:,:,:,None,None,None]
  tmp +=  (fvFIG13[:,:,:,None] + fvFIG23[:,:,:,None] + fvFIG33[:,:,:,None]*main.wpedge2[None,None,None,:,1,None,None,None]  - (fvBIG13[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None] + fvBIG23[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None] + fvBIG33[:,:,:,None]*main.wpedge2[None,None,None,:,0,None,None,None]) )*dzi[None,:,:,:,None,None,None] 
  tmp +=  (fvR2I[:,None,:,:] - fvL2I[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None])*dxi[None,:,:,:,None,None,None] + (fvU2I[:,:,None,:] - fvD2I[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None])*dyi[None,:,:,:,None,None,None] + (fvF2I[:,:,:,None] - fvB2I[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None])*dzi[None,:,:,:,None,None,None]
 
  if (main.source):
    force = np.zeros(np.shape(fvGX))
    for i in range(0,main.nvars):
      force[i] = main.source_mag[i]
    tmp += volIntegrateGlob(main, force ,main.w0,main.w1,main.w2)*scale[None,:,:,:,None,None,None]

  main.RHS = tmp
  main.comm.Barrier()


def getRHS_INVISCID(main,MZ,eqns,args=[]):
  t0 = time.time()
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,MZ,eqns,args )

  ### Get interior vol terms
  nargs = np.shape(args)[0]
  args_u = []
  for i in range(0,nargs):
    tmp = reconstructUGeneral(main,args[i])
    args_u.append(tmp)
  eqns.evalFluxX(main.a.u,main.iFlux.fx,args_u)
  eqns.evalFluxY(main.a.u,main.iFlux.fy,args_u)
  eqns.evalFluxZ(main.a.u,main.iFlux.fz,args_u)
  # Now form RHS
  t1 = time.time()
  ## This is important. Do partial integrations in each direction to avoid doing for each ijk
  ord_arr0= np.linspace(0,MZ.order[0]-1,MZ.order[0])
  ord_arr1= np.linspace(0,MZ.order[1]-1,MZ.order[1])
  ord_arr2= np.linspace(0,MZ.order[2]-1,MZ.order[2])

  scale =  (2.*ord_arr0[:,None,None] + 1.)*(2.*ord_arr1[None,:,None] + 1.)*(2.*ord_arr2[None,None,:]+1.)/8.
#  print('got here')

  dxi = 2./main.dx*scale
  dyi = 2./main.dy*scale
  dzi = 2./main.dz*scale
  v1ijk = volIntegrateGlob(main,main.iFlux.fx ,MZ.wp0,MZ.w1,MZ.w2)*dxi[None,:,:,:,None,None,None]
  v2ijk = volIntegrateGlob(main,main.iFlux.fy ,MZ.w0,MZ.wp1,MZ.w2)*dyi[None,:,:,:,None,None,None]
  v3ijk = volIntegrateGlob(main,main.iFlux.fz ,MZ.w0,MZ.w1,MZ.wp2)*dzi[None,:,:,:,None,None,None]

  tmp = v1ijk + v2ijk + v3ijk
  tmp +=  (-main.iFlux.fRI[:,None,:,:] + main.iFlux.fLI[:,None,:,:]*MZ.altarray0[None,:,None,None,None,None,None])*dxi[None,:,:,:,None,None,None]
  tmp +=  (-main.iFlux.fUI[:,:,None,:] + main.iFlux.fDI[:,:,None,:]*MZ.altarray1[None,None,:,None,None,None,None])*dyi[None,:,:,:,None,None,None]
  tmp +=  (-main.iFlux.fFI[:,:,:,None] + main.iFlux.fBI[:,:,:,None]*MZ.altarray2[None,None,None,:,None,None,None])*dzi[None,:,:,:,None,None,None]
  main.RHS = tmp
  main.comm.Barrier()


def computeJump(uR,uL,uU,uD,uF,uB,uR_edge,uL_edge,uU_edge,uD_edge,uF_edge,uB_edge):
  nvars,order1,order2,Npx,Npy,Npz = np.shape(uR)
  nvars,order0,order1,Npx,Npy,Npz = np.shape(uF)
  jumpR = np.zeros((nvars,order1,order2,Npx,Npy,Npz))
  jumpL = np.zeros((nvars,order1,order2,Npx,Npy,Npz))
  jumpU = np.zeros((nvars,order0,order2,Npx,Npy,Npz))
  jumpD = np.zeros((nvars,order0,order2,Npx,Npy,Npz))
  jumpF = np.zeros((nvars,order0,order1,Npx,Npy,Npz))
  jumpB = np.zeros((nvars,order0,order1,Npx,Npy,Npz))

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


def getViscousFlux(main,eqns):
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

#  fvRG11 = np.sum(G11R*(main.a.uR - uhatR),axis=1)
#  fvLG11 = np.sum(G11L*(main.a.uL - uhatL),axis=1)
#  fvRG21 = np.sum(G21R*(main.a.uR - uhatR),axis=1)
#  fvLG21 = np.sum(G21L*(main.a.uL - uhatL),axis=1)
#  fvRG31 = np.sum(G31R*(main.a.uR - uhatR),axis=1)
#  fvLG31 = np.sum(G31L*(main.a.uL - uhatL),axis=1)
#
#  fvUG12 = np.sum(G12U*(main.a.uU - uhatU),axis=1)
#  fvDG12 = np.sum(G12D*(main.a.uD - uhatD),axis=1)
#  fvUG22 = np.sum(G22U*(main.a.uU - uhatU),axis=1)
#  fvDG22 = np.sum(G22D*(main.a.uD - uhatD),axis=1)
#  fvUG32 = np.sum(G32U*(main.a.uU - uhatU),axis=1)
#  fvDG32 = np.sum(G32D*(main.a.uD - uhatD),axis=1)

#  fvFG13 = np.sum(G13F*(main.a.uF - uhatF),axis=1)
#  fvBG13 = np.sum(G13B*(main.a.uB - uhatB),axis=1)
#  fvFG23 = np.sum(G23F*(main.a.uF - uhatF),axis=1)
#  fvBG23 = np.sum(G23B*(main.a.uB - uhatB),axis=1)
#  fvFG33 = np.sum(G33F*(main.a.uF - uhatF),axis=1)
#  fvBG33 = np.sum(G33B*(main.a.uB - uhatB),axis=1)

  
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
  #print(np.linalg.norm(UzR),np.linalg.norm(UzL),np.linalg.norm(UzU),np.linalg.norm(UzD),np.linalg.norm(UzF),np.linalg.norm(UzB))

#  UxR_edge,UxL_edge,UxU_edge,UxD_edge,UxF_edge,UxB_edge = sendEdgesGeneralSlab_Derivs(UxL,UxR,UxD,UxU,UxB,UxF,main)
#  UyR_edge,UyL_edge,UyU_edge,UyD_edge,UyF_edge,UyB_edge = sendEdgesGeneralSlab_Derivs(UyL,UyR,UyD,UyU,UyB,UyF,main)
#  UzR_edge,UzL_edge,UzU_edge,UzD_edge,UzF_edge,UzB_edge = sendEdgesGeneralSlab_Derivs(UzL,UzR,UzD,UzU,UzB,UzF,main)
  #UxR_edge*=0 
  #UxL_edge*=0
  #UyR_edge*=0 
  #UyL_edge*=0
  #UzR_edge*=0 
  #UzL_edge*=0

  #UxR_edge,UxL_edge,UxU_edge,UxD_edge,UxF_edge,UxB_edge = diffUXEdge_edge(main)
  #UyR_edge,UyL_edge,UyU_edge,UyD_edge,UyF_edge,UyB_edge = diffUYEdge_edge(main)
  #UzR_edge,UzL_edge,UzU_edge,UzD_edge,UzF_edge,UzB_edge = diffUZEdge_edge(main)

#  print(np.linalg.norm(UzB_edge2 - UzB_edge))

  fvxR = eqns.evalViscousFluxX(main,main.a.uR,UxR,UyR,UzR)
  fvxL = eqns.evalViscousFluxX(main,main.a.uL,UxL,UyL,UzL)
  #fvxR_edge = eqns.evalViscousFluxX(main,main.a.uR_edge,UxR_edge,UyR_edge,UzR_edge)
  #fvxL_edge = eqns.evalViscousFluxX(main,main.a.uL_edge,UxL_edge,UyL_edge,UyL_edge)
#  fvxR_edge = fvxR[:,:,:,-1,:,:]
#  fvxL_edge = fvxL[:,:,:,0,:,:]

  fvyU = eqns.evalViscousFluxY(main,main.a.uU,UxU,UyU,UzU)
  fvyD = eqns.evalViscousFluxY(main,main.a.uD,UxD,UyD,UzD)
#  fvyU_edge = eqns.evalViscousFluxY(main,main.a.uU_edge,UxU_edge,UyU_edge,UzU_edge)
#  fvyD_edge = eqns.evalViscousFluxY(main,main.a.uD_edge,UxD_edge,UyD_edge,UzD_edge)
#  fvyU_edge = fvyU[:,:,:,:,-1,:]
#  fvyD_edge = fvyD[:,:,:,:,0 ,:]


  fvzF = eqns.evalViscousFluxZ(main,main.a.uF,UxF,UyF,UzF)
  fvzB = eqns.evalViscousFluxZ(main,main.a.uB,UxB,UyB,UzB)
#  fvzF_edge = eqns.evalViscousFluxZ(main,main.a.uF_edge,UxF_edge,UyF_edge,UzF_edge)
#  fvzB_edge = eqns.evalViscousFluxZ(main,main.a.uB_edge,UxB_edge,UyB_edge,UzB_edge)
#  fvzF_edge = fvzF[:,:,:,:,:,-1]
#  fvzB_edge = fvzB[:,:,:,:,:, 0]

  fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge = sendEdgesGeneralSlab_Derivs(fvxL,fvxR,fvyD,fvyU,fvzB,fvzF,main)


  shatR,shatL,shatU,shatD,shatF,shatB = centralFluxGeneral(fvxR,fvxL,fvyU,fvyD,fvzF,fvzB,fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge)
  jumpR,jumpL,jumpU,jumpD,jumpF,jumpB = computeJump(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
  fvR2 = shatR - 6.*main.mu*jumpR*3**2/main.dx
  fvL2 = shatL - 6.*main.mu*jumpL*3**2/main.dx
  fvU2 = shatU - 6.*main.mu*jumpU*3**2/main.dy
  fvD2 = shatD - 6.*main.mu*jumpD*3**2/main.dy
  fvF2 = shatF - 6.*main.mu*jumpF*3**2/main.dz
  fvB2 = shatB - 6.*main.mu*jumpB*3**2/main.dz
  fvRIG11 = faceIntegrateGlob(main,fvRG11,main.w1,main.w2,main.weights1,main.weights2) 
  fvLIG11 = faceIntegrateGlob(main,fvLG11,main.w1,main.w2,main.weights1,main.weights2)  
  fvRIG21 = faceIntegrateGlob(main,fvRG21,main.wp1,main.w2,main.weights1,main.weights2) 
  fvLIG21 = faceIntegrateGlob(main,fvLG21,main.wp1,main.w2,main.weights1,main.weights2) 
  fvRIG31 = faceIntegrateGlob(main,fvRG31,main.w1,main.wp2,main.weights1,main.weights2) 
  fvLIG31 = faceIntegrateGlob(main,fvLG31,main.w1,main.wp2,main.weights1,main.weights2) 

  fvUIG12 = faceIntegrateGlob(main,fvUG12,main.wp0,main.w2,main.weights0,main.weights2)  
  fvDIG12 = faceIntegrateGlob(main,fvDG12,main.wp0,main.w2,main.weights0,main.weights2)  
  fvUIG22 = faceIntegrateGlob(main,fvUG22,main.w0,main.w2,main.weights0,main.weights2)  
  fvDIG22 = faceIntegrateGlob(main,fvDG22,main.w0,main.w2,main.weights0,main.weights2)  
  fvUIG32 = faceIntegrateGlob(main,fvUG32,main.w0,main.wp2,main.weights0,main.weights2)  
  fvDIG32 = faceIntegrateGlob(main,fvDG32,main.w0,main.wp2,main.weights0,main.weights2)  

  fvFIG13 = faceIntegrateGlob(main,fvFG13,main.wp0,main.w1,main.weights0,main.weights1)   
  fvBIG13 = faceIntegrateGlob(main,fvBG13,main.wp0,main.w1,main.weights0,main.weights1)  
  fvFIG23 = faceIntegrateGlob(main,fvFG23,main.w0,main.wp1,main.weights0,main.weights1)  
  fvBIG23 = faceIntegrateGlob(main,fvBG23,main.w0,main.wp1,main.weights0,main.weights1)  
  fvFIG33 = faceIntegrateGlob(main,fvFG33,main.w0,main.w1,main.weights0,main.weights1)  
  fvBIG33 = faceIntegrateGlob(main,fvBG33,main.w0,main.w1,main.weights0,main.weights1)  

  fvR2I = faceIntegrateGlob(main,fvR2,main.w1,main.w2,main.weights1,main.weights2)    
  fvL2I = faceIntegrateGlob(main,fvL2,main.w1,main.w2,main.weights1,main.weights2)  
  fvU2I = faceIntegrateGlob(main,fvU2,main.w0,main.w2,main.weights0,main.weights2)  
  fvD2I = faceIntegrateGlob(main,fvD2,main.w0,main.w2,main.weights0,main.weights2)  
  fvF2I = faceIntegrateGlob(main,fvF2,main.w0,main.w1,main.weights0,main.weights1)
  fvB2I = faceIntegrateGlob(main,fvB2,main.w0,main.w1,main.weights0,main.weights1)
#  print( np.linalg.norm(fvFIG13) , np.linalg.norm(fvBIG13) ,np.linalg.norm(fvFIG23), np.linalg.norm(fvBIG23),  np.linalg.norm(fvFIG33),np.linalg.norm(fvBIG33), np.linalg.norm(fvB2I),np.linalg.norm(fvF2I))
  return fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I


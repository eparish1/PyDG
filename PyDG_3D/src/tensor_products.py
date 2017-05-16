import numpy as np

def reconstructU_einsum(main,var):
  var.u[:] = 0.
  #var.u = np.einsum('lmpq,klmij->kpqij',main.w[:,None,:,None]*main.w[None,:,None,:],var.a) ## this is actually much slower than the two line code
  tmp =  np.einsum('rn,zpqr...->zpqn...',main.w,var.a)
  tmp = np.einsum('qm,zpqn...->zpmn...',main.w,tmp)
  var.u = np.einsum('pl,zpmn...->zlmn...',main.w,tmp)

def reconstructUGeneral_einsum(main,a):
  tmp =  np.einsum('rn,zpqr...->zpqn...',main.w,a)
  tmp = np.einsum('qm,zpqn...->zpmn...',main.w,tmp)
  return np.einsum('pl,zpmn...->zlmn...',main.w,tmp)


def reconstructU(main,var):
  var.u[:] = 0.
  tmp = np.tensordot(var.a,main.w0,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0])) 
#  var.u = np.swapaxes( np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5), 3, 6)
  var.u = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

def reconstructUGeneral(main,a):
  tmp = np.tensordot(a,main.w0,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0])) 
  #return np.swapaxes( np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5), 3, 6)
  return np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)


def reconstructEdgesGeneral(a,main):
  nvars = np.shape(a)[0]
  aR = np.einsum('zpqrijk->zqrijk',a)
  aL = np.einsum('zpqrijk->zqrijk',a*main.altarray0[None,:,None,None,None,None,None])

  aU = np.einsum('zpqrijk->zprijk',a)
  aD = np.einsum('zpqrijk->zprijk',a*main.altarray1[None,None,:,None,None,None,None])

  aF = np.einsum('zpqrijk->zpqijk',a)
  aB = np.einsum('zpqrijk->zpqijk',a*main.altarray2[None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uR = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uL = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uU = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uD = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uF = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uB = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  return uR,uL,uU,uD,uF,uB



def reconstructEdgesGeneral_einsum(a,main):
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
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray0[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray1[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray2[None,None,None,:,None,None])
#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uR = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uL = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uU = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uD = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uF = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uB = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  return uR,uL,uU,uD,uF,uB

def reconstructEdgeEdgesGeneral_einsum(main):
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


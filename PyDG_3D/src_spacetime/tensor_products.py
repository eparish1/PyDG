import numpy as np

def diffU_tensordot(a,main):
  tmp = np.tensordot(a,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  ux = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)
 
  tmpu = np.tensordot(a,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmpu,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uy = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

  #tmp = np.tensordot(a,main.w,axes=([1],[0]))
  tmp = np.tensordot(tmpu,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uz = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

  ux *= 2./main.dx2[None,None,None,None,None,:,None,None,None]
  uy *= 2./main.dy2[None,None,None,None,None,None,:,None,None]
  uz *= 2./main.dz2[None,None,None,None,None,None,None,:,None]
  return ux,uy,uz



def diffUXEdge_edge_tensordot(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge*main.wpedge0[:,1][None,:,None,None,None,None,None,None])
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.wpedge0[:,0][None,:,None,None,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray1[None,None,:,None,None,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray2[None,None,None,:,None,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxR = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxL = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aU,main.wp0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxU = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aD,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxD = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aF,main.wp0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aB,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  return uxR,uxL,uxU,uxD,uxF,uxB

def diffUYEdge_edge_tensordot(main):
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


def diffUZEdge_edge_tensordot(main):
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



def diffUX_edge_tensordot(a,main):
  aR = np.einsum('zpqr...->zqr...',a*main.wpedge0[:,1][None,:,None,None,None,None,None,None,None])
  aL = np.einsum('zpqr...->zqr...',a*main.wpedge0[:,0] [None,:,None,None,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray1[None,None,:,None,None,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray2[None,None,None,:,None,None,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]

  tmp = np.tensordot(aU,main.wp0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]
  tmp = np.tensordot(aD,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]

  tmp = np.tensordot(aF,main.wp0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]
  tmp = np.tensordot(aB,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]
  return uxR,uxL,uxU,uxD,uxF,uxB

def diffUY_edge_tensordot(a,main):
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray0[None,:,None,None,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a*main.wpedge1[:,1][None,None,:,None,None,None,None,None,None])
  aD = np.einsum('zpqr...->zpr...',a*main.wpedge1[:,0][None,None,:,None,None,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray2[None,None,None,:,None,None,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.wp1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]
  tmp = np.tensordot(aL,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]
  return uyR,uyL,uyU,uyD,uyF,uyB



def diffUZ_edge_tensordot(a,main):
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray0[None,:,None,None,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray1[None,None,:,None,None,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a*main.wpedge2[:,1][None,None,None,:,None,None,None,None,None])
  aB = np.einsum('zpqr...->zpq...',a*main.wpedge2[:,0][None,None,None,:,None,None,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]
  return uzR,uzL,uzU,uzD,uzF,uzB




def diffU_einsum(a,main):
  tmp =  np.einsum('rn,zpqr...->zpqn...',main.w,a) #reconstruct along third axis 
  tmp2 = np.einsum('qm,zpqn...->zpmn...',main.w,tmp) #reconstruct along second axis
  ux = np.einsum('pl,zpmn...->zlmn...',main.wp,tmp2) # get ux by differentiating along the first axis

  tmp2 = np.einsum('qm,zpqn...->zpmn...',main.wp,tmp) #diff tmp along second axis
  uy = np.einsum('pl,zpmn...->zlmn...',main.w,tmp2) # get uy by reconstructing along the first axis

  tmp =  np.einsum('rn,zpqr...->zpqn...',main.wp,a) #diff along third axis 
  tmp = np.einsum('qm,zpqn...->zpmn...',main.w,tmp) #reconstruct along second axis
  uz = np.einsum('pl,zpmn...->zlmn...',main.w,tmp) # reconstruct along the first axis

  ux *= 2./main.dx2[None,None,None,None,:,None,None]
  uy *= 2./main.dy2[None,None,None,None,None,:,None]
  uz *= 2./main.dz2[None,None,None,None,None,None,:]

  return ux,uy,uz

def diffUXEdge_edge_einsum(main):
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


def diffUX_edge_einsum(a,main):
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

def diffUYEdge_edge_einsum(main):
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




def diffUY_edge_einsum(a,main):
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



def diffUZEdge_edge_einsum(main):
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




def diffUZ_edge_einsum(a,main):
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


def volIntegrate(weights0,weights1,weights2,f):
  return  np.einsum('zpqrijk->zijk',weights0[None,:,None,None,None,None,None]*weights1[None,None,:,None,None,None,None]*weights2[None,None,None,:,None,None,None]*f[:,:,:,:,:])


def volIntegrateGlob_tensordot(main,f,w0,w1,w2,w3):
  tmp = np.tensordot(main.weights0[None,:,None,None,None,None,None,None,None]*f  ,w0,axes=([1],[1]))
  tmp = np.tensordot(main.weights1[None,:,None,None,None,None,None,None,None]*tmp,w1,axes=([1],[1]))
  tmp = np.tensordot(main.weights2[None,:,None,None,None,None,None,None,None]*tmp,w2,axes=([1],[1]))
  tmp = np.tensordot(main.weights3[None,:,None,None,None,None,None,None,None]*tmp,w3,axes=([1],[1]))
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

def volIntegrateGlob_tensordot_collocate(main,f,w0,w1,w2,w3):
  tmp = np.tensordot(main.weights0_c[None,:,None,None,None,None,None,None,None]*f  ,w0,axes=([1],[1]))
  tmp = np.tensordot(main.weights1_c[None,:,None,None,None,None,None,None,None]*tmp,w1,axes=([1],[1]))
  tmp = np.tensordot(main.weights1_c[None,:,None,None,None,None,None,None,None]*tmp,w2,axes=([1],[1]))
  tmp = np.tensordot(main.weights3_c[None,:,None,None,None,None,None,None,None]*tmp,w3,axes=([1],[1]))
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)


def volIntegrateGlob_einsum(main,f,w1,w2,w3):
  tmp = np.einsum('nr,zpqrijk->zpqnijk',w3,main.weights2[None,None,None,:,None,None,None]*f)
  tmp = np.einsum('mq,zpqnijk->zpmnijk',w2,main.weights1[None,None,:,None,None,None,None]*tmp)
  return np.einsum('lp,zpmnijk->zlmnijk',w1,main.weights0[None,:,None,None,None,None,None]*tmp)

def faceIntegrate(weights,f):
  return np.einsum('zpqijk->zijk',weights[None,:,None,None,None,None]*weights[None,None,:,None,None,None]*f)

def faceIntegrateGlob_tensordot(main,f,w1,w2,w3,weights1,weights2,weights3):
  tmp = np.tensordot(weights1[None,:,None,None,None,None,None,None]*f,w1,axes=([1],[1]))
  tmp = np.tensordot(weights2[None,:,None,None,None,None,None,None]*tmp,w2,axes=([1],[1]))
  tmp = np.tensordot(weights3[None,:,None,None,None,None,None,None]*tmp,w3,axes=([1],[1]))
  #return np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5)
  return np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2),-1,3)


def faceIntegrateGlob_einsum(main,f,w1,w2,weights0,weights1):
  tmp = np.einsum('nr,zqrijk->zqnijk',w2,weights1[None,None,:,None,None,None]*f)
  return np.einsum('mq,zqnijk->zmnijk',w1,weights0[None,:,None,None,None,None]*tmp)



def reconstructU_einsum(main,var):
  var.u[:] = 0.
  #var.u = np.einsum('lmpq,klmij->kpqij',main.w[:,None,:,None]*main.w[None,:,None,:],var.a) ## this is actually much slower than the two line code
  tmp =  np.einsum('so,zpqrs...->zpqro...',main.w3,var.a)
  tmp =  np.einsum('rn,zpqro...->zpqno...',main.w2,var.a)
  tmp = np.einsum('qm,zpqno...->zpmno...',main.w1,tmp)
  var.u = np.einsum('pl,zpmno...->zlmno...',main.w0,tmp)

def reconstructUGeneral_einsum(main,a):
  tmp =  np.einsum('so,zpqrs...->zpqro...',main.w3,a)
  tmp =  np.einsum('rn,zpqro...->zpqno...',main.w2,a)
  tmp = np.einsum('qm,zpqno...->zpmno...',main.w1,tmp)
  return np.einsum('pl,zpmno...->zlmno...',main.w0,tmp)


def reconstructU_tensordot(main,var):
  var.u[:] = 0.
  tmp = np.tensordot(var.a,main.w0,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0])) 
#  var.u = np.swapaxes( np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5), 3, 6)
  var.u = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)


def reconstructUGeneral_tensordot(main,a):
  tmp = np.tensordot(a,main.w0,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0])) 
  #return np.swapaxes( np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5), 3, 6)
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4 )




def reconstructEdgesGeneral_tensordot(a,main):
  nvars = np.shape(a)[0]
  aR = np.einsum('zpqr...->zqr...',a)
  #aL = np.einsum('zpqr...->zqr...',a*main.altarray0[None,:,None,None,None,None,None])
  aL = np.tensordot(main.altarray0,a,axes=([0],[1]) )
  
  aU = np.einsum('zpqr...->zpr...',a)
  #aD = np.einsum('zpqr...->zpr...',a*main.altarray1[None,None,:,None,None,None,None])
  aD = np.tensordot(main.altarray1,a,axes=([0],[2]) )

  aF = np.einsum('zpqr...->zpq...',a)
#  aB = np.einsum('zpqr...->zpq...',a*main.altarray2[None,None,None,:,None,None,None])
  aB = np.tensordot(main.altarray2,a,axes=([0],[3]) )

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uR = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uL = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uU = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uD = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uF = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uB = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  return uR,uL,uU,uD,uF,uB


def reconstructEdgesGeneralTime_tensordot(a,main):
  nvars = np.shape(a)[0]
  aFuture = np.einsum('zpqrl...->zpqr...',a)
  aPast = np.tensordot(main.altarray3,a,axes=([0],[4]) )

  tmp = np.tensordot(aFuture,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uFuture = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aPast,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uPast = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  return uFuture,uPast


def reconstructEdgesGeneral_einsum(a,main):
  nvars = np.shape(a)[0]
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray0[None,:,None,None,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray1[None,None,:,None,None,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray2[None,None,None,:,None,None,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('so,zqrs...->zqro...',main.w3,aR)
  tmp = np.einsum('rn,zqro...->zqno...',main.w2,tmp)
  uR  = np.einsum('qm,zqno...->zmno...',main.w1,tmp)
  tmp = np.einsum('so,zqrs...->zqro...',main.w3,aL)
  tmp = np.einsum('rn,zqro...->zqno...',main.w2,tmp)
  uL  = np.einsum('qm,zqno...->zmno...',main.w1,tmp)

  tmp = np.einsum('so,zprs...->zpro...',main.w3,aU)
  tmp = np.einsum('rn,zpro...->zpno...',main.w2,tmp)
  uU  = np.einsum('pl,zpno...->zlno...',main.w0,tmp)
  tmp = np.einsum('so,zprs...->zpro...',main.w3,aD)
  tmp = np.einsum('rn,zpro...->zpno...',main.w2,tmp)
  uD  = np.einsum('pl,zpno...->zlno...',main.w0,tmp)

  tmp = np.einsum('so,zpqs...->zpqo...',main.w3,aF)
  tmp = np.einsum('qm,zpqo...->zpmo...',main.w1,tmp)
  uF  = np.einsum('pl,zpmo...->zlmo...',main.w0,tmp)
  tmp = np.einsum('so,zpqs...->zpqo...',main.w3,aB)
  tmp = np.einsum('qm,zpqo...->zpmo...',main.w1,tmp)
  uB  = np.einsum('pl,zpmo...->zlmo...',main.w0,tmp)
  return uR,uL,uU,uD,uF,uB


def reconstructEdgeEdgesGeneral_tensordot(main):
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





































def reconstructU_entropy(main,var):
  var.u[:] = 0.
  tmp = np.tensordot(var.a,main.w0,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0])) 
  v = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  var.u[:] = entropy_to_conservative(v)

def entropy_to_conservative(v):
  gamma = 1.4
  gamma1   = gamma-1.0      # gamma-1
  igamma1  = 1.0/gamma1    # 1/(gamma-1)
  gmogm1   = gamma*igamma1 #  gamma/(gamma-1) 
  iu4 = 1./v[4]
  u = -iu4*v[1]
  v = -iu4*v[2]
  w = -iu4*v[3]
  t0 = -0.5*iu4*(+u[1]**2+u[2]**2+u[3]**2) # 0.5*rho*v^2/p
  t1 = v[0] - gmogm1+t0 # -s/(gamma-1)
  t2 = exp(-igamma1*log(-v[4])) # pow(-u4,-igamma1)
  t3 = exp(t1)
  rho = t2*t3          
  H = -iu4*(gmogm1+t0) 
  E = (H+iu4)          # total enery
  rhou = rho*u         # x-momentum
  rhov = rho*v         # y-momentum
  rhow = rho*w         # z-momentum
  rhoE = rho*E
  u = np.zeros(np.shape(v))
  u[0] = rho
  u[1] = rhou
  u[2] = rhov
  u[3] = rhow
  u[4] = rhoE
  return u
 

def reconstructUGeneral_entropy(main,a):
  tmp = np.tensordot(a,main.w0,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0])) 
  v = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  ## This is the same, but first we need to make the transformation from entropy variables 
  ## to conservative variables (adopted from murman)
  return entropy_to_conservative(v)






def reconstructEdgesGeneral_entropy(a,main):
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
  uR = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uL = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uU = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uD = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uF = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uB = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  return uR,uL,uU,uD,uF,uB


def reconstructEdgeEdgesGeneral_entropy(main):
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
  uR = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uL = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uU = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uD = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uF = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uB = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  return uR,uL,uU,uD,uF,uB




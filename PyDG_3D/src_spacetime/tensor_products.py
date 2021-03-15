import numpy as np
import numexpr as ne

def applyMassMatrix(main,RHS):
  RHS[:] = np.sum(main.Minv[None]*RHS[:,None,None,None,None],axis=(5,6,7,8) )
  '''
  Note that einsum doesn't work with pyadolc, the above sum does but is quite slow.
  Replace with tensordot at some time
  '''
  #RHS[:] = np.einsum('abcdpqrs...,zpqrs...->zabcd...',main.Minv,RHS)

def applyMassMatrix_orthogonal(main,RHS):
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arr3= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arr0[:,None,None,None] + 1.)*(2.*ord_arr1[None,:,None,None] + 1.)*(2.*ord_arr2[None,None,:,None]+1.)*(2.*ord_arr3[None,None,None,:] + 1. )/16.
  RHS[:] = RHS*scale[None,:,:,:,:,None,None,None,None]/main.Jdet[None,0,0,0,None,None,None,None,:,:,:,None]


def applyVolIntegralAdjoint(region,f1,f2,f3,RHS):
  f = f1 + f2 + f3
  f *= region.Jdet[None,:,:,:,None,:,:,:,None]
  RHS[:] += region.basis.volIntegrateGlob(region,f,region.w0,region.w1,region.w2,region.w3)

def applyVolIntegralAdjoint_indices(region,f1,f2,f3,RHS,cell_ijk):
  f = f1 + f2 + f3
  f *= region.Jdet[None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  RHS[:] += volIntegrateGlob_tensordot_indices(region,f,region.w0,region.w1,region.w2,region.w3)
  return RHS

def applyVolIntegral(region,f1,f2,f3,RHS):
  f = f1*region.Jinv[0,0][None,:,:,:,None,:,:,:,None]
  f += f2*region.Jinv[0,1][None,:,:,:,None,:,:,:,None]
  f += f3*region.Jinv[0,2][None,:,:,:,None,:,:,:,None]
  f *= region.Jdet[None,:,:,:,None,:,:,:,None]
  RHS[:] += region.basis.volIntegrateGlob(region,f,region.wp0,region.w1,region.w2,region.w3)


  f = f1*region.Jinv[1,0][None,:,:,:,None,:,:,:,None]
  f += f2*region.Jinv[1,1][None,:,:,:,None,:,:,:,None]
  f += f3*region.Jinv[1,2][None,:,:,:,None,:,:,:,None]
  f *= region.Jdet[None,:,:,:,None,:,:,:,None]
  RHS[:] += region.basis.volIntegrateGlob(region,f,region.w0,region.wp1,region.w2,region.w3)

  f = f1*region.Jinv[2,0][None,:,:,:,None,:,:,:,None]
  f += f2*region.Jinv[2,1][None,:,:,:,None,:,:,:,None]
  f += f3*region.Jinv[2,2][None,:,:,:,None,:,:,:,None]
  f *= region.Jdet[None,:,:,:,None,:,:,:,None]
  RHS[:] += region.basis.volIntegrateGlob(region,f,region.w0,region.w1,region.wp2,region.w3)

def applyVolIntegral_indices(region,f1,f2,f3,RHS,cell_ijk):
  f = f1*region.Jinv[0,0][None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f += f2*region.Jinv[0,1][None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f += f3*region.Jinv[0,2][None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f *= region.Jdet[None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  RHS[:] += volIntegrateGlob_tensordot_indices(region,f,region.wp0,region.w1,region.w2,region.w3)


  f = f1*region.Jinv[1,0][None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f += f2*region.Jinv[1,1][None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f += f3*region.Jinv[1,2][None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f *= region.Jdet[None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  RHS[:] += volIntegrateGlob_tensordot_indices(region,f,region.w0,region.wp1,region.w2,region.w3)

  f = f1*region.Jinv[2,0][None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f += f2*region.Jinv[2,1][None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f += f3*region.Jinv[2,2][None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f *= region.Jdet[None,:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  RHS[:] += volIntegrateGlob_tensordot_indices(region,f,region.w0,region.w1,region.wp2,region.w3)

  return RHS

## This doesn't work, should figure it out sometime
def applyVolIntegral_indices_2(region,f1,f2,f3,RHS,cell_ijk):
  print(np.shape(f1),np.shape(region.Jinv[0,0][None,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]))
  f = f1*region.Jinv[0,0][None,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f += f2*region.Jinv[0,1][None,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f += f3*region.Jinv[0,2][None,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  f *= region.Jdet[None,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
  RHS[:] += volIntegrateGlob_tensordot_indices2(region,f,region.wp0,region.w1,region.w2,region.w3)


#  f = f1*region.Jinv[1,0][:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
#  f += f2*region.Jinv[1,1][:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
#  f += f3*region.Jinv[1,2][:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
#  f *= region.Jdet[:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
#  RHS[:] += volIntegrateGlob_tensordot_indices(region,f,region.w0,region.wp1,region.w2,region.w3)
#
#  f = f1*region.Jinv[2,0][:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
#  f += f2*region.Jinv[2,1][:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
#  f += f3*region.Jinv[2,2][:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
#  f *= region.Jdet[:,:,:,None,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
#  RHS[:] += volIntegrateGlob_tensordot_indices(region,f,region.w0,region.w1,region.wp2,region.w3)

  return RHS



def applyVolIntegral_numexpr(main,f1,f2,f3,RHS):
  J1 = main.Jinv[0,0][None,:,:,:,None,:,:,:,None]
  J2 = main.Jinv[0,1][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[0,2][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[0,2][None,:,:,:,None,:,:,:,None]
  JD = main.Jdet[None,:,:,:,None,:,:,:,None]
  f = ne.evaluate("JD*(f1*J1 + f2*J2 + f3*J3)")
  f = main.basis.volIntegrateGlob(main,f,main.wp0,main.w1,main.w2,main.w3)
  ne.evaluate("RHS+f",out=RHS)

  J1 = main.Jinv[1,0][None,:,:,:,None,:,:,:,None]
  J2 = main.Jinv[1,1][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[1,2][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[1,2][None,:,:,:,None,:,:,:,None]
  #f = ne.re_evaluate()#"JD*(a*J1 + b*J2 + c*J3)")
  f = ne.evaluate("JD*(f1*J1 + f2*J2 + f3*J3)")
  f = main.basis.volIntegrateGlob(main,f,main.w0,main.wp1,main.w2,main.w3)
  ne.evaluate("RHS+f",out=RHS)

  J1 = main.Jinv[2,0][None,:,:,:,None,:,:,:,None]
  J2 = main.Jinv[2,1][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[2,2][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[2,2][None,:,:,:,None,:,:,:,None]
  #f = ne.re_evaluate()#"JD*(a*J1 + b*J2 + c*J3)")
  f = ne.evaluate("JD*(f1*J1 + f2*J2 + f3*J3)")
  f = main.basis.volIntegrateGlob(main,f,main.w0,main.w1,main.wp2,main.w3)
  ne.evaluate("RHS+f",out=RHS)

def applyVolIntegral_numexpr_orthogonal(main,f1,f2,f3,RHS):
  J1 = main.Jinv[0,0][None,:,:,:,None,:,:,:,None]
  JD = main.Jdet[None,:,:,:,None,:,:,:,None]
  f = ne.evaluate("JD*(f1*J1)")
  RHS[:] += main.basis.volIntegrateGlob(main,f,main.wp0,main.w1,main.w2,main.w3)

  J2 = main.Jinv[1,1][None,:,:,:,None,:,:,:,None]
  f = ne.evaluate("JD*(f2*J2)")
  RHS[:] += main.basis.volIntegrateGlob(main,f,main.w0,main.wp1,main.w2,main.w3)

  J3 = main.Jinv[2,2][None,:,:,:,None,:,:,:,None]
  f = ne.evaluate("JD*(f3*J3)")
  RHS[:] += main.basis.volIntegrateGlob(main,f,main.w0,main.w1,main.wp2,main.w3)



def diffU_tensordot_sample(a,main,Jinv):
  tmp = np.tensordot(a,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzeta = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)
  tmpu = np.tensordot(a,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmpu,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  ueta = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

  #tmp = np.tensordot(a,main.w,axes=([1],[0]))
  tmp = np.tensordot(tmpu,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umu = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

  ux = uzeta*Jinv[0,0][None,:,:,:,None] + ueta*Jinv[1,0][None,:,:,:,None] + umu*Jinv[2,0][None,:,:,:,None]
  uy = uzeta*Jinv[0,1][None,:,:,:,None] + ueta*Jinv[1,1][None,:,:,:,None] + umu*Jinv[2,1][None,:,:,:,None]
  uz = uzeta*Jinv[0,2][None,:,:,:,None] + ueta*Jinv[1,2][None,:,:,:,None] + umu*Jinv[2,2][None,:,:,:,None]

  return ux,uy,uz




def diffU_tensordot(a,main):
  tmp = np.tensordot(a,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzeta = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)
  tmpu = np.tensordot(a,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmpu,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  ueta = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

  #tmp = np.tensordot(a,main.w,axes=([1],[0]))
  tmp = np.tensordot(tmpu,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umu = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

#  uzeta *= 2./main.dx2[None,None,None,None,None,:,None,None,None]
#  ueta *= 2./main.dy2[None,None,None,None,None,None,:,None,None]
#  umu *= 2./main.dz2[None,None,None,None,None,None,None,:,None]
  
  ux = uzeta*main.Jinv[0,0][None,:,:,:,None,:,:,:,None] + ueta*main.Jinv[1,0][None,:,:,:,None,:,:,:,None] + umu*main.Jinv[2,0][None,:,:,:,None,:,:,:,None]
  uy = uzeta*main.Jinv[0,1][None,:,:,:,None,:,:,:,None] + ueta*main.Jinv[1,1][None,:,:,:,None,:,:,:,None] + umu*main.Jinv[2,1][None,:,:,:,None,:,:,:,None]
  uz = uzeta*main.Jinv[0,2][None,:,:,:,None,:,:,:,None] + ueta*main.Jinv[1,2][None,:,:,:,None,:,:,:,None] + umu*main.Jinv[2,2][None,:,:,:,None,:,:,:,None]

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



def diffUXYZ_edge_tensordot_hyper(a,main,Jinv):
  aR = np.tensordot(a,main.wpedge0[:,1],axes=([1],[0]))
  aL = np.tensordot(a,main.wpedge0[:,0],axes=([1],[0]))

  aU = np.sum(a,axis=2)
  aD = np.tensordot(a,main.altarray1[:],axes=([2],[0]))

  aF = np.sum(a,axis=3)
  aB = np.tensordot(a,main.altarray2[:],axes=([3],[0]))



#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1[:,:,None,None,None,None]*main.w2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  #tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzetaR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aL,main.w1[:,:,None,None,None,None]*main.w2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzetaL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aU,main.wp0[:,:,None,None,None,None]*main.w2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aU,main.wp0,axes=([1],[0])) #reconstruct in x and z 
  #tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzetaU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aD,main.wp0[:,:,None,None,None,None]*main.w2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aD,main.wp0,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzetaD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)


  tmp = np.tensordot(aF,main.wp0[:,:,None,None,None,None]*main.w1[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aF,main.wp0,axes=([1],[0])) #reconstruct in x and y 
  #tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzetaF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aB,main.wp0[:,:,None,None,None,None]*main.w1[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aB,main.wp0,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzetaB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)


  aR = np.sum(a,axis=1)
  aL = np.tensordot(a,main.altarray0,axes=([1],[0]))

  aU = np.tensordot(a,main.wpedge1[:,1],axes=([2],[0]))
  aD = np.tensordot(a,main.wpedge1[:,0],axes=([2],[0]))

  #aF = np.sum(a,axis=3)
  #aB = np.tensordot(a,main.altarray2[:],axes=([3],[0]))



#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.wp1[:,:,None,None,None,None]*main.w2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aR,main.wp1,axes=([1],[0])) #reconstruct in y and z
  #tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uetaR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aL,main.wp1[:,:,None,None,None,None]*main.w2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aL,main.wp1,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uetaL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aU,main.w0[:,:,None,None,None,None]*main.w2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  #tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uetaU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aD,main.w0[:,:,None,None,None,None]*main.w2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uetaD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aF,main.w0[:,:,None,None,None,None]*main.wp1[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  #tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uetaF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aB,main.w0[:,:,None,None,None,None]*main.wp1[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uetaB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)


  #aR = np.sum(a,axis=1)
  #aL = np.tensordot(a,main.altarray0,axes=([1],[0]))

  aU = np.sum(a,axis=2)
  aD = np.tensordot(a,main.altarray1[:],axes=([2],[0]))

  aF = np.tensordot(a,main.wpedge2[:,1],axes=([3],[0]))
  aB = np.tensordot(a,main.wpedge2[:,0],axes=([3],[0]))


#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1[:,:,None,None,None,None]*main.wp2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  #tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umuR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aL,main.w1[:,:,None,None,None,None]*main.wp2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umuL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)


  tmp = np.tensordot(aU,main.w0[:,:,None,None,None,None]*main.wp2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  #tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umuU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aD,main.w0[:,:,None,None,None,None]*main.wp2[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umuD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aF,main.w0[:,:,None,None,None,None]*main.w1[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  #tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umuF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aB,main.w0[:,:,None,None,None,None]*main.w1[None,None,:,:,None,None]*main.w3[None,None,None,None,:],axes=([1,2,3],[0,2,4]))
  #tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  #tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umuB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)



  uxR = uzetaR*Jinv[0,0][None,-1,:,:,None] + uetaR*Jinv[1,0][None,-1,:,:,None] + umuR*Jinv[2,0][None,-1,:,:,None]
  uxL = uzetaL*Jinv[0,0][None,0,:,:,None] + uetaL*Jinv[1,0][None,0,:,:,None] + umuL*Jinv[2,0][None,0,:,:,None]
  uxU = uzetaU*Jinv[0,0][None,:,-1,:,None] + uetaU*Jinv[1,0][None,:,-1,:,None] + umuU*Jinv[2,0][None,:,-1,:,None]
  uxD = uzetaD*Jinv[0,0][None,:,0,:,None] + uetaD*Jinv[1,0][None,:,0,:,None] + umuD*Jinv[2,0][None,:,0,:,None]
  uxF = uzetaF*Jinv[0,0][None,:,:,-1,None] + uetaF*Jinv[1,0][None,:,:,-1,None] + umuF*Jinv[2,0][None,:,:,-1,None]
  uxB = uzetaB*Jinv[0,0][None,:,:,0,None] + uetaB*Jinv[1,0][None,:,:,0,None] + umuB*Jinv[2,0][None,:,:,0,None]


  uyR = uzetaR*Jinv[0,1][None,-1,:,:,None] + uetaR*Jinv[1,1][None,-1,:,:,None] + umuR*Jinv[2,1][None,-1,:,:,None]
  uyL = uzetaL*Jinv[0,1][None,0,:,:,None] + uetaL*Jinv[1,1][None,0,:,:,None] + umuL*Jinv[2,1][None,0,:,:,None]
  uyU = uzetaU*Jinv[0,1][None,:,-1,:,None] + uetaU*Jinv[1,1][None,:,-1,:,None] + umuU*Jinv[2,1][None,:,-1,:,None]
  uyD = uzetaD*Jinv[0,1][None,:,0,:,None] + uetaD*Jinv[1,1][None,:,0,:,None] + umuD*Jinv[2,1][None,:,0,:,None]
  uyF = uzetaF*Jinv[0,1][None,:,:,-1,None] + uetaF*Jinv[1,1][None,:,:,-1,None] + umuF*Jinv[2,1][None,:,:,-1,None]
  uyB = uzetaB*Jinv[0,1][None,:,:,0,None] + uetaB*Jinv[1,1][None,:,:,0,None] + umuB*Jinv[2,1][None,:,:,0,None]

  uzR = uzetaR*Jinv[0,2][None,-1,:,:,None] + uetaR*Jinv[1,2][None,-1,:,:,None] + umuR*Jinv[2,2][None,-1,:,:,None]
  uzL = uzetaL*Jinv[0,2][None,0,:,:,None] + uetaL*Jinv[1,2][None,0,:,:,None] + umuL*Jinv[2,2][None,0,:,:,None]
  uzU = uzetaU*Jinv[0,2][None,:,-1,:,None] + uetaU*Jinv[1,2][None,:,-1,:,None] + umuU*Jinv[2,2][None,:,-1,:,None]
  uzD = uzetaD*Jinv[0,2][None,:,0,:,None] + uetaD*Jinv[1,2][None,:,0,:,None] + umuD*Jinv[2,2][None,:,0,:,None]
  uzF = uzetaF*Jinv[0,2][None,:,:,-1,None] + uetaF*Jinv[1,2][None,:,:,-1,None] + umuF*Jinv[2,2][None,:,:,-1,None]
  uzB = uzetaB*Jinv[0,2][None,:,:,0,None] + uetaB*Jinv[1,2][None,:,:,0,None] + umuB*Jinv[2,2][None,:,:,0,None]

  return uxR,uxL,uxU,uxD,uxF,uxB , uyR,uyL,uyU,uyD,uyF,uyB , uzR,uzL,uzU,uzD,uzF,uzB




def diffUXYZ_edge_tensordot(a,main):
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
  uzetaR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzetaL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aU,main.wp0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzetaU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aD,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzetaD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aF,main.wp0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzetaF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aB,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzetaB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)


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
  uetaR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aL,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uetaL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uetaU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uetaD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uetaF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uetaB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

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
  umuR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umuL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umuU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umuD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umuF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umuB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)



  uxR = uzetaR*main.Jinv[0,0][None,-1,:,:,None,:,:,:,None] + uetaR*main.Jinv[1,0][None,-1,:,:,None,:,:,:,None] + umuR*main.Jinv[2,0][None,-1,:,:,None,:,:,:,None]    
  uxL = uzetaL*main.Jinv[0,0][None,0,:,:,None,:,:,:,None] + uetaL*main.Jinv[1,0][None,0,:,:,None,:,:,:,None] + umuL*main.Jinv[2,0][None,0,:,:,None,:,:,:,None]    
  uxU = uzetaU*main.Jinv[0,0][None,:,-1,:,None,:,:,:,None] + uetaU*main.Jinv[1,0][None,:,-1,:,None,:,:,:,None] + umuU*main.Jinv[2,0][None,:,-1,:,None,:,:,:,None]    
  uxD = uzetaD*main.Jinv[0,0][None,:,0,:,None,:,:,:,None] + uetaD*main.Jinv[1,0][None,:,0,:,None,:,:,:,None] + umuD*main.Jinv[2,0][None,:,0,:,None,:,:,:,None]    
  uxF = uzetaF*main.Jinv[0,0][None,:,:,-1,None,:,:,:,None] + uetaF*main.Jinv[1,0][None,:,:,-1,None,:,:,:,None] + umuF*main.Jinv[2,0][None,:,:,-1,None,:,:,:,None]    
  uxB = uzetaB*main.Jinv[0,0][None,:,:,0,None,:,:,:,None] + uetaB*main.Jinv[1,0][None,:,:,0,None,:,:,:,None] + umuB*main.Jinv[2,0][None,:,:,0,None,:,:,:,None]    


  uyR = uzetaR*main.Jinv[0,1][None,-1,:,:,None,:,:,:,None] + uetaR*main.Jinv[1,1][None,-1,:,:,None,:,:,:,None] + umuR*main.Jinv[2,1][None,-1,:,:,None,:,:,:,None]    
  uyL = uzetaL*main.Jinv[0,1][None,0,:,:,None,:,:,:,None] + uetaL*main.Jinv[1,1][None,0,:,:,None,:,:,:,None] + umuL*main.Jinv[2,1][None,0,:,:,None,:,:,:,None]    
  uyU = uzetaU*main.Jinv[0,1][None,:,-1,:,None,:,:,:,None] + uetaU*main.Jinv[1,1][None,:,-1,:,None,:,:,:,None] + umuU*main.Jinv[2,1][None,:,-1,:,None,:,:,:,None]    
  uyD = uzetaD*main.Jinv[0,1][None,:,0,:,None,:,:,:,None] + uetaD*main.Jinv[1,1][None,:,0,:,None,:,:,:,None] + umuD*main.Jinv[2,1][None,:,0,:,None,:,:,:,None]    
  uyF = uzetaF*main.Jinv[0,1][None,:,:,-1,None,:,:,:,None] + uetaF*main.Jinv[1,1][None,:,:,-1,None,:,:,:,None] + umuF*main.Jinv[2,1][None,:,:,-1,None,:,:,:,None]    
  uyB = uzetaB*main.Jinv[0,1][None,:,:,0,None,:,:,:,None] + uetaB*main.Jinv[1,1][None,:,:,0,None,:,:,:,None] + umuB*main.Jinv[2,1][None,:,:,0,None,:,:,:,None]    

  uzR = uzetaR*main.Jinv[0,2][None,-1,:,:,None,:,:,:,None] + uetaR*main.Jinv[1,2][None,-1,:,:,None,:,:,:,None] + umuR*main.Jinv[2,2][None,-1,:,:,None,:,:,:,None]    
  uzL = uzetaL*main.Jinv[0,2][None,0,:,:,None,:,:,:,None] + uetaL*main.Jinv[1,2][None,0,:,:,None,:,:,:,None] + umuL*main.Jinv[2,2][None,0,:,:,None,:,:,:,None]    
  uzU = uzetaU*main.Jinv[0,2][None,:,-1,:,None,:,:,:,None] + uetaU*main.Jinv[1,2][None,:,-1,:,None,:,:,:,None] + umuU*main.Jinv[2,2][None,:,-1,:,None,:,:,:,None]    
  uzD = uzetaD*main.Jinv[0,2][None,:,0,:,None,:,:,:,None] + uetaD*main.Jinv[1,2][None,:,0,:,None,:,:,:,None] + umuD*main.Jinv[2,2][None,:,0,:,None,:,:,:,None]    
  uzF = uzetaF*main.Jinv[0,2][None,:,:,-1,None,:,:,:,None] + uetaF*main.Jinv[1,2][None,:,:,-1,None,:,:,:,None] + umuF*main.Jinv[2,2][None,:,:,-1,None,:,:,:,None]    
  uzB = uzetaB*main.Jinv[0,2][None,:,:,0,None,:,:,:,None] + uetaB*main.Jinv[1,2][None,:,:,0,None,:,:,:,None] + umuB*main.Jinv[2,2][None,:,:,0,None,:,:,:,None]    

  return uxR,uxL,uxU,uxD,uxF,uxB , uyR,uyL,uyU,uyD,uyF,uyB , uzR,uzL,uzU,uzD,uzF,uzB



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
  nvars,orderx,ordery,orderz,ordert,Nelx,Nely,Nelz,Nelt = np.shape(a)
  order = np.array([orderx,ordery,orderz,ordert])
  ax = np.zeros((nvars,order[0],order[1],order[2],Nelx,Nely,Nelz,Nelt))
  ay = np.zeros((nvars,order[0],order[1],order[2],Nelx,Nely,Nelz,Nelt))
  az = np.zeros((nvars,order[0],order[1],order[2],Nelx,Nely,Nelz,Nelt))

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


def volIntegrate(weights0,weights1,weights2,weights3,f):
  return  np.sum(weights0[None,:,None,None,None,None,None,None,None]*weights1[None,None,:,None,None,None,None,None,None]*\
          weights2[None,None,None,:,None,None,None,None,None]*weights3[None,None,None,None,:,None,None,None,None]*f,axis=(1,2,3,4))


#def volIntegrate(weights0,weights1,weights2,weights3,f):
#  return  np.einsum('zpqrl...->z...',weights0[None,:,None,None,None,None,None,None,None]*weights1[None,None,:,None,None,None,None,None,None]*\
#          weights2[None,None,None,:,None,None,None,None,None]*weights3[None,None,None,None,:,None,None,None,None]*f)


### Volume integral for when we use specific indices.
def volIntegrateGlob_tensordot_indices2(main,f,w0,w1,w2,w3):
  nf = np.size(np.shape(f))
  tmp = np.rollaxis(np.tensordot(w0*main.weights0[:],f,axes=([0],[0])),0,nf)
  tmp = np.rollaxis(np.tensordot(w1*main.weights1[:],tmp,axes=([0],[0])) , 0 , nf)
  tmp = np.rollaxis(np.tensordot(w2*main.weights2[:],tmp,axes=([0],[0])) , 0 , nf)
  tmp = np.rollaxis(np.tensordot(w3*main.weights3[:],tmp,axes=([0],[0])) , 0 , nf)
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 0) , -3 , 1), -2, 2), -1, 3)


### Volume integral for when we use specific indices.
def volIntegrateGlob_tensordot_indices(main,f,w0,w1,w2,w3):
  nf = np.size(np.shape(f))
  tmp = np.rollaxis(np.tensordot(w0*main.weights0[None,:],f,axes=([1],[1])),0,nf)
  tmp = np.rollaxis(np.tensordot(w1*main.weights1[None,:],tmp,axes=([1],[1])) , 0 , nf)
  tmp = np.rollaxis(np.tensordot(w2*main.weights2[None,:],tmp,axes=([1],[1])) , 0 , nf)
  tmp = np.rollaxis(np.tensordot(w3*main.weights3[None,:],tmp,axes=([1],[1])) , 0 , nf)
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)


def volIntegrateGlob_tensordot(main,f,w0,w1,w2,w3):
  tmp = np.rollaxis(np.tensordot(w0*main.weights0[None,:],f,axes=([1],[1])),0,9)
  tmp = np.rollaxis(np.tensordot(w1*main.weights1[None,:],tmp,axes=([1],[1])) , 0 , 9)
  tmp = np.rollaxis(np.tensordot(w2*main.weights2[None,:],tmp,axes=([1],[1])) , 0 , 9)
  tmp = np.rollaxis(np.tensordot(w3*main.weights3[None,:],tmp,axes=([1],[1])) , 0 , 9)
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)



def volIntegrateGlob_einsumMM2(main,f,w0,w1,w2,w3):
  tmp = np.einsum('dos...,zpqrs...->zpqrdo...',w3[:,None]*w3[None,:]*main.weights3[None,None,:],f)
  tmp = np.einsum('cnr...,zpqrdo...->zpqcndo...',w2[:,None]*w2[None,:]*main.weights2[None,None,:],tmp)
  tmp = np.einsum('bmq...,zpqcndo...->zpbmcndo...',w1[:,None]*w1[None,:]*main.weights1[None,None,:],tmp)
  tmp = np.einsum('alp...,zpbmcndo...->zalbmcndo...',w0[:,None]*w0[None,:]*main.weights0[None,None,:],tmp)
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp, 7,9) , 5,8) ,3 , 7) , 1 , 6)

def volIntegrateGlob_einsumMM3(main,f,w0,w1,w2,w3):
  tmp = np.einsum('os...,zdpqras...->zdpqrao...',w3[None,:]*main.weights3[None,None,:],f[:,:,:,:,None]*main.w3[None,None,None,None,:,:,None,None,None,None])
  tmp = np.einsum('nr...,zdpqro...->zcdpqno...',w2[:,None]*w2[None,:]*main.weights2[None,None,:],tmp)
  tmp = np.einsum('mq...,zcdpqno...->zbcdpmno...',w1[:,None]*w1[None,:]*main.weights1[None,None,:],tmp)
  return np.einsum('lp...,zbcdpmno...->zabcdlmno...',w0[:,None]*w0[None,:]*main.weights0[None,None,:],tmp)


def volIntegrateGlob_einsumMM(main,f,w0,w1,w2,w3):
  tmp = np.einsum('os...,zabcdpqrs...->zabcdpqro...',w3*main.weights3[None,:],f)
  tmp = np.einsum('nr...,zabcdpqro...->zabcdpqno...',w2*main.weights2[None,:],tmp)
  tmp = np.einsum('mq...,zabcdpqno...->zabcdpmno...',w1*main.weights1[None,:],tmp)
  return np.einsum('lp...,zabcdpmno...->zabcdlmno...',w0*main.weights0[None,:],tmp)

def volIntegrateGlob_tensordotMM(main,f,w0,w1,w2,w3):
  tmp = np.rollaxis(np.tensordot(w0*main.weights0[None,:],f,axes=([1],[5])),0,13)
  tmp = np.rollaxis(np.tensordot(w1*main.weights1[None,:],tmp,axes=([1],[5])) , 0 , 13)
  tmp = np.rollaxis(np.tensordot(w2*main.weights2[None,:],tmp,axes=([1],[5])) , 0 , 13)
  tmp = np.rollaxis(np.tensordot(w3*main.weights3[None,:],tmp,axes=([1],[5])) , 0 , 13)
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 5) , -3 , 6), -2, 7), -1, 8)


def volIntegrateGlob_einsum(main,f,w0,w1,w2,w3):
  tmp = np.einsum('os...,zpqrs...->zpqro...',w3,main.weights3[None,None,None,None,:,None,None,None,None]*f)
  tmp = np.einsum('nr...,zpqro...->zpqno...',w2,main.weights2[None,None,None,:,None,None,None,None,None]*tmp)
  tmp = np.einsum('mq...,zpqno...->zpmno...',w1,main.weights1[None,None,:,None,None,None,None,None,None]*tmp)
  return np.einsum('lp...,zpmno...->zlmno...',w0,main.weights0[None,:,None,None,None,None,None,None,None]*tmp)



def faceIntegrateGlob_tensordot(main,f,w1,w2,w3,weights1,weights2,weights3):
  tmp = np.tensordot(f,w1*weights1[None,:],axes=([1],[1]))
  tmp = np.tensordot(tmp,w2*weights2[None,:],axes=([1],[1]))
  tmp = np.tensordot(tmp,w3*weights3[None,:],axes=([1],[1]))
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
  tmp = np.rollaxis(np.tensordot(main.w0,var.a,axes=([0],[1])) ,0,9)
  tmp = np.rollaxis(np.tensordot(main.w1,tmp,axes=([0],[1])) ,0,9)
  tmp = np.rollaxis(np.tensordot(main.w2,tmp,axes=([0],[1])) ,0,9)
  tmp = np.rollaxis(np.tensordot(main.w3,tmp,axes=([0],[1])) ,0,9)
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
  aR = np.sum(a,axis=1)
  #aR = np.einsum('zpqr...->zqr...',a)
  #aL = np.einsum('zpqr...->zqr...',a*main.altarray0[None,:,None,None,None,None,None])
  aL = np.tensordot(main.altarray0,a,axes=([0],[1]) )
  
  aU = np.sum(a,axis=2)
  #aU = np.einsum('zpqr...->zpr...',a)
  #aD = np.einsum('zpqr...->zpr...',a*main.altarray1[None,None,:,None,None,None,None])
  aD = np.tensordot(main.altarray1,a,axes=([0],[2]) )

  aF = np.sum(a,axis=3)
#  aF = np.einsum('zpqr...->zpq...',a)
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




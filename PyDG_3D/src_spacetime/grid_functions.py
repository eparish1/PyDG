import numpy as np
import numpy.linalg 
from tensor_products import *
def getMassMatrix(main):
  M = np.zeros((main.order[0],main.order[1],main.order[2],main.order[3],\
                main.order[0],main.order[1],main.order[2],main.order[3],\
                main.Npx,main.Npy,main.Npz,1 ) )

  f = main.w0[:,None,None,None,:,None,None,None]*main.w1[None,:,None,None,None,:,None,None]\
     *main.w2[None,None,:,None,None,None,:,None]*main.w3[None,None,None,:,None,None,None,:]
  norder = main.order[0]*main.order[1]*main.order[2]*main.order[3]
  M2 = np.zeros((norder,norder,\
                main.Npx,main.Npy,main.Npz,1 ) )
  count = 0
  for i in range(0,main.order[0]):
    for j in range(0,main.order[1]):
      for k in range(0,main.order[2]):
        for l in range(0,main.order[3]):
          #M2[count] =np.reshape( volIntegrateGlob_einsum_2(main,(f*f[i,j,k,l])[None,:,:,:,:,:,:,:,:,None,None,None,None]*main.Jdet[None,None,None,None,None,:,:,:,None,:,:,:,None]) , np.shape(M2[0]))
          M2[count] =np.reshape( volIntegrateGlob_tensordot(main,f[i,j,k,l][None,:,:,:,:,None,None,None,None]*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.w3) , np.shape(M2[0]))
          count += 1
  tmp = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis(M2,2,0),3,1),4,2),5,3)
  tmp = np.linalg.inv(tmp)
  tmp = np.rollaxis(np.rollaxis(tmp,4,0),5,1)
  count += 1
  return np.reshape(tmp,np.shape(M))

def getGlobGrid(main,x,y,z,zeta0,zeta1,zeta2):
#  dx = x[1] - x[0]
#  dy = y[1] - y[0]
#  dz = z[1] - z[0]
  Npx,Npy,Npz = np.shape(x)
  nqx = np.size(zeta0)
  nqy = np.size(zeta1)
  nqz = np.size(zeta2)
  xG = np.zeros((nqx,nqy,nqz,Npx-1,Npy-1,Npz-1))
  yG = np.zeros((nqx,nqy,nqz,Npx-1,Npy-1,Npz-1))
  zG = np.zeros((nqx,nqy,nqz,Npx-1,Npy-1,Npz-1))
  zeta,eta,mu = np.meshgrid(zeta0,zeta1,zeta2,indexing='ij')
  zeta = zeta[:,:,:,None,None,None]
  eta = eta[:,:,:,None,None,None]
  mu = mu[:,:,:,None,None,None]
  x0 = x[None,None,None,0:-1,0:-1,0:-1]
  x1 = x[None,None,None,1:: ,0:-1,0:-1]
  x2 = x[None,None,None,0:-1,1:: ,0:-1]
  x3 = x[None,None,None,1:: ,1:: ,0:-1]
  x4 = x[None,None,None,0:-1,0:-1,1:: ]
  x5 = x[None,None,None,1:: ,0:-1,1:: ]
  x6 = x[None,None,None,0:-1,1:: ,1:: ]
  x7 = x[None,None,None,1:: ,1:: ,1:: ]
  xG = (x1*(eta - 1)*(mu - 1)*(zeta + 1))/8 - (x0*(eta - 1)*(mu - 1)*(zeta - 1))/8 + (x2*(eta + 1)*(mu - 1)*(zeta - 1))/8 - (x3*(eta + 1)*(mu - 1)*(zeta + 1))/8 + (x4*(eta - 1)*(mu + 1)*(zeta - 1))/8     - (x5*(eta - 1)*(mu + 1)*(zeta + 1))/8 - (x6*(eta + 1)*(mu + 1)*(zeta - 1))/8 + (x7*(eta + 1)*(mu + 1)*(zeta + 1))/8
  x0 = y[None,None,None,0:-1,0:-1,0:-1]
  x1 = y[None,None,None,1:: ,0:-1,0:-1]
  x2 = y[None,None,None,0:-1,1:: ,0:-1]
  x3 = y[None,None,None,1:: ,1:: ,0:-1]
  x4 = y[None,None,None,0:-1,0:-1,1:: ]
  x5 = y[None,None,None,1:: ,0:-1,1:: ]
  x6 = y[None,None,None,0:-1,1:: ,1:: ]
  x7 = y[None,None,None,1:: ,1:: ,1:: ]

  yG = (x1*(eta - 1)*(mu - 1)*(zeta + 1))/8 - (x0*(eta - 1)*(mu - 1)*(zeta - 1))/8 + (x2*(eta + 1)*(mu - 1)*(zeta - 1))/8 - (x3*(eta + 1)*(mu - 1)*(zeta + 1))/8 + (x4*(eta - 1)*(mu + 1)*(zeta - 1))/8     - (x5*(eta - 1)*(mu + 1)*(zeta + 1))/8 - (x6*(eta + 1)*(mu + 1)*(zeta - 1))/8 + (x7*(eta + 1)*(mu + 1)*(zeta + 1))/8

  x0 = z[None,None,None,0:-1,0:-1,0:-1]
  x1 = z[None,None,None,1:: ,0:-1,0:-1]
  x2 = z[None,None,None,0:-1,1:: ,0:-1]
  x3 = z[None,None,None,1:: ,1:: ,0:-1]
  x4 = z[None,None,None,0:-1,0:-1,1:: ]
  x5 = z[None,None,None,1:: ,0:-1,1:: ]
  x6 = z[None,None,None,0:-1,1:: ,1:: ]
  x7 = z[None,None,None,1:: ,1:: ,1:: ]

  zG = (x1*(eta - 1)*(mu - 1)*(zeta + 1))/8 - (x0*(eta - 1)*(mu - 1)*(zeta - 1))/8 + (x2*(eta + 1)*(mu - 1)*(zeta - 1))/8 - (x3*(eta + 1)*(mu - 1)*(zeta + 1))/8 + (x4*(eta - 1)*(mu + 1)*(zeta - 1))/8     - (x5*(eta - 1)*(mu + 1)*(zeta + 1))/8 - (x6*(eta + 1)*(mu + 1)*(zeta - 1))/8 + (x7*(eta + 1)*(mu + 1)*(zeta + 1))/8

  return xG[:,:,:,main.sx,main.sy,:],yG[:,:,:,main.sx,main.sy,:],zG[:,:,:,main.sx,main.sy,:]



def get_Xel(X,sx,sy):
  Nelx,Nely,Nelz = np.shape(X[0])
  Nelx -= 1
  Nely -= 1
  Nelz -= 1
  X_el = np.zeros((8,3,Nelx,Nely,Nelz))
  X_el[0] = X[:,0:-1,0:-1,0:-1]
  X_el[1] = X[:,1:: ,0:-1,0:-1]
  X_el[2]=  X[:,0:-1,1:: ,0:-1]
  X_el[3] = X[:,1:: ,1:: ,0:-1]
  X_el[4] = X[:,0:-1,0:-1,1:: ]
  X_el[5] = X[:,1:: ,0:-1,1:: ]
  X_el[6] = X[:,0:-1,1:: ,1:: ]
  X_el[7] = X[:,1:: ,1:: ,1:: ]
  return X_el[:,:,sx,sy]

def computeJacobian(X_el,zeta0,zeta1,zeta2):
  orderx = np.size(zeta0)
  ordery = np.size(zeta1)
  orderz = np.size(zeta2)

  zeta,eta,mu = np.meshgrid(zeta0,zeta1,zeta2,indexing='ij')
  dum,dum,Nelx,Nely,Nelz = np.shape(X_el)
  #Nelx,Nely,Nelz = np.shape(X_el)[0,0,0],np.shape(X_el)[0,0,1],np.shape(X_el)[0,0,2]
  J = np.zeros((3,3,orderx,ordery,orderz,Nelx,Nely,Nelz))
  JRL = np.zeros((3,2,ordery,orderz,Nelx+1,Nely,Nelz))
  JUD = np.zeros((3,2,orderx,orderz,Nelx,Nely+1,Nelz))
  JFB = np.zeros((3,2,orderx,ordery,Nelx,Nely,Nelz+1))
  JRLinv = np.zeros((3,2,ordery,orderz,Nelx+1,Nely,Nelz))
  JUDinv = np.zeros((3,2,orderx,orderz,Nelx,Nely+1,Nelz))
  JFBinv = np.zeros((3,2,orderx,ordery,Nelx,Nely,Nelz+1))
  JRLdet = np.zeros((ordery,orderz,Nelx+1,Nely,Nelz))
  JUDdet = np.zeros((orderx,orderz,Nelx,Nely+1,Nelz))
  JFBdet = np.zeros((orderx,ordery,Nelx,Nely,Nelz+1))

  J_edge = [JRL,JUD,JFB]
  J_edge_inv = [JRLinv,JUDinv,JFBinv]
  J_edge_det = [JRLdet,JUDdet,JFBdet]
  Jinv = np.zeros((3,3,orderx,ordery,orderz,Nelx,Nely,Nelz))
  Jdet = np.zeros((orderx,ordery,orderz,Nelx,Nely,Nelz))
  normals = np.zeros((6,3,Nelx,Nely,Nelz))

  x0 = X_el[0,0]
  x1 = X_el[1,0]
  x2 = X_el[2,0]
  x3 = X_el[3,0]
  x4 = X_el[4,0]
  x5 = X_el[5,0]
  x6 = X_el[6,0]
  x7 = X_el[7,0]
  

  J[0,0] = (x1[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] - 1))/8 - (x0[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] - 1))/8 + \
           (x2[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] - 1))/8 - (x3[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] - 1))/8 + \
           (x4[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] + 1))/8 - (x5[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] + 1))/8 - \
           (x6[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] + 1))/8 + (x7[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] + 1))/8
  J[0,1] = (x1[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 - (x0[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 + \
           (x2[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x3[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 + \
           (x4[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x5[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8 - \
           (x6[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 + (x7[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8
  J[0,2] = (x1[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 - (x0[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 + \
           (x2[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x3[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8 + \
           (x4[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x5[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 - \
           (x6[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 + (x7[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8
  x0 = X_el[0,1]
  x1 = X_el[1,1]
  x2 = X_el[2,1]
  x3 = X_el[3,1]
  x4 = X_el[4,1]
  x5 = X_el[5,1]
  x6 = X_el[6,1]
  x7 = X_el[7,1]

  J[1,0] = (x1[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] - 1))/8 - (x0[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] - 1))/8 + \
           (x2[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] - 1))/8 - (x3[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] - 1))/8 + \
           (x4[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] + 1))/8 - (x5[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] + 1))/8 - \
           (x6[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] + 1))/8 + (x7[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] + 1))/8
  J[1,1] = (x1[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 - (x0[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 + \
           (x2[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x3[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 + \
           (x4[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x5[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8 - \
           (x6[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 + (x7[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8
  J[1,2] = (x1[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 - (x0[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 + \
           (x2[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x3[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8 + \
           (x4[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x5[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 - \
           (x6[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 + (x7[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8


  x0 = X_el[0,2]
  x1 = X_el[1,2]
  x2 = X_el[2,2]
  x3 = X_el[3,2]
  x4 = X_el[4,2]
  x5 = X_el[5,2]
  x6 = X_el[6,2]
  x7 = X_el[7,2]

  J[2,0] = (x1[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] - 1))/8 - (x0[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] - 1))/8 + \
           (x2[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] - 1))/8 - (x3[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] - 1))/8 + \
           (x4[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] + 1))/8 - (x5[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(mu[:,:,:,None,None,None] + 1))/8 - \
           (x6[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] + 1))/8 + (x7[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(mu[:,:,:,None,None,None] + 1))/8
  J[2,1] = (x1[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 - (x0[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 + \
           (x2[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x3[None,None,None]*(mu[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 + \
           (x4[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x5[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8 - \
           (x6[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 + (x7[None,None,None]*(mu[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8
  J[2,2] = (x1[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 - (x0[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 + \
           (x2[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x3[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8 + \
           (x4[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] - 1))/8 - (x5[None,None,None]*(eta[:,:,:,None,None,None] - 1)*(zeta[:,:,:,None,None,None] + 1))/8 - \
           (x6[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] - 1))/8 + (x7[None,None,None]*(eta[:,:,:,None,None,None] + 1)*(zeta[:,:,:,None,None,None] + 1))/8

  for i in range(0,Nelx):
    for j in range(0,Nely):
      for k in range(0,Nelz):
        for p in range(0,orderx):
          for q in range(0,ordery):
            for r in range(0,orderz):
              Jinv[:,:,p,q,r,i,j,k] = np.linalg.inv(J[:,:,p,q,r,i,j,k])
              Jdet[p,q,r,i,j,k] = np.linalg.det(J[:,:,p,q,r,i,j,k])

  def faceMetrics(J_edge,J_edge_inv,J_edge_det,zeta,eta,x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3):
    J_edge[0,0] = (x0[None,None]*(eta[:,:,None,None,None] - 1))/4 - (x1[None,None]*(eta[:,:,None,None,None] - 1))/4 - \
            (x2[None,None]*(eta[:,:,None,None,None] + 1))/4 + (x3[None,None]*(eta[:,:,None,None,None] + 1))/4
    J_edge[0,1] = (x0[None,None]*(zeta[:,:,None,None,None] - 1))/4 - (x1[None,None]*(zeta[:,:,None,None,None] + 1))/4 - \
            (x2[None,None]*(zeta[:,:,None,None,None] - 1))/4 + (x3[None,None]*(zeta[:,:,None,None,None] + 1))/4
    J_edge[1,0] = (y0[None,None]*(eta[:,:,None,None,None] - 1))/4 - (y1[None,None]*(eta[:,:,None,None,None] - 1))/4 - \
            (y2[None,None]*(eta[:,:,None,None,None] + 1))/4 + (y3[None,None]*(eta[:,:,None,None,None] + 1))/4
    J_edge[1,1] = (y0[None,None]*(zeta[:,:,None,None,None] - 1))/4 - (y1[None,None]*(zeta[:,:,None,None,None] + 1))/4 - \
            (y2[None,None]*(zeta[:,:,None,None,None] - 1))/4 + (y3[None,None]*(zeta[:,:,None,None,None] + 1))/4
    J_edge[2,0] = (z0[None,None]*(eta[:,:,None,None,None] - 1))/4 - (z1[None,None]*(eta[:,:,None,None,None] - 1))/4 - \
               (z2[None,None]*(eta[:,:,None,None,None] + 1))/4 + (z3[None,None]*(eta[:,:,None,None,None] + 1))/4
    J_edge[2,1] = (z0[None,None]*(zeta[:,:,None,None,None] - 1))/4 - (z1[None,None]*(zeta[:,:,None,None,None] + 1))/4 - \
               (z2[None,None]*(zeta[:,:,None,None,None] - 1))/4 + (z3[None,None]*(zeta[:,:,None,None,None] + 1))/4

    # compute surface area magnitude dA
    mag1 = J_edge[0,0]*J_edge[1,1] - J_edge[1,0]*J_edge[0,1] 
    mag2 = J_edge[1,0]*J_edge[2,1] - J_edge[2,0]*J_edge[1,1] 
    mag3 = J_edge[2,0]*J_edge[0,1] - J_edge[0,0]*J_edge[2,1] 
    J_edge_det = np.sqrt(mag1**2 + mag2**2 + mag3**2)              
    return J_edge_det



  zeta,eta = np.meshgrid(zeta1,zeta2,indexing='ij')
  x0R,y0R,z0R = X_el[1,1],X_el[1,2],X_el[1,0]
  x1R,y1R,z1R = X_el[3,1],X_el[3,2],X_el[3,0]
  x2R,y2R,z2R = X_el[5,1],X_el[5,2],X_el[5,0]
  x3R,y3R,z3R = X_el[7,1],X_el[7,2],X_el[7,0]
  x0,y0,z0 = X_el[0,1],X_el[0,2],X_el[0,0]
  x1,y1,z1 = X_el[2,1],X_el[2,2],X_el[2,0]
  x2,y2,z2 = X_el[4,1],X_el[4,2],X_el[4,0]
  x3,y3,z3 = X_el[6,1],X_el[6,2],X_el[6,0]

  x0 = np.append(x0,x0R[-1][None],axis=0)
  x1 = np.append(x1,x1R[-1][None],axis=0)
  x2 = np.append(x2,x2R[-1][None],axis=0)
  x3 = np.append(x3,x3R[-1][None],axis=0)
  y0 = np.append(y0,y0R[-1][None],axis=0)
  y1 = np.append(y1,y1R[-1][None],axis=0)
  y2 = np.append(y2,y2R[-1][None],axis=0)
  y3 = np.append(y3,y3R[-1][None],axis=0)
  z0 = np.append(z0,z0R[-1][None],axis=0)
  z1 = np.append(z1,z1R[-1][None],axis=0)
  z2 = np.append(z2,z2R[-1][None],axis=0)
  z3 = np.append(z3,z3R[-1][None],axis=0)
  J_edge_det[0] = faceMetrics(J_edge[0],J_edge_inv[0],J_edge_det[0],zeta,eta,x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3)

  zeta,eta = np.meshgrid(zeta0,zeta2,indexing='ij')
  x0U,y0U,z0U = X_el[2,0],X_el[2,2],X_el[2,1]
  x1U,y1U,z1U = X_el[3,0],X_el[3,2],X_el[3,1]
  x2U,y2U,z2U = X_el[6,0],X_el[6,2],X_el[6,1]
  x3U,y3U,z3U = X_el[7,0],X_el[7,2],X_el[7,1]
  x0,y0,z0 = X_el[0,0],X_el[0,2],X_el[0,1]
  x1,y1,z1 = X_el[1,0],X_el[1,2],X_el[1,1]
  x2,y2,z2 = X_el[4,0],X_el[4,2],X_el[4,1]
  x3,y3,z3 = X_el[5,0],X_el[5,2],X_el[5,1]
  x0 = np.append(x0,x0U[:,-1][:,None],axis=1)
  x1 = np.append(x1,x1U[:,-1][:,None],axis=1)
  x2 = np.append(x2,x2U[:,-1][:,None],axis=1)
  x3 = np.append(x3,x3U[:,-1][:,None],axis=1)
  y0 = np.append(y0,y0U[:,-1][:,None],axis=1)
  y1 = np.append(y1,y1U[:,-1][:,None],axis=1)
  y2 = np.append(y2,y2U[:,-1][:,None],axis=1)
  y3 = np.append(y3,y3U[:,-1][:,None],axis=1)
  z0 = np.append(z0,z0U[:,-1][:,None],axis=1)
  z1 = np.append(z1,z1U[:,-1][:,None],axis=1)
  z2 = np.append(z2,z2U[:,-1][:,None],axis=1)
  z3 = np.append(z3,z3U[:,-1][:,None],axis=1)

  J_edge_det[1] = faceMetrics(J_edge[1],J_edge_inv[1],J_edge_det[1],zeta,eta,x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3)

  zeta,eta = np.meshgrid(zeta0,zeta1,indexing='ij')
  x0F,y0F,z0F = X_el[4,0],X_el[4,1],X_el[4,2]
  x1F,y1F,z1F = X_el[5,0],X_el[5,1],X_el[5,2]
  x2F,y2F,z2F = X_el[6,0],X_el[6,1],X_el[6,2]
  x3F,y3F,z3F = X_el[7,0],X_el[7,1],X_el[7,2]
  x0,y0,z0 = X_el[0,0],X_el[0,1],X_el[0,2]
  x1,y1,z1 = X_el[1,0],X_el[1,1],X_el[1,2]
  x2,y2,z2 = X_el[2,0],X_el[2,1],X_el[2,2]
  x3,y3,z3 = X_el[3,0],X_el[3,1],X_el[3,2]
  x0 = np.append(x0,x0F[:,:,-1][:,:,None],axis=2)
  x1 = np.append(x1,x1F[:,:,-1][:,:,None],axis=2)
  x2 = np.append(x2,x2F[:,:,-1][:,:,None],axis=2)
  x3 = np.append(x3,x3F[:,:,-1][:,:,None],axis=2)
  y0 = np.append(y0,y0F[:,:,-1][:,:,None],axis=2)
  y1 = np.append(y1,y1F[:,:,-1][:,:,None],axis=2)
  y2 = np.append(y2,y2F[:,:,-1][:,:,None],axis=2)
  y3 = np.append(y3,y3F[:,:,-1][:,:,None],axis=2)
  z0 = np.append(z0,z0F[:,:,-1][:,:,None],axis=2)
  z1 = np.append(z1,z1F[:,:,-1][:,:,None],axis=2)
  z2 = np.append(z2,z2F[:,:,-1][:,:,None],axis=2)
  z3 = np.append(z3,z3F[:,:,-1][:,:,None],axis=2)

  J_edge_det[2] = faceMetrics(J_edge[2],J_edge_inv[2],J_edge_det[2],zeta,eta,x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3)

  a = X_el[1] - X_el[5]
  b = X_el[3] - X_el[5]
  normals[0,0] =   a[1]*b[2] - a[2]*b[1]
  normals[0,1] =-( a[0]*b[2] - a[2]*b[0] )
  normals[0,2] =   a[0]*b[1] - a[1]*b[0]
  mag = np.sqrt(normals[0,0]**2 + normals[0,1]**2 + normals[0,2]**2)
  normals[0,:] /= mag[None,:]

  a = X_el[4] - X_el[0]
  b = X_el[6] - X_el[0]
  normals[1,0] =   a[1]*b[2] - a[2]*b[1]
  normals[1,1] =-( a[0]*b[2] - a[2]*b[0] )
  normals[1,2] =   a[0]*b[1] - a[1]*b[0]
  mag = np.sqrt(normals[1,0]**2 + normals[1,1]**2 + normals[1,2]**2)
  normals[1,:] /= mag[None,:]

  a = X_el[3] - X_el[6]
  b = X_el[2] - X_el[6]
  normals[2,0] =   a[1]*b[2] - a[2]*b[1]
  normals[2,1] =-( a[0]*b[2] - a[2]*b[0] )
  normals[2,2] =   a[0]*b[1] - a[1]*b[0]
  mag = np.sqrt(normals[2,0]**2 + normals[2,1]**2 + normals[2,2]**2)
  normals[2,:] /= mag[None,:]

  a = X_el[5] - X_el[1]
  b = X_el[4] - X_el[1]
  normals[3,0] =   a[1]*b[2] - a[2]*b[1]
  normals[3,1] =-( a[0]*b[2] - a[2]*b[0] )
  normals[3,2] =   a[0]*b[1] - a[1]*b[0]
  mag = np.sqrt(normals[3,0]**2 + normals[3,1]**2 + normals[3,2]**2)
  normals[3,:] /= mag[None,:]

  a = X_el[1] - X_el[0]
  b = X_el[3] - X_el[0]
  normals[4,0] =   a[1]*b[2] - a[2]*b[1]
  normals[4,1] =-( a[0]*b[2] - a[2]*b[0] )
  normals[4,2] =   a[0]*b[1] - a[1]*b[0]
  mag = np.sqrt(normals[4,0]**2 + normals[4,1]**2 + normals[4,2]**2)
  normals[4,:] /= mag[None,:]

  a = X_el[5] - X_el[6]
  b = X_el[4] - X_el[6]
  normals[5,0] =   a[1]*b[2] - a[2]*b[1]
  normals[5,1] =-( a[0]*b[2] - a[2]*b[0] )
  normals[5,2] =   a[0]*b[1] - a[1]*b[0]
  mag = np.sqrt(normals[5,0]**2 + normals[5,1]**2 + normals[5,2]**2)
  normals[5,:] /= mag[None,:]

  return J,Jinv,Jdet,J_edge_det,normals

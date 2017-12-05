import numpy as np
def nonreflecting_flux_bc_right(U,FR,FRm1,args,main):
  ## Compute right and left eigenvectors
  # A = R Lam L (I do like CFD vol2  pg 77,78)
  gamma = 1.4
  p = (gamma - 1.)*(U[4] - 0.5*U[1]**2/U[0] - 0.5*U[2]**2/U[0] - 0.5*U[3]**2/U[0])
  c = np.sqrt(gamma*p/U[0])
  u = U[1] / U[0]
  v = U[2] / U[0]
  w = U[3] / U[0]
  nx = 1.
  ny = 0.
  nz = 0.
  mx = 0.
  my = 1.
  mz = 0.
  lx = 0.
  ly = 0.
  lz = 1.
  K = gamma - 1.
  ql = 0
  qm = 0
  qn = u

  sizeu = np.array([5,5])#np.shape(main.a.u)[0]
  sizeu = np.append(sizeu,np.shape(U[0]))
  L = np.zeros(sizeu)
  R = np.zeros(sizeu)

  q = np.sqrt(u**2 + v**2 + w**2)
  L[0,0] = K*q**2/(4.*c**2) + qn/(2.*c)
  L[0,1] = -(K/(2.*c**2)*u + nx/(2.*c))
  L[0,2] = -(K/(2.*c**2)*v + ny/(2.*c))
  L[0,3] = -(K/(2.*c**2)*w + nz/(2.*c))
  L[0,4] = K/(2.*c**2)
  L[1,0] = 1. - K*q**2/(2.*c**2)
  L[1,1] = K*u/c**2
  L[1,2] = K*v/c**2
  L[1,3] = K*w/c**2
  L[1,4] = -K/c**2
  L[2,0] = K*q**2/(4.*c**2) - qn/(2.*c)
  L[2,1] = -(K/(2.*c**2)*u - nx/(2.*c) )
  L[2,2] = -(K/(2.*c**2)*v - ny/(2.*c) )
  L[2,3] = -(K/(2.*c**2)*w - nz/(2.*c) )
  L[2,4] = K/(2.*c**2)
  L[3,0] = -ql
  L[3,1] = lx
  L[3,2] = ly
  L[3,3] = lz
  L[4,0] = -qm
  L[4,1] = mx
  L[4,2] = my
  L[4,3] = mz

  # compute H in three steps (H = E + p/rho)
  H = (gamma - 1.)*(U[4] - 0.5*U[0]*q**2) #compute pressure
  H += U[4]
  H /= U[0]

  R[0,0] = 1.
  R[0,1] = 1.
  R[0,2] = 1.
  R[1,0] = u - c*nx
  R[1,1] = u
  R[1,2] = u + c*nx
  R[1,3] = lx
  R[1,4] = mx
  R[2,0] = v - c*ny
  R[2,1] = v
  R[2,2] = v + c*ny
  R[2,3] = ly
  R[2,4] = my
  R[3,0] = w - c*nz
  R[3,1] = w
  R[3,2] = w + c*nz
  R[3,3] = lz
  R[3,4] = mz
  R[4,0] = H - qn*c
  R[4,1] = q**2/2.
  R[4,2] = H + qn*c
  R[4,3] = ql
  R[4,4] = qm

  ###====== To get non-reflecting bcs, cast the flux in chacteristic form
  # L u_t +  L R Lam L U_x = 0
  # L u_t + Lam L U_x = 0
  # set Lam L U_x \approx F_x = 0
#  print(np.linalg.norm(FR[1]))
  FcR  =  np.einsum('ij...,j...->i...',L,FR[0:5])
  ## First row corresponds to the negative pressure wave, third to the positive
  FcRm1  =  np.einsum('ij...,j...->i...',L,FRm1[0:5])
  FcR[0] = FcRm1[0]
  # now transform back
  FR[0:5] = np.einsum('ij...,j...->i...',R,FcR[0:5])
  #test = np.einsum('ij...,jk...->ik...',L[0:5],R[0:5])
  #print(np.linalg.norm(FR2[0:5] - FR[0:5] ))
  return FR


def nonreflecting_flux_bc_left(U,FL,FLp1,args,main):
  ## Compute right and left eigenvectors
  # A = R Lam L (I do like CFD vol2  pg 77,78)
  gamma = 1.4
  p = (gamma - 1.)*(U[4] - 0.5*U[1]**2/U[0] - 0.5*U[2]**2/U[0] - 0.5*U[3]**2/U[0])
  c = np.sqrt(gamma*p/U[0])
  u = U[1] / U[0]
  v = U[2] / U[0]
  w = U[3] / U[0]
  nx = 1.
  ny = 0.
  nz = 0.
  mx = 0.
  my = 1.
  mz = 0.
  lx = 0.
  ly = 0.
  lz = 1.
  K = gamma - 1.
  ql = 0
  qm = 0
  qn = u*1.

  sizeu = np.array([5,5])#np.shape(main.a.u)[0]
  sizeu = np.append(sizeu,np.shape(U[0]))
  L = np.zeros(sizeu)
  R = np.zeros(sizeu)

  q = np.sqrt(u**2 + v**2 + w**2)
  L[0,0] = K*q**2/(4.*c**2) + qn/(2.*c)
  L[0,1] = -(K/(2.*c**2)*u + nx/(2.*c))
  L[0,2] = -(K/(2.*c**2)*v + ny/(2.*c))
  L[0,3] = -(K/(2.*c**2)*w + nz/(2.*c))
  L[0,4] = K/(2.*c**2)
  L[1,0] = 1. - K*q**2/(2.*c**2)
  L[1,1] = K*u/c**2
  L[1,2] = K*v/c**2
  L[1,3] = K*w/c**2
  L[1,4] = -K/c**2
  L[2,0] = K*q**2/(4.*c**2) - qn/(2.*c)
  L[2,1] = -(K/(2.*c**2)*u - nx/(2.*c) )
  L[2,2] = -(K/(2.*c**2)*v - ny/(2.*c) )
  L[2,3] = -(K/(2.*c**2)*w - nz/(2.*c) )
  L[2,4] = K/(2.*c**2)
  L[3,0] = -ql
  L[3,1] = lx
  L[3,2] = ly
  L[3,3] = lz
  L[4,0] = -qm
  L[4,1] = mx
  L[4,2] = my
  L[4,3] = mz

  # compute H in three steps (H = E + p/rho)
  H = (gamma - 1.)*(U[4] - 0.5*U[0]*q**2) #compute pressure
  H += U[4]
  H /= U[0]

  R[0,0] = 1.
  R[0,1] = 1.
  R[0,2] = 1.
  R[1,0] = u - c*nx
  R[1,1] = u
  R[1,2] = u + c*nx
  R[1,3] = lx
  R[1,4] = mx
  R[2,0] = v - c*ny
  R[2,1] = v
  R[2,2] = v + c*ny
  R[2,3] = ly
  R[2,4] = my
  R[3,0] = w - c*nz
  R[3,1] = w
  R[3,2] = w + c*nz
  R[3,3] = lz
  R[3,4] = mz
  R[4,0] = H - qn*c
  R[4,1] = q**2/2.
  R[4,2] = H + qn*c
  R[4,3] = ql
  R[4,4] = qm

  ###====== To get non-reflecting bcs, cast the flux in chacteristic form
  # L u_t +  L R Lam L U_x = 0
  # L u_t + Lam L U_x = 0
  # set Lam L U_x \approx F_x = 0
#  print(np.linalg.norm(FR[1]))
  FcL  =  np.einsum('ij...,j...->i...',L,FL[0:5])
  ## First row corresponds to the negative pressure wave, third to the positive
  FcLp1  =  np.einsum('ij...,j...->i...',L,FLp1[0:5])
  FcL[2] = FcLp1[2]
  # now transform back
  FL[0:5] = np.einsum('ij...,j...->i...',R,FcL[0:5])
  #test = np.einsum('ij...,jk...->ik...',L[0:5],R[0:5])
  return FL 


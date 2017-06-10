import numpy as np
import matplotlib.pyplot as plt
def computeSmagViscosity(main,Ux,Uy,Uz,mu0,u):
  ux = 1./u[0]*(Ux[1] - u[1]/u[0]*Ux[0])
  vx = 1./u[0]*(Ux[2] - u[2]/u[0]*Ux[0])
  wx = 1./u[0]*(Ux[3] - u[3]/u[0]*Ux[0])
  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  uy = 1./u[0]*(Uy[1] - u[1]/u[0]*Uy[0])
  vy = 1./u[0]*(Uy[2] - u[2]/u[0]*Uy[0])
  wy = 1./u[0]*(Uy[3] - u[3]/u[0]*Uy[0])
  # ->  u_z = 1/rho d/dz(rho u) - rho u /rho^2 rho_z
  uz = 1./u[0]*(Uz[1] - u[1]/u[0]*Uz[0])
  vz = 1./u[0]*(Uz[2] - u[2]/u[0]*Uz[0])
  wz = 1./u[0]*(Uz[3] - u[3]/u[0]*Uz[0])
  DG_Delta = main.order[0]**(-1.21)*2.041
  Delta = (main.dx*main.dy*main.dz)**(1./3.)*DG_Delta
  S11 = ux
  S22 = vy
  S33 = wz
  S12 = 0.5*(uy + vx)
  S13 = 0.5*(uz + wx)
  S23 = 0.5*(vz + wy)
  S_mag = np.sqrt( 2.*(S11**2 + S22**2 + S33**2 + 2.*S12**2 + 2.*S13**2 + 2.*S23**2) )
  mut = u[0]*0.16**2*Delta**2*np.abs(S_mag)
  return mu0 + mut

def staticSmagorinsky(main):
  main.mu[:] = computeSmagViscosity(main,main.a.Upx,main.a.Upy,main.a.Upz,main.mu0,main.a.u)
  main.muR[:] = computeSmagViscosity(main,main.a.UxR,main.a.UyR,main.a.UzR,main.mu0R,main.a.uR)
  main.muL[:] = computeSmagViscosity(main,main.a.UxL,main.a.UyL,main.a.UzL,main.mu0L,main.a.uL)
  main.muU[:] = computeSmagViscosity(main,main.a.UxU,main.a.UyU,main.a.UzU,main.mu0U,main.a.uU)
  main.muD[:] = computeSmagViscosity(main,main.a.UxD,main.a.UyD,main.a.UzD,main.mu0D,main.a.uD)
  main.muF[:] = computeSmagViscosity(main,main.a.UxF,main.a.UyF,main.a.UzF,main.mu0F,main.a.uF)
  main.muB[:] = computeSmagViscosity(main,main.a.UxB,main.a.UyB,main.a.UzB,main.mu0B,main.a.uB)
  #print(np.mean(main.mu0),np.amax(main.mu))

### For some reason this keeps giving a negative C^2
def computeDynSmagViscosity(main,Ux,Uy,Uz,mu0,u):
  ratio = np.amin(main.order)/2
  # need to compute filtered quantities first
  filt_array = np.zeros(np.shape(main.a.a[0]))
  # filtered stuff
  filt_array[0:main.order[0]/ratio,0:main.order[1]/ratio,0:main.order[2]/ratio] = 1.
  af = filt_array[None]*main.a.a
  Uf = main.basis.reconstructUGeneral(main,af)
  U  = main.basis.reconstructUGeneral(main,main.a.a)
  uf = Uf[1] / Uf[0]
  vf = Uf[2] / Uf[0]
  wf = Uf[3] / Uf[0]
 
  u = U[1] / U[0]
  v = U[2] / U[0]
  w = U[3] / U[0]

  ## Make Leonard Stress Tensor. To do the filtering, we need to project to modal 
  # space, filter, and then project back.
  Lshape = np.append(6,np.shape(main.a.u[0]))
  L = np.zeros(Lshape)
  L[0] = U[0]*u**2 
  L[1] = U[0]*v**2 
  L[2] = U[0]*w**2  
  L[3] = U[0]*u*v  
  L[4] = U[0]*u*w  
  L[5] = U[0]*v*w 
  ord_arrx= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arry= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arrz= np.linspace(0,main.order[2]-1,main.order[2])
  scale =  (2.*ord_arrx[:,None,None] + 1.)*(2.*ord_arry[None,:,None] + 1.)*(2.*ord_arrz[None,None    ,:]+1.)/8.
  L_modal = main.basis.volIntegrateGlob(main,L,main.w0,main.w1,main.w2)*scale[None,:,:,:,None,None,None]
  L_modal *= filt_array[None]
  L[:] = main.basis.reconstructUGeneral(main,L_modal)

  L1shape = np.append(6,np.shape(main.a.u[0]))
  L1 = np.zeros(Lshape)

  L1[0] = Uf[1]*Uf[1]/Uf[0]
  L1[1] = Uf[2]*Uf[2]/Uf[0]
  L1[2] = Uf[3]*Uf[3]/Uf[0]
  L1[3] = Uf[1]*Uf[2]/Uf[0] 
  L1[4] = Uf[1]*Uf[3]/Uf[0]
  L1[5] = Uf[2]*Uf[3]/Uf[0]

  L1_modal = main.basis.volIntegrateGlob(main,L1,main.w0,main.w1,main.w2)*scale[None,:,:,:,None,None,None]
  L1[:] =main.basis.reconstructUGeneral(main,L1_modal)
  L[:] = L[:] - L1[:] 
  # now subtract deviatoric part
  L[0] -= 1./3.*(L[0] + L[1] + L[2])
  L[1] -= 1./3.*(L[0] + L[1] + L[2])
  L[2] -= 1./3.*(L[0] + L[1] + L[2])

  ## Now make strain rate tensor. First need derivs
  Uxf,Uyf,Uzf = main.basis.diffU(af,main)
  # now compute filtered strain tensor
  uxf = 1./Uf[0]*(Uxf[1] - Uf[1]/Uf[0]*Uxf[0])
  vxf = 1./Uf[0]*(Uxf[2] - Uf[2]/Uf[0]*Uxf[0])
  wxf = 1./Uf[0]*(Uxf[3] - Uf[3]/Uf[0]*Uxf[0])
  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  uyf = 1./Uf[0]*(Uyf[1] - Uf[1]/Uf[0]*Uyf[0])
  vyf = 1./Uf[0]*(Uyf[2] - Uf[2]/Uf[0]*Uyf[0])
  wyf = 1./Uf[0]*(Uyf[3] - Uf[3]/Uf[0]*Uyf[0])
  # ->  u_z = 1/rho d/dz(rho u) - rho u /rho^2 rho_z
  uzf = 1./Uf[0]*(Uzf[1] - Uf[1]/Uf[0]*Uzf[0])
  vzf = 1./Uf[0]*(Uzf[2] - Uf[2]/Uf[0]*Uzf[0])
  wzf = 1./Uf[0]*(Uzf[3] - Uf[3]/Uf[0]*Uzf[0])
  S11f = uxf
  S22f = vyf
  S33f = wzf
  S12f = 0.5*(uyf + vxf)
  S13f = 0.5*(uzf + wxf)
  S23f = 0.5*(vzf + wyf)
  S_magf = np.sqrt( 2.*(S11f**2 + S22f**2 + S33f**2 + 2.*S12f**2 + 2.*S13f**2 + 2.*S23f**2) )


  ## Now do unfiltered quantities
  # d/dx (rho u) = rho d/dx u + u d/dx rho
  # du / dx = 1/rho*( d/dx rho u - u d/dx rho )
  ux = 1./U[0]*(Ux[1] - U[1]/U[0]*Ux[0])
  vx = 1./U[0]*(Ux[2] - U[2]/U[0]*Ux[0])
  wx = 1./U[0]*(Ux[3] - U[3]/U[0]*Ux[0])
  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  uy = 1./U[0]*(Uy[1] - U[1]/U[0]*Uy[0])
  vy = 1./U[0]*(Uy[2] - U[2]/U[0]*Uy[0])
  wy = 1./U[0]*(Uy[3] - U[3]/U[0]*Uy[0])
  # ->  u_z = 1/rho d/dz(rho u) - rho u /rho^2 rho_z
  uz = 1./U[0]*(Uz[1] - U[1]/U[0]*Uz[0])
  vz = 1./U[0]*(Uz[2] - U[2]/U[0]*Uz[0])
  wz = 1./U[0]*(Uz[3] - U[3]/U[0]*Uz[0])
  S11 = ux
  S22 = vy
  S33 = wz
  div = ux + vy + wz
  div2 = Ux[1] + Uy[2] + Uz[3]

  plt.plot(div2[0,0,0,:,0,0])
  plt.plot(wz[0,0,0,:,0,0],'--')

  plt.pause(0.001)
  S12 = 0.5*(uy + vx)
  S13 = 0.5*(uz + wx)
  S23 = 0.5*(vz + wy)
  S_mag = np.sqrt( 2.*(S11**2 + S22**2 + S33**2 + 2.*S12**2 + 2.*S13**2 + 2.*S23**2) )
  ## Now create Mhat tensor
  M = np.zeros(Lshape)
  S_magSF = np.zeros(Lshape)
  S_magSF[0] = S_mag*(S11 - 1./3.*(S11 + S22 + S33) )
  S_magSF[1] = S_mag*(S22 - 1./3.*(S11 + S22 + S33) )
  S_magSF[2] = S_mag*(S33 - 1./3.*(S11 + S22 + S33) )
  S_magSF[3] = S_mag*S12
  S_magSF[4] = S_mag*S13
  S_magSF[5] = S_mag*S23
  S_magSF_modal = main.basis.volIntegrateGlob(main,S_magSF,main.w0,main.w1,main.w2)*scale[None,:,:,:,None,None,None]
  S_magSF_modal = filt_array[None]*S_magSF_modal
  S_magSF[:] = main.basis.reconstructUGeneral(main,S_magSF_modal)
  DG_Delta = main.order[0]**(-1.21)*2.041
  Delta = (main.dx*main.dy*main.dz)**(1./3.)*DG_Delta
  DSmag_alpha = ( (main.order[0]/ratio)**(-1.21)*2.041)/( (main.order[0])**(-1.21)*2.041)
  Sf_magSF = np.zeros(Lshape)
  Sf_magSF[0] = S_magf*(S11f - 1./3.*(S11f + S22f + S33f) )
  Sf_magSF[1] = S_magf*(S22f - 1./3.*(S11f + S22f + S33f) )
  Sf_magSF[2] = S_magf*(S33f - 1./3.*(S11f + S22f + S33f) )
  Sf_magSF[3] = S_magf*S12f
  Sf_magSF[4] = S_magf*S13f
  Sf_magSF[5] = S_magf*S23f
  Sf_magSF_modal = main.basis.volIntegrateGlob(main,Sf_magSF,main.w0,main.w1,main.w2)*scale[None,:,:,:,None,None,None]
  Sf_magSF[:] = main.basis.reconstructUGeneral(main,Sf_magSF_modal)
  M[:] = (U[0]*S_magSF[:]) - DSmag_alpha**2*Uf[0]*Sf_magSF[:]
#  M[:] = -2.*Delta**2*(DSmag_alpha**2*Sf_magSF[:] - S_magSF[:])
  Cs_sqr =1./Delta**2*0.5* np.mean(np.sum(M*L,axis=0) + np.sum(M[3::]*L[3::]*2.,axis=0 ) , axis=(0,1,2)) / np.mean( np.sum(M*M,axis=0) + 2.*np.sum(M[3::]**2,axis=0) +  1e-40 ,axis=(0,1,2) ) 
  print('MPI_RANK = ' + str(main.mpi_rank) + '   Cs_sqr = ' + str(np.mean(Cs_sqr)) + ' rat = ' + str(DSmag_alpha))
  print('==================================================')
  #print(np.mean(Sf_magSF**2),np.mean(S_magSF**2),np.mean(np.sum(M*L,axis=0)))


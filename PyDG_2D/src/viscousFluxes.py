import numpy as np

def getGs(u):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G11 = np.zeros(ashape)
  G12 = np.zeros(ashape)
  G21 = np.zeros(ashape)
  G22 = np.zeros(ashape)
  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  E = u[3]/u[0]

  vsqr = v1**2 + v2**2
  G11[1,0] = -4./3.*v1
  G11[1,1] = 4./3.
  G11[2,0] = -v2
  G11[2,2] = 1.
  G11[3,0] = -(4./3.*v1**2 + v2**2 + gamma/Pr*(E - vsqr) )
  G11[3,1] = (4./3. - gamma/Pr)*v1
  G11[3,2] = (1. - gamma/Pr)*v2
  G11[3,3] = gamma/Pr

  G12[1,0] = 2./3.*v2
  G12[1,2] = -2./3.
  G12[2,0] = -v1
  G12[2,1] = 1.
  G12[3,0] = -1./3.*v1*v2
  G12[3,1] = v2
  G12[3,2] = -2./3.*v1

  G21[1,0] = -v2
  G21[1,2] = 1.
  G21[2,0] = 2./3.*v1
  G21[2,1] = -2./3.
  G21[3,0] = -1./3.*v1*v2
  G21[3,1] = -2./3.*v2
  G21[3,2] = v1

  G22[1,0] = -v1
  G22[1,1] = 1.
  G22[2,0] = -4./3.*v2
  G22[2,2] = 4./3.
  G22[3,0] = -(v1**2 + 4./3.*v2**2 + gamma/Pr*(E - vsqr) )
  G22[3,1] = (1. - gamma/Pr)*v1
  G22[3,2] = (4./3. - gamma/Pr)*v2
  G22[3,3] = gamma/Pr
  G11 = G11*mu/u[0]
  G12 = G12*mu/u[0]
  G21 = G21*mu/u[0]
  G22 = G22*mu/u[0]
  return G11,G12,G21,G22


def evalViscousFluxX(u,ux,uy,vx,vy,kTx,kTy):
  fx = np.zeros(np.shape(u))
  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  fx[1] = mu*(4./3.*ux - 2./3.*vy)
  fx[2] = mu*(vx + uy)
  fx[3] = fx[1]*v1 + fx[2]*v2 + kTx
  return fx

def evalViscousFluxY(u,ux,uy,vx,vy,kTx,kTy):
  fy = np.zeros(np.shape(u))
  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  fy[1] = mu*(vx + uy)
  fy[2] = mu*(4./3.*vy - 2./3.*ux)
  fy[3] = fy[1]*v1 + fy[2]*v2 + kTy
  return fy


def diffCoeffs(a):
  atmp = np.zeros(np.shape(a))
  atmp[:] = a[:]
  nvars,order,order,Nelx,Nely = np.shape(a)
  ax = np.zeros((nvars,order,order,Nelx,Nely))
  ay = np.zeros((nvars,order,order,Nelx,Nely))
  for j in range(order-1,2,-1):
    ax[:,j-1,:] = (2.*j-1)*atmp[:,j,:]
    atmp[:,j-2,:] = atmp[:,j-2,:] + atmp[:,j,:]
  ax[:,1,:] = 3.*atmp[:,2,:]
  ax[:,0,:] = atmp[:,1,:]

  atmp[:] = a[:]
  for j in range(order-1,2,-1):
    ay[:,:,j-1] = (2.*j-1)*atmp[:,:,j]
    atmp[:,:,j-2] = atmp[:,:,j-2] + atmp[:,:,j]
  ay[:,:,1] = 3.*atmp[:,:,2]
  ay[:,:,0] = atmp[:,:,1]
  return ax,ay



def diffU(a):
  nvars = np.shape(a)[0]
  order = np.shape(a)[1]
  ux = np.zeros(np.shape(a))
  uy = np.zeros(np.shape(a))
  for l in range(0,order):
    for m in range(0,order):
      for k in range(0,nvars):
        ux[k,:,:,:,:] += wp[l][:,None,None,None]*w[m][None,:,None,None]*a[k,l,m,:,:]
        uy[k,:,:,:,:] += w[l][:,None,None,None]*wp[m][None,:,None,None]*a[k,l,m,:,:]
  return ux,uy


def computeJump(uR,uL,uU,uD,uU_edge,uD_edge):
  nvars,order,Npx,Npy = np.shape(uR)
  jumpR = np.zeros((nvars,order,Npx,Npy))
  jumpL = np.zeros((nvars,order,Npx,Npy))
  jumpU = np.zeros((nvars,order,Npx,Npy))
  jumpD = np.zeros((nvars,order,Npx,Npy))

  jumpR[:,:,0:-1,:] = uR[:,:,0:-1,:] - uL[:,:,1::,:]
  jumpR[:,:,-1   ,:] = uR[:,:,  -1,:] - uL[:,:,0  ,:]
  jumpL[:,:,1:: ,:] = jumpR[:,:,0:-1,:]
  jumpL[:,:,0   ,:] = jumpR[:,:,  -1,:]
  jumpU[:,:,:,0:-1] = uU[:,:,:,0:-1] - uD[:,:,:,1::]
  jumpU[:,:,:,  -1] = uU[:,:,:,  -1] - uU_edge
  jumpD[:,:,:,1:: ] = jumpU[:,:,:,0:-1]
  jumpD[:,:,:,0   ] = uD_edge - uD[:,:,:,   0]
  return jumpR,jumpL,jumpU,jumpD



def getViscousFlux(a):
  nvars,order,order,Npx,Npy = np.shape(a)
  fvRG11 = np.zeros((nvars,order,Npx,Npy))
  fvLG11 = np.zeros((nvars,order,Npx,Npy))

  fvRG21 = np.zeros((nvars,order,Npx,Npy))
  fvLG21 = np.zeros((nvars,order,Npx,Npy))

  fvUG12 = np.zeros((nvars,order,Npx,Npy))
  fvDG12 = np.zeros((nvars,order,Npx,Npy))

  fvUG22 = np.zeros((nvars,order,Npx,Npy))
  fvDG22 = np.zeros((nvars,order,Npx,Npy))

  fvR2 = np.zeros((nvars,order,Npx,Npy))
  fvL2 = np.zeros((nvars,order,Npx,Npy))
  fvU2 = np.zeros((nvars,order,Npx,Npy))
  fvD2 = np.zeros((nvars,order,Npx,Npy))

  # now construct star state
  #uR,uL,uU,uD = reconstructEdges(a)
  #uU_edge,uD_edge = sendEdges(uD,uU)
  ## Here is where you would get the viscous fluxes...
  uhatR,uhatL,uhatU,uhatD = centralFlux(uR,uL,uU,uD,uU_edge,uD_edge)
  G11R,G12R,G21R,G22R = getGs(main.a.uR)
  G11L,G12L,G21L,G22L = getGs(main.a.uL)
  G11U,G12U,G21U,G22U = getGs(main.a.uU)
  G11D,G12D,G21D,G22D = getGs(main.a.uD)

  for i in range(0,nvars):
    for j in range(0,nvars):
      fvRG11[i] += G11R[i,j]*(main.a.uR[j] - uhatR[j])
      fvLG11[i] += G11L[i,j]*(main.a.uL[j] - uhatL[j])

      fvRG21[i] += G21R[i,j]*(main.a.uR[j] - uhatR[j])
      fvLG21[i] += G21L[i,j]*(main.a.uL[j] - uhatL[j])

      fvUG12[i] += G12U[i,j]*(main.a.uU[j] - uhatU[j])
      fvDG12[i] += G12D[i,j]*(main.a.uD[j] - uhatD[j])

      fvUG22[i] += G22U[i,j]*(main.a.uU[j] - uhatU[j])
      fvDG22[i] += G22D[i,j]*(main.a.uD[j] - uhatD[j])


  apx,apy = diffCoeffs(main.a.a)
  apx = apx*2./dx
  apy = apy*2./dy
  UxR,UxL,UxU,UxD = reconstructEdges(apx)
  UyR,UyL,UyU,UyD = reconstructEdges(apy)
  UxU_edge,UxD_edge = sendEdges(UxD,UxU)
  UyU_edge,UyD_edge = sendEdges(UyD,UyU)
  ### now we need to do modifications to individual derivs of u,v,etc.
  ##ex : d/dx(rho u ) = rho u_x + u rho_x
  ## ->  u_x = 1/rho d/dx(rho u) - rho u /rho^2 rho_x
  uxR = 1./uR[0]*UxR[1] - uR[1]/uR[0]**2*UxR[0]
  uxL = 1./uL[0]*UxL[1] - uL[1]/uL[0]**2*UxL[0]
  uxU = 1./uU[0]*UxU[1] - uU[1]/uU[0]**2*UxU[0]
  uxD = 1./uD[0]*UxD[1] - uD[1]/uD[0]**2*UxD[0]
  uxU_edge = 1./uU_edge[0]*UxU_edge[1] - uU_edge[1]/uU_edge[0]**2*UxU_edge[0]
  uxD_edge = 1./uD_edge[0]*UxD_edge[1] - uD_edge[1]/uD_edge[0]**2*UxD_edge[0]

  ## ->  v_x = 1/rho d/dx(rho v) - rho v /rho^2 rho_x
  vxR = 1./uR[0]*UxR[2] - uR[2]/uR[0]**2*UxR[0]
  vxL = 1./uL[0]*UxL[2] - uL[2]/uL[0]**2*UxL[0]
  vxU = 1./uU[0]*UxU[2] - uU[2]/uU[0]**2*UxU[0]
  vxD = 1./uD[0]*UxD[2] - uD[2]/uD[0]**2*UxD[0]
  vxU_edge = 1./uU_edge[0]*UxU_edge[2] - uU_edge[2]/uU_edge[0]**2*UxU_edge[0]
  vxD_edge = 1./uD_edge[0]*UxD_edge[2] - uD_edge[2]/uD_edge[0]**2*UxD_edge[0]

  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  uyR = 1./uR[0]*UyR[1] - uR[1]/uR[0]**2*UyR[0]
  uyL = 1./uL[0]*UyL[1] - uL[1]/uL[0]**2*UyL[0]
  uyU = 1./uU[0]*UyU[1] - uU[1]/uU[0]**2*UyU[0]
  uyD = 1./uD[0]*UyD[1] - uD[1]/uD[0]**2*UyD[0]
  uyU_edge = 1./uU_edge[0]*UyU_edge[1] - uU_edge[1]/uU_edge[0]**2*UyU_edge[0]
  uyD_edge = 1./uD_edge[0]*UyD_edge[1] - uD_edge[1]/uD_edge[0]**2*UyD_edge[0]

  ## ->  v_y = 1/rho d/dy(rho v) - rho v /rho^2 rho_y
  vyR = 1./uR[0]*UyR[2] - uR[2]/uR[0]**2*UyR[0]
  vyL = 1./uL[0]*UyL[2] - uL[2]/uL[0]**2*UyL[0]
  vyU = 1./uU[0]*UyU[2] - uU[2]/uU[0]**2*UyU[0]
  vyD = 1./uD[0]*UyD[2] - uD[2]/uD[0]**2*UyD[0]
  vyU_edge = 1./uU_edge[0]*UyU_edge[2] - uU_edge[2]/uU_edge[0]**2*UyU_edge[0]
  vyD_edge = 1./uD_edge[0]*UyD_edge[2] - uD_edge[2]/uD_edge[0]**2*UyD_edge[0]

  ## -> (kT)_x =d/dx[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dx - 0.5 d/dx(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dx - (v1 v1_x + v2 v2_x) ]
  ## ->  E_x = 1/rho d/x(rho E) - rho E /rho^2 rho_x
  kTxR =( 1./uR[0]*UxR[3] - uR[3]/uR[0]**2*UxR[0] - 1./uR[0]*(uR[1]*uxR + uR[2]*vxR)  )*mu*gamma/Pr
  kTxL =( 1./uL[0]*UxL[3] - uL[3]/uL[0]**2*UxL[0] - 1./uL[0]*(uL[1]*uxL + uL[2]*vxL)  )*mu*gamma/Pr
  kTxU =( 1./uU[0]*UxU[3] - uU[3]/uU[0]**2*UxU[0] - 1./uU[0]*(uU[1]*uxU + uU[2]*vxU)  )*mu*gamma/Pr
  kTxD =( 1./uD[0]*UxD[3] - uD[3]/uD[0]**2*UxD[0] - 1./uD[0]*(uD[1]*uxD + uD[2]*vxD)  )*mu*gamma/Pr
  kTxU_edge =( 1./uU_edge[0]*UxU_edge[3] - uU_edge[3]/uU_edge[0]**2*UxU_edge[0] - 1./uU_edge[0]*(uU_edge[1]*uxU_edge + uU_edge[2]*vxU_edge) )*mu*gamma/Pr
  kTxD_edge =( 1./uD_edge[0]*UxD_edge[3] - uD_edge[3]/uD_edge[0]**2*UxD_edge[0] - 1./uD_edge[0]*(uD_edge[1]*uxD_edge + uD_edge[2]*vxD_edge) )*mu*gamma/Pr

  ## -> (kT)_y =d/dy[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dy - 0.5 d/dy(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dy - (v1 v1_y + v2 v2_y) ]
  ## ->  E_x = 1/rho d/y(rho E) - rho E /rho^2 rho_y
  kTyR =( 1./uR[0]*UyR[3] - uR[3]/uR[0]**2*UyR[0] - 1./uR[0]*(uR[1]*uyR + uR[2]*vyR)  )*mu*gamma/Pr
  kTyL =( 1./uL[0]*UyL[3] - uL[3]/uL[0]**2*UyL[0] - 1./uL[0]*(uL[1]*uyL + uL[2]*vyL)  )*mu*gamma/Pr
  kTyU =( 1./uU[0]*UyU[3] - uU[3]/uU[0]**2*UyU[0] - 1./uU[0]*(uU[1]*uyU + uU[2]*vyU)  )*mu*gamma/Pr
  kTyD =( 1./uD[0]*UyD[3] - uD[3]/uD[0]**2*UyD[0] - 1./uD[0]*(uD[1]*uyD + uD[2]*vyD)  )*mu*gamma/Pr
  kTyU_edge =( 1./uU_edge[0]*UyU_edge[3] - uU_edge[3]/uU_edge[0]**2*UyU_edge[0] - 1./uU_edge[0]*(uU_edge[1]*uyU_edge + uU_edge[2]*vyU_edge) )*mu*gamma/Pr
  kTyD_edge =( 1./uD_edge[0]*UyD_edge[3] - uD_edge[3]/uD_edge[0]**2*UyD_edge[0] - 1./uD_edge[0]*(uD_edge[1]*uyD_edge + uD_edge[2]*vyD_edge) )*mu*gamma/Pr

  fvxR = evalViscousFluxX(uR,uxR,uyR,vxR,vyR,kTxR,kTyR)
  fvxL = evalViscousFluxX(uL,uxL,uyL,vxL,vyL,kTxL,kTyL)
  fvyU = evalViscousFluxY(uU,uxU,uyU,vxU,vyU,kTxU,kTyU)
  fvyD = evalViscousFluxY(uD,uxD,uyD,vxD,vyD,kTxD,kTyD)
  fvyU_edge = evalViscousFluxY(uU_edge,uxU_edge,uyU_edge,vxU_edge,vyU_edge,kTxU_edge,kTyU_edge)
  fvyD_edge = evalViscousFluxY(uD_edge,uxD_edge,uyD_edge,vxD_edge,vyD_edge,kTxD_edge,kTyD_edge)

  shatR,shatL,shatU,shatD = centralFlux(fvxR,fvxL,fvyU,fvyD,fvyU_edge,fvyD_edge)
  jumpR,jumpL,jumpU,jumpD = computeJump(uR,uL,uU,uD,uU_edge,uD_edge)
  fvR2[:] = shatR[:] - 2.*mu*jumpR[:]*order**2/dx
  fvL2[:] = shatL[:] - 2.*mu*jumpL[:]*order**2/dx
  fvU2[:] = shatU[:] - 2.*mu*jumpU[:]*order**2/dx
  fvD2[:] = shatD[:] - 2.*mu*jumpD[:]*order**2/dx

 # now we need to integrate along the boundary 
  fvRIG11 = np.zeros((nvars,order,Npx,Npy))
  fvLIG11 = np.zeros((nvars,order,Npx,Npy))

  fvRIG21 = np.zeros((nvars,order,Npx,Npy))
  fvLIG21 = np.zeros((nvars,order,Npx,Npy))

  fvUIG12 = np.zeros((nvars,order,Npx,Npy))
  fvDIG12 = np.zeros((nvars,order,Npx,Npy))

  fvUIG22 = np.zeros((nvars,order,Npx,Npy))
  fvDIG22 = np.zeros((nvars,order,Npx,Npy))


  fvR2I = np.zeros((nvars,order,Npx,Npy))
  fvL2I = np.zeros((nvars,order,Npx,Npy))
  fvU2I = np.zeros((nvars,order,Npx,Npy))
  fvD2I = np.zeros((nvars,order,Npx,Npy))


  for i in range(0,order):
    fvRIG11[:,i] = faceIntegrate(w[i],fvRG11)
    fvLIG11[:,i] = faceIntegrate(w[i],fvLG11)

    fvRIG21[:,i] = faceIntegrate(wp[i],fvRG21)
    fvLIG21[:,i] = faceIntegrate(wp[i],fvLG21)

    fvUIG12[:,i] = faceIntegrate(wp[i],fvUG12)
    fvDIG12[:,i] = faceIntegrate(wp[i],fvDG12)

    fvUIG22[:,i] = faceIntegrate(w[i],fvUG22)
    fvDIG22[:,i] = faceIntegrate(w[i],fvDG22)

    fvR2I[:,i] = faceIntegrate(w[i],fvR2)
    fvL2I[:,i] = faceIntegrate(w[i],fvL2)
    fvU2I[:,i] = faceIntegrate(w[i],fvU2)
    fvD2I[:,i] = faceIntegrate(w[i],fvD2)

  return fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvR2I,fvL2I,fvU2I,fvD2I


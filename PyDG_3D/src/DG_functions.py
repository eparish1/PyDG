from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs#,sendaEdgesGeneralSlab
from fluxSchemes import *
from navier_stokes import evalViscousFluxZNS_IP
from navier_stokes import evalViscousFluxYNS_IP
from navier_stokes import evalViscousFluxXNS_IP
from tensor_products import *
import matplotlib.pyplot as plt
import time




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

def shockCapturingSetViscosity(main):
  ### Shock capturing
  filta = np.zeros(np.shape(main.a.a))
  filta[:,0:main.order[0]-1,main.order[1]-1,main.order[2]-1] = 1.
  af = main.a.a*filta    #make filtered state
  uf = reconstructUGeneral(main,af)
  udff = (main.a.u - uf)**2
  # now compute switch
  Se_num = volIntegrate(main.weights0,main.weights1,main.weights2,udff) 
  Se_den = volIntegrate(main.weights0,main.weights1,main.weights2,main.a.u**2)
  Se = (Se_num + 1e-10)/(Se_den + 1.e-30)
  eps0 = 1.*main.dx/main.order[0]
  s0 =1./main.order[0]**4
  kap = 5.
  se = np.log10(Se)
  #print(np.amax(udff))
  epse = eps0/2.*(1. + np.sin(np.pi/(2.*kap)*(se - s0) ) )
  epse[se<s0-kap] = 0.
  epse[se>s0  + kap] = eps0
  #plt.clf()
  #print(np.amax(epse),np.amin(epse))
  #plt.plot(epse[0,:,0,0])
  #plt.ylim([1e-9,0.005])
  #plt.pause(0.001)
  #print(np.shape(main.mu),np.shape(epse) )
  main.mu = main.mu0 + epse[0]
  main.muR = main.mu0R + epse[0]
  main.muL = main.mu0L + epse[0]
  main.muU = main.mu0F + epse[0]
  main.muD = main.mu0D + epse[0]
  main.muF = main.mu0U + epse[0]
  main.muB = main.mu0B + epse[0]

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
  Delta = (main.dx/main.order[0]*main.dy/main.order[1]*main.dz/main.order[2])**(1./3.)
  S11 = ux
  S22 = vy
  S33 = wz
  S12 = 0.5*(uy + vx)
  S13 = 0.5*(uz + wx)
  S23 = 0.5*(vz + wy)
  S_mag = np.sqrt( 2.*(S11**2 + S22**2 + S33**2 + 2.*S12**2 + 2.*S13**2 + 2.*S23**2) )
  mut = u[0]*0.16**2*Delta**2*np.abs(S_mag)
  return mu0 + mut
#  print(np.mean(np.abs(mu)))

def getRHS_IP(main,MZ,eqns,args=[]):
  t0 = time.time()
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,MZ,eqns,args)
  ### Get interior vol terms
  eqns.evalFluxX(main.a.u,main.iFlux.fx,args)
  eqns.evalFluxY(main.a.u,main.iFlux.fy,args)
  eqns.evalFluxZ(main.a.u,main.iFlux.fz,args)

  upx,upy,upz = diffU(main.a.a,main)
  upx = upx*2./main.dx2[None,None,None,None,:,None,None]
  upy = upy*2./main.dy2[None,None,None,None,None,:,None]
  upz = upz*2./main.dz2[None,None,None,None,None,None,:]

#  print(np.mean(main.mu))
  UxR,UxL,UxU,UxD,UxF,UxB = diffUX_edge(main.a.a,main)
  UyR,UyL,UyU,UyD,UyF,UyB = diffUY_edge(main.a.a,main)
  UzR,UzL,UzU,UzD,UzF,UzB = diffUZ_edge(main.a.a,main)
  main.mu[:] = computeSmagViscosity(main,upx,upy,upz,main.mu0,main.a.u)
  main.muR[:] = computeSmagViscosity(main,UxR,UyR,UzR,main.mu0R,main.a.uR)
  main.muL[:] = computeSmagViscosity(main,UxL,UyL,UzL,main.mu0L,main.a.uL)
  main.muU[:] = computeSmagViscosity(main,UxU,UyU,UzU,main.mu0U,main.a.uU)
  main.muD[:] = computeSmagViscosity(main,UxD,UyD,UzD,main.mu0D,main.a.uD)
  main.muF[:] = computeSmagViscosity(main,UxF,UyF,UzF,main.mu0F,main.a.uF)
  main.muB[:] = computeSmagViscosity(main,UxB,UyB,UzB,main.mu0B,main.a.uB)
#  print(np.mean(main.mu[-1]))
#  print(np.mean(main.muR))
#  print(np.mean(main.muL))

  if (main.shock_capturing):
    shockCapturingSetViscosity(main)

  fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I = getViscousFlux(main,eqns) ##takes roughly 20% of the time


  fvGX = eqns.evalViscousFluxX(main,main.a.u,upx,upy,upz,main.mu)
  fvGY = eqns.evalViscousFluxY(main,main.a.u,upx,upy,upz,main.mu)
  fvGZ = eqns.evalViscousFluxZ(main,main.a.u,upx,upy,upz,main.mu)
  #print(np.linalg.norm(fvGX2 - fvGX))
  # Now form RHS
  t1 = time.time()
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])

  scale =  (2.*ord_arr0[:,None,None] + 1.)*(2.*ord_arr1[None,:,None] + 1.)*(2.*ord_arr2[None,None,:]+1.)/8.
  dxi = 2./main.dx2[None,None,None,:,None,None]*scale[:,:,:,None,None,None]
  dyi = 2./main.dy2[None,None,None,None,:,None]*scale[:,:,:,None,None,None]
  dzi = 2./main.dz2[None,None,None,None,None,:]*scale[:,:,:,None,None,None]

  v1ijk = volIntegrateGlob(main,main.iFlux.fx - fvGX,main.wp0,main.w1,main.w2)*dxi[None]
  v2ijk = volIntegrateGlob(main,main.iFlux.fy - fvGY,main.w0,main.wp1,main.w2)*dyi[None]
  v3ijk = volIntegrateGlob(main,main.iFlux.fz - fvGZ,main.w0,main.w1,main.wp2)*dzi[None]

  tmp = v1ijk + v2ijk + v3ijk
  tmp +=  (-main.iFlux.fRI[:,None,:,:] + main.iFlux.fLI[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None])*dxi[None]
  tmp +=  (-main.iFlux.fUI[:,:,None,:] + main.iFlux.fDI[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None])*dyi[None]
  tmp +=  (-main.iFlux.fFI[:,:,:,None] + main.iFlux.fBI[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None])*dzi[None]
  tmp +=  (fvRIG11[:,None,:,:]*main.wpedge0[None,:,None,None,1,None,None,None] + fvRIG21[:,None,:,:] + fvRIG31[:,None,:,:]  - (fvLIG11[:,None,:,:]*main.wpedge0[None,:,None,None,0,None,None,None] + fvLIG21[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None] + fvLIG31[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None]) )*dxi[None]
  tmp +=  (fvUIG12[:,:,None,:] + fvUIG22[:,:,None,:]*main.wpedge1[None,None,:,None,1,None,None,None] + fvUIG32[:,:,None,:]  - (fvDIG12[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None] + fvDIG22[:,:,None,:]*main.wpedge1[None,None,:,None,0,None,None,None] + fvDIG32[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None]) )*dyi[None]
  tmp +=  (fvFIG13[:,:,:,None] + fvFIG23[:,:,:,None] + fvFIG33[:,:,:,None]*main.wpedge2[None,None,None,:,1,None,None,None]  - (fvBIG13[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None] + fvBIG23[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None] + fvBIG33[:,:,:,None]*main.wpedge2[None,None,None,:,0,None,None,None]) )*dzi[None] 
  tmp +=  (fvR2I[:,None,:,:] - fvL2I[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None])*dxi[None] + (fvU2I[:,:,None,:] - fvD2I[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None])*dyi[None] + (fvF2I[:,:,:,None] - fvB2I[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None])*dzi[None]
 
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
  #dxi = 2./main.dx*scale
  #dyi = 2./main.dy*scale
  #dzi = 2./main.dz*scale
  dxi = 2./main.dx2[None,None,None,:,None,None]*scale[:,:,:,None,None,None]
  dyi = 2./main.dy2[None,None,None,None,:,None]*scale[:,:,:,None,None,None]
  dzi = 2./main.dz2[None,None,None,None,None,:]*scale[:,:,:,None,None,None]

  v1ijk = volIntegrateGlob(main,main.iFlux.fx ,MZ.wp0,MZ.w1,MZ.w2)*dxi[None]#,:,:,:,ne]
  v2ijk = volIntegrateGlob(main,main.iFlux.fy ,MZ.w0,MZ.wp1,MZ.w2)*dyi[None]#,:,:,:,None,None,None]
  v3ijk = volIntegrateGlob(main,main.iFlux.fz ,MZ.w0,MZ.w1,MZ.wp2)*dzi[None]#,:,:,:,None,None,None]

  tmp = v1ijk + v2ijk + v3ijk
  tmp +=  (-main.iFlux.fRI[:,None,:,:] + main.iFlux.fLI[:,None,:,:]*MZ.altarray0[None,:,None,None,None,None,None])*dxi[None]#,:,:,:,None,None,None]
  tmp +=  (-main.iFlux.fUI[:,:,None,:] + main.iFlux.fDI[:,:,None,:]*MZ.altarray1[None,None,:,None,None,None,None])*dyi[None]#,:,:,:,None,None,None]
  tmp +=  (-main.iFlux.fFI[:,:,:,None] + main.iFlux.fBI[:,:,:,None]*MZ.altarray2[None,None,None,:,None,None,None])*dzi[None]#,:,:,:,None,None,None]
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
 
  G11R,G21R,G31R = eqns.getGsX(main.a.uR,main,main.muR)
  G11L,G21L,G31L = eqns.getGsX(main.a.uL,main,main.muL)
  G12U,G22U,G32U = eqns.getGsY(main.a.uU,main,main.muU)
  G12D,G22D,G32D = eqns.getGsY(main.a.uD,main,main.muD)
  G13F,G23F,G33F = eqns.getGsZ(main.a.uF,main,main.muF)
  G13B,G23B,G33B = eqns.getGsZ(main.a.uB,main,main.muB)

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

  fvxR = eqns.evalViscousFluxX(main,main.a.uR,UxR,UyR,UzR,main.muR)
  fvxL = eqns.evalViscousFluxX(main,main.a.uL,UxL,UyL,UzL,main.muL)
  #fvxR_edge = eqns.evalViscousFluxX(main,main.a.uR_edge,UxR_edge,UyR_edge,UzR_edge)
  #fvxL_edge = eqns.evalViscousFluxX(main,main.a.uL_edge,UxL_edge,UyL_edge,UyL_edge)
#  fvxR_edge = fvxR[:,:,:,-1,:,:]
#  fvxL_edge = fvxL[:,:,:,0,:,:]

  fvyU = eqns.evalViscousFluxY(main,main.a.uU,UxU,UyU,UzU,main.muU)
  fvyD = eqns.evalViscousFluxY(main,main.a.uD,UxD,UyD,UzD,main.muD)
#  fvyU_edge = eqns.evalViscousFluxY(main,main.a.uU_edge,UxU_edge,UyU_edge,UzU_edge)
#  fvyD_edge = eqns.evalViscousFluxY(main,main.a.uD_edge,UxD_edge,UyD_edge,UzD_edge)
#  fvyU_edge = fvyU[:,:,:,:,-1,:]
#  fvyD_edge = fvyD[:,:,:,:,0 ,:]


  fvzF = eqns.evalViscousFluxZ(main,main.a.uF,UxF,UyF,UzF,main.muF)
  fvzB = eqns.evalViscousFluxZ(main,main.a.uB,UxB,UyB,UzB,main.muB)
#  fvzF_edge = eqns.evalViscousFluxZ(main,main.a.uF_edge,UxF_edge,UyF_edge,UzF_edge)
#  fvzB_edge = eqns.evalViscousFluxZ(main,main.a.uB_edge,UxB_edge,UyB_edge,UzB_edge)
#  fvzF_edge = fvzF[:,:,:,:,:,-1]
#  fvzB_edge = fvzB[:,:,:,:,:, 0]

  fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge = sendEdgesGeneralSlab_Derivs(fvxL,fvxR,fvyD,fvyU,fvzB,fvzF,main)


  shatR,shatL,shatU,shatD,shatF,shatB = centralFluxGeneral(fvxR,fvxL,fvyU,fvyD,fvzF,fvzB,fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge)
  jumpR,jumpL,jumpU,jumpD,jumpF,jumpB = computeJump(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
#  print(np.linalg.norm(main.muR))
  fvR2 = shatR - 6.*main.muR*3**2*jumpR/main.dx
  fvL2 = shatL - 6.*main.muL*3**2*jumpL/main.dx
  fvU2 = shatU - 6.*main.muU*3**2*jumpU/main.dy
  fvD2 = shatD - 6.*main.muD*3**2*jumpD/main.dy
  fvF2 = shatF - 6.*main.muF*3**2*jumpF/main.dz
  fvB2 = shatB - 6.*main.muB*3**2*jumpB/main.dz
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


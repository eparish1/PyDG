from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs#,sendaEdgesGeneralSlab
from fluxSchemes import *
from navier_stokes import evalViscousFluxZNS_IP
from navier_stokes import evalViscousFluxYNS_IP
from navier_stokes import evalViscousFluxXNS_IP
from navier_stokes import getGsNSX_FAST,getGsNSY_FAST,getGsNSZ_FAST
from tensor_products import *
import matplotlib.pyplot as plt
from smagorinsky import *
import time




def getFlux(main,MZ,eqns,args):
  # first reconstruct states
  main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB = main.basis.reconstructEdgesGeneral(main.a.a,main)
  #main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  #main.a.aR_edge[:],main.a.aL_edge[:],main.a.aU_edge[:],main.a.aD_edge[:],main.a.aF_edge[:],main.a.aB_edge[:] = sendaEdgesGeneralSlab(main.a.a,main)
  #main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = reconstructEdgeEdgesGeneral(main)
  inviscidFluxGen(main,eqns,main.iFlux,main.a,args)
  # now we need to integrate along the boundary 
  main.iFlux.fRI = main.basis.faceIntegrateGlob(main,main.iFlux.fRS,MZ.w1,MZ.w2,MZ.weights1,MZ.weights2)
  main.iFlux.fLI = main.basis.faceIntegrateGlob(main,main.iFlux.fLS,MZ.w1,MZ.w2,MZ.weights1,MZ.weights2)
  main.iFlux.fUI = main.basis.faceIntegrateGlob(main,main.iFlux.fUS,MZ.w0,MZ.w2,MZ.weights0,MZ.weights2)
  main.iFlux.fDI = main.basis.faceIntegrateGlob(main,main.iFlux.fDS,MZ.w0,MZ.w2,MZ.weights0,MZ.weights2)
  main.iFlux.fFI = main.basis.faceIntegrateGlob(main,main.iFlux.fFS,MZ.w0,MZ.w1,MZ.weights0,MZ.weights1)
  main.iFlux.fBI = main.basis.faceIntegrateGlob(main,main.iFlux.fBS,MZ.w0,MZ.w1,MZ.weights0,MZ.weights1)


def getRHS_reacting(main,MZ,eqns,args=[]):
  t0 = time.time()
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,MZ,eqns,args)
  ### Get interior vol terms
  eqns.evalFluxX(main.a.u,main.iFlux.fx,args)
  eqns.evalFluxY(main.a.u,main.iFlux.fy,args)
  eqns.evalFluxZ(main.a.u,main.iFlux.fz,args)

  t1 = time.time()
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])

  scale =  (2.*ord_arr0[:,None,None] + 1.)*(2.*ord_arr1[None,:,None] + 1.)*(2.*ord_arr2[None,None,:]+1.)/8.
  dxi = 2./main.dx2[None,None,None,:,None,None]*scale[:,:,:,None,None,None]
  dyi = 2./main.dy2[None,None,None,None,:,None]*scale[:,:,:,None,None,None]
  dzi = 2./main.dz2[None,None,None,None,None,:]*scale[:,:,:,None,None,None]

  if (eqns.viscous):
    fvGX,fvGY,fvGZ,fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I = getViscousFlux(main,eqns) ##takes roughly 20% of the 
    main.iFlux.fx -= fvGX
    main.iFlux.fy -= fvGY
    main.iFlux.fz -= fvGZ

  v1ijk = volIntegrateGlob(main,main.iFlux.fx,main.wp0,main.w1,main.w2)*dxi[None]
  v2ijk = volIntegrateGlob(main,main.iFlux.fy,main.w0,main.wp1,main.w2)*dyi[None]
  v3ijk = volIntegrateGlob(main,main.iFlux.fz,main.w0,main.w1,main.wp2)*dzi[None]

  tmp = v1ijk + v2ijk + v3ijk
  tmp +=  (-main.iFlux.fRI[:,None,:,:] + main.iFlux.fLI[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None])*dxi[None]
  tmp +=  (-main.iFlux.fUI[:,:,None,:] + main.iFlux.fDI[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None])*dyi[None]
  tmp +=  (-main.iFlux.fFI[:,:,:,None] + main.iFlux.fBI[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None])*dzi[None]
  if (eqns.viscous):
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


def getRHS(main,MZ,eqns,args=[]):
  t0 = time.time()
  main.basis.reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,MZ,eqns,args)
  ### Get interior vol terms
  eqns.evalFluxX(main.a.u,main.iFlux.fx,args)
  eqns.evalFluxY(main.a.u,main.iFlux.fy,args)
  eqns.evalFluxZ(main.a.u,main.iFlux.fz,args)

  t1 = time.time()
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])

  scale =  (2.*ord_arr0[:,None,None] + 1.)*(2.*ord_arr1[None,:,None] + 1.)*(2.*ord_arr2[None,None,:]+1.)/8.
  dxi = 2./main.dx2[None,None,None,:,None,None]*scale[:,:,:,None,None,None]
  dyi = 2./main.dy2[None,None,None,None,:,None]*scale[:,:,:,None,None,None]
  dzi = 2./main.dz2[None,None,None,None,None,:]*scale[:,:,:,None,None,None]

  if (eqns.viscous):
    fvGX,fvGY,fvGZ,fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I = getViscousFlux(main,eqns) ##takes roughly 20% of the 
    main.iFlux.fx -= fvGX
    main.iFlux.fy -= fvGY
    main.iFlux.fz -= fvGZ

  v1ijk = main.basis.volIntegrateGlob(main,main.iFlux.fx,main.wp0,main.w1,main.w2)*dxi[None]
  v2ijk = main.basis.volIntegrateGlob(main,main.iFlux.fy,main.w0,main.wp1,main.w2)*dyi[None]
  v3ijk = main.basis.volIntegrateGlob(main,main.iFlux.fz,main.w0,main.w1,main.wp2)*dzi[None]

  tmp = v1ijk + v2ijk + v3ijk
  tmp +=  (-main.iFlux.fRI[:,None,:,:] + main.iFlux.fLI[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None])*dxi[None]
  tmp +=  (-main.iFlux.fUI[:,:,None,:] + main.iFlux.fDI[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None])*dyi[None]
  tmp +=  (-main.iFlux.fFI[:,:,:,None] + main.iFlux.fBI[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None])*dzi[None]
  if (eqns.viscous):
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

  main.a.Upx,main.a.Upy,main.a.Upz = main.basis.diffU(main.a.a,main)

  main.a.UxR,main.a.UxL,main.a.UxU,main.a.UxD,main.a.UxF,main.a.UxB = main.basis.diffUX_edge(main.a.a,main)
  main.a.UyR,main.a.UyL,main.a.UyU,main.a.UyD,main.a.UyF,main.a.UyB = main.basis.diffUY_edge(main.a.a,main)
  main.a.UzR,main.a.UzL,main.a.UzU,main.a.UzD,main.a.UzF,main.a.UzB = main.basis.diffUZ_edge(main.a.a,main)
  if (eqns.turb_str == 'Smagorinsky'):
    staticSmagorinsky(main)

  fvGX = eqns.evalViscousFluxX(main,main.a.u,main.a.Upx,main.a.Upy,main.a.Upz,main.mu)
  fvGY = eqns.evalViscousFluxY(main,main.a.u,main.a.Upx,main.a.Upy,main.a.Upz,main.mu)
  fvGZ = eqns.evalViscousFluxZ(main,main.a.u,main.a.Upx,main.a.Upy,main.a.Upz,main.mu)

  uhatR,uhatL,uhatU,uhatD,uhatF,uhatB = centralFluxGeneral(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
 
  fvRG11,fvRG21,fvRG31 = getGsNSX_FAST(main.a.uR,main,main.muR,main.a.uR - uhatR)
  fvLG11,fvLG21,fvLG31 = getGsNSX_FAST(main.a.uL,main,main.muL,main.a.uL - uhatL)

  fvUG12,fvUG22,fvUG32 = getGsNSY_FAST(main.a.uU,main,main.muU,main.a.uU-uhatU)
  fvDG12,fvDG22,fvDG32 = getGsNSY_FAST(main.a.uD,main,main.muD,main.a.uD-uhatD)

  fvFG13,fvFG23,fvFG33 = getGsNSZ_FAST(main.a.uF,main,main.muF,main.a.uF-uhatF)
  fvBG13,fvBG23,fvBG33 = getGsNSZ_FAST(main.a.uB,main,main.muB,main.a.uB-uhatB)


  fvxR = eqns.evalViscousFluxX(main,main.a.uR,main.a.UxR,main.a.UyR,main.a.UzR,main.muR)
  fvxL = eqns.evalViscousFluxX(main,main.a.uL,main.a.UxL,main.a.UyL,main.a.UzL,main.muL)

  fvyU = eqns.evalViscousFluxY(main,main.a.uU,main.a.UxU,main.a.UyU,main.a.UzU,main.muU)
  fvyD = eqns.evalViscousFluxY(main,main.a.uD,main.a.UxD,main.a.UyD,main.a.UzD,main.muD)

  fvzF = eqns.evalViscousFluxZ(main,main.a.uF,main.a.UxF,main.a.UyF,main.a.UzF,main.muF)
  fvzB = eqns.evalViscousFluxZ(main,main.a.uB,main.a.UxB,main.a.UyB,main.a.UzB,main.muB)

  fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge = sendEdgesGeneralSlab_Derivs(fvxL,fvxR,fvyD,fvyU,fvzB,fvzF,main)

  shatR,shatL,shatU,shatD,shatF,shatB = centralFluxGeneral(fvxR,fvxL,fvyU,fvyD,fvzF,fvzB,fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge)
  jumpR,jumpL,jumpU,jumpD,jumpF,jumpB = computeJump(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
  fvR2 = shatR - 6.*main.muR*3**2*jumpR/main.dx2[None,None,None,:,None,None]
  fvL2 = shatL - 6.*main.muL*3**2*jumpL/main.dx2[None,None,None,:,None,None]
  fvU2 = shatU - 6.*main.muU*3**2*jumpU/main.dy2[None,None,None,None,:,None]
  fvD2 = shatD - 6.*main.muD*3**2*jumpD/main.dy2[None,None,None,None,:,None]
  fvF2 = shatF - 6.*main.muF*3**2*jumpF/main.dz2[None,None,None,None,None,:]
  fvB2 = shatB - 6.*main.muB*3**2*jumpB/main.dz2[None,None,None,None,None,:]
  fvRIG11 = main.basis.faceIntegrateGlob(main,fvRG11,main.w1,main.w2,main.weights1,main.weights2) 
  fvLIG11 = main.basis.faceIntegrateGlob(main,fvLG11,main.w1,main.w2,main.weights1,main.weights2)  
  fvRIG21 = main.basis.faceIntegrateGlob(main,fvRG21,main.wp1,main.w2,main.weights1,main.weights2) 
  fvLIG21 = main.basis.faceIntegrateGlob(main,fvLG21,main.wp1,main.w2,main.weights1,main.weights2) 
  fvRIG31 = main.basis.faceIntegrateGlob(main,fvRG31,main.w1,main.wp2,main.weights1,main.weights2) 
  fvLIG31 = main.basis.faceIntegrateGlob(main,fvLG31,main.w1,main.wp2,main.weights1,main.weights2) 

  fvUIG12 = main.basis.faceIntegrateGlob(main,fvUG12,main.wp0,main.w2,main.weights0,main.weights2)  
  fvDIG12 = main.basis.faceIntegrateGlob(main,fvDG12,main.wp0,main.w2,main.weights0,main.weights2)  
  fvUIG22 = main.basis.faceIntegrateGlob(main,fvUG22,main.w0,main.w2,main.weights0,main.weights2)  
  fvDIG22 = main.basis.faceIntegrateGlob(main,fvDG22,main.w0,main.w2,main.weights0,main.weights2)  
  fvUIG32 = main.basis.faceIntegrateGlob(main,fvUG32,main.w0,main.wp2,main.weights0,main.weights2)  
  fvDIG32 = main.basis.faceIntegrateGlob(main,fvDG32,main.w0,main.wp2,main.weights0,main.weights2)  

  fvFIG13 = main.basis.faceIntegrateGlob(main,fvFG13,main.wp0,main.w1,main.weights0,main.weights1)   
  fvBIG13 = main.basis.faceIntegrateGlob(main,fvBG13,main.wp0,main.w1,main.weights0,main.weights1)  
  fvFIG23 = main.basis.faceIntegrateGlob(main,fvFG23,main.w0,main.wp1,main.weights0,main.weights1)  
  fvBIG23 = main.basis.faceIntegrateGlob(main,fvBG23,main.w0,main.wp1,main.weights0,main.weights1)  
  fvFIG33 = main.basis.faceIntegrateGlob(main,fvFG33,main.w0,main.w1,main.weights0,main.weights1)  
  fvBIG33 = main.basis.faceIntegrateGlob(main,fvBG33,main.w0,main.w1,main.weights0,main.weights1)  

  fvR2I = main.basis.faceIntegrateGlob(main,fvR2,main.w1,main.w2,main.weights1,main.weights2)    
  fvL2I = main.basis.faceIntegrateGlob(main,fvL2,main.w1,main.w2,main.weights1,main.weights2)  
  fvU2I = main.basis.faceIntegrateGlob(main,fvU2,main.w0,main.w2,main.weights0,main.weights2)  
  fvD2I = main.basis.faceIntegrateGlob(main,fvD2,main.w0,main.w2,main.weights0,main.weights2)  
  fvF2I = main.basis.faceIntegrateGlob(main,fvF2,main.w0,main.w1,main.weights0,main.weights1)
  fvB2I = main.basis.faceIntegrateGlob(main,fvB2,main.w0,main.w1,main.weights0,main.weights1)
  return fvGX,fvGY,fvGZ,fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I




#def dVdU(V):
#  U = entropy_to_conservative(V)
#  V_U[0,0] = (U1**2*gamma + U2**4*gamma + U3**4*gamma + U1**4 + U2**4 + U3**4 + 2*U1**2*U2**2 + 2*U1**2*U3**2 + 2*U2**2*U3**2 + \
#           2*U1**2*U2**2*gamma + 4*U0**2*U4**2*gamma + 2*U1**2*U3**2*gamma + 2*U2**2*U3**2*gamma - 4*U0*U1**2*U4 - 4*U0*U2**2*U4 - 4*U0*U3**2*U4 - \
#           4*U0*U1**2*U4*gamma - 4*U0*U2**2*U4*gamma - 4*U0*U3**2*U4*gamma)/(U0*(gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[0,1] = -(2*U1*(U1**2 + U2**2 + U3**2 - 4*U0*U4))/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[0,2] = -(2*U2*(U1**2 + U2**2 + U3**2 - 4*U0*U4))/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[0,3] = -(2*U3*(U1**2 + U2**2 + U3**2 - 4*U0*U4))/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[0,4] = -(4*U0**2*U4)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[1,0] = -(2*U1*(U1**2 + U2**2 + U3**2))/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[1,1] = (2*U0*(U1**2 - U2**2 - U3**2 + 2*U0*U4))/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[1,2] = (4*U0*U1*U2)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[1,3] = (4*U0*U1*U3)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[1,4] = -(4*U0**2*U1)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[2,0] = -(2*U2*(U1**2 + U2**2 + U3**2))/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[2,1] = (4*U0*U1*U2)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[2,2] = (2*U0*(- U1**2 + U2**2 - U3**2 + 2*U0*U4))/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[2,3] = (4*U0*U2*U3)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[2,4] = -(4*U0**2*U2)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[3,0] = -(2*U3*(U1**2 + U2**2 + U3**2))/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[3,1] = (4*U0*U1*U3)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[3,2] = (4*U0*U2*U3)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[3,3] = (2*U0*(- U1**2 - U2**2 + U3**2 + 2*U0*U4))/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[3,4] = -(4*U0**2*U3)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[4,1] = -(4*U0**2*U1)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[4,2] = -(4*U0**2*U2)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[4,3] = -(4*U0**2*U3)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)
#  V_U[4,4] = (4*U0**3)/((gamma - 1)*(U1**2 + U2**2 + U3**2 - 2*U0*U4)**2)


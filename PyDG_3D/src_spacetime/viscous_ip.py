from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs#,sendaEdgesGeneralSlab
from fluxSchemes import *
from navier_stokes import evalViscousFluxZNS_IP
from navier_stokes import evalViscousFluxYNS_IP
from navier_stokes import evalViscousFluxXNS_IP
from navier_stokes import getGsNSX_FAST,getGsNSY_FAST,getGsNSZ_FAST
from eos_functions import *
from tensor_products import *
from chemistry_values import *
from smagorinsky import *
import numexpr as ne
import time


def computeJump(uR,uL,uU,uD,uF,uB,uR_edge,uL_edge,uU_edge,uD_edge,uF_edge,uB_edge):
  nvars,order1,order2,order3,Npx,Npy,Npz,Npt = np.shape(uR)
  nvars,order0,order1,order3,Npx,Npy,Npz,Npt = np.shape(uF)
#  nvars,order0,order1,Npx,Npy,Npz = np.shape(uF)
  jumpRLS = np.zeros((nvars,order1,order2,order3,Npx+1,Npy,Npz,Npt))
  jumpUDS = np.zeros((nvars,order0,order2,order3,Npx,Npy+1,Npz,Npt))
  jumpFBS = np.zeros((nvars,order0,order1,order3,Npx,Npy,Npz+1,Npt))

  jumpRLS[:,:,:,:,1:-1,:,:] = uR[:,:,:,:,0:-1,:,:] - uL[:,:,:,:,1::,:,:]
  jumpRLS[:,:,:,:,-1  ,:,:] = uR[:,:,:,:,  -1,:,:] - uR_edge
  jumpRLS[:,:,:,:,0   ,:,:] = uL_edge - uL[:,:,:,:,  0,:,:]
  jumpUDS[:,:,:,:,:,1:-1,:] = uU[:,:,:,:,:,0:-1,:] - uD[:,:,:,:,:,1::,:]
  jumpUDS[:,:,:,:,:,  -1,:] = uU[:,:,:,:,:,  -1,:] - uU_edge
  jumpUDS[:,:,:,:,:,0   ,:] = uD_edge - uD[:,:,:,:,:,   0,:]
  jumpFBS[:,:,:,:,:,:,1:-1] = uF[:,:,:,:,:,:,0:-1] - uB[:,:,:,:,:,:,1::]
  jumpFBS[:,:,:,:,:,:,  -1] = uF[:,:,:,:,:,:,  -1] - uF_edge
  jumpFBS[:,:,:,:,:,:,0   ] = uB_edge - uB[:,:,:,:,:,:,   0]

  return jumpRLS,jumpUDS,jumpFBS


def addViscousContribution_IP(main,MZ,eqns):
  gamma = 1.4
  Pr = 0.72
  a = main.a.a

  main.a.Upx,main.a.Upy,main.a.Upz = main.basis.diffU(main.a.a,main)

  main.a.UxR,main.a.UxL,main.a.UxU,main.a.UxD,main.a.UxF,main.a.UxB = main.basis.diffUX_edge(main.a.a,main)
  main.a.UyR,main.a.UyL,main.a.UyU,main.a.UyD,main.a.UyF,main.a.UyB = main.basis.diffUY_edge(main.a.a,main)
  main.a.UzR,main.a.UzL,main.a.UzU,main.a.UzD,main.a.UzF,main.a.UzB = main.basis.diffUZ_edge(main.a.a,main)
  if (eqns.turb_str == 'Smagorinsky'):
  #  staticSmagorinsky(main)
    computeDynSmagViscosity(main,main.a.Upx,main.a.Upy,main.a.Upz,main.mu0,main.a.u)
  if (main.reacting):
    main.mu[1::] = computeDiffusionConstants(main.a.u[5::]/main.a.u[0],main.a.u,main)
  fvGX = eqns.evalViscousFluxX(main,main.a.u,main.a.Upx,main.a.Upy,main.a.Upz,main.mus)
  fvGY = eqns.evalViscousFluxY(main,main.a.u,main.a.Upx,main.a.Upy,main.a.Upz,main.mus)
  fvGZ = eqns.evalViscousFluxZ(main,main.a.u,main.a.Upx,main.a.Upy,main.a.Upz,main.mus)
  main.iFlux.fx -= fvGX
  main.iFlux.fy -= fvGY
  main.iFlux.fz -= fvGZ


  ## Compute the penalty term
  centralFluxGeneral2(main.iFlux.fRLS,main.iFlux.fUDS,main.iFlux.fFBS,main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
  fvRG11,fvRG21,fvRG31 = eqns.getGsX(main.a.uR,main,main.mus,main.a.uR - main.iFlux.fRLS[:,:,:,:,1::])
  fvLG11,fvLG21,fvLG31 = eqns.getGsX(main.a.uL,main,main.mus,main.a.uL - main.iFlux.fRLS[:,:,:,:,0:-1])

  fvUG12,fvUG22,fvUG32 = eqns.getGsY(main.a.uU,main,main.mus,main.a.uU-main.iFlux.fUDS[:,:,:,:,:,1::])
  fvDG12,fvDG22,fvDG32 = eqns.getGsY(main.a.uD,main,main.mus,main.a.uD-main.iFlux.fUDS[:,:,:,:,:,0:-1])

  fvFG13,fvFG23,fvFG33 = eqns.getGsZ(main.a.uF,main,main.mus,main.a.uF-main.iFlux.fFBS[:,:,:,:,:,:,1::])
  fvBG13,fvBG23,fvBG33 = eqns.getGsZ(main.a.uB,main,main.mus,main.a.uB-main.iFlux.fFBS[:,:,:,:,:,:,0:-1])

  ## Integrate over the faces
  main.iFlux.fRLI[:,:,:,:,1::] = main.basis.faceIntegrateGlob(main,fvRG11*main.J_edge_det[0][None,:,:,None,1::,:,:,None],main.w1,main.w2,main.w3,main.weights1,main.weights2,main.weights3) 
  main.RHS[:] += main.iFlux.fRLI[:,None,:,:,:,1::]*main.wpedge0[None,:,None,None,1,None,None,None,None,None] 
  main.iFlux.fRLI[:,:,:,:,0:-1] = main.basis.faceIntegrateGlob(main,fvLG11*main.J_edge_det[0][None,:,:,None,0:-1,:,:,None],main.w1,main.w2,main.w3,main.weights1,main.weights2,main.weights3)  
  main.RHS[:] -= main.iFlux.fRLI[:,None,:,:,:,0:-1]*main.wpedge0[None,:,None,None,0,None,None,None,None,None] 

  main.iFlux.fRLI[:,:,:,:,1::] = main.basis.faceIntegrateGlob(main,fvRG21*main.J_edge_det[0][None,:,:,None,1::,:,:,None],main.wp1,main.w2,main.w3,main.weights1,main.weights2,main.weights3) 
  main.RHS[:] += main.iFlux.fRLI[:,None,:,:,:,1::]
  main.iFlux.fRLI[:,:,:,:,0:-1] = main.basis.faceIntegrateGlob(main,fvLG21*main.J_edge_det[0][None,:,:,None,0:-1,:,:,None],main.wp1,main.w2,main.w3,main.weights1,main.weights2,main.weights3) 
  main.RHS[:] -= main.iFlux.fRLI[:,None,:,:,:,0:-1]*main.altarray0[None,:,None,None,None,None,None,None,None]

  main.iFlux.fRLI[:,:,:,:,1::] = main.basis.faceIntegrateGlob(main,fvRG31*main.J_edge_det[0][None,:,:,None,1::,:,:,None],main.w1,main.wp2,main.w3,main.weights1,main.weights2,main.weights3) 
  main.RHS[:] += main.iFlux.fRLI[:,None,:,:,:,1::] 
  main.iFlux.fRLI[:,:,:,:,0:-1] = main.basis.faceIntegrateGlob(main,fvLG31*main.J_edge_det[0][None,:,:,None,0:-1,:,:,None],main.w1,main.wp2,main.w3,main.weights1,main.weights2,main.weights3) 
  main.RHS[:] -= main.iFlux.fRLI[:,None,:,:,:,0:-1]*main.altarray0[None,:,None,None,None,None,None,None,None]

  main.iFlux.fUDI[:,:,:,:,:,1::] = main.basis.faceIntegrateGlob(main,fvUG12*main.J_edge_det[1][None,:,:,None,:,1::,:,None],main.wp0,main.w2,main.w3,main.weights0,main.weights2,main.weights3)  
  main.RHS[:] += main.iFlux.fUDI[:,:,None,:,:,:,1::]
  main.iFlux.fUDI[:,:,:,:,:,0:-1] = main.basis.faceIntegrateGlob(main,fvDG12*main.J_edge_det[1][None,:,:,None,:,0:-1,:,None],main.wp0,main.w2,main.w3,main.weights0,main.weights2,main.weights3)  
  main.RHS[:] -= main.iFlux.fUDI[:,:,None,:,:,:,0:-1]*main.altarray1[None,None,:,None,None,None,None,None,None] 

  main.iFlux.fUDI[:,:,:,:,:,1::] = main.basis.faceIntegrateGlob(main,fvUG22*main.J_edge_det[1][None,:,:,None,:,1::,:,None],main.w0,main.w2,main.w3,main.weights0,main.weights2,main.weights3)  
  main.RHS[:] += main.iFlux.fUDI[:,:,None,:,:,:,1::]*main.wpedge1[None,None,:,None,1,None,None,None,None,None]
  main.iFlux.fUDI[:,:,:,:,:,0:-1] = main.basis.faceIntegrateGlob(main,fvDG22*main.J_edge_det[1][None,:,:,None,:,0:-1,:,None],main.w0,main.w2,main.w3,main.weights0,main.weights2,main.weights3)  
  main.RHS[:] -= main.iFlux.fUDI[:,:,None,:,:,:,0:-1]*main.wpedge1[None,None,:,None,0,None,None,None,None,None] 

  main.iFlux.fUDI[:,:,:,:,:,1::] = main.basis.faceIntegrateGlob(main,fvUG32*main.J_edge_det[1][None,:,:,None,:,1::,:,None],main.w0,main.wp2,main.w3,main.weights0,main.weights2,main.weights3)  
  main.RHS[:] += main.iFlux.fUDI[:,:,None,:,:,:,1::]
  main.iFlux.fUDI[:,:,:,:,:,0:-1] = main.basis.faceIntegrateGlob(main,fvDG32*main.J_edge_det[1][None,:,:,None,:,0:-1,:,None],main.w0,main.wp2,main.w3,main.weights0,main.weights2,main.weights3)  
  main.RHS[:] -= main.iFlux.fUDI[:,:,None,:,:,:,0:-1]*main.altarray1[None,None,:,None,None,None,None,None,None]

  main.iFlux.fFBI[:,:,:,:,:,:,1::] = main.basis.faceIntegrateGlob(main,fvFG13*main.J_edge_det[2][None,:,:,None,:,:,1::,None],main.wp0,main.w1,main.w3,main.weights0,main.weights1,main.weights3)   
  main.RHS[:] += main.iFlux.fFBI[:,:,:,None,:,:,:,1::]
  main.iFlux.fFBI[:,:,:,:,:,:,0:-1] = main.basis.faceIntegrateGlob(main,fvBG13*main.J_edge_det[2][None,:,:,None,:,:,0:-1,None],main.wp0,main.w1,main.w3,main.weights0,main.weights1,main.weights3)  
  main.RHS[:] -= main.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*main.altarray2[None,None,None,:,None,None,None,None,None]

  main.iFlux.fFBI[:,:,:,:,:,:,1::] = main.basis.faceIntegrateGlob(main,fvFG23*main.J_edge_det[2][None,:,:,None,:,:,1::,None],main.w0,main.wp1,main.w3,main.weights0,main.weights1,main.weights3)  
  main.RHS[:] += main.iFlux.fFBI[:,:,:,None,:,:,:,1::]
  main.iFlux.fFBI[:,:,:,:,:,:,0:-1] = main.basis.faceIntegrateGlob(main,fvBG23*main.J_edge_det[2][None,:,:,None,:,:,0:-1,None],main.w0,main.wp1,main.w3,main.weights0,main.weights1,main.weights3)  
  main.RHS[:] -= main.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*main.altarray2[None,None,None,:,None,None,None,None,None]


  main.iFlux.fFBI[:,:,:,:,:,:,1::] = main.basis.faceIntegrateGlob(main,fvFG33*main.J_edge_det[2][None,:,:,None,:,:,1::,None],main.w0,main.w1,main.w3,main.weights0,main.weights1,main.weights3)  
  main.RHS[:] += main.iFlux.fFBI[:,:,:,None,:,:,:,1::]*main.wpedge2[None,None,None,:,1,None,None,None,None,None]
  main.iFlux.fFBI[:,:,:,:,:,:,0:-1] = main.basis.faceIntegrateGlob(main,fvBG33*main.J_edge_det[2][None,:,:,None,:,:,0:-1,None],main.w0,main.w1,main.w3,main.weights0,main.weights1,main.weights3)  
  main.RHS[:] -= main.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*main.wpedge2[None,None,None,:,0,None,None,None,None,None]




  main.iFlux.fR[:] = eqns.evalViscousFluxX(main,main.a.uR,main.a.UxR,main.a.UyR,main.a.UzR,main.mu)
  main.iFlux.fL[:] = eqns.evalViscousFluxX(main,main.a.uL,main.a.UxL,main.a.UyL,main.a.UzL,main.mu)

  main.iFlux.fU[:] = eqns.evalViscousFluxY(main,main.a.uU,main.a.UxU,main.a.UyU,main.a.UzU,main.mu)
  main.iFlux.fD[:] = eqns.evalViscousFluxY(main,main.a.uD,main.a.UxD,main.a.UyD,main.a.UzD,main.mu)

  main.iFlux.fF[:] = eqns.evalViscousFluxZ(main,main.a.uF,main.a.UxF,main.a.UyF,main.a.UzF,main.mu)
  main.iFlux.fB[:] = eqns.evalViscousFluxZ(main,main.a.uB,main.a.UxB,main.a.UyB,main.a.UzB,main.mu)

  fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge = sendEdgesGeneralSlab_Derivs(main.iFlux.fL,main.iFlux.fR,main.iFlux.fD,main.iFlux.fU,main.iFlux.fB,main.iFlux.fF,main)

  centralFluxGeneral2(main.iFlux.fRLS,main.iFlux.fUDS,main.iFlux.fFBS,main.iFlux.fR,main.iFlux.fL,main.iFlux.fU,main.iFlux.fD,main.iFlux.fF,main.iFlux.fB,fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge)
  jumpRLS,jumpUDS,jumpFBS = computeJump(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
  fvRL2 = main.iFlux.fRLS - 1.*main.mus*main.order[0]**2*jumpRLS/main.dx
  fvUD2 = main.iFlux.fUDS - 1.*main.mus*main.order[1]**2*jumpUDS/main.dy
  fvFB2 = main.iFlux.fFBS - 1.*main.mus*main.order[2]**2*jumpFBS/main.dz



  main.iFlux.fRLI[:] = main.basis.faceIntegrateGlob(main,fvRL2*main.J_edge_det[0][None,:,:,None,:,:,:,None],main.w1,main.w2,main.w3,main.weights1,main.weights2,main.weights3)    
  main.iFlux.fUDI[:] = main.basis.faceIntegrateGlob(main,fvUD2*main.J_edge_det[1][None,:,:,None,:,:,:,None],main.w0,main.w2,main.w3,main.weights0,main.weights2,main.weights3)  
  main.iFlux.fFBI[:] = main.basis.faceIntegrateGlob(main,fvFB2*main.J_edge_det[2][None,:,:,None,:,:,:,None],main.w0,main.w1,main.w3,main.weights0,main.weights1,main.weights3)

  main.RHS[:] +=  main.iFlux.fRLI[:,None,:,:,:,1::]
  tmp = main.iFlux.fRLI[:,None,:,:,:,0:-1] * main.altarray0[None,:,None,None,None,None,None,None,None]
  main.RHS[:] -=  tmp
  main.RHS[:] +=  main.iFlux.fUDI[:,:,None,:,:,:,1::]
  tmp = main.iFlux.fUDI[:,:,None,:,:,:,0:-1] * main.altarray1[None,None,:,None,None,None,None,None,None]
  main.RHS[:] -=  tmp
  main.RHS[:] +=  main.iFlux.fFBI[:,:,:,None,:,:,:,1::]
  tmp = main.iFlux.fFBI[:,:,:,None,:,:,:,0:-1] * main.altarray2[None,None,None,:,None,None,None,None,None]
  main.RHS[:] -=  tmp


from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs
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


def addViscousContribution_IP(regionManager,eqns):
  #print('IP Not up to date. Use BR1')
  #sys.exit()
  gamma = 1.4
  Pr = 0.72
  for region in regionManager.region:
    a = region.a.a
    region.a.Upx,region.a.Upy,region.a.Upz = region.basis.diffU(region.a.a,region)
    region.a.UxR,region.a.UxL,region.a.UxU,region.a.UxD,region.a.UxF,region.a.UxB , region.a.UyR,region.a.UyL,region.a.UyU,region.a.UyD,region.a.UyF,region.a.UyB , region.a.UzR,region.a.UzL,region.a.UzU,region.a.UzD,region.a.UzF,region.a.UzB = region.basis.diffUXYZ_edge(region.a.a,region)

    #region.a.UxR,region.a.UxL,region.a.UxU,region.a.UxD,region.a.UxF,region.a.UxB = region.basis.diffUX_edge(region.a.a,region)
    #region.a.UyR,region.a.UyL,region.a.UyU,region.a.UyD,region.a.UyF,region.a.UyB = region.basis.diffUY_edge(region.a.a,region)
    #region.a.UzR,region.a.UzL,region.a.UzU,region.a.UzD,region.a.UzF,region.a.UzB = region.basis.diffUZ_edge(region.a.a,region)
    if (eqns.turb_str == 'Smagorinsky'):
    #  staticSmagorinsky(region)
      computeDynSmagViscosity(region,region.a.Upx,region.a.Upy,region.a.Upz,region.mu0,region.a.u)
    if (region.reacting):
      region.mu[1::] = computeDiffusionConstants(region.a.u[5::]/region.a.u[0],region.a.u,region)
    fvGX = eqns.evalViscousFluxX(region,region.a.u,region.a.Upx,region.a.Upy,region.a.Upz,region.mus)
    fvGY = eqns.evalViscousFluxY(region,region.a.u,region.a.Upx,region.a.Upy,region.a.Upz,region.mus)
    fvGZ = eqns.evalViscousFluxZ(region,region.a.u,region.a.Upx,region.a.Upy,region.a.Upz,region.mus)
    region.iFlux.fx -= fvGX
    region.iFlux.fy -= fvGY
    region.iFlux.fz -= fvGZ


    ## Compute the penalty term
    centralFluxGeneral2(region.iFlux.fRLS,region.iFlux.fUDS,region.iFlux.fFBS,region.a.uR,region.a.uL,region.a.uU,region.a.uD,region.a.uF,region.a.uB,region.a.uR_edge,region.a.uL_edge,region.a.uU_edge,region.a.uD_edge,region.a.uF_edge,region.a.uB_edge)
    fvRG11,fvRG21,fvRG31 = eqns.getGsX(region.a.uR,region,region.mus,region.a.uR - region.iFlux.fRLS[:,:,:,:,1::])
    fvLG11,fvLG21,fvLG31 = eqns.getGsX(region.a.uL,region,region.mus,region.a.uL - region.iFlux.fRLS[:,:,:,:,0:-1])
  
    fvUG12,fvUG22,fvUG32 = eqns.getGsY(region.a.uU,region,region.mus,region.a.uU-region.iFlux.fUDS[:,:,:,:,:,1::])
    fvDG12,fvDG22,fvDG32 = eqns.getGsY(region.a.uD,region,region.mus,region.a.uD-region.iFlux.fUDS[:,:,:,:,:,0:-1])
  
    fvFG13,fvFG23,fvFG33 = eqns.getGsZ(region.a.uF,region,region.mus,region.a.uF-region.iFlux.fFBS[:,:,:,:,:,:,1::])
    fvBG13,fvBG23,fvBG33 = eqns.getGsZ(region.a.uB,region,region.mus,region.a.uB-region.iFlux.fFBS[:,:,:,:,:,:,0:-1])
  
    ## Integrate over the faces
    region.iFlux.fRLI[:,:,:,:,1::] = region.basis.faceIntegrateGlob(region,fvRG11*region.J_edge_det[0][None,:,:,None,1::,:,:,None],region.w1,region.w2,region.w3,region.weights1,region.weights2,region.weights3) 
    region.RHS[:] += region.iFlux.fRLI[:,None,:,:,:,1::]*region.wpedge0[None,:,None,None,1,None,None,None,None,None] 
    region.iFlux.fRLI[:,:,:,:,0:-1] = region.basis.faceIntegrateGlob(region,fvLG11*region.J_edge_det[0][None,:,:,None,0:-1,:,:,None],region.w1,region.w2,region.w3,region.weights1,region.weights2,region.weights3)  
    region.RHS[:] -= region.iFlux.fRLI[:,None,:,:,:,0:-1]*region.wpedge0[None,:,None,None,0,None,None,None,None,None] 
  
    region.iFlux.fRLI[:,:,:,:,1::] = region.basis.faceIntegrateGlob(region,fvRG21*region.J_edge_det[0][None,:,:,None,1::,:,:,None],region.wp1,region.w2,region.w3,region.weights1,region.weights2,region.weights3) 
    region.RHS[:] += region.iFlux.fRLI[:,None,:,:,:,1::]
    region.iFlux.fRLI[:,:,:,:,0:-1] = region.basis.faceIntegrateGlob(region,fvLG21*region.J_edge_det[0][None,:,:,None,0:-1,:,:,None],region.wp1,region.w2,region.w3,region.weights1,region.weights2,region.weights3) 
    region.RHS[:] -= region.iFlux.fRLI[:,None,:,:,:,0:-1]*region.altarray0[None,:,None,None,None,None,None,None,None]
  
    region.iFlux.fRLI[:,:,:,:,1::] = region.basis.faceIntegrateGlob(region,fvRG31*region.J_edge_det[0][None,:,:,None,1::,:,:,None],region.w1,region.wp2,region.w3,region.weights1,region.weights2,region.weights3) 
    region.RHS[:] += region.iFlux.fRLI[:,None,:,:,:,1::] 
    region.iFlux.fRLI[:,:,:,:,0:-1] = region.basis.faceIntegrateGlob(region,fvLG31*region.J_edge_det[0][None,:,:,None,0:-1,:,:,None],region.w1,region.wp2,region.w3,region.weights1,region.weights2,region.weights3) 
    region.RHS[:] -= region.iFlux.fRLI[:,None,:,:,:,0:-1]*region.altarray0[None,:,None,None,None,None,None,None,None]
 
    region.iFlux.fUDI[:,:,:,:,:,1::] = region.basis.faceIntegrateGlob(region,fvUG12*region.J_edge_det[1][None,:,:,None,:,1::,:,None],region.wp0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS[:] += region.iFlux.fUDI[:,:,None,:,:,:,1::]
    region.iFlux.fUDI[:,:,:,:,:,0:-1] = region.basis.faceIntegrateGlob(region,fvDG12*region.J_edge_det[1][None,:,:,None,:,0:-1,:,None],region.wp0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS[:] -= region.iFlux.fUDI[:,:,None,:,:,:,0:-1]*region.altarray1[None,None,:,None,None,None,None,None,None] 
  
    region.iFlux.fUDI[:,:,:,:,:,1::] = region.basis.faceIntegrateGlob(region,fvUG22*region.J_edge_det[1][None,:,:,None,:,1::,:,None],region.w0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS[:] += region.iFlux.fUDI[:,:,None,:,:,:,1::]*region.wpedge1[None,None,:,None,1,None,None,None,None,None]
    region.iFlux.fUDI[:,:,:,:,:,0:-1] = region.basis.faceIntegrateGlob(region,fvDG22*region.J_edge_det[1][None,:,:,None,:,0:-1,:,None],region.w0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS[:] -= region.iFlux.fUDI[:,:,None,:,:,:,0:-1]*region.wpedge1[None,None,:,None,0,None,None,None,None,None] 
  
    region.iFlux.fUDI[:,:,:,:,:,1::] = region.basis.faceIntegrateGlob(region,fvUG32*region.J_edge_det[1][None,:,:,None,:,1::,:,None],region.w0,region.wp2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS[:] += region.iFlux.fUDI[:,:,None,:,:,:,1::]
    region.iFlux.fUDI[:,:,:,:,:,0:-1] = region.basis.faceIntegrateGlob(region,fvDG32*region.J_edge_det[1][None,:,:,None,:,0:-1,:,None],region.w0,region.wp2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS[:] -= region.iFlux.fUDI[:,:,None,:,:,:,0:-1]*region.altarray1[None,None,:,None,None,None,None,None,None]
  
    region.iFlux.fFBI[:,:,:,:,:,:,1::] = region.basis.faceIntegrateGlob(region,fvFG13*region.J_edge_det[2][None,:,:,None,:,:,1::,None],region.wp0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)   
    region.RHS[:] += region.iFlux.fFBI[:,:,:,None,:,:,:,1::]
    region.iFlux.fFBI[:,:,:,:,:,:,0:-1] = region.basis.faceIntegrateGlob(region,fvBG13*region.J_edge_det[2][None,:,:,None,:,:,0:-1,None],region.wp0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)  
    region.RHS[:] -= region.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*region.altarray2[None,None,None,:,None,None,None,None,None]
  
    region.iFlux.fFBI[:,:,:,:,:,:,1::] = region.basis.faceIntegrateGlob(region,fvFG23*region.J_edge_det[2][None,:,:,None,:,:,1::,None],region.w0,region.wp1,region.w3,region.weights0,region.weights1,region.weights3)  
    region.RHS[:] += region.iFlux.fFBI[:,:,:,None,:,:,:,1::]
    region.iFlux.fFBI[:,:,:,:,:,:,0:-1] = region.basis.faceIntegrateGlob(region,fvBG23*region.J_edge_det[2][None,:,:,None,:,:,0:-1,None],region.w0,region.wp1,region.w3,region.weights0,region.weights1,region.weights3)  
    region.RHS[:] -= region.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*region.altarray2[None,None,None,:,None,None,None,None,None]
  
  
    region.iFlux.fFBI[:,:,:,:,:,:,1::] = region.basis.faceIntegrateGlob(region,fvFG33*region.J_edge_det[2][None,:,:,None,:,:,1::,None],region.w0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)  
    region.RHS[:] += region.iFlux.fFBI[:,:,:,None,:,:,:,1::]*region.wpedge2[None,None,None,:,1,None,None,None,None,None]
    region.iFlux.fFBI[:,:,:,:,:,:,0:-1] = region.basis.faceIntegrateGlob(region,fvBG33*region.J_edge_det[2][None,:,:,None,:,:,0:-1,None],region.w0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)  
    region.RHS[:] -= region.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*region.wpedge2[None,None,None,:,0,None,None,None,None,None]
  
 


    region.iFlux.fR[:] = eqns.evalViscousFluxX(region,region.a.uR,region.a.UxR,region.a.UyR,region.a.UzR,region.mu)
    region.iFlux.fL[:] = eqns.evalViscousFluxX(region,region.a.uL,region.a.UxL,region.a.UyL,region.a.UzL,region.mu)
  
    region.iFlux.fU[:] = eqns.evalViscousFluxY(region,region.a.uU,region.a.UxU,region.a.UyU,region.a.UzU,region.mu)
    region.iFlux.fD[:] = eqns.evalViscousFluxY(region,region.a.uD,region.a.UxD,region.a.UyD,region.a.UzD,region.mu)
  
    region.iFlux.fF[:] = eqns.evalViscousFluxZ(region,region.a.uF,region.a.UxF,region.a.UyF,region.a.UzF,region.mu)
    region.iFlux.fB[:] = eqns.evalViscousFluxZ(region,region.a.uB,region.a.UxB,region.a.UyB,region.a.UzB,region.mu)
  
    fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge = sendEdgesGeneralSlab_Derivs(region.iFlux.fL,region.iFlux.fR,region.iFlux.fD,region.iFlux.fU,region.iFlux.fB,region.iFlux.fF,region,regionManager)
  
    centralFluxGeneral2(region.iFlux.fRLS,region.iFlux.fUDS,region.iFlux.fFBS,region.iFlux.fR,region.iFlux.fL,region.iFlux.fU,region.iFlux.fD,region.iFlux.fF,region.iFlux.fB,fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge)
    jumpRLS,jumpUDS,jumpFBS = computeJump(region.a.uR,region.a.uL,region.a.uU,region.a.uD,region.a.uF,region.a.uB,region.a.uR_edge,region.a.uL_edge,region.a.uU_edge,region.a.uD_edge,region.a.uF_edge,region.a.uB_edge)
    fvRL2 = region.iFlux.fRLS - 1.*region.mus*region.order[0]**2*jumpRLS/region.dx
    fvUD2 = region.iFlux.fUDS - 1.*region.mus*region.order[1]**2*jumpUDS/region.dy
    fvFB2 = region.iFlux.fFBS - 1.*region.mus*region.order[2]**2*jumpFBS/region.dz
 
 
 
    region.iFlux.fRLI[:] = region.basis.faceIntegrateGlob(region,fvRL2*region.J_edge_det[0][None,:,:,None,:,:,:,None],region.w1,region.w2,region.w3,region.weights1,region.weights2,region.weights3)    
    region.iFlux.fUDI[:] = region.basis.faceIntegrateGlob(region,fvUD2*region.J_edge_det[1][None,:,:,None,:,:,:,None],region.w0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.iFlux.fFBI[:] = region.basis.faceIntegrateGlob(region,fvFB2*region.J_edge_det[2][None,:,:,None,:,:,:,None],region.w0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)
  
    region.RHS[:] +=  region.iFlux.fRLI[:,None,:,:,:,1::]
    tmp = region.iFlux.fRLI[:,None,:,:,:,0:-1] * region.altarray0[None,:,None,None,None,None,None,None,None]
    region.RHS[:] -=  tmp
    region.RHS[:] +=  region.iFlux.fUDI[:,:,None,:,:,:,1::]
    tmp = region.iFlux.fUDI[:,:,None,:,:,:,0:-1] * region.altarray1[None,None,:,None,None,None,None,None,None]
    region.RHS[:] -=  tmp
    region.RHS[:] +=  region.iFlux.fFBI[:,:,:,None,:,:,:,1::]
    tmp = region.iFlux.fFBI[:,:,:,None,:,:,:,0:-1] * region.altarray2[None,None,None,:,None,None,None,None,None]
    region.RHS[:] -=  tmp
  

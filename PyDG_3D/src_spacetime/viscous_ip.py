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
from pylab import *

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


def computeArtificialViscocity(region):
    pmax = np.amax(region.order)
    epsilon = np.zeros(np.shape(region.a.a[0]),dtype=region.a.a.dtype)
    filtarray = np.zeros(np.shape(region.a.a))
    filtarray[:,0:-1,0:-1,0:-1,:] = 1.
    af = region.a.a*1.
    af[:,-1::,-1::,-1::] = 0.
    u = region.a.u
    uf = region.basis.reconstructUGeneral(region,af)
    #print('ufn',np.linalg.norm(uf))
    Se = region.basis.volIntegrate(region.weights0,region.weights1,region.weights2,region.weights3, (u - uf)*(u - uf) )/( region.basis.volIntegrate( region.weights0,region.weights1,region.weights2,region.weights3, u*u + 1e-30) ) 
    sensor = np.log10(Se[0] + 1e-6)
    kappa = 2. 
    s0  = 1./np.amax(region.order)**4 
    tol = s0
    eps0 = region.dx/pmax
    epsilon[0,0,0,0,:] =  0.1*eps0/2.*np.exp(sensor)#(1. + np.sin( np.pi*(sensor - s0)/ (2.*kappa) ) )
    #epsilon[0,0,0,0,:] =  eps0/2.*(1. + np.sin( np.pi*(sensor - s0)/ (2.*kappa) ) )

    #print(np.shape(sensor))
    #print(np.amin(sensor),np.amax(sensor))
    #print(np.amin(epsilon),np.amax(epsilon))
    #plot(epsilon[0,0,0,0].flatten())
    #pause(0.0001)
    #clf()
    return epsilon

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


    epsilon_a = computeArtificialViscocity(region)
    epsilon_u = reconstructUGeneral_tensordot(region,epsilon_a[None])

    epsR,epsL,epsU,epsD,epsF,epsB = reconstructEdgesGeneral_tensordot(epsilon_a[None],region)
    epsRL = np.append(epsL,epsR[:,:,:,:,-1,None],axis=4)
    epsUD = np.append(epsD,epsU[:,:,:,:,:,-1,None],axis=5)
    epsFB = np.append(epsB,epsF[:,:,:,:,:,-1,None],axis=6)

    fvGX = eqns.evalViscousFluxX(region,region.a.u,region.a.Upx,region.a.Upy,region.a.Upz,region.mus + epsilon_u[0])
    fvGY = eqns.evalViscousFluxY(region,region.a.u,region.a.Upx,region.a.Upy,region.a.Upz,region.mus + epsilon_u[0])
    fvGZ = eqns.evalViscousFluxZ(region,region.a.u,region.a.Upx,region.a.Upy,region.a.Upz,region.mus + epsilon_u[0])
    region.iFlux.fx -= fvGX
    region.iFlux.fy -= fvGY
    region.iFlux.fz -= fvGZ


    ## Compute the penalty term
    centralFluxGeneral2(region.iFlux.fRLS,region.iFlux.fUDS,region.iFlux.fFBS,region.a.uR,region.a.uL,region.a.uU,region.a.uD,region.a.uF,region.a.uB,region.a.uR_edge,region.a.uL_edge,region.a.uU_edge,region.a.uD_edge,region.a.uF_edge,region.a.uB_edge)
    tmp = region.iFlux.fRLS[:,:,:,:,1::]

    fvRG11,fvRG21,fvRG31 = eqns.getGsX(region.a.uR,region,region.mus + epsR[0],region.a.uR - region.iFlux.fRLS[:,:,:,:,1::])
    fvLG11,fvLG21,fvLG31 = eqns.getGsX(region.a.uL,region,region.mus + epsL[0],region.a.uL - region.iFlux.fRLS[:,:,:,:,0:-1])
  
    fvUG12,fvUG22,fvUG32 = eqns.getGsY(region.a.uU,region,region.mus + epsU[0],region.a.uU-region.iFlux.fUDS[:,:,:,:,:,1::])
    fvDG12,fvDG22,fvDG32 = eqns.getGsY(region.a.uD,region,region.mus + epsD[0],region.a.uD-region.iFlux.fUDS[:,:,:,:,:,0:-1])
  
    fvFG13,fvFG23,fvFG33 = eqns.getGsZ(region.a.uF,region,region.mus + epsF[0],region.a.uF-region.iFlux.fFBS[:,:,:,:,:,:,1::])
    fvBG13,fvBG23,fvBG33 = eqns.getGsZ(region.a.uB,region,region.mus + epsB[0],region.a.uB-region.iFlux.fFBS[:,:,:,:,:,:,0:-1])

    ## Integrate over the faces
    region.iFlux.fRLI[:,:,:,:,1::] = region.basis.faceIntegrateGlob(region,fvRG11*region.J_edge_det[0][None,:,:,None,1::,:,:,None],region.w1,region.w2,region.w3,region.weights1,region.weights2,region.weights3) 
    region.RHS[:] += region.iFlux.fRLI[:,None,:,:,:,1::]*region.wpedge0[None,:,None,None,None,None,None,None,None,1] 

    region.iFlux.fRLI[:,:,:,:,0:-1] = region.basis.faceIntegrateGlob(region,fvLG11*region.J_edge_det[0][None,:,:,None,0:-1,:,:,None],region.w1,region.w2,region.w3,region.weights1,region.weights2,region.weights3)  
    region.RHS[:] -= region.iFlux.fRLI[:,None,:,:,:,0:-1]*region.wpedge0[None,:,None,None,0,None,None,None,None,None]

    tmp = region.iFlux.fRLI[:,:,:,:,0:-1]


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
    fvRL2 = region.iFlux.fRLS - (region.mus + epsRL)*region.order[0]**2*jumpRLS/region.dx
    fvUD2 = region.iFlux.fUDS - (region.mus + epsUD)*region.order[1]**2*jumpUDS/region.dy
    fvFB2 = region.iFlux.fFBS - (region.mus + epsFB)*region.order[2]**2*jumpFBS/region.dz
 
 
 
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
  


def computeJump_hyper(uR,uL,uU,uD,uF,uB,uR_edge,uL_edge,uU_edge,uD_edge,uF_edge,uB_edge,region):
  cell_ijk = region.cell_ijk
  jumpR = uR[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] - uL[:,:,:,:, (cell_ijk[5][0]+1)%region.Npx,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]
  jumpL = uR[:,:,:,:,cell_ijk[5][0]-1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] - uL[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]

  jumpU = uU[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] - uD[:,:,:,:, cell_ijk[5][0],(cell_ijk[6][0]+1)%region.Npy,cell_ijk[7][0],cell_ijk[8][0]]
  jumpD = uU[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0]-1,cell_ijk[7][0],cell_ijk[8][0]] - uD[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]

  jumpF = uF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] - uB[:,:,:,:, cell_ijk[5][0],cell_ijk[6][0],(cell_ijk[7][0]+1)%region.Npz,cell_ijk[8][0]]
  jumpB = uF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]-1,cell_ijk[8][0]] - uB[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]

  return jumpR,jumpL,jumpU,jumpD,jumpF,jumpB 




def addViscousContribution_IP_hyper(regionManager,eqns):
  #print('IP Not up to date. Use BR1')
  #sys.exit()
  gamma = 1.4
  Pr = 0.72
  for regionCounter in range(0,np.size(regionManager.region)):
    region = regionManager.region[regionCounter]
    regionSampleMesh = regionManager.regionSampleMesh[regionCounter]
    a = region.a.a
    cell_ijk = region.cell_ijk
    stencil_ijk = region.stencil_ijk
    region.a.a_hyper_cell = region.a.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]
    region.a.a_hyper_stencil = region.a.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]

    region.a.Upx_hyper_cell,region.a.Upy_hyper_cell,region.a.Upz_hyper_cell = diffU_tensordot_sample(region.a.a_hyper_cell,region,regionSampleMesh.Jinv_cell)
    region.a.Upx_hyper_stencil,region.a.Upy_hyper_stencil,region.a.Upz_hyper_stencil = diffU_tensordot_sample(region.a.a_hyper_stencil,region,regionSampleMesh.Jinv_stencil)

    region.a.UxR,region.a.UxL,region.a.UxU,region.a.UxD,region.a.UxF,region.a.UxB , region.a.UyR,region.a.UyL,region.a.UyU,region.a.UyD,region.a.UyF,region.a.UyB , region.a.UzR,region.a.UzL,region.a.UzU,region.a.UzD,region.a.UzF,region.a.UzB = diffUXYZ_edge_tensordot_hyper(region.a.a_hyper_stencil,region,regionSampleMesh.Jinv_stencil)

    fvGX = eqns.evalViscousFluxX(region,region.a.u_hyper_cell,region.a.Upx_hyper_cell,region.a.Upy_hyper_cell,region.a.Upz_hyper_cell,region.mus)
    fvGY = eqns.evalViscousFluxY(region,region.a.u_hyper_cell,region.a.Upx_hyper_cell,region.a.Upy_hyper_cell,region.a.Upz_hyper_cell,region.mus)
    fvGZ = eqns.evalViscousFluxZ(region,region.a.u_hyper_cell,region.a.Upx_hyper_cell,region.a.Upy_hyper_cell,region.a.Upz_hyper_cell,region.mus)

    region.iFlux.fx_hyper -= fvGX
    region.iFlux.fy_hyper -= fvGY
    region.iFlux.fz_hyper -= fvGZ


    ## Compute the penalty term
    #centralFluxGeneral2(region.iFlux.fRLS,region.iFlux.fUDS,region.iFlux.fFBS,region.a.uR,region.a.uL,region.a.uU,region.a.uD,region.a.uF,region.a.uB,region.a.uR_edge,region.a.uL_edge,region.a.uU_edge,region.a.uD_edge,region.a.uF_edge,region.a.uB_edge)
    #region.iFlux.fRS[:],region.iFlux.fLS[:],region.iFlux.fUS[:],region.iFlux.fDS[:],region.iFlux.fFS[:], region.iFlux.fBS[:] = centralFluxGeneral(region.a.uR,region.a.uL,region.a.uU,region.a.uD,region.a.uF,region.a.uB,region.a.uR_edge,region.a.uL_edge,region.a.uU_edge,region.a.uD_edge,region.a.uF_edge,region.a.uB_edge)
    region.iFlux.fLS[:] = 0.
    region.iFlux.fRS[:] = 0.

    generalFluxGen_hyper(region,eqns,region.iFlux,region.a,centralFlux,cell_ijk,[])
    #print(np.shape(region.a.uL_edge),np.shape(region.a.uL))
    fvRG11,fvRG21,fvRG31 = eqns.getGsX(region.a.uR[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region,region.mus,region.a.uR[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] - region.iFlux.fRS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
    fvLG11,fvLG21,fvLG31 = eqns.getGsX(region.a.uL[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region,region.mus,region.a.uL[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] - region.iFlux.fLS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
  
    fvUG12,fvUG22,fvUG32 = eqns.getGsY(region.a.uU[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region,region.mus,region.a.uU[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]-region.iFlux.fUS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
    fvDG12,fvDG22,fvDG32 = eqns.getGsY(region.a.uD[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region,region.mus,region.a.uD[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]-region.iFlux.fDS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
  
    fvFG13,fvFG23,fvFG33 = eqns.getGsZ(region.a.uF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region,region.mus,region.a.uF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]-region.iFlux.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
    fvBG13,fvBG23,fvBG33 = eqns.getGsZ(region.a.uB[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region,region.mus,region.a.uB[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]-region.iFlux.fBS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])

    ## Integrate over the faces
    region.iFlux.fRI_hyper = region.basis.faceIntegrateGlob(region,fvRG11*region.J_edge_det_hyper_x1p1[None,:,:,None],region.w1,region.w2,region.w3,region.weights1,region.weights2,region.weights3)

    region.RHS_hyper[:] += region.iFlux.fRI_hyper[:,None]*region.wpedge0[None,:,None,None,None,None,1] 

    region.iFlux.fLI_hyper[:] = region.basis.faceIntegrateGlob(region,fvLG11*region.J_edge_det_hyper_x1[None,:,:,None],region.w1,region.w2,region.w3,region.weights1,region.weights2,region.weights3)  

    region.RHS_hyper[:] -= region.iFlux.fLI_hyper[:,None]*region.wpedge0[None,:,None,None,None,None,0] 
    region.iFlux.fRI_hyper[:] = region.basis.faceIntegrateGlob(region,fvRG21*region.J_edge_det_hyper_x1p1[None,:,:,None],region.wp1,region.w2,region.w3,region.weights1,region.weights2,region.weights3) 
    region.RHS_hyper[:] += region.iFlux.fRI_hyper[:,None]

    region.iFlux.fLI_hyper[:] = region.basis.faceIntegrateGlob(region,fvLG21*region.J_edge_det_hyper_x1[None,:,:,None],region.wp1,region.w2,region.w3,region.weights1,region.weights2,region.weights3) 
    region.RHS_hyper[:] -= region.iFlux.fLI_hyper[:,None]*region.altarray0[None,:,None,None,None,None]
      
    region.iFlux.fRI_hyper[:] = region.basis.faceIntegrateGlob(region,fvRG31*region.J_edge_det_hyper_x1p1[None,:,:,None],region.w1,region.wp2,region.w3,region.weights1,region.weights2,region.weights3) 
    region.RHS_hyper[:] += region.iFlux.fRI_hyper[:,None] 
    region.iFlux.fLI_hyper[:] = region.basis.faceIntegrateGlob(region,fvLG31*region.J_edge_det_hyper_x1[None,:,:,None],region.w1,region.wp2,region.w3,region.weights1,region.weights2,region.weights3) 
    region.RHS_hyper[:] -= region.iFlux.fLI_hyper[:,None]*region.altarray0[None,:,None,None,None,None]
    region.iFlux.fUI_hyper[:] = region.basis.faceIntegrateGlob(region,fvUG12*region.J_edge_det_hyper_x2p1[None,:,:,None],region.wp0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS_hyper[:] += region.iFlux.fUI_hyper[:,:,None]
    region.iFlux.fDI_hyper[:] = region.basis.faceIntegrateGlob(region,fvDG12*region.J_edge_det_hyper_x2[None,:,:,None],region.wp0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS_hyper[:] -= region.iFlux.fDI_hyper[:,:,None]*region.altarray1[None,None,:,None,None,None] 
  
    region.iFlux.fUI_hyper[:] = region.basis.faceIntegrateGlob(region,fvUG22*region.J_edge_det_hyper_x2p1[None,:,:,None],region.w0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS_hyper[:] += region.iFlux.fUI_hyper[:,:,None]*region.wpedge1[None,None,:,1,None,None,None]

    region.iFlux.fDI_hyper[:] = region.basis.faceIntegrateGlob(region,fvDG22*region.J_edge_det_hyper_x2[None,:,:,None],region.w0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS_hyper[:] -= region.iFlux.fDI_hyper[:,:,None]*region.wpedge1[None,None,:,0,None,None,None] 
  
    region.iFlux.fUI_hyper[:] = region.basis.faceIntegrateGlob(region,fvUG32*region.J_edge_det_hyper_x2p1[None,:,:,None],region.w0,region.wp2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS_hyper[:] += region.iFlux.fUI_hyper[:,:,None]
    region.iFlux.fDI_hyper[:] = region.basis.faceIntegrateGlob(region,fvDG32*region.J_edge_det_hyper_x2[None,:,:,None],region.w0,region.wp2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.RHS_hyper[:] -= region.iFlux.fDI_hyper[:,:,None]*region.altarray1[None,None,:,None,None,None]
 
    region.iFlux.fFI_hyper[:] = region.basis.faceIntegrateGlob(region,fvFG13*region.J_edge_det_hyper_x3p1[None,:,:,None],region.wp0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)   
    region.RHS_hyper[:] += region.iFlux.fFI_hyper[:,:,:,None]
    region.iFlux.fBI_hyper[:] = region.basis.faceIntegrateGlob(region,fvBG13*region.J_edge_det_hyper_x3[None,:,:,None],region.wp0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)  
    region.RHS_hyper[:] -= region.iFlux.fBI_hyper[:,:,:,None]*region.altarray2[None,None,None,:,None,None]
  
    region.iFlux.fFI_hyper[:] = region.basis.faceIntegrateGlob(region,fvFG23*region.J_edge_det_hyper_x3p1[None,:,:,None],region.w0,region.wp1,region.w3,region.weights0,region.weights1,region.weights3)  
    region.RHS_hyper[:] += region.iFlux.fFI_hyper[:,:,:,None]
    region.iFlux.fBI_hyper[:] = region.basis.faceIntegrateGlob(region,fvBG23*region.J_edge_det_hyper_x3[None,:,:,None],region.w0,region.wp1,region.w3,region.weights0,region.weights1,region.weights3)  
    region.RHS_hyper[:] -= region.iFlux.fBI_hyper[:,:,:,None]*region.altarray2[None,None,None,:,None,None]
  
  
    region.iFlux.fFI_hyper[:] = region.basis.faceIntegrateGlob(region,fvFG33*region.J_edge_det_hyper_x3p1[None,:,:,None],region.w0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)  
    region.RHS_hyper[:] += region.iFlux.fFI_hyper[:,:,:,None]*region.wpedge2[None,None,None,:,1,None,None] 
    region.iFlux.fBI_hyper[:] = region.basis.faceIntegrateGlob(region,fvBG33*region.J_edge_det_hyper_x3[None,:,:,None],region.w0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)  
    region.RHS_hyper[:] -= region.iFlux.fBI_hyper[:,:,:,None]*region.wpedge2[None,None,None,:,0,None,None]

    region.iFlux.fR[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = eqns.evalViscousFluxX(region,region.a.uR[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],region.a.UxR,region.a.UyR,region.a.UzR,region.mu)
    region.iFlux.fL[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = eqns.evalViscousFluxX(region,region.a.uL[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],region.a.UxL,region.a.UyL,region.a.UzL,region.mu)
  
    region.iFlux.fU[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = eqns.evalViscousFluxY(region,region.a.uU[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],region.a.UxU,region.a.UyU,region.a.UzU,region.mu)
    region.iFlux.fD[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = eqns.evalViscousFluxY(region,region.a.uD[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],region.a.UxD,region.a.UyD,region.a.UzD,region.mu)
  
    region.iFlux.fF[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = eqns.evalViscousFluxZ(region,region.a.uF[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],region.a.UxF,region.a.UyF,region.a.UzF,region.mu)
    region.iFlux.fB[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = eqns.evalViscousFluxZ(region,region.a.uB[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],region.a.UxB,region.a.UyB,region.a.UzB,region.mu)
    
 
    fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge = sendEdgesGeneralSlab_Derivs(region.iFlux.fL,region.iFlux.fR,region.iFlux.fD,region.iFlux.fU,region.iFlux.fB,region.iFlux.fF,region,regionManager)

#    centralFluxGeneral2(region.iFlux.fRLS,region.iFlux.fUDS,region.iFlux.fFBS,region.iFlux.fR,region.iFlux.fL,region.iFlux.fU,region.iFlux.fD,region.iFlux.fF,region.iFlux.fB,fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge)
#=================
    ## Cells
    #cell_ijk = region.cell_ijk
    fluxArgs = []
    ftmp = np.zeros(np.shape( region.iFlux.fRS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fRS.dtype)
    centralFlux(eqns,ftmp,region,region.iFlux.fR[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.iFlux.fL[:,:,:,:, (cell_ijk[5][0]+1)%region.Npx,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.normals[0][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
    region.iFlux.fRS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
    ##now to same thing for fL
    ftmp = np.zeros(np.shape( region.iFlux.fLS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fLS.dtype)
    centralFlux(eqns,ftmp,region,region.iFlux.fR[:,:,:,:,cell_ijk[5][0]-1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.iFlux.fL[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],-region.normals[1][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
    region.iFlux.fLS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
    ## now get contributions from boundary
    ftmp = np.zeros(np.shape(region.iFlux.fRS[:,:,:,:,  -1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fRS.dtype)
    centralFlux(eqns,ftmp,region,region.iFlux.fR[:,:,:,:,  -1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.iFlux.fR_edge[:,:,:,:,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.normals[0][:,-1,cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
    region.iFlux.fRS[:,:,:,:,-1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
    ftmp = np.zeros(np.shape(region.iFlux.fLS[:,:,:,:,  0,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fLS.dtype)
    centralFlux(eqns,ftmp,region,region.iFlux.fL_edge[:,:,:,:,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.iFlux.fL[:,:,:,:,0,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],-region.normals[1][:,0,cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
    region.iFlux.fLS[:,:,:,:,0 ,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
  
    ## Get the up and down fluxes
    ftmp = np.zeros(np.shape( region.iFlux.fUS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fUS.dtype)
    centralFlux(eqns,ftmp,region,region.iFlux.fU[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.iFlux.fD[:,:,:,:, cell_ijk[5][0],(cell_ijk[6][0]+1)%region.Npy,cell_ijk[7][0],cell_ijk[8][0]],region.normals[2][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
    region.iFlux.fUS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
  
    ftmp = np.zeros(np.shape( region.iFlux.fDS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fDS.dtype)
    centralFlux(eqns,ftmp,region,region.iFlux.fU[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0]-1,cell_ijk[7][0],cell_ijk[8][0]],region.iFlux.fD[:,:,:,:, cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],-region.normals[3][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
    region.iFlux.fDS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
    ## now get contributions from the boundary
    ftmp = np.zeros(np.shape(region.iFlux.fUS[:,:,:,:,cell_ijk[5][0],-1,cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fUS.dtype)
    centralFlux(eqns,ftmp,region,region.iFlux.fU[:,:,:,:,cell_ijk[5][0],  -1,cell_ijk[7][0],cell_ijk[8][0]],region.iFlux.fU_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[7][0],cell_ijk[8][0]],region.normals[2][:,cell_ijk[5][0],-1,cell_ijk[7][0] ],fluxArgs)
    region.iFlux.fUS[:,:,:,:,cell_ijk[5][0],-1,cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
  
    ftmp = np.zeros(np.shape(region.iFlux.fDS[:,:,:,:,cell_ijk[5][0],0,cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fDS.dtype)
    centralFlux(eqns,ftmp,region,region.iFlux.fD_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[7][0],cell_ijk[8][0]],region.iFlux.fD[:,:,:,:,cell_ijk[5][0],0,cell_ijk[7][0],cell_ijk[8][0]],-region.normals[3][:,cell_ijk[5][0],0,cell_ijk[7][0] ],fluxArgs)
    region.iFlux.fDS[:,:,:,:,cell_ijk[5][0],0,cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
  
    # now get up and down
    ftmp = np.zeros(np.shape( region.iFlux.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fFS.dtype)
    centralFlux(eqns,ftmp,region,region.iFlux.fF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.iFlux.fB[:,:,:,:, cell_ijk[5][0],cell_ijk[6][0],(cell_ijk[7][0]+1)%region.Npz,cell_ijk[8][0]],region.normals[4][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
    region.iFlux.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
  
    ftmp = np.zeros(np.shape( region.iFlux.fBS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fBS.dtype)
    centralFlux(eqns,ftmp,region,region.iFlux.fF[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]-1,cell_ijk[8][0]],region.iFlux.fB[:,:,:,:, cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],-region.normals[5][:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0] ],fluxArgs)
    region.iFlux.fBS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = ftmp[:]*1.
    ## now get contributions from the boundary
    ftmp = np.zeros(np.shape(region.iFlux.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],-1,cell_ijk[8][0]] ) ,dtype=region.iFlux.fFS.dtype)
    centralFlux(eqns,ftmp,region,region.iFlux.fF[:,:,:,:,cell_ijk[5][0], cell_ijk[6][0],-1,cell_ijk[8][0]],region.iFlux.fF_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[8][0]],region.normals[4][:,cell_ijk[5][0],cell_ijk[6][0],-1 ],fluxArgs)
    region.iFlux.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],-1,cell_ijk[8][0]] = ftmp[:]*1.
  
    centralFlux(eqns,ftmp,region,region.iFlux.fB_edge[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[8][0]],region.iFlux.fB[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],0,cell_ijk[8][0]],-region.normals[5][:,cell_ijk[5][0],cell_ijk[6][0],0 ],fluxArgs)
    region.iFlux.fBS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],0,cell_ijk[8][0]] = ftmp[:]*1.
  
    #generalFluxGen_hyper(region,eqns,region.iFlux,region.a,centralFlux,cell_ijk,[])

    jumpR,jumpL,jumpU,jumpD,jumpF,jumpB = computeJump_hyper(region.a.uR,region.a.uL,region.a.uU,region.a.uD,region.a.uF,region.a.uB,region.a.uR_edge,region.a.uL_edge,region.a.uU_edge,region.a.uD_edge,region.a.uF_edge,region.a.uB_edge,region)

    fvR2 = region.iFlux.fRS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],0,cell_ijk[8][0]] - 1.*region.mus*region.order[0]**2*jumpR/region.dx
    fvL2 = region.iFlux.fLS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],0,cell_ijk[8][0]] - 1.*region.mus*region.order[0]**2*jumpL/region.dx

    fvU2 = region.iFlux.fUS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],0,cell_ijk[8][0]] - 1.*region.mus*region.order[1]**2*jumpU/region.dy
    fvD2 = region.iFlux.fDS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],0,cell_ijk[8][0]] - 1.*region.mus*region.order[1]**2*jumpD/region.dy

    fvF2 = region.iFlux.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],0,cell_ijk[8][0]] - 1.*region.mus*region.order[2]**2*jumpF/region.dz
    fvB2 = region.iFlux.fBS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],0,cell_ijk[8][0]] - 1.*region.mus*region.order[2]**2*jumpB/region.dz


 
    region.iFlux.fRI_hyper[:] = region.basis.faceIntegrateGlob(region,fvR2*region.J_edge_det_hyper_x1p1[None,:,:,None],region.w1,region.w2,region.w3,region.weights1,region.weights2,region.weights3)   
    region.iFlux.fLI_hyper[:] = region.basis.faceIntegrateGlob(region,fvL2*region.J_edge_det_hyper_x1[None,:,:,None],region.w1,region.w2,region.w3,region.weights1,region.weights2,region.weights3)   

    region.iFlux.fUI_hyper[:] = region.basis.faceIntegrateGlob(region,fvU2*region.J_edge_det_hyper_x2p1[None,:,:,None],region.w0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)  
    region.iFlux.fDI_hyper[:] = region.basis.faceIntegrateGlob(region,fvD2*region.J_edge_det_hyper_x2[None,:,:,None],region.w0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)  

    region.iFlux.fFI_hyper[:] = region.basis.faceIntegrateGlob(region,fvF2*region.J_edge_det_hyper_x3p1[None,:,:,None],region.w0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)
    region.iFlux.fBI_hyper[:] = region.basis.faceIntegrateGlob(region,fvB2*region.J_edge_det_hyper_x3[None,:,:,None],region.w0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)
 


    region.RHS_hyper[:] += region.iFlux.fRI_hyper[:,None]
    region.RHS_hyper[:] -=  region.iFlux.fLI_hyper[:,None]*region.altarray0[None,:,None,None,None,None]
    region.RHS_hyper[:] +=  region.iFlux.fUI_hyper[:,:,None]
    region.RHS_hyper[:] -=  region.iFlux.fDI_hyper[:,:,None]*region.altarray1[None,None,:,None,None,None]
    region.RHS_hyper[:] +=  region.iFlux.fFI_hyper[:,:,:,None]
    region.RHS_hyper[:] -=  region.iFlux.fBI_hyper[:,:,:,None]*region.altarray2[None,None,None,:,None,None]

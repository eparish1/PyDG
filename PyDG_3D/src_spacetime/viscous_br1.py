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





def addSecondaryViscousContribution_BR1(region,MZ,eqns):
  # get interior viscous flux
  # note we use the vFlux array since BR1 has more unknowns
  eqns.evalViscousFluxX(region,region.a.u,region.vFlux.fx)
  eqns.evalViscousFluxY(region,region.a.u,region.vFlux.fy)
  eqns.evalViscousFluxZ(region,region.a.u,region.vFlux.fz)
  # first reconstruct states

  generalFluxGen(region,eqns,region.vFlux,region.a,eqns.evalViscousFlux,[])
#  print('test')

  # now we need to integrate along the boundary
  a = region.vFlux.fRLS
  Jdet = region.J_edge_det[0][None,:,:,None,:,:,:,None]
  #f = ne.evaluate("a*Jdet")  
  f = a*Jdet
  region.vFlux.fRLI[:] = region.basis.faceIntegrateGlob(region,f,MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  a = region.vFlux.fUDS
  Jdet = region.J_edge_det[1][None,:,:,None,:,:,:,None]
  #f = ne.re_evaluate()  
  f = a*Jdet
  region.vFlux.fUDI[:] = region.basis.faceIntegrateGlob(region,f,MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  a = region.vFlux.fFBS
  Jdet = region.J_edge_det[2][None,:,:,None,:,:,:,None]
  #f = ne.re_evaluate()  
  f = a*Jdet
  region.vFlux.fFBI[:] = region.basis.faceIntegrateGlob(region,f,MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)

   
  region.b.a[:] =  region.vFlux.fRLI[:,None,:,:,:,1::]
  region.b.a[:] += region.vFlux.fUDI[:,:,None,:,:,:,1::]
  region.b.a[:] += region.vFlux.fFBI[:,:,:,None,:,:,:,1::]
  region.b.a[:] -= region.vFlux.fRLI[:,None,:,:,:,0:-1]*region.altarray0[None,:,None,None,None,None,None,None,None]
  region.b.a[:] -= region.vFlux.fUDI[:,:,None,:,:,:,0:-1]*region.altarray1[None,None,:,None,None,None,None,None,None]
  region.b.a[:] -= region.vFlux.fFBI[:,:,:,None,:,:,:,0:-1]*region.altarray2[None,None,None,:,None,None,None,None,None]
#  fRLI = region.vFlux.fRLI[:,None,:,:,:,0:-1]
#  alt0 = region.altarray0[None,:,None,None,None,None,None,None,None]
#  fUDI = region.vFlux.fUDI[:,:,None,:,:,:,0:-1]
#  alt1 = region.altarray1[None,None,:,None,None,None,None,None,None]
#  fFBI = region.vFlux.fFBI[:,:,:,None,:,:,:,0:-1]
#  alt2 = region.altarray2[None,None,None,:,None,None,None,None,None]
#  region.b.a[:] -= ne.evaluate("fRLI*alt0 + fUDI*alt1 + fFBI*alt2")  

def addViscousContribution_BR1(regionManager,eqns):
  ##first do quadrature
  for region in regionManager.region:
    addSecondaryViscousContribution_BR1(region,region,eqns)
    region.basis.applyVolIntegral(region,-region.vFlux.fx,-region.vFlux.fy,-region.vFlux.fz,region.b.a)
    region.basis.applyMassMatrix(region,region.b.a)
    ## Now reconstruct tau and get edge states for later flux computations
    region.basis.reconstructU(region,region.b)
    region.b.uR[:],region.b.uL[:],region.b.uU[:],region.b.uD[:],region.b.uF[:],region.b.uB[:] = region.basis.reconstructEdgesGeneral(region.b.a,region)

  for region in regionManager.region:
    region.b.uR_edge[:],region.b.uL_edge[:],region.b.uU_edge[:],region.b.uD_edge[:],region.b.uF_edge[:],region.b.uB_edge[:] = sendEdgesGeneralSlab_Derivs(region.b.uL,region.b.uR,region.b.uD,region.b.uU,region.b.uB,region.b.uF,region,regionManager)

  for region in regionManager.region:
    MZ = region
    generalFluxGen(region,eqns,region.iFlux,region.a,eqns.evalTauFlux,[region.b])
    eqns.evalTauFluxX(region,region.b.u,region.a.u,region.vFlux2.fx,region.mus,region.cgas_field)
    eqns.evalTauFluxY(region,region.b.u,region.a.u,region.vFlux2.fy,region.mus,region.cgas_field)
    eqns.evalTauFluxZ(region,region.b.u,region.a.u,region.vFlux2.fz,region.mus,region.cgas_field)
    region.iFlux.fx -= region.vFlux2.fx
    region.iFlux.fy -= region.vFlux2.fy
    region.iFlux.fz -= region.vFlux2.fz
    #ne.evaluate("ifx - vfx",out=region.iFlux.fx, local_dict = {'ifx':region.iFlux.fx, 'vfx': region.vFlux2.fx})
    #ne.evaluate("ify - vfy",out=region.iFlux.fy, local_dict = {'ify':region.iFlux.fy, 'vfy': region.vFlux2.fy})
    #ne.evaluate("ifz - vfz",out=region.iFlux.fz, local_dict = {'ifz':region.iFlux.fz, 'vfz': region.vFlux2.fz})
  
    region.iFlux.fRLI = region.basis.faceIntegrateGlob(region,region.iFlux.fRLS*region.J_edge_det[0][None,:,:,None,:,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
    region.iFlux.fUDI = region.basis.faceIntegrateGlob(region,region.iFlux.fUDS*region.J_edge_det[1][None,:,:,None,:,:,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
    region.iFlux.fFBI = region.basis.faceIntegrateGlob(region,region.iFlux.fFBS*region.J_edge_det[2][None,:,:,None,:,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
    region.RHS[:] +=  region.iFlux.fRLI[:,None,:,:,:,1::] 
    region.RHS[:] -=  region.iFlux.fRLI[:,None,:,:,:,0:-1]*region.altarray0[None,:,None,None,None,None,None,None,None]
    region.RHS[:] +=  region.iFlux.fUDI[:,:,None,:,:,:,1::] 
    region.RHS[:] -=  region.iFlux.fUDI[:,:,None,:,:,:,0:-1]*region.altarray1[None,None,:,None,None,None,None,None,None]
    region.RHS[:] +=  region.iFlux.fFBI[:,:,:,None,:,:,:,1::] 
    region.RHS[:] -=  region.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*region.altarray2[None,None,None,:,None,None,None,None,None]
    #cell_ijk = region.cell_ijk ## Have to update everything over the stencil
    #print(np.linalg.norm(region.RHS[cell_ijk[0][0],:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])) 
def addSecondaryViscousContribution_BR1_hyper(region,MZ,eqns):
  cell_ijk = region.cell_ijk ## Have to update everything over the stencil
  stencil_ijk = region.stencil_ijk ## Have to update everything over the stencil
#  tmpfx = np.zeros(np.shape(region.vFlux.fx[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]),dtype=region.iFlux.fx.dtype)
#  tmpfy = np.zeros(np.shape(region.vFlux.fy[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]),dtype=region.iFlux.fy.dtype)
#  tmpfz = np.zeros(np.shape(region.vFlux.fz[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]),dtype=region.iFlux.fz.dtype)
#  eqns.evalViscousFluxX(region,region.a.u[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],tmpfx)
#  region.vFlux.fx[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = tmpfx[:]
#  eqns.evalViscousFluxY(region,region.a.u[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],tmpfy)
#  region.vFlux.fy[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = tmpfy[:]
#  eqns.evalViscousFluxZ(region,region.a.u[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],tmpfz)
#  region.vFlux.fz[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = tmpfz[:]
  eqns.evalViscousFluxX(region,region.a.u_hyper_stencil,region.vFlux.fx_hyper)
  eqns.evalViscousFluxY(region,region.a.u_hyper_stencil,region.vFlux.fy_hyper)
  eqns.evalViscousFluxZ(region,region.a.u_hyper_stencil,region.vFlux.fz_hyper)

  generalFluxGen_hyper(region,eqns,region.vFlux,region.a,eqns.evalViscousFlux,stencil_ijk,[])

  #region.vFlux.fRI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fRS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det[0][None,:,:,None,(stencil_ijk[5][0]+1)%(region.Npx+1),stencil_ijk[6][0],stencil_ijk[7][0]],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  #region.vFlux.fLI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fLS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det[0][None,:,:,None,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0]],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)

  #region.vFlux.fUI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fUS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det[1][None,:,:,None,stencil_ijk[5][0],(stencil_ijk[6][0]+1)%(region.Npy+1),stencil_ijk[7][0]],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  #region.vFlux.fDI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fDS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det[1][None,:,:,None,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0]],MZ.w0,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)

  #region.vFlux.fFI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fFS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det[2][None,:,:,None,stencil_ijk[5][0],stencil_ijk[6][0],(stencil_ijk[7][0]+1)%(region.Npz+1)],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
  #region.vFlux.fBI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fBS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det[2][None,:,:,None,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0]],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)


  region.vFlux.fRI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fRS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det_stencil_x1p1[None,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  region.vFlux.fLI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fLS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det_stencil_x1[None,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  region.vFlux.fUI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fUS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det_stencil_x2p1[None,:,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  region.vFlux.fDI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fDS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det_stencil_x2[None,:,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  region.vFlux.fFI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fFS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det_stencil_x3p1[None,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
  region.vFlux.fBI_hyper[:] = region.basis.faceIntegrateGlob(region,region.vFlux.fBS[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.J_edge_det_stencil_x3[None,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)



  region.b.a_hyper[:] =  region.vFlux.fRI_hyper[:,None]
  region.b.a_hyper[:] -=  region.vFlux.fLI_hyper[:,None]*region.altarray0[None,:,None,None,None,None]
  region.b.a_hyper[:] +=  region.vFlux.fUI_hyper[:,:,None]
  region.b.a_hyper[:] -=  region.vFlux.fDI_hyper[:,:,None]*region.altarray1[None,None,:,None,None,None]
  region.b.a_hyper[:] +=  region.vFlux.fFI_hyper[:,:,:,None]
  region.b.a_hyper[:] -=  region.vFlux.fBI_hyper[:,:,:,None]*region.altarray2[None,None,None,:,None,None]

def addViscousContribution_BR1_hyper(regionManager,eqns):
  ##first do quadrature
  for region in regionManager.region:
    cell_ijk = region.cell_ijk
    stencil_ijk = region.stencil_ijk
    addSecondaryViscousContribution_BR1_hyper(region,region,eqns)
    region.b.a_hyper[:] = applyVolIntegral_indices(region,-region.vFlux.fx_hyper,-region.vFlux.fy_hyper,-region.vFlux.fz_hyper, region.b.a_hyper[:],stencil_ijk)
    region.b.a_hyper[:] = np.sum(region.Minv[None,:,:,:,:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*region.b.a_hyper[:,None,None,None,None],axis=(5,6,7,8) )
    #region.basis.applyMassMatrix(region,region.b.a)
    ## Now reconstruct tau and get edge states for later flux computations
    #region.basis.reconstructU(region,region.b) 
    #region.b.u[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] =   reconstructUGeneral_tensordot(region,region.b.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]) #reconstructUGeneral_einsum(region,region.b.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]])
    region.b.u[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] =  reconstructUGeneral_tensordot(region,region.b.a_hyper[:])
    #region.b.uR[:],region.b.uL[:],region.b.uU[:],region.b.uD[:],region.b.uF[:],region.b.uB[:] = region.basis.reconstructEdgesGeneral(region.b.a,region)
    v1,v2,v3,v4,v5,v6 = region.basis.reconstructEdgesGeneral(region.b.a_hyper[:],region)
    region.b.uR[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v1[:]
    region.b.uL[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v2[:]
    region.b.uU[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v3[:]
    region.b.uD[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v4[:]
    region.b.uF[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v5[:]
    region.b.uB[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v6[:]
    

  for region in regionManager.region:
    region.b.uR_edge[:],region.b.uL_edge[:],region.b.uU_edge[:],region.b.uD_edge[:],region.b.uF_edge[:],region.b.uB_edge[:] = sendEdgesGeneralSlab_Derivs(region.b.uL,region.b.uR,region.b.uD,region.b.uU,region.b.uB,region.b.uF,region,regionManager)

  for region in regionManager.region:
    MZ = region
    
    #region.iFlux.fRLS[:,:,:,:,1::] = region.iFlux.fR[:]
    #region.iFlux.fRLS[:,:,:,:,0] = region.iFlux.fL[:,:,:,:,0]
    #region.iFlux.fUDS[:,:,:,:,:,1::] = region.iFlux.fU[:]
    #region.iFlux.fUDS[:,:,:,:,:,0] = region.iFlux.fD[:,:,:,:,:,0]

#    tmpfx = np.zeros(np.shape(region.vFlux2.fx[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]),dtype=region.vFlux2.fx.dtype)
#    tmpfy = np.zeros(np.shape(region.vFlux2.fy[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]),dtype=region.vFlux2.fy.dtype)
#    tmpfz = np.zeros(np.shape(region.vFlux2.fz[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]),dtype=region.vFlux2.fz.dtype)

    eqns.evalTauFluxX(region,region.b.u[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.a.u_hyper_cell,region.vFlux2.fx_hyper,region.mus,region.cgas_field)
    #region.iFlux.fx[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] -= tmpfx[:]
    region.iFlux.fx_hyper[:] -= region.vFlux2.fx_hyper[:]
    eqns.evalTauFluxY(region,region.b.u[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.a.u_hyper_cell,region.vFlux2.fy_hyper,region.mus,region.cgas_field)
    #region.iFlux.fy[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] -= tmpfy[:]
    region.iFlux.fy_hyper[:] -= region.vFlux2.fy_hyper[:]
    eqns.evalTauFluxZ(region,region.b.u[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region.a.u_hyper_cell,region.vFlux2.fz_hyper,region.mus,region.cgas_field)
    #region.iFlux.fz[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] -= tmpfz[:]
    region.iFlux.fz_hyper[:] -= region.vFlux2.fz_hyper[:]
  

    generalFluxGen_hyper(region,eqns,region.iFlux,region.a,eqns.evalTauFlux,cell_ijk,[region.b])

    region.iFlux.fRI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fRS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x1p1[None,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
    region.iFlux.fLI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fLS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x1[None,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
    region.iFlux.fUI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fUS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x2p1[None,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
    region.iFlux.fDI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fDS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x2[None,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
    region.iFlux.fFI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x3p1[None,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
    region.iFlux.fBI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fBS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x3[None,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
    region.RHS_hyper +=  region.iFlux.fRI_hyper[:,None]
    region.RHS_hyper -=  region.iFlux.fLI_hyper[:,None]*region.altarray0[None,:,None,None,None,None]
    region.RHS_hyper +=  region.iFlux.fUI_hyper[:,:,None]
    region.RHS_hyper -=  region.iFlux.fDI_hyper[:,:,None]*region.altarray1[None,None,:,None,None,None]
    region.RHS_hyper +=  region.iFlux.fFI_hyper[:,:,:,None]
    region.RHS_hyper -=  region.iFlux.fBI_hyper[:,:,:,None]*region.altarray2[None,None,None,:,None,None]


from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs
from fluxSchemes import *
from navier_stokes import evalViscousFluxZNS_IP
from navier_stokes import evalViscousFluxYNS_IP
from navier_stokes import evalViscousFluxXNS_IP
from navier_stokes import getGsNSX_FAST,getGsNSY_FAST,getGsNSZ_FAST
from bfer_mechanism import *
from eos_functions import *
from tensor_products import *
from chemistry_values import *
from smagorinsky import *
import numexpr as ne
import time
from viscous_br1 import addViscousContribution_BR1_hyper
def addSource_SWE(main):
    force = np.zeros(np.shape(main.iFlux.fx))
    g = 9.8/100000.
    force[0] = 0.
    force[1] = g*(main.a.u[0] - main.a.u[3])*main.surface_height_grad_x #- 0.5*g*(ux2[0] - main.surface_height_grad_x2) 
    force[2] = g*(main.a.u[0] - main.a.u[3])*main.surface_height_grad_y #- 0.5*g*(uy2[0] - main.surface_height_grad_y2)
    main.RHS[:] += main.basis.volIntegrateGlob(main, force*main.Jdet[None,:,:,:,None,:,:,:,None] ,main.w0,main.w1,main.w2,main.w3)

def addSource_kOmega(region):
  betaStar = 0.09
  beta = 3./40.
  gammaTurb = 13./25.
  Ux,Uy,Uz = region.basis.diffU(region.a.a,region)
  U = region.a.u
  # Compute turbulent stress tensor, tau_{ij}
  force = np.zeros(np.shape(region.iFlux.fx))
  ux = 1./U[0]*(Ux[1] - U[1]/U[0]*Ux[0])
  vx = 1./U[0]*(Ux[2] - U[2]/U[0]*Ux[0])
  wx = 1./U[0]*(Ux[3] - U[3]/U[0]*Ux[0])
  uy = 1./U[0]*(Uy[1] - U[1]/U[0]*Uy[0])
  vy = 1./U[0]*(Uy[2] - U[2]/U[0]*Uy[0])
  wy = 1./U[0]*(Uy[3] - U[3]/U[0]*Uy[0])
  uz = 1./U[0]*(Uz[1] - U[1]/U[0]*Uz[0])
  vz = 1./U[0]*(Uz[2] - U[2]/U[0]*Uz[0])
  wz = 1./U[0]*(Uz[3] - U[3]/U[0]*Uz[0]) 
  u_div = ux + vy + wz
  kx =     1./U[0]*(Ux[5] - U[5]/U[0]*Ux[0])
  omegax = 1./U[0]*(Ux[6] - U[6]/U[0]*Ux[0])
  ky =     1./U[0]*(Uy[5] - U[5]/U[0]*Uy[0])
  omegay = 1./U[0]*(Uy[6] - U[6]/U[0]*Uy[0])
  kz =     1./U[0]*(Uz[5] - U[5]/U[0]*Uz[0])
  omegaz = 1./U[0]*(Uz[6] - U[6]/U[0]*Uz[0])
  shp = np.shape(ux)
  shp = np.append( np.array([3,3]),shp)
  S = np.zeros(shp)
  tau = np.zeros(shp)
  mut = U[5]/U[6]*U[0]
  S[0,0] = 0.5*( ux + ux )
  S[0,1] = 0.5*( uy + vx )
  S[0,2] = 0.5*( uz + wx )
  S[1,0] = 0.5*( vx + uy )
  S[1,1] = 0.5*( vy + vy )
  S[1,2] = 0.5*( vz + wy )
  S[2,0] = 0.5*( wx + uz )
  S[2,1] = 0.5*( wy + vz )
  S[2,2] = 0.5*( wz + wz )
  tau[0,0] = mut*(2.*S[0,0] - 2./3.*u_div ) - 2./3.*U[-2]
  tau[0,1] = mut*(2.*S[0,1]               )
  tau[0,2] = mut*(2.*S[0,2]               )
  tau[1,0] = mut*(2.*S[1,0]               )
  tau[1,1] = mut*(2.*S[1,1] - 2./3.*u_div ) - 2./3.*U[-2]
  tau[1,2] = mut*(2.*S[1,2]               )
  tau[2,0] = mut*(2.*S[2,0]               )
  tau[2,1] = mut*(2.*S[1,1]               )
  tau[2,2] = mut*(2.*S[2,2] - 2./3.*u_div ) - 2./3.*U[-2]
  P = tau[0,0]*ux + tau[0,1]*uy + tau[0,2]*uz + tau[1,0]*vx + tau[1,1]*vy + tau[1,2]*vz + tau[2,0]*wx + tau[2,1]*wy + tau[2,2]*wz
  force[-2] = P - betaStar*U[5]*U[6]/U[0]**2
  force[-1] = gammaTurb * U[-1]/U[-2] * P  - beta*U[-1]**2/U[0]
  #final_term = U[0]**2 * sigma_d / U[6] * (k_x*omega_x + k_y*omega_y + k_z*omega_z) 
  #force[-1] += final_term
  region.RHS[:] += region.basis.volIntegrateGlob(region, force*region.Jdet[None,:,:,:,None,:,:,:,None] ,region.w0,region.w1,region.w2,region.w3)  

def addSource(main):
  if (main.fsource):
#    rates = getNetProductionRates(main,main.a.u,main.W)
#    sources = rates*main.W[0:-1,None,None,None,None,None,None,None,None]*1000.
#    print(np.shape(rates),np.shape(main.a.u))
    force = np.zeros(np.shape(main.iFlux.fx))
#    sources2 = main.cgas_field.net_production_rates[:,:]*main.cgas_field.molecular_weights[None,:]
#    #main.source_hook(main,force)
    for i in range(0,main.nvars):
      force[i] = main.source_mag[i]#*main.a.u[i]
#    for i in range(5,main.nvars):
#      force[i] = sources[i-5]#np.reshape(rates[:,i-5],np.shape(main.a.u[0]))
#      force[4] -= force[i]*main.delta_h0[i-5]
#    force[4] -= main.delta_h0[-1]*np.reshape(sources[:,-1],np.shape(main.a.u[0]))
    main.RHS[:] += main.basis.volIntegrateGlob(main, force*main.Jdet[None,:,:,:,None,:,:,:,None] ,main.w0,main.w1,main.w2,main.w3)


def addInviscidFlux_hyper(regionManager,eqns,args=[],args_phys=[]):
  for region in regionManager.region:
    MZ = region 
    # get fluxes at select cells
    cell_ijk = region.cell_ijk
    stencil_ijk = region.stencil_ijk
    generalFluxGen_hyper(region,eqns,region.iFlux,region.a,eqns.inviscidFlux,cell_ijk,args)
    region.iFlux.fRI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fRS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x1p1[None,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
    region.iFlux.fLI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fLS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x1[None,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
    region.iFlux.fUI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fUS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x2p1[None,:,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
    region.iFlux.fDI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fDS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x2[None,:,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
    region.iFlux.fFI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fFS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x3p1[None,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
    region.iFlux.fBI_hyper[:] = region.basis.faceIntegrateGlob(region,region.iFlux.fBS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.J_edge_det_hyper_x3[None,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)

    region.RHS_hyper[:] =  -region.iFlux.fRI_hyper[:,None]
    region.RHS_hyper[:] +=  region.iFlux.fLI_hyper[:,None]*region.altarray0[None,:,None,None,None,None]
    region.RHS_hyper[:] -=  region.iFlux.fUI_hyper[:,:,None]
    region.RHS_hyper[:] +=  region.iFlux.fDI_hyper[:,:,None]*region.altarray1[None,None,:,None,None,None]
    region.RHS_hyper[:] -=  region.iFlux.fFI_hyper[:,:,:,None]
    region.RHS_hyper[:] +=  region.iFlux.fBI_hyper[:,:,:,None]*region.altarray2[None,None,None,:,None,None]




def addVolume_and_Viscous_hyper(regionManager,eqns,args=[],args_phys=[]):
  for region in regionManager.region:
    #cell_list = range(0,region.Npx*region.Npy*region.Npz*region.Npt)
    cell_ijk = region.cell_ijk
    eqns.evalFluxXYZ(eqns,region,region.a.u_hyper_cell,region.iFlux.fx_hyper,region.iFlux.fy_hyper,region.iFlux.fz_hyper,args_phys)

  eqns.addViscousContribution_hyper(regionManager,eqns)
  for region in regionManager.region:
    region.RHS_hyper[:] = applyVolIntegral_indices(region,region.iFlux.fx_hyper*1.,region.iFlux.fy_hyper*1.,region.iFlux.fz_hyper*1.,region.RHS_hyper,cell_ijk)
  #region.RHS[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = region.RHS_hyper[:]
  #region.RHS = region.RHS_hyper[:]



def addInviscidFlux(regionManager,eqns,args=[],args_phys=[]):
  for region in regionManager.region:
    # first compute contribution from flux at faces
    generalFluxGen(region,eqns,region.iFlux,region.a,eqns.inviscidFlux,args)
    region.iFlux.fRLI = region.basis.faceIntegrateGlob(region,region.iFlux.fRLS*region.J_edge_det[0][None,:,:,None,:,:,:,None],region.w1,region.w2,region.w3,region.weights1,region.weights2,region.weights3)
    region.iFlux.fUDI = region.basis.faceIntegrateGlob(region,region.iFlux.fUDS*region.J_edge_det[1][None,:,:,None,:,:,:,None],region.w0,region.w2,region.w3,region.weights0,region.weights2,region.weights3)
    region.iFlux.fFBI = region.basis.faceIntegrateGlob(region,region.iFlux.fFBS*region.J_edge_det[2][None,:,:,None,:,:,:,None],region.w0,region.w1,region.w3,region.weights0,region.weights1,region.weights3)
    # now add inviscid flux contribution to the RHS
    region.RHS[:] =  -region.iFlux.fRLI[:,None,:,:,:,1::] 
    region.RHS[:] += region.iFlux.fRLI[:,None,:,:,:,0:-1]*region.altarray0[None,:,None,None,None,None,None,None,None]
    region.RHS[:] -= region.iFlux.fUDI[:,:,None,:,:,:,1::]
    region.RHS[:] += region.iFlux.fUDI[:,:,None,:,:,:,0:-1]*region.altarray1[None,None,:,None,None,None,None,None,None]
    region.RHS[:] -= region.iFlux.fFBI[:,:,:,None,:,:,:,1::] 
    region.RHS[:] += region.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*region.altarray2[None,None,None,:,None,None,None,None,None]




def addVolume_and_Viscous(regionManager,eqns,args=[],args_phys=[]):
  for region in regionManager.region:
    ## evaluate fluxes at indices
    tmpfx = np.zeros(np.shape(region.iFlux.fx),dtype=region.iFlux.fx.dtype)
    tmpfy = np.zeros(np.shape(region.iFlux.fy),dtype=region.iFlux.fy.dtype)
    tmpfz = np.zeros(np.shape(region.iFlux.fz),dtype=region.iFlux.fz.dtype)
    eqns.evalFluxXYZ(eqns,region,region.a.u,region.iFlux.fx,region.iFlux.fy,region.iFlux.fz,args_phys)

  eqns.addViscousContribution(regionManager,eqns) 
  for region in regionManager.region:
    region.basis.applyVolIntegral(region,region.iFlux.fx,region.iFlux.fy,region.iFlux.fz,region.RHS)


def getRHS(regionManager,eqns,args=[],args_phys=[]):
  t0 = time.time()
  addInviscidFlux(regionManager,eqns,args,args_phys)
  addVolume_and_Viscous(regionManager,eqns,args,args_phys)
  ### Get interior vol terms
  for region in regionManager.region:
    addSource(region)
    #addSource_kOmega(region)
    region.comm.Barrier()


def getRHS_strong(regionManager,eqns,args=[],args_phys=[]):
  t0 = time.time()
  vol_resid = [] 
  flux_resid = np.zeros(0,dtype=regionManager.region[0].a.a.dtype) 

  for region in regionManager.region:
    region.basis.reconstructU(region,region.a)
    region.a.uR[:],region.a.uL[:],region.a.uU[:],region.a.uD[:],region.a.uF[:],region.a.uB[:] = region.basis.reconstructEdgesGeneral(region.a.a,region)

  for region in regionManager.region:
    region.a.uR_edge[:],region.a.uL_edge[:],region.a.uU_edge[:],region.a.uD_edge[:],region.a.uF_edge[:],region.a.uB_edge[:] = sendEdgesGeneralSlab(region.a.uL,region.a.uR,region.a.uD,region.a.uU,region.a.uB,region.a.uF,region,regionManager)

  for region in regionManager.region:
    vol_resid.append(eqns.strongFormResidual(region,region.a.a,None) )
    #region.iFlux[:] = 0.
    generalFluxGenStrong(region,eqns,region.iFlux,region.a,eqns.inviscidFlux,args)
    flux_resid = np.append(flux_resid,region.iFlux.fRS.flatten())
    #flux_resid = np.append(flux_resid,region.iFlux.fLS.flatten())
    #flux_resid = np.append(flux_resid,region.iFlux.fUS.flatten())
    #flux_resid = np.append(flux_resid,region.iFlux.fDS.flatten())
    #flux_resid = np.append(flux_resid,region.iFlux.fFS.flatten())
    #flux_resid = np.append(flux_resid,region.iFlux.fBS.flatten())
  return vol_resid,flux_resid 



def getRHS_hyper(regionManager,eqns,args=[],args_phys=[]):
  t0 = time.time()
  addInviscidFlux_hyper(regionManager,eqns,args,args_phys)
  addVolume_and_Viscous_hyper(regionManager,eqns,args,args_phys)
  ### Get interior vol terms
  for main in regionManager.region:
    addSource(main)
    main.comm.Barrier()

def getRHS_zeta(main,MZ,eqns,args=[],args_phys=[]):
  t0 = time.time()
  main.basis.reconstructU(main,main.a)
  main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)

  addInviscidFlux_zeta(main,MZ,eqns,args,args_phys)
  addVolume_and_Viscous_zeta(main,MZ,eqns,args,args_phys)
  main.comm.Barrier()

def addInviscidFlux_zeta(main,MZ,eqns,args=[],args_phys=[]):
  # first compute contribution from flux at faces
  generalFluxGen(main,eqns,main.iFlux,main.a,eqns.inviscidFlux,args)
  # now we need to integrate along the boundary 
  main.iFlux.fRLI = main.basis.faceIntegrateGlob(main,main.iFlux.fRLS*main.J_edge_det[0][None,:,:,None,:,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  # now add inviscid flux contribution to the RHS
  main.RHS[:] =  -main.iFlux.fRLI[:,None,:,:,:,1::] 
  main.RHS[:] += main.iFlux.fRLI[:,None,:,:,:,0:-1]*main.altarray0[None,:,None,None,None,None,None,None,None]

def addVolume_and_Viscous_zeta(main,MZ,eqns,args=[],args_phys=[]):
  eqns.evalFluxX(main,main.a.u,main.iFlux.fx,args_phys)
  main.iFlux.fy[:] = 0.
  main.iFlux.fz[:] = 0.
  eqns.addViscousContribution(main,MZ,eqns) 
  main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,main.RHS)


def getRHS_eta(main,MZ,eqns,args=[],args_phys=[]):
  t0 = time.time()
  main.basis.reconstructU(main,main.a)
  main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)

  addInviscidFlux_eta(main,MZ,eqns,args,args_phys)
  addVolume_and_Viscous_eta(main,MZ,eqns,args,args_phys)
  main.comm.Barrier()



def addInviscidFlux_eta(main,MZ,eqns,args=[],args_phys=[]):
  # first compute contribution from flux at faces
  generalFluxGen(main,eqns,main.iFlux,main.a,eqns.inviscidFlux,args)
  # now we need to integrate along the boundary 
  main.iFlux.fUDI = main.basis.faceIntegrateGlob(main,main.iFlux.fUDS*main.J_edge_det[1][None,:,:,None,:,:,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  # now add inviscid flux contribution to the RHS
  main.RHS[:] = -main.iFlux.fUDI[:,:,None,:,:,:,1::]
  main.RHS[:] += main.iFlux.fUDI[:,:,None,:,:,:,0:-1]*main.altarray1[None,None,:,None,None,None,None,None,None]

def addVolume_and_Viscous_eta(main,MZ,eqns,args=[],args_phys=[]):
  main.iFlux.fx[:] = 0.
  eqns.evalFluxY(main,main.a.u,main.iFlux.fy,args_phys)
  main.iFlux.fz[:] = 0.
  eqns.addViscousContribution(main,MZ,eqns) 
  main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,main.RHS)



def getRHS_element_zeta(main,MZ,eqns,args=[],args_phys=[]):
  t0 = time.time()
  main.basis.reconstructU(main,main.a)
  main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  addInviscidFlux_element_zeta(main,MZ,eqns,args,args_phys)
  addVolume_and_Viscous_zeta(main,MZ,eqns,args,args_phys)
  main.comm.Barrier()


def addInviscidFlux_element_zeta(main,MZ,eqns,args=[],args_phys=[]):
  # first compute contribution from flux at faces
  generalFluxGen_element(main,eqns,main.iFlux,main.a,eqns.inviscidFlux,args)
  # now we need to integrate along the boundary 
  main.iFlux.fRI = main.basis.faceIntegrateGlob(main,main.iFlux.fR*main.J_edge_det[0][None,:,:,None,1::,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.iFlux.fLI = main.basis.faceIntegrateGlob(main,main.iFlux.fL*main.J_edge_det[0][None,:,:,None,0:-1,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.RHS[:] =  -main.iFlux.fRI[:,None,:,:,:,:] 
  main.RHS[:] += main.iFlux.fLI[:,None,:,:,:,:]*main.altarray0[None,:,None,None,None,None,None,None,None]


def getRHS_element_eta(main,MZ,eqns,args=[],args_phys=[]):
  t0 = time.time()
  main.basis.reconstructU(main,main.a)
  main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  addInviscidFlux_element_eta(main,MZ,eqns,args,args_phys)
  addVolume_and_Viscous_eta(main,MZ,eqns,args,args_phys)
  main.comm.Barrier()


def addInviscidFlux_element_eta(main,MZ,eqns,args=[],args_phys=[]):
  # first compute contribution from flux at faces
  generalFluxGen_element(main,eqns,main.iFlux,main.a,eqns.inviscidFlux,args)
  # now we need to integrate along the boundary 
  main.iFlux.fUI = main.basis.faceIntegrateGlob(main,main.iFlux.fU*main.J_edge_det[1][None,:,:,None,:,1::,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.iFlux.fDI = main.basis.faceIntegrateGlob(main,main.iFlux.fD*main.J_edge_det[1][None,:,:,None,:,0:-1,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.RHS[:] = -main.iFlux.fUI[:,:,None,:,:,:,:]
  main.RHS[:] += main.iFlux.fDI[:,:,None,:,:,:,:]*main.altarray1[None,None,:,None,None,None,None,None,None]




########## FUNCTIONS FOR ELEMENT LOCAL EVALUATIONS
#===================
def getRHS_element(regionManager,eqns,args=[],args_phys=[]):
  t0 = time.time()
  addInviscidFlux_element(regionManager,eqns,args,args_phys)
  addVolume_and_Viscous_element(regionManager,eqns,args,args_phys)
  regionManager.comm.Barrier()

def addInviscidFlux_element(regionManager,eqns,args=[],args_phys=[]):
  for region in regionManager.region:
    MZ = region 
    # first compute contribution from flux at faces
    generalFluxGen_element(region,eqns,region.iFlux,region.a,eqns.inviscidFlux,args)
    # now we need to integrate along the boundary 
    region.iFlux.fRI = region.basis.faceIntegrateGlob(region,region.iFlux.fR[:]*region.J_edge_det[0][None,:,:,None,1::,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
    region.iFlux.fLI = region.basis.faceIntegrateGlob(region,region.iFlux.fL[:]*region.J_edge_det[0][None,:,:,None,0:-1,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)

    region.iFlux.fUI = region.basis.faceIntegrateGlob(region,region.iFlux.fU[:]*region.J_edge_det[1][None,:,:,None,:,1::,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
    region.iFlux.fDI = region.basis.faceIntegrateGlob(region,region.iFlux.fD[:]*region.J_edge_det[1][None,:,:,None,:,0:-1,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  
    region.iFlux.fFI = region.basis.faceIntegrateGlob(region,region.iFlux.fF[:]*region.J_edge_det[2][None,:,:,None,:,:,1::,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
    region.iFlux.fBI = region.basis.faceIntegrateGlob(region,region.iFlux.fB[:]*region.J_edge_det[2][None,:,:,None,:,:,0:-1,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
  
    # now add inviscid flux contribution to the RHS
    region.RHS[:] =  -region.iFlux.fRI[:,None,:,:,:,:] 
    region.RHS[:] += region.iFlux.fLI[:,None,:,:,:,:]*region.altarray0[None,:,None,None,None,None,None,None,None]
    region.RHS[:] -= region.iFlux.fUI[:,:,None,:,:,:,:]
    region.RHS[:] += region.iFlux.fDI[:,:,None,:,:,:,:]*region.altarray1[None,None,:,None,None,None,None,None,None]
    region.RHS[:] -= region.iFlux.fFI[:,:,:,None,:,:,:,:] 
    region.RHS[:] += region.iFlux.fBI[:,:,:,None,:,:,:,:]*region.altarray2[None,None,None,:,None,None,None,None,None]

def addVolume_and_Viscous_element(regionManager,eqns,args=[],args_phys=[]):
  for region in regionManager.region:
    eqns.evalFluxXYZ(region,region.a.u,region.iFlux.fx,region.iFlux.fy,region.iFlux.fz,args_phys)
  for region in regionManager.region:
    region.basis.applyVolIntegral(region,region.iFlux.fx,region.iFlux.fy,region.iFlux.fz,region.RHS)
#=======================





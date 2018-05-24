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

def addInviscidFlux(regionManager,eqns,args=[],args_phys=[]):
  for main in regionManager.region:
    MZ = main
    # first compute contribution from flux at faces
    generalFluxGen(main,eqns,main.iFlux,main.a,eqns.inviscidFlux,args)
    # now we need to integrate along the boundary 
    main.iFlux.fRLI = main.basis.faceIntegrateGlob(main,main.iFlux.fRLS*main.J_edge_det[0][None,:,:,None,:,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
    main.iFlux.fUDI = main.basis.faceIntegrateGlob(main,main.iFlux.fUDS*main.J_edge_det[1][None,:,:,None,:,:,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
    main.iFlux.fFBI = main.basis.faceIntegrateGlob(main,main.iFlux.fFBS*main.J_edge_det[2][None,:,:,None,:,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
    # now add inviscid flux contribution to the RHS
    main.RHS[:] =  -main.iFlux.fRLI[:,None,:,:,:,1::] 
    main.RHS[:] += main.iFlux.fRLI[:,None,:,:,:,0:-1]*main.altarray0[None,:,None,None,None,None,None,None,None]
    main.RHS[:] -= main.iFlux.fUDI[:,:,None,:,:,:,1::]
    main.RHS[:] += main.iFlux.fUDI[:,:,None,:,:,:,0:-1]*main.altarray1[None,None,:,None,None,None,None,None,None]
    main.RHS[:] -= main.iFlux.fFBI[:,:,:,None,:,:,:,1::] 
    main.RHS[:] += main.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*main.altarray2[None,None,None,:,None,None,None,None,None]

def addVolume_and_Viscous(regionManager,eqns,args=[],args_phys=[]):
  for main in regionManager.region:
    eqns.evalFluxXYZ(main,main.a.u,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,args_phys)

  eqns.addViscousContribution(regionManager,eqns) 

  for main in regionManager.region:
    main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,main.RHS)


def getRHS(regionManager,eqns,args=[],args_phys=[]):
  t0 = time.time()
  addInviscidFlux(regionManager,eqns,args,args_phys)
  addVolume_and_Viscous(regionManager,eqns,args,args_phys)
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



def getRHS_element(main,MZ,eqns,args=[],args_phys=[]):
  t0 = time.time()
  main.basis.reconstructU(main,main.a)
  main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)
  main.RHS[:] = 0.
  addInviscidFlux_element(main,MZ,eqns,args,args_phys)
  addVolume_and_Viscous(main,MZ,eqns,args,args_phys)

  main.comm.Barrier()


def addInviscidFlux_element(main,MZ,eqns,args=[],args_phys=[]):
  # first compute contribution from flux at faces
  main.iFlux.fRI[:] = 0.
  main.iFlux.fLI[:] = 0.
  main.iFlux.fR[:] = 0.
  main.iFlux.fL[:] = 0.

  generalFluxGen_element(main,eqns,main.iFlux,main.a,eqns.inviscidFlux,args)
  # now we need to integrate along the boundary 
  main.iFlux.fRI[:] = main.basis.faceIntegrateGlob(main,main.iFlux.fR[:]*main.J_edge_det[0][None,:,:,None,1::,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.iFlux.fLI[:] = main.basis.faceIntegrateGlob(main,main.iFlux.fL[:]*main.J_edge_det[0][None,:,:,None,0:-1,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)

  main.iFlux.fUI = main.basis.faceIntegrateGlob(main,main.iFlux.fU[:]*main.J_edge_det[1][None,:,:,None,:,1::,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.iFlux.fDI = main.basis.faceIntegrateGlob(main,main.iFlux.fD[:]*main.J_edge_det[1][None,:,:,None,:,0:-1,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)

  main.iFlux.fFI = main.basis.faceIntegrateGlob(main,main.iFlux.fF[:]*main.J_edge_det[2][None,:,:,None,:,:,1::,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
  main.iFlux.fBI = main.basis.faceIntegrateGlob(main,main.iFlux.fB[:]*main.J_edge_det[2][None,:,:,None,:,:,0:-1,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)

  # now add inviscid flux contribution to the RHS
  main.RHS[:] =  -main.iFlux.fRI[:,None,:,:,:,:] 
  main.RHS[:] += main.iFlux.fLI[:,None,:,:,:,:]*main.altarray0[None,:,None,None,None,None,None,None,None]
  main.RHS[:] -= main.iFlux.fUI[:,:,None,:,:,:,:]
  main.RHS[:] += main.iFlux.fDI[:,:,None,:,:,:,:]*main.altarray1[None,None,:,None,None,None,None,None,None]
  main.RHS[:] -= main.iFlux.fFI[:,:,:,None,:,:,:,:] 
  main.RHS[:] += main.iFlux.fBI[:,:,:,None,:,:,:,:]*main.altarray2[None,None,None,:,None,None,None,None,None]

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



from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs#,sendaEdgesGeneralSlab
from fluxSchemes import *
from navier_stokes import evalViscousFluxZNS_IP
from navier_stokes import evalViscousFluxYNS_IP
from navier_stokes import evalViscousFluxXNS_IP
from navier_stokes import getGsNSX_FAST,getGsNSY_FAST,getGsNSZ_FAST
from eos_functions import *
from tensor_products import *
from chemistry_values import *
import matplotlib.pyplot as plt
from smagorinsky import *
import time

def getViscousFluxes_BR1(main,MZ,eqns):
  eqns.evalViscousFluxX(main,main.a.u,main.vFlux.fx,main.a.T)
  eqns.evalViscousFluxY(main,main.a.u,main.vFlux.fy,main.a.T)
  eqns.evalViscousFluxZ(main,main.a.u,main.vFlux.fz,main.a.T)
  # first reconstruct states

  eqns.evalViscousFluxX(main,main.a.uR,main.vFlux.fR,main.a.TR)
  eqns.evalViscousFluxX(main,main.a.uL,main.vFlux.fL,main.a.TL)
  eqns.evalViscousFluxY(main,main.a.uU,main.vFlux.fU,main.a.TU)
  eqns.evalViscousFluxY(main,main.a.uD,main.vFlux.fD,main.a.TD)
  eqns.evalViscousFluxZ(main,main.a.uF,main.vFlux.fF,main.a.TF)
  eqns.evalViscousFluxZ(main,main.a.uB,main.vFlux.fB,main.a.TB)


  eqns.evalViscousFluxX(main,main.a.uR_edge,main.vFlux.fR_edge,main.a.TR_edge)
  eqns.evalViscousFluxX(main,main.a.uL_edge,main.vFlux.fL_edge,main.a.TL_edge)
  eqns.evalViscousFluxY(main,main.a.uU_edge,main.vFlux.fU_edge,main.a.TU_edge)
  eqns.evalViscousFluxY(main,main.a.uD_edge,main.vFlux.fD_edge,main.a.TD_edge)
  eqns.evalViscousFluxZ(main,main.a.uF_edge,main.vFlux.fF_edge,main.a.TF_edge)
  eqns.evalViscousFluxZ(main,main.a.uB_edge,main.vFlux.fB_edge,main.a.TB_edge)

  # now construct star state

  main.vFlux.fRS[:],main.vFlux.fLS[:],main.vFlux.fUS[:],main.vFlux.fDS[:],main.vFlux.fFS[:],main.vFlux.fBS[:] = centralFluxGeneral(main.vFlux.fR,main.vFlux.fL,main.vFlux.fU,main.vFlux.fD,main.vFlux.fF,main.vFlux.fB,main.vFlux.fR_edge,main.vFlux.fL_edge,main.vFlux.fU_edge,main.vFlux.fD_edge,main.vFlux.fF_edge,main.vFlux.fB_edge)
  # now we need to integrate along the boundary 
  main.vFlux.fRI = main.basis.faceIntegrateGlob(main,main.vFlux.fRS,MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.vFlux.fLI = main.basis.faceIntegrateGlob(main,main.vFlux.fLS,MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.vFlux.fUI = main.basis.faceIntegrateGlob(main,main.vFlux.fUS,MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.vFlux.fDI = main.basis.faceIntegrateGlob(main,main.vFlux.fDS,MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.vFlux.fFI = main.basis.faceIntegrateGlob(main,main.vFlux.fFS,MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
  main.vFlux.fBI = main.basis.faceIntegrateGlob(main,main.vFlux.fBS,MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)


def solveb(main,MZ,eqns,dxi,dyi,dzi):
  ##first do quadrature
  getViscousFluxes_BR1(main,MZ,eqns)

  v1ijk = main.basis.volIntegrateGlob(main,main.vFlux.fx,main.wp0,main.w1,main.w2,main.w3)*dxi[None]
  v2ijk = main.basis.volIntegrateGlob(main,main.vFlux.fy,main.w0,main.wp1,main.w2,main.w3)*dyi[None]
  v3ijk = main.basis.volIntegrateGlob(main,main.vFlux.fz,main.w0,main.w1,main.wp2,main.w3)*dzi[None]
  tmp = -v1ijk - v2ijk - v3ijk
  tmp +=  (main.vFlux.fRI[:,None,:,:] - main.vFlux.fLI[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None,None,None])*dxi[None]
  tmp +=  (main.vFlux.fUI[:,:,None,:] - main.vFlux.fDI[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None,None,None])*dyi[None]
  tmp +=  (main.vFlux.fFI[:,:,:,None] - main.vFlux.fBI[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None,None,None])*dzi[None]
  main.b.a[:] = tmp[:]

  ## Now reconstruct tau and get edge states for later flux computations
  main.basis.reconstructU(main,main.b)
  main.b.uR[:],main.b.uL[:],main.b.uU[:],main.b.uD[:],main.b.uF[:],main.b.uB[:] = main.basis.reconstructEdgesGeneral(main.b.a,main)
  main.b.uR_edge[:],main.b.uL_edge[:],main.b.uU_edge[:],main.b.uD_edge[:],main.b.uF_edge[:],main.b.uB_edge[:] = sendEdgesGeneralSlab_Derivs(main.b.uL,main.b.uR,main.b.uD,main.b.uU,main.b.uB,main.b.uF,main)
  eqns.evalTauFluxX(main,main.b.uR,main.a.uR,main.vFlux2.fR,main.mus,main.cgas_field_R)
  eqns.evalTauFluxX(main,main.b.uL,main.a.uL,main.vFlux2.fL,main.mus,main.cgas_field_L)
  eqns.evalTauFluxY(main,main.b.uU,main.a.uU,main.vFlux2.fU,main.mus,main.cgas_field_U)
  eqns.evalTauFluxY(main,main.b.uD,main.a.uD,main.vFlux2.fD,main.mus,main.cgas_field_D)
  eqns.evalTauFluxZ(main,main.b.uF,main.a.uF,main.vFlux2.fF,main.mus,main.cgas_field_F)
  eqns.evalTauFluxZ(main,main.b.uB,main.a.uB,main.vFlux2.fB,main.mus,main.cgas_field_B)

  eqns.evalTauFluxX(main,main.b.uR_edge,main.a.uR_edge,main.vFlux2.fR_edge,main.mus,main.cgas_field_R_edge)
  eqns.evalTauFluxX(main,main.b.uL_edge,main.a.uL_edge,main.vFlux2.fL_edge,main.mus,main.cgas_field_L_edge)
  eqns.evalTauFluxY(main,main.b.uU_edge,main.a.uU_edge,main.vFlux2.fU_edge,main.mus,main.cgas_field_U_edge)
  eqns.evalTauFluxY(main,main.b.uD_edge,main.a.uD_edge,main.vFlux2.fD_edge,main.mus,main.cgas_field_D_edge)
  eqns.evalTauFluxZ(main,main.b.uF_edge,main.a.uF_edge,main.vFlux2.fF_edge,main.mus,main.cgas_field_F_edge)
  eqns.evalTauFluxZ(main,main.b.uB_edge,main.a.uB_edge,main.vFlux2.fB_edge,main.mus,main.cgas_field_B_edge)
  main.vFlux2.fRS[:],main.vFlux2.fLS[:],main.vFlux2.fUS[:],main.vFlux2.fDS[:],main.vFlux2.fFS[:],main.vFlux2.fBS[:] = centralFluxGeneral(main.vFlux2.fR,main.vFlux2.fL,main.vFlux2.fU,main.vFlux2.fD,main.vFlux2.fF,main.vFlux2.fB,main.vFlux2.fR_edge,main.vFlux2.fL_edge,main.vFlux2.fU_edge,main.vFlux2.fD_edge,main.vFlux2.fF_edge,main.vFlux2.fB_edge)

  eqns.evalTauFluxX(main,main.b.u,main.a.u,main.vFlux2.fx,main.mus,main.cgas_field)
  eqns.evalTauFluxY(main,main.b.u,main.a.u,main.vFlux2.fy,main.mus,main.cgas_field)
  eqns.evalTauFluxZ(main,main.b.u,main.a.u,main.vFlux2.fz,main.mus,main.cgas_field)

  main.vFlux2.fRI = main.basis.faceIntegrateGlob(main,main.vFlux2.fRS,MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.vFlux2.fLI = main.basis.faceIntegrateGlob(main,main.vFlux2.fLS,MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.vFlux2.fUI = main.basis.faceIntegrateGlob(main,main.vFlux2.fUS,MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.vFlux2.fDI = main.basis.faceIntegrateGlob(main,main.vFlux2.fDS,MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.vFlux2.fFI = main.basis.faceIntegrateGlob(main,main.vFlux2.fFS,MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
  main.vFlux2.fBI = main.basis.faceIntegrateGlob(main,main.vFlux2.fBS,MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)





def getFlux(main,MZ,eqns,args):
  # first reconstruct states
#  main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)
  #main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
#  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  #main.a.aR_edge[:],main.a.aL_edge[:],main.a.aU_edge[:],main.a.aD_edge[:],main.a.aF_edge[:],main.a.aB_edge[:] = sendaEdgesGeneralSlab(main.a.a,main)
  #main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = reconstructEdgeEdgesGeneral(main)
  inviscidFlux(main,eqns,main.iFlux,main.a,args)
  # now we need to integrate along the boundary 
  main.iFlux.fRI = main.basis.faceIntegrateGlob(main,main.iFlux.fRS,MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.iFlux.fLI = main.basis.faceIntegrateGlob(main,main.iFlux.fLS,MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.iFlux.fUI = main.basis.faceIntegrateGlob(main,main.iFlux.fUS,MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.iFlux.fDI = main.basis.faceIntegrateGlob(main,main.iFlux.fDS,MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.iFlux.fFI = main.basis.faceIntegrateGlob(main,main.iFlux.fFS,MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
  main.iFlux.fBI = main.basis.faceIntegrateGlob(main,main.iFlux.fBS,MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)


def getRHS_BR1(main,MZ,eqns,args=[],args_phys=[]):
  t0 = time.time()
  main.basis.reconstructU(main,main.a)
  main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)
  #main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)

#  print(np.amax(main.a.a[0,1::,1::]))
#  print(np.amin(main.a.u[0]),np.amax(main.a.u[0]))
  if (main.eq_str[0:-2] == 'Navier-Stokes Reacting'):
    update_state(main)
    #main.a.p[:],main.a.T[:] = computePressure_and_Temperature(main,main.a.u)

  # evaluate inviscid flux
  getFlux(main,MZ,eqns,args)
  ### Get interior vol terms
  eqns.evalFluxX(main,main.a.u,main.iFlux.fx,args_phys)
  eqns.evalFluxY(main,main.a.u,main.iFlux.fy,args_phys)
  eqns.evalFluxZ(main,main.a.u,main.iFlux.fz,args_phys)

  t1 = time.time()
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arr3= np.linspace(0,main.order[3]-1,main.order[3])

  scale =  (2.*ord_arr0[:,None,None,None] + 1.)*(2.*ord_arr1[None,:,None,None] + 1.)*(2.*ord_arr2[None,None,:,None]+1.)*(2.*ord_arr3[None,None,None,:] + 1.)/16.
  dxi = 2./main.dx2[None,None,None,None,:,None,None,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
  dyi = 2./main.dy2[None,None,None,None,None,:,None,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
  dzi = 2./main.dz2[None,None,None,None,None,None,:,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
  solveb(main,MZ,eqns,dxi,dyi,dzi) 

  main.iFlux.fx -= main.vFlux2.fx
  main.iFlux.fy -= main.vFlux2.fy
  main.iFlux.fz -= main.vFlux2.fz
  v1ijk = main.basis.volIntegrateGlob(main,main.iFlux.fx,main.wp0,main.w1,main.w2,main.w3)*dxi[None]
  v2ijk = main.basis.volIntegrateGlob(main,main.iFlux.fy,main.w0,main.wp1,main.w2,main.w3)*dyi[None]
  v3ijk = main.basis.volIntegrateGlob(main,main.iFlux.fz,main.w0,main.w1,main.wp2,main.w3)*dzi[None]
  tmp = v1ijk + v2ijk + v3ijk
  tmp +=  (-main.iFlux.fRI[:,None,:,:] + main.iFlux.fLI[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None,None,None])*dxi[None]
  tmp +=  (-main.iFlux.fUI[:,:,None,:] + main.iFlux.fDI[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None,None,None])*dyi[None]
  tmp +=  (-main.iFlux.fFI[:,:,:,None] + main.iFlux.fBI[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None,None,None])*dzi[None]
  tmp +=  (main.vFlux2.fRI[:,None,:,:] - main.vFlux2.fLI[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None,None,None])*dxi[None]
  tmp +=  (main.vFlux2.fUI[:,:,None,:] - main.vFlux2.fDI[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None,None,None])*dyi[None]
  tmp +=  (main.vFlux2.fFI[:,:,:,None] - main.vFlux2.fBI[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None,None,None])*dzi[None]

  if (main.source):
    force = np.zeros(np.shape(main.iFlux.fx))
    sources = main.cgas_field.net_production_rates[:,0:-1]*main.cgas_field.molecular_weights[None,0:-1]
    #main.source_hook(main,force)
#    for i in range(0,main.nvars):
#      force[i] = main.source_mag[i]#*main.a.u[i]
    for i in range(5,main.nvars):
      force[i] = np.reshape(sources[:,i-5],np.shape(main.a.u[0]))
    tmp += main.basis.volIntegrateGlob(main, force ,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  main.RHS = tmp
  main.comm.Barrier()


def getRHS(main,MZ,eqns,args=[],args_phys=[]):
  t0 = time.time()
  main.basis.reconstructU(main,main.a)
  main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)
  #main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)

  if (main.eq_str[0:-2] == 'Navier-Stokes Reacting'):
    update_state(main)
    #main.a.p[:],main.a.T[:] = computePressure_and_Temperature(main,main.a.u)
  # evaluate inviscid flux
  getFlux(main,MZ,eqns,args)
  ### Get interior vol terms
  eqns.evalFluxX(main,main.a.u,main.iFlux.fx,args_phys)
  eqns.evalFluxY(main,main.a.u,main.iFlux.fy,args_phys)
  eqns.evalFluxZ(main,main.a.u,main.iFlux.fz,args_phys)

  t1 = time.time()
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arr3= np.linspace(0,main.order[3]-1,main.order[3])

  scale =  (2.*ord_arr0[:,None,None,None] + 1.)*(2.*ord_arr1[None,:,None,None] + 1.)*(2.*ord_arr2[None,None,:,None]+1.)*(2.*ord_arr3[None,None,None,:] + 1.)/16.
  dxi = 2./main.dx2[None,None,None,None,:,None,None,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
  dyi = 2./main.dy2[None,None,None,None,None,:,None,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
  dzi = 2./main.dz2[None,None,None,None,None,None,:,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
  if (eqns.viscous):
    fvGX,fvGY,fvGZ,fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I = getViscousFlux(main,eqns) ##takes roughly 20% of the
    main.iFlux.fx -= fvGX
    main.iFlux.fy -= fvGY
    main.iFlux.fz -= fvGZ

  v1ijk = main.basis.volIntegrateGlob(main,main.iFlux.fx,main.wp0,main.w1,main.w2,main.w3)*dxi[None]
  v2ijk = main.basis.volIntegrateGlob(main,main.iFlux.fy,main.w0,main.wp1,main.w2,main.w3)*dyi[None]
  v3ijk = main.basis.volIntegrateGlob(main,main.iFlux.fz,main.w0,main.w1,main.wp2,main.w3)*dzi[None]

  tmp = v1ijk + v2ijk + v3ijk
  tmp +=  (-main.iFlux.fRI[:,None,:,:] + main.iFlux.fLI[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None,None,None])*dxi[None]
  tmp +=  (-main.iFlux.fUI[:,:,None,:] + main.iFlux.fDI[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None,None,None])*dyi[None]
  tmp +=  (-main.iFlux.fFI[:,:,:,None] + main.iFlux.fBI[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None,None,None])*dzi[None]
  if (eqns.viscous):
    tmp +=  (fvRIG11[:,None,:,:]*main.wpedge0[None,:,None,None,1,None,None,None,None,None] + fvRIG21[:,None,:,:] + fvRIG31[:,None,:,:]  - (fvLIG11[:,None,:,:]*main.wpedge0[None,:,None,None,0,None,None,None,None,None] + fvLIG21[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None,None,None] + fvLIG31[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None,None,None]) )*dxi[None]
    tmp +=  (fvUIG12[:,:,None,:] + fvUIG22[:,:,None,:]*main.wpedge1[None,None,:,None,1,None,None,None,None,None] + fvUIG32[:,:,None,:]  - (fvDIG12[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None,None,None] + fvDIG22[:,:,None,:]*main.wpedge1[None,None,:,None,0,None,None,None,None,None] + fvDIG32[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None,None,None]) )*dyi[None]
    tmp +=  (fvFIG13[:,:,:,None] + fvFIG23[:,:,:,None] + fvFIG33[:,:,:,None]*main.wpedge2[None,None,None,:,1,None,None,None,None,None]  - (fvBIG13[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None,None,None] + fvBIG23[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None,None,None] + fvBIG33[:,:,:,None]*main.wpedge2[None,None,None,:,0,None,None,None,None,None]) )*dzi[None]
    tmp +=  (fvR2I[:,None,:,:] - fvL2I[:,None,:,:]*main.altarray0[None,:,None,None,None,None,None,None,None])*dxi[None] + (fvU2I[:,:,None,:] - fvD2I[:,:,None,:]*main.altarray1[None,None,:,None,None,None,None,None,None])*dyi[None] + (fvF2I[:,:,:,None] - fvB2I[:,:,:,None]*main.altarray2[None,None,None,:,None,None,None,None,None])*dzi[None]


  if (main.fsource):
    force = np.zeros(np.shape(main.iFlux.fx))
    sources = main.cgas_field.net_production_rates[:,0:-1]*main.cgas_field.molecular_weights[None,0:-1]
    #main.source_hook(main,force)
#    for i in range(0,main.nvars):
#      force[i] = main.source_mag[i]#*main.a.u[i]
    for i in range(5,main.nvars):
      force[i] = np.reshape(sources[:,i-5],np.shape(main.a.u[0]))
    tmp += main.basis.volIntegrateGlob(main, force ,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]

  main.RHS = tmp
  main.comm.Barrier()


def computeJump(uR,uL,uU,uD,uF,uB,uR_edge,uL_edge,uU_edge,uD_edge,uF_edge,uB_edge):
  nvars,order1,order2,order3,Npx,Npy,Npz,Npt = np.shape(uR)
  nvars,order0,order1,order3,Npx,Npy,Npz,Npt = np.shape(uF)
#  nvars,order0,order1,Npx,Npy,Npz = np.shape(uF)
  jumpR = np.zeros((nvars,order1,order2,order3,Npx,Npy,Npz,Npt))
  jumpL = np.zeros((nvars,order1,order2,order3,Npx,Npy,Npz,Npt))
  jumpU = np.zeros((nvars,order0,order2,order3,Npx,Npy,Npz,Npt))
  jumpD = np.zeros((nvars,order0,order2,order3,Npx,Npy,Npz,Npt))
  jumpF = np.zeros((nvars,order0,order1,order3,Npx,Npy,Npz,Npt))
  jumpB = np.zeros((nvars,order0,order1,order3,Npx,Npy,Npz,Npt))

  jumpR[:,:,:,:,0:-1,:,:] = uR[:,:,:,:,0:-1,:,:] - uL[:,:,:,:,1::,:,:]
  jumpR[:,:,:,:,-1  ,:,:] = uR[:,:,:,:,  -1,:,:] - uR_edge
  jumpL[:,:,:,:,1:: ,:,:] = jumpR[:,:,:,:,0:-1,:,:]
  jumpL[:,:,:,:,0   ,:,:] = uL_edge - uL[:,:,:,:,  0,:,:]
  jumpU[:,:,:,:,:,0:-1,:] = uU[:,:,:,:,:,0:-1,:] - uD[:,:,:,:,:,1::,:]
  jumpU[:,:,:,:,:,  -1,:] = uU[:,:,:,:,:,  -1,:] - uU_edge
  jumpD[:,:,:,:,:,1:: ,:] = jumpU[:,:,:,:,:,0:-1,:]
  jumpD[:,:,:,:,:,0   ,:] = uD_edge - uD[:,:,:,:,:,   0,:]
  jumpF[:,:,:,:,:,:,0:-1] = uF[:,:,:,:,:,:,0:-1] - uB[:,:,:,:,:,:,1::]
  jumpF[:,:,:,:,:,:,  -1] = uF[:,:,:,:,:,:,  -1] - uF_edge
  jumpB[:,:,:,:,:,:,1:: ] = jumpF[:,:,:,:,:,:,0:-1]
  jumpB[:,:,:,:,:,:,0   ] = uB_edge - uB[:,:,:,:,:,:,   0]

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
  #  staticSmagorinsky(main)
    computeDynSmagViscosity(main,main.a.Upx,main.a.Upy,main.a.Upz,main.mu0,main.a.u)
  if (main.reacting):
    main.mu[1::] = computeDiffusionConstants(main.a.u[5::]/main.a.u[0],main.a.u,main)
  fvGX = eqns.evalViscousFluxX(main,main.a.u,main.a.Upx,main.a.Upy,main.a.Upz,main.mu)
  fvGY = eqns.evalViscousFluxY(main,main.a.u,main.a.Upx,main.a.Upy,main.a.Upz,main.mu)
  fvGZ = eqns.evalViscousFluxZ(main,main.a.u,main.a.Upx,main.a.Upy,main.a.Upz,main.mu)
  uhatR,uhatL,uhatU,uhatD,uhatF,uhatB = centralFluxGeneral(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
  fvRG11,fvRG21,fvRG31 = eqns.getGsX(main.a.uR,main,main.muR,main.a.uR - uhatR)
  fvLG11,fvLG21,fvLG31 = eqns.getGsX(main.a.uL,main,main.muL,main.a.uL - uhatL)

  fvUG12,fvUG22,fvUG32 = eqns.getGsY(main.a.uU,main,main.muU,main.a.uU-uhatU)
  fvDG12,fvDG22,fvDG32 = eqns.getGsY(main.a.uD,main,main.muD,main.a.uD-uhatD)

  fvFG13,fvFG23,fvFG33 = eqns.getGsZ(main.a.uF,main,main.muF,main.a.uF-uhatF)
  fvBG13,fvBG23,fvBG33 = eqns.getGsZ(main.a.uB,main,main.muB,main.a.uB-uhatB)


  fvxR = eqns.evalViscousFluxX(main,main.a.uR,main.a.UxR,main.a.UyR,main.a.UzR,main.muR)
  fvxL = eqns.evalViscousFluxX(main,main.a.uL,main.a.UxL,main.a.UyL,main.a.UzL,main.muL)

  fvyU = eqns.evalViscousFluxY(main,main.a.uU,main.a.UxU,main.a.UyU,main.a.UzU,main.muU)
  fvyD = eqns.evalViscousFluxY(main,main.a.uD,main.a.UxD,main.a.UyD,main.a.UzD,main.muD)

  fvzF = eqns.evalViscousFluxZ(main,main.a.uF,main.a.UxF,main.a.UyF,main.a.UzF,main.muF)
  fvzB = eqns.evalViscousFluxZ(main,main.a.uB,main.a.UxB,main.a.UyB,main.a.UzB,main.muB)

  fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge = sendEdgesGeneralSlab_Derivs(fvxL,fvxR,fvyD,fvyU,fvzB,fvzF,main)

  shatR,shatL,shatU,shatD,shatF,shatB = centralFluxGeneral(fvxR,fvxL,fvyU,fvyD,fvzF,fvzB,fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge,fvzF_edge,fvzB_edge)
  jumpR,jumpL,jumpU,jumpD,jumpF,jumpB = computeJump(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
  fvR2 = shatR - 1.*main.muR[0]*main.order[0]**2*jumpR/main.dx2[None,None,None,None,:,None,None,None]
  fvL2 = shatL - 1.*main.muL[0]*main.order[0]**2*jumpL/main.dx2[None,None,None,None,:,None,None,None]
  fvU2 = shatU - 1.*main.muU[0]*main.order[0]**2*jumpU/main.dy2[None,None,None,None,None,:,None,None]
  fvD2 = shatD - 1.*main.muD[0]*main.order[0]**2*jumpD/main.dy2[None,None,None,None,None,:,None,None]
  fvF2 = shatF - 1.*main.muF[0]*main.order[0]**2*jumpF/main.dz2[None,None,None,None,None,None,:,None]
  fvB2 = shatB - 1.*main.muB[0]*main.order[0]**2*jumpB/main.dz2[None,None,None,None,None,None,:,None]
  fvRIG11 = main.basis.faceIntegrateGlob(main,fvRG11,main.w1,main.w2,main.w3,main.weights1,main.weights2,main.weights3) 
  fvLIG11 = main.basis.faceIntegrateGlob(main,fvLG11,main.w1,main.w2,main.w3,main.weights1,main.weights2,main.weights3)  
  fvRIG21 = main.basis.faceIntegrateGlob(main,fvRG21,main.wp1,main.w2,main.w3,main.weights1,main.weights2,main.weights3) 
  fvLIG21 = main.basis.faceIntegrateGlob(main,fvLG21,main.wp1,main.w2,main.w3,main.weights1,main.weights2,main.weights3) 
  fvRIG31 = main.basis.faceIntegrateGlob(main,fvRG31,main.w1,main.wp2,main.w3,main.weights1,main.weights2,main.weights3) 
  fvLIG31 = main.basis.faceIntegrateGlob(main,fvLG31,main.w1,main.wp2,main.w3,main.weights1,main.weights2,main.weights3) 

  fvUIG12 = main.basis.faceIntegrateGlob(main,fvUG12,main.wp0,main.w2,main.w3,main.weights0,main.weights2,main.weights3)  
  fvDIG12 = main.basis.faceIntegrateGlob(main,fvDG12,main.wp0,main.w2,main.w3,main.weights0,main.weights2,main.weights3)  
  fvUIG22 = main.basis.faceIntegrateGlob(main,fvUG22,main.w0,main.w2,main.w3,main.weights0,main.weights2,main.weights3)  
  fvDIG22 = main.basis.faceIntegrateGlob(main,fvDG22,main.w0,main.w2,main.w3,main.weights0,main.weights2,main.weights3)  
  fvUIG32 = main.basis.faceIntegrateGlob(main,fvUG32,main.w0,main.wp2,main.w3,main.weights0,main.weights2,main.weights3)  
  fvDIG32 = main.basis.faceIntegrateGlob(main,fvDG32,main.w0,main.wp2,main.w3,main.weights0,main.weights2,main.weights3)  

  fvFIG13 = main.basis.faceIntegrateGlob(main,fvFG13,main.wp0,main.w1,main.w3,main.weights0,main.weights1,main.weights3)   
  fvBIG13 = main.basis.faceIntegrateGlob(main,fvBG13,main.wp0,main.w1,main.w3,main.weights0,main.weights1,main.weights3)  
  fvFIG23 = main.basis.faceIntegrateGlob(main,fvFG23,main.w0,main.wp1,main.w3,main.weights0,main.weights1,main.weights3)  
  fvBIG23 = main.basis.faceIntegrateGlob(main,fvBG23,main.w0,main.wp1,main.w3,main.weights0,main.weights1,main.weights3)  
  fvFIG33 = main.basis.faceIntegrateGlob(main,fvFG33,main.w0,main.w1,main.w3,main.weights0,main.weights1,main.weights3)  
  fvBIG33 = main.basis.faceIntegrateGlob(main,fvBG33,main.w0,main.w1,main.w3,main.weights0,main.weights1,main.weights3)  

  fvR2I = main.basis.faceIntegrateGlob(main,fvR2,main.w1,main.w2,main.w3,main.weights1,main.weights2,main.weights3)    
  fvL2I = main.basis.faceIntegrateGlob(main,fvL2,main.w1,main.w2,main.w3,main.weights1,main.weights2,main.weights3)  
  fvU2I = main.basis.faceIntegrateGlob(main,fvU2,main.w0,main.w2,main.w3,main.weights0,main.weights2,main.weights3)  
  fvD2I = main.basis.faceIntegrateGlob(main,fvD2,main.w0,main.w2,main.w3,main.weights0,main.weights2,main.weights3)  
  fvF2I = main.basis.faceIntegrateGlob(main,fvF2,main.w0,main.w1,main.w3,main.weights0,main.weights1,main.weights3)
  fvB2I = main.basis.faceIntegrateGlob(main,fvB2,main.w0,main.w1,main.w3,main.weights0,main.weights1,main.weights3)
  return fvGX,fvGY,fvGZ,fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvRIG31,fvLIG31,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvUIG32,fvDIG32,fvFIG13,fvBIG13,fvFIG23,fvBIG23,fvFIG33,fvBIG33,fvR2I,fvL2I,fvU2I,fvD2I,fvF2I,fvB2I


def computeDiffusionConstants(Yk,u,main):
  Ykshape = np.shape(Yk)
  n_species = Ykshape[0]
  Djkshape = np.append(n_species,Ykshape)
  Dk = np.zeros( np.shape(Yk) )
  Djk = np.zeros(Djkshape)
  ## compute diffusion constants for the Hisrchfelder and Curtis approximation
  # Y is the mass fraction. W is the molecular weight. X is mole fraction
  winv =  np.einsum('i...,ijk...->jk...',1./main.W,u[5::]) #compute inverse of mean molecular weight
  #Xk = 1./winv*Yk/main.W #compute the molar fraction
  Xk =  np.einsum('i...,ijk...->jk...',1./main.W,1./winv*Yk) #compute inverse of mean molecular weight
  p,T = computePressure_and_Temperature(main,u)
  for i in range(0,n_species):
    for j in range(0,n_species):
      Djk[i,j] =  computeDAB_Fuller(main.W[i],main.W[j],p,T,main.D_Vols[i],main.D_Vols[j])

  for i in range(0,n_species):
    Dk[i] = 1. - Yk[i]
    Dk_den = 0.
    for j in range(0,n_species):
      if (i != j):
        Dk_den += Xk[j]/Djk[i,j]
    Dk[i] /= Dk_den
  print(np.linalg.norm(Dk))
  return Dk




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


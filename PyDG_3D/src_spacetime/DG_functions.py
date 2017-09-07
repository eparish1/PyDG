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
  # get interior viscous flux
  # note we use the vFlux array since BR1 has more unknowns
  eqns.evalViscousFluxX(main,main.a.u,main.vFlux.fx,main.a.T)
  eqns.evalViscousFluxY(main,main.a.u,main.vFlux.fy,main.a.T)
  eqns.evalViscousFluxZ(main,main.a.u,main.vFlux.fz,main.a.T)
  # first reconstruct states
  generalFluxGen(main,eqns,main.vFlux,main.a,eqns.evalViscousFlux,[])

  # now we need to integrate along the boundary 
  main.vFlux.fRLI[:] = main.basis.faceIntegrateGlob(main,main.vFlux.fRLS*main.J_edge_det[0][None,:,:,None,:,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.vFlux.fUDI[:] = main.basis.faceIntegrateGlob(main,main.vFlux.fUDS*main.J_edge_det[1][None,:,:,None,:,:,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.vFlux.fFBI[:] = main.basis.faceIntegrateGlob(main,main.vFlux.fFBS*main.J_edge_det[2][None,:,:,None,:,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)


  main.b.a[:] =  main.vFlux.fRLI[:,None,:,:,:,1::]
  main.b.a[:] -= main.vFlux.fRLI[:,None,:,:,:,0:-1]*main.altarray0[None,:,None,None,None,None,None,None,None]
  main.b.a[:] += main.vFlux.fUDI[:,:,None,:,:,:,1::]
  main.b.a[:] -= main.vFlux.fUDI[:,:,None,:,:,:,0:-1]*main.altarray1[None,None,:,None,None,None,None,None,None]
  main.b.a[:] += main.vFlux.fFBI[:,:,:,None,:,:,:,1::] 
  main.b.a[:] -= main.vFlux.fFBI[:,:,:,None,:,:,:,0:-1]*main.altarray2[None,None,None,:,None,None,None,None,None]


def solveb(main,MZ,eqns):
  ##first do quadrature
  t0 = time.time()
  getViscousFluxes_BR1(main,MZ,eqns)
  t1 = time.time()
  f = main.vFlux.fx[:,None,None,None,None]*main.Jdet[None,None,None,None,None,:,:,:,None,:,:,:,None]*main.wx[None,:,:,:,:,:,:,:,:]
  main.b.a[:] -= volIntegrateGlob_einsum_2(main,f)
  f = main.vFlux.fy[:,None,None,None,None]*main.Jdet[None,None,None,None,None,:,:,:,None,:,:,:,None]*main.wy[None,:,:,:,:,:,:,:,:]
  main.b.a[:] -= volIntegrateGlob_einsum_2(main,f)
  f = main.vFlux.fz[:,None,None,None,None]*main.Jdet[None,None,None,None,None,:,:,:,None,:,:,:,None]*main.wz[None,:,:,:,:,:,:,:,:]
  main.b.a[:] -= volIntegrateGlob_einsum_2(main,f)
#  main.b.a[:] -= main.basis.volIntegrateGlob(main,main.vFlux.fz*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.wp2,main.w3)
  t2 = time.time()

  main.b.a[:] = np.einsum('ijklpqrs...,zpqrs...->zijkl...',main.Minv,main.b.a)
  t3 = time.time()

#  print('b = ' , np.linalg.norm(main.b.a))
  ## Now reconstruct tau and get edge states for later flux computations
  main.basis.reconstructU(main,main.b)
  main.b.uR[:],main.b.uL[:],main.b.uU[:],main.b.uD[:],main.b.uF[:],main.b.uB[:] = main.basis.reconstructEdgesGeneral(main.b.a,main)
  main.b.uR_edge[:],main.b.uL_edge[:],main.b.uU_edge[:],main.b.uD_edge[:],main.b.uF_edge[:],main.b.uB_edge[:] = sendEdgesGeneralSlab_Derivs(main.b.uL,main.b.uR,main.b.uD,main.b.uU,main.b.uB,main.b.uF,main)
  generalFluxGen(main,eqns,main.iFlux,main.a,eqns.evalTauFlux,[main.b])
  eqns.evalTauFluxX(main,main.b.u,main.a.u,main.vFlux2.fx,main.mus,main.cgas_field)
  eqns.evalTauFluxY(main,main.b.u,main.a.u,main.vFlux2.fy,main.mus,main.cgas_field)
  eqns.evalTauFluxZ(main,main.b.u,main.a.u,main.vFlux2.fz,main.mus,main.cgas_field)
  main.iFlux.fx -= main.vFlux2.fx
  main.iFlux.fy -= main.vFlux2.fy
  main.iFlux.fz -= main.vFlux2.fz

  t4 = time.time()
  print(t1-t0,t2-t1,t3-t2,t4-t3)
  main.iFlux.fRLI = main.basis.faceIntegrateGlob(main,main.iFlux.fRLS*main.J_edge_det[0][None,:,:,None,:,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.iFlux.fUDI = main.basis.faceIntegrateGlob(main,main.iFlux.fUDS*main.J_edge_det[1][None,:,:,None,:,:,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.iFlux.fFBI = main.basis.faceIntegrateGlob(main,main.iFlux.fFBS*main.J_edge_det[2][None,:,:,None,:,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)

  main.RHS[:] +=  main.iFlux.fRLI[:,None,:,:,:,1::] 
  main.RHS[:] -=  main.iFlux.fRLI[:,None,:,:,:,0:-1]*main.altarray0[None,:,None,None,None,None,None,None,None]
  main.RHS[:] +=  main.iFlux.fUDI[:,:,None,:,:,:,1::] 
  main.RHS[:] -=  main.iFlux.fUDI[:,:,None,:,:,:,0:-1]*main.altarray1[None,None,:,None,None,None,None,None,None]
  main.RHS[:] +=  main.iFlux.fFBI[:,:,:,None,:,:,:,1::] 
  main.RHS[:] -=  main.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*main.altarray2[None,None,None,:,None,None,None,None,None]




def getFlux(main,MZ,eqns,args):
  # first reconstruct states
  inviscidFlux(main,eqns,main.iFlux,main.a,args)
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

def getRHS_BR1(main,MZ,eqns,args=[],args_phys=[]):
  t0 = time.time()
  main.basis.reconstructU(main,main.a)
  main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  if (main.eq_str[0:-2] == 'Navier-Stokes Reacting'):
    update_state(main)
    #main.a.p[:],main.a.T[:] = computePressure_and_Temperature(main,main.a.u)

  # evaluate inviscid flux and add contribution to RHS
  getFlux(main,MZ,eqns,args)
  ### Get interior vol terms
  eqns.evalFluxX(main,main.a.u,main.iFlux.fx,args_phys)
  eqns.evalFluxY(main,main.a.u,main.iFlux.fy,args_phys)
  eqns.evalFluxZ(main,main.a.u,main.iFlux.fz,args_phys)
  solveb(main,MZ,eqns) 

  f = main.iFlux.fx[:,None,None,None,None]*main.Jdet[None,None,None,None,None,:,:,:,None,:,:,:,None]*main.wx[None,:,:,:,:,:,:,:,:]
  main.RHS[:] += volIntegrateGlob_einsum_2(main,f)

  f = main.iFlux.fy[:,None,None,None,None]*main.Jdet[None,None,None,None,None,:,:,:,None,:,:,:,None]*main.wy[None,:,:,:,:,:,:,:,:]
  main.RHS[:] += volIntegrateGlob_einsum_2(main,f)

  f = main.iFlux.fz[:,None,None,None,None]*main.Jdet[None,None,None,None,None,:,:,:,None,:,:,:,None]*main.wz[None,:,:,:,:,:,:,:,:]
  main.RHS[:] += volIntegrateGlob_einsum_2(main,f)


#  main.a.Upx,main.a.Upy,main.a.Upz = main.basis.diffU(main.a.a,main)
#  force = np.zeros(np.shape(main.iFlux.fx))
#  force[3] =  main.a.Upx[0]**2 + main.a.Upy[1]**2 + main.a.Upz[2]**2 + 2.*main.a.Upy[0]*main.a.Upx[1] + 2.*main.a.Upz[0]*main.a.Upx[2] + 2.*main.a.Upz[1]*main.a.Upy[2]
#  #force[3] = main.a.Upx[0] + main.a.Upy[1] + main.a.Upz[2]
#  vol_int = main.basis.volIntegrateGlob(main,force,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
#  tmp[3] += vol_int[3]
#  #f =  main.a.Upx[0]**2 + main.a.Upy[1]**2 + main.a.Upz[2]**2 + 2.*main.a.Upy[0]*main.a.Upx[1] + 2.*main.a.Upz[0]*main.a.Upx[2] + 2.*main.a.Upz[1]*main.a.Upy[2]
  if (main.fsource):
    force = np.zeros(np.shape(main.iFlux.fx))
#    #sources = main.cgas_field.net_production_rates[:,:]*main.cgas_field.molecular_weights[None,:]
#    #main.source_hook(main,force)
    for i in range(0,main.nvars):
      force[i] = main.source_mag[i]#*main.a.u[i]
#    for i in range(5,main.nvars):
#      force[i] = np.reshape(sources[:,i-5],np.shape(main.a.u[0]))
#      force[4] -= force[i]*main.delta_h0[i-5]
#    force[4] -= main.delta_h0[-1]*np.reshape(sources[:,-1],np.shape(main.a.u[0]))
    main.RHS[:] += main.basis.volIntegrateGlob(main, force*main.Jdet[None,:,:,:,None,:,:,:,None] ,main.w0,main.w1,main.w2,main.w3)

  main.comm.Barrier()


def getRHS(main,MZ,eqns,args=[],args_phys=[]):
  t0 = time.time()
  main.basis.reconstructU(main,main.a)
  main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)

  if (main.eq_str[0:-2] == 'Navier-Stokes Reacting'):
    update_state(main)
    #main.a.p[:],main.a.T[:] = computePressure_and_Temperature(main,main.a.u)
  # evaluate inviscid flux and add contribution to RHS
  getFlux(main,MZ,eqns,args)
  ### Get interior vol terms
  eqns.evalFluxX(main,main.a.u,main.iFlux.fx,args_phys)
  eqns.evalFluxY(main,main.a.u,main.iFlux.fy,args_phys)
  eqns.evalFluxZ(main,main.a.u,main.iFlux.fz,args_phys)

  t1 = time.time()
  if (eqns.viscous):
    getViscousFlux(main,eqns) ##takes roughly 20% of the
#  w =  main.wp0[:,None,None,None,:,None,None,None,None,None,None,None]*main.Jinv[0,0][None,None,None,None,:,:,:,None,:,:,:,None]*main.w1[None,:,None,None,None,:,None,None,None,None,None,None]*\
#       main.w2[None,None,:,None,None,None,:,None,None,None,None,None]*main.w3[None,None,None,:,None,None,None,:,None,None,None,None] + \
#       main.w0[:,None,None,None,:,None,None,None,None,None,None,None]*main.wp1[None,:,None,None,None,:,None,None,None,None,None,None]*main.Jinv[1,0][None,None,None,None,:,:,:,None,:,:,:,None]*\
#       main.w2[None,None,:,None,None,None,:,None,None,None,None,None]*main.w3[None,None,None,:,None,None,None,:,None,None,None,None] + \
#       main.w0[:,None,None,None,:,None,None,None,None,None,None,None]*main.w1[None,:,None,None,None,:,None,None,None,None,None,None]*\
#       main.wp2[None,None,:,None,None,None,:,None,None,None,None,None]*main.Jinv[2,0][None,None,None,None,:,:,:,None,:,:,:,None]*main.w3[None,None,None,:,None,None,None,:,None,None,None,None] 
  f = main.iFlux.fx[:,None,None,None,None]*main.Jdet[None,None,None,None,None,:,:,:,None,:,:,:,None]*main.wx[None,:,:,:,:,:,:,:,:]
  main.RHS[:] += volIntegrateGlob_einsum_2(main,f)

#  w =  main.wp0[:,None,None,None,:,None,None,None,None,None,None,None]*main.Jinv[0,1][None,None,None,None,:,:,:,None,:,:,:,None]*main.w1[None,:,None,None,None,:,None,None,None,None,None,None]*\
#       main.w2[None,None,:,None,None,None,:,None,None,None,None,None]*main.w3[None,None,None,:,None,None,None,:,None,None,None,None] + \
#       main.w0[:,None,None,None,:,None,None,None,None,None,None,None]*main.wp1[None,:,None,None,None,:,None,None,None,None,None,None]*main.Jinv[1,1][None,None,None,None,:,:,:,None,:,:,:,None]*\
#       main.w2[None,None,:,None,None,None,:,None,None,None,None,None]*main.w3[None,None,None,:,None,None,None,:,None,None,None,None] + \
#       main.w0[:,None,None,None,:,None,None,None,None,None,None,None]*main.w1[None,:,None,None,None,:,None,None,None,None,None,None]*\
#       main.wp2[None,None,:,None,None,None,:,None,None,None,None,None]*main.Jinv[2,1][None,None,None,None,:,:,:,None,:,:,:,None]*main.w3[None,None,None,:,None,None,None,:,None,None,None,None] 
  f = main.iFlux.fy[:,None,None,None,None]*main.Jdet[None,None,None,None,None,:,:,:,None,:,:,:,None]*main.wy[None,:,:,:,:,:,:,:,:]
  main.RHS[:] += volIntegrateGlob_einsum_2(main,f)

#  w =  main.wp0[:,None,None,None,:,None,None,None,None,None,None,None]*main.Jinv[0,2][None,None,None,None,:,:,:,None,:,:,:,None]*main.w1[None,:,None,None,None,:,None,None,None,None,None,None]*\
#       main.w2[None,None,:,None,None,None,:,None,None,None,None,None]*main.w3[None,None,None,:,None,None,None,:,None,None,None,None] + \
#       main.w0[:,None,None,None,:,None,None,None,None,None,None,None]*main.wp1[None,:,None,None,None,:,None,None,None,None,None,None]*main.Jinv[1,2][None,None,None,None,:,:,:,None,:,:,:,None]*\
#       main.w2[None,None,:,None,None,None,:,None,None,None,None,None]*main.w3[None,None,None,:,None,None,None,:,None,None,None,None] + \
#       main.w0[:,None,None,None,:,None,None,None,None,None,None,None]*main.w1[None,:,None,None,None,:,None,None,None,None,None,None]*\
#       main.wp2[None,None,:,None,None,None,:,None,None,None,None,None]*main.Jinv[2,2][None,None,None,None,:,:,:,None,:,:,:,None]*main.w3[None,None,None,:,None,None,None,:,None,None,None,None] 

  f = main.iFlux.fz[:,None,None,None,None]*main.Jdet[None,None,None,None,None,:,:,:,None,:,:,:,None]*main.wz[None,:,:,:,:,:,:,:,:]
  main.RHS[:] += volIntegrateGlob_einsum_2(main,f)


#  main.a.Upx,main.a.Upy,main.a.Upz = main.basis.diffU(main.a.a,main)
#  force = np.zeros(np.shape(main.iFlux.fx))
#  force[3] =  main.a.Upx[0]**2 + main.a.Upy[1]**2 + main.a.Upz[2]**2 + 2.*main.a.Upy[0]*main.a.Upx[1] + 2.*main.a.Upz[0]*main.a.Upx[2] + 2.*main.a.Upz[1]*main.a.Upy[2]
#  tmp += main.basis.volIntegrateGlob(main, force ,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  if (main.fsource):
    force = np.zeros(np.shape(main.iFlux.fx))
#    sources = main.cgas_field.net_production_rates[:,0:-1]*main.cgas_field.molecular_weights[None,0:-1]
#    #main.source_hook(main,force)
    for i in range(0,main.nvars):
      force[i] = main.source_mag[i]#*main.a.u[i]
#    for i in range(5,main.nvars):
#      force[i] = np.reshape(sources[:,i-5],np.shape(main.a.u[0]))
#      force[4] -= force[i]*main.delta_h0[i-5]
#    force[4] -= main.delta_h0[-1]*np.reshape(sources[:,-1],np.shape(main.a.u[0]))
    main.RHS[:] += main.basis.volIntegrateGlob(main, force*main.Jdet[None,:,:,:,None,:,:,:,None] ,main.w0,main.w1,main.w2,main.w3)

  main.comm.Barrier()



def getRHS_SOURCE(main,MZ,eqns,args=[],args_phys=[]):
  main.basis.reconstructU(main,main.a)
  main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)
  main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main)
  update_state(main)
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arr3= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arr0[:,None,None,None] + 1.)*(2.*ord_arr1[None,:,None,None] + 1.)*(2.*ord_arr2[None,None,:,None]+1.)*(2.*ord_arr3[None,None,None,:] + 1.)/16.
  dxi = 2./main.dx2[None,None,None,None,:,None,None,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
  dyi = 2./main.dy2[None,None,None,None,None,:,None,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
  dzi = 2./main.dz2[None,None,None,None,None,None,:,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
  tmp = 0
  force = np.zeros(np.shape(main.iFlux.fx))
  main.a.Upx,main.a.Upy,main.a.Upz = main.basis.diffU(main.a.a,main)
  source[0] = (main.a.Upx[0] + main.a.Upy[1] + main.a.Upz[2])/main.dt
#  sources = main.cgas_field.net_production_rates[:,0:-1]*main.cgas_field.molecular_weights[None,0:-1]
#  for i in range(5,main.nvars):
#    force[i] = np.reshape(sources[:,i-5],np.shape(main.a.u[0]))
#    force[4] -= force[i]*main.delta_h0[i-5]
#  force[4] -= main.delta_h0[-1]*np.reshape(sources[:,-1],np.shape(main.a.u[0]))
#  tmp += main.basis.volIntegrateGlob(main, force ,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  main.RHS = tmp
  main.comm.Barrier()


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
  main.iFlux.fx -= fvGX
  main.iFlux.fy -= fvGY
  main.iFlux.fz -= fvGZ


  ## Compute the penalty term
  centralFluxGeneral2(main.iFlux.fRLS,main.iFlux.fUDS,main.iFlux.fFBS,main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uF,main.a.uB,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge,main.a.uF_edge,main.a.uB_edge)
  fvRG11,fvRG21,fvRG31 = eqns.getGsX(main.a.uR,main,main.muR,main.a.uR - main.iFlux.fRLS[:,:,:,:,1::])
  fvLG11,fvLG21,fvLG31 = eqns.getGsX(main.a.uL,main,main.muL,main.a.uL - main.iFlux.fRLS[:,:,:,:,0:-1])

  fvUG12,fvUG22,fvUG32 = eqns.getGsY(main.a.uU,main,main.muU,main.a.uU-main.iFlux.fUDS[:,:,:,:,:,1::])
  fvDG12,fvDG22,fvDG32 = eqns.getGsY(main.a.uD,main,main.muD,main.a.uD-main.iFlux.fUDS[:,:,:,:,:,0:-1])

  fvFG13,fvFG23,fvFG33 = eqns.getGsZ(main.a.uF,main,main.muF,main.a.uF-main.iFlux.fFBS[:,:,:,:,:,:,1::])
  fvBG13,fvBG23,fvBG33 = eqns.getGsZ(main.a.uB,main,main.muB,main.a.uB-main.iFlux.fFBS[:,:,:,:,:,:,0:-1])

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




  main.iFlux.fR[:] = eqns.evalViscousFluxX(main,main.a.uR,main.a.UxR,main.a.UyR,main.a.UzR,main.muR)
  main.iFlux.fL[:] = eqns.evalViscousFluxX(main,main.a.uL,main.a.UxL,main.a.UyL,main.a.UzL,main.muL)

  main.iFlux.fU[:] = eqns.evalViscousFluxY(main,main.a.uU,main.a.UxU,main.a.UyU,main.a.UzU,main.muU)
  main.iFlux.fD[:] = eqns.evalViscousFluxY(main,main.a.uD,main.a.UxD,main.a.UyD,main.a.UzD,main.muD)

  main.iFlux.fF[:] = eqns.evalViscousFluxZ(main,main.a.uF,main.a.UxF,main.a.UyF,main.a.UzF,main.muF)
  main.iFlux.fB[:] = eqns.evalViscousFluxZ(main,main.a.uB,main.a.UxB,main.a.UyB,main.a.UzB,main.muB)

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


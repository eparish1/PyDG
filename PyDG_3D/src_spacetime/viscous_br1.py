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



def addSecondaryViscousContribution_BR1(main,MZ,eqns):
  # get interior viscous flux
  # note we use the vFlux array since BR1 has more unknowns
  eqns.evalViscousFluxXYZ(main,main.a.u,main.vFlux.fx,main.vFlux.fy,main.vFlux.fz)
  #eqns.evalViscousFluxX(main,main.a.u,main.vFlux.fx)
  #eqns.evalViscousFluxY(main,main.a.u,main.vFlux.fy)
  #eqns.evalViscousFluxZ(main,main.a.u,main.vFlux.fz)
  # first reconstruct states
  generalFluxGen(main,eqns,main.vFlux,main.a,eqns.evalViscousFlux,[])

  # now we need to integrate along the boundary
  a = main.vFlux.fRLS
  Jdet = main.J_edge_det[0][None,:,:,None,:,:,:,None]
  f = ne.evaluate("a*Jdet")  
  main.vFlux.fRLI[:] = main.basis.faceIntegrateGlob(main,f,MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  a = main.vFlux.fUDS
  Jdet = main.J_edge_det[1][None,:,:,None,:,:,:,None]
  f = ne.re_evaluate()  
  main.vFlux.fUDI[:] = main.basis.faceIntegrateGlob(main,f,MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  a = main.vFlux.fFBS
  Jdet = main.J_edge_det[2][None,:,:,None,:,:,:,None]
  f = ne.re_evaluate()  
  main.vFlux.fFBI[:] = main.basis.faceIntegrateGlob(main,f,MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)


   
  main.b.a[:] =  main.vFlux.fRLI[:,None,:,:,:,1::]
  main.b.a[:] += main.vFlux.fUDI[:,:,None,:,:,:,1::]
  main.b.a[:] += main.vFlux.fFBI[:,:,:,None,:,:,:,1::]
  main.b.a[:] -= main.vFlux.fRLI[:,None,:,:,:,0:-1]*main.altarray0[None,:,None,None,None,None,None,None,None]
  main.b.a[:] -= main.vFlux.fUDI[:,:,None,:,:,:,0:-1]*main.altarray1[None,None,:,None,None,None,None,None,None]
  main.b.a[:] -= main.vFlux.fFBI[:,:,:,None,:,:,:,0:-1]*main.altarray2[None,None,None,:,None,None,None,None,None]
#  fRLI = main.vFlux.fRLI[:,None,:,:,:,0:-1]
#  alt0 = main.altarray0[None,:,None,None,None,None,None,None,None]
#  fUDI = main.vFlux.fUDI[:,:,None,:,:,:,0:-1]
#  alt1 = main.altarray1[None,None,:,None,None,None,None,None,None]
#  fFBI = main.vFlux.fFBI[:,:,:,None,:,:,:,0:-1]
#  alt2 = main.altarray2[None,None,None,:,None,None,None,None,None]
#  main.b.a[:] -= ne.evaluate("fRLI*alt0 + fUDI*alt1 + fFBI*alt2")  

def addViscousContribution_BR1(main,MZ,eqns):
  ##first do quadrature
  addSecondaryViscousContribution_BR1(main,MZ,eqns)

  main.basis.applyVolIntegral(main,-main.vFlux.fx,-main.vFlux.fy,-main.vFlux.fz,main.b.a)
  main.basis.applyMassMatrix(main,main.b.a)

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
  #ne.evaluate("ifx - vfx",out=main.iFlux.fx, local_dict = {'ifx':main.iFlux.fx, 'vfx': main.vFlux2.fx})
  #ne.evaluate("ify - vfy",out=main.iFlux.fy, local_dict = {'ify':main.iFlux.fy, 'vfy': main.vFlux2.fy})
  #ne.evaluate("ifz - vfz",out=main.iFlux.fz, local_dict = {'ifz':main.iFlux.fz, 'vfz': main.vFlux2.fz})

  main.iFlux.fRLI = main.basis.faceIntegrateGlob(main,main.iFlux.fRLS*main.J_edge_det[0][None,:,:,None,:,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
  main.iFlux.fUDI = main.basis.faceIntegrateGlob(main,main.iFlux.fUDS*main.J_edge_det[1][None,:,:,None,:,:,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
  main.iFlux.fFBI = main.basis.faceIntegrateGlob(main,main.iFlux.fFBS*main.J_edge_det[2][None,:,:,None,:,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)

  main.RHS[:] +=  main.iFlux.fRLI[:,None,:,:,:,1::] 
  main.RHS[:] -=  main.iFlux.fRLI[:,None,:,:,:,0:-1]*main.altarray0[None,:,None,None,None,None,None,None,None]
  main.RHS[:] +=  main.iFlux.fUDI[:,:,None,:,:,:,1::] 
  main.RHS[:] -=  main.iFlux.fUDI[:,:,None,:,:,:,0:-1]*main.altarray1[None,None,:,None,None,None,None,None,None]
  main.RHS[:] +=  main.iFlux.fFBI[:,:,:,None,:,:,:,1::] 
  main.RHS[:] -=  main.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*main.altarray2[None,None,None,:,None,None,None,None,None]


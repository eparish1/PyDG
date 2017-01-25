import numpy as np
from MPI_functions import sendEdges
def reconstructU(main,var):
  var.u[:] = 0.
  for l in range(0,var.order):
    for m in range(0,var.order):
      for k in range(0,var.nvars):
        var.u[k,:,:,:,:] += main.w[l][:,None,None,None]*main.w[m][None,:,None,None]*var.a[k,l,m,:,:]


#def reconstructU(w,u,a):
#  nvars,order,order,Nelx,Nely = np.shape(a)
#  u[:] = 0.
#  for l in range(0,order):
#    for m in range(0,order):
#      for k in range(0,nvars):
#        u[k,:,:,:,:] += w[l][:,None,None,None]*w[m][None,:,None,None]*a[k,l,m,:,:]


def reconstructEdges(main,var):
  var.aU[:] =  np.sum(var.a,axis=(2))
  var.aD[:] =  np.sum(var.a*main.altarray[None,None,:,None,None],axis=(2))
  var.aR[:] =  np.sum(var.a,axis=(1))
  var.aL[:] =  np.sum(var.a*main.altarray[None,:,None,None,None],axis=(1))
  var.uU[:] = 0. 
  var.uD[:] = 0.
  var.uL[:] = 0. 
  var.uR[:] = 0.
  for m in range(0,var.order):
    for k in range(0,var.nvars):
       var.uU[k,:,:,:] += main.w[m,:,None,None]*var.aU[k,m,:,:]
       var.uD[k,:,:,:] += main.w[m,:,None,None]*var.aD[k,m,:,:]
       var.uR[k,:,:,:] += main.w[m,:,None,None]*var.aR[k,m,:,:]
       var.uL[k,:,:,:] += main.w[m,:,None,None]*var.aL[k,m,:,:]

#def reconstructEdges(w,a,aR,aL,aU,aD,altarray,uR,uL,uU,uD):
#  nvars,order,order,Nelx,Nely = np.shape(a)
#  aU[:] =  np.sum(a,axis=(2))
#  aD[:] =  np.sum(a*altarray[None,None,:,None,None],axis=(2))
#  aR[:] =  np.sum(a,axis=(1))
#  aL[:] =  np.sum(a*altarray[None,:,None,None,None],axis=(1))
#  uU[:] = 0. 
#  uD[:] = 0.
#  uL[:] = 0. 
#  uR[:] = 0.
#  for m in range(0,order):
#    for k in range(0,nvars):
#       uU[k,:,:,:] += w[m,:,None,None]*aU[k,m,:,:]
#       uD[k,:,:,:] += w[m,:,None,None]*aD[k,m,:,:]
#       uR[k,:,:,:] += w[m,:,None,None]*aR[k,m,:,:]
#       uL[k,:,:,:] += w[m,:,None,None]*aL[k,m,:,:]
#  #return uR,uL,uU,uD

def volIntegrate(weights,w,u):
  nvars,order,order,Nelx,Nely = np.shape(u)
  integ = np.zeros((nvars,Nelx,Nely))
  integ[:,:,:] =  np.sum(weights[None,:,None,None,None]*weights[None,None,:,None,None]*\
                       w[None,:,:,None,None]*u[:,:,:,:,:],axis=(1,2))
  return integ


def faceIntegrate(weights,w,f):
  return np.sum(weights[None,:,None,None]*w[None,:,None,None]*f,axis=1)


def getFlux(main,eqns,schemes):
  # first reconstruct states
  reconstructEdges(main,main.a)
  sendEdges(main,main.a)
  eqns.evalFluxX(main.a.uR,main.iFlux.fR)
  eqns.evalFluxX(main.a.uL,main.iFlux.fL)
  eqns.evalFluxY(main.a.uU,main.iFlux.fU)
  eqns.evalFluxY(main.a.uD,main.iFlux.fD)
  eqns.evalFluxY(main.a.uU_edge,main.iFlux.fU_edge)
  eqns.evalFluxY(main.a.uD_edge,main.iFlux.fD_edge)
  # now construct star state
  #starState(main.uR,main.uL,main.uU,main.uD,main.uU_edge,main.uD_edge,main.uRLS,main.uUDS)
  #eigsRL,eigsUD = eqns.getEigs(uRLS,uUDS)
  #fRS,fLS,fUS,fDS = centralFlux(fR,fL,fU,fD,fU_edge,fD_edge)
  #fRS,fLS,fUS,fDS = rusanovFlux(fR,fL,fU,fD,fU_edge,fD_edge,eigsRL,eigsUD,uR,uL,uU,uD,uU_edge,uD_edge)
  schemes.inviscidFlux(main,eqns,schemes,main.iFlux,main.a)
  # now we need to integrate along the boundary 
  for i in range(0,main.order):
    main.iFlux.fRI[:,i] = faceIntegrate(main.weights,main.w[i],main.iFlux.fRS)
    main.iFlux.fLI[:,i] = faceIntegrate(main.weights,main.w[i],main.iFlux.fLS)
    main.iFlux.fUI[:,i] = faceIntegrate(main.weights,main.w[i],main.iFlux.fUS)
    main.iFlux.fDI[:,i] = faceIntegrate(main.weights,main.w[i],main.iFlux.fDS)


def getRHS(main,eqns,schemes):
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,eqns,schemes)
  # now get viscous flux
  solveb(main,eqns,schemes)
  ### Quadratures
  eqns.evalFluxX(main.a.u,main.iFlux.fx)
  eqns.evalFluxY(main.a.u,main.iFlux.fy)
  for i in range(0,main.order):
    for j in range(0,main.order):
      main.RHS[:,i,j] = volIntegrate(main.weights,main.wp[i][:,None]*main.w[j][None,:],main.iFlux.fx - main.nu*main.vFlux2.fx)*2./main.dx + volIntegrate(main.weights,main.w[i][:,None]*main.wp[j][None,:],main.iFlux.fy - main.nu*main.vFlux2.fy )*2./main.dy + (-main.iFlux.fRI[:,j] + main.iFlux.fLI[:,j]*main.altarray[i])*2./main.dx + (-main.iFlux.fUI[:,i] + main.iFlux.fDI[:,i]*main.altarray[j])*2./main.dy \
                  + main.nu*(main.vFlux2.fRI[:,j] - main.vFlux2.fLI[:,j]*main.altarray[i])*2./main.dx + main.nu*(main.vFlux2.fUI[:,i] - main.vFlux2.fDI[:,i]*main.altarray[j])*2./main.dy
      main.RHS[:,i,j] = main.RHS[:,i,j]*(2.*i + 1.)*(2.*j + 1.)/4.


def getViscousFluxes(main,eqns,schemes):
  eqns.evalViscousFluxX(main.a.u,main.vFlux.fx)
  eqns.evalViscousFluxY(main.a.u,main.vFlux.fy)
  # first reconstruct states
  eqns.evalViscousFluxX(main.a.uR,main.vFlux.fR)
  eqns.evalViscousFluxX(main.a.uL,main.vFlux.fL)
  eqns.evalViscousFluxY(main.a.uU,main.vFlux.fU)
  eqns.evalViscousFluxY(main.a.uD,main.vFlux.fD)
  eqns.evalViscousFluxY(main.a.uU_edge,main.vFlux.fU_edge)
  eqns.evalViscousFluxY(main.a.uD_edge,main.vFlux.fD_edge)
  # now construct star state
  schemes.viscousFlux(main,eqns,schemes,main.vFlux,main.b)
  # now we need to integrate along the boundary 
  for i in range(0,main.order):
    main.vFlux.fRI[:,i] = faceIntegrate(main.weights,main.w[i],main.vFlux.fRS)
    main.vFlux.fLI[:,i] = faceIntegrate(main.weights,main.w[i],main.vFlux.fLS)
    main.vFlux.fUI[:,i] = faceIntegrate(main.weights,main.w[i],main.vFlux.fUS)
    main.vFlux.fDI[:,i] = faceIntegrate(main.weights,main.w[i],main.vFlux.fDS)


def solveb(main,eqns,schemes):
  ##first do quadrature
  getViscousFluxes(main,eqns,schemes)
  for i in range(0,main.order):
    for j in range(0,main.order):
      main.b.a[:,i,j] = -volIntegrate(main.weights,main.wp[i][:,None]*main.w[j][None,:],main.vFlux.fx)*2./main.dx - volIntegrate(main.weights,main.w[i][:,None]*main.wp[j][None,:],main.vFlux.fy)*2./main.dy + (main.vFlux.fRI[:,j] - main.vFlux.fLI[:,j]*main.altarray[i])*2./main.dx + (main.vFlux.fUI[:,i] - main.vFlux.fDI[:,i]*main.altarray[j])*2./main.dy
      main.b.a[:,i,j] = main.b.a[:,i,j]*(2.*i + 1.)*(2.*j + 1.)/4.
  ## Now reconstruct tau and get edge states for later flux computations
  reconstructU(main,main.b)
  reconstructEdges(main,main.b)
  sendEdges(main,main.b)
  eqns.evalTauFluxX(main.b.uR,main.a.uR,main.vFlux2.fR)
  eqns.evalTauFluxX(main.b.uL,main.a.uL,main.vFlux2.fL)
  eqns.evalTauFluxY(main.b.uU,main.a.uU,main.vFlux2.fU)
  eqns.evalTauFluxY(main.b.uD,main.a.uD,main.vFlux2.fD)
  eqns.evalTauFluxY(main.b.uU_edge,main.a.uU_edge,main.vFlux2.fU_edge)
  eqns.evalTauFluxY(main.b.uD_edge,main.a.uD_edge,main.vFlux2.fD_edge)
  eqns.evalTauFluxX(main.b.u,main.a.u,main.vFlux2.fx)  
  eqns.evalTauFluxY(main.b.u,main.a.u,main.vFlux2.fy)  
  schemes.viscousFlux(main,eqns,schemes,main.vFlux2,main.b)
  for i in range(0,main.order):
     main.vFlux2.fRI[:,i] = faceIntegrate(main.weights,main.w[i],main.vFlux2.fRS)
     main.vFlux2.fLI[:,i] = faceIntegrate(main.weights,main.w[i],main.vFlux2.fLS)
     main.vFlux2.fUI[:,i] = faceIntegrate(main.weights,main.w[i],main.vFlux2.fUS)
     main.vFlux2.fDI[:,i] = faceIntegrate(main.weights,main.w[i],main.vFlux2.fDS)

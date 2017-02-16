import numpy as np
from MPI_functions import sendEdges,sendEdgesGeneral,sendEdgesSlab,sendEdgesGeneralSlab
from fluxSchemes import *
from scipy import weave
from scipy.weave import converters
def reconstructU2(main,var):
  var.u[:] = 0.
  for l in range(0,var.order):
    for m in range(0,var.order):
      for k in range(0,var.nvars):
        var.u[k,:,:,:,:] += main.w[l][:,None,None,None]*main.w[m][None,:,None,None]*var.a[k,l,m,:,:]


def reconstructU(main,var):
  var.u[:] = 0.
  #var.u = np.einsum('lmpq,klmij->kpqij',main.w[:,None,:,None]*main.w[None,:,None,:],var.a) ## this is actually much slower than the two line code
  tmp =  np.einsum('mq,klmij->klqij',main.w,var.a)
  var.u = np.einsum('lp,klqij->kpqij',main.w,tmp)
  #for l in range(0,var.order):
  #  for m in range(0,var.order):
  #    var.u[:,:,:,:,:] += main.w[l][None,:,None,None,None]*main.w[m][None,None,:,None,None]*var.a[:,l,m,:,:]
  #print('hi',np.linalg.norm(u - var.u))


def reconstructUF(main,var):
  var.u[:] = 0.
  for l in range(0,var.filt_order):
    for m in range(0,var.filt_order):
      for k in range(0,var.nvars):
        var.u[k,:,:,:,:] += main.w[l][:,None,None,None]*main.w[m][None,:,None,None]*var.a[k,l,m,:,:]

#def reconstructU(w,u,a):
#  nvars,order,order,Nelx,Nely = np.shape(a)
#  u[:] = 0.
#  for l in range(0,order):
#    for m in range(0,order):
#      for k in range(0,nvars):
#        u[k,:,:,:,:] += w[l][:,None,None,None]*w[m][None,:,None,None]*a[k,l,m,:,:]


def diffU(a,main):
  tmp =  np.einsum('mq,klmij->klqij',main.w,a)
  ux = np.einsum('lp,klqij->kpqij',main.wp,tmp)
  tmp =  np.einsum('mq,klmij->klqij',main.wp,a)
  uy = np.einsum('lp,klqij->kpqij',main.w,tmp)
  return ux,uy

def diffU2(a,main):
  nvars = np.shape(a)[0]
  order = np.shape(a)[1]
  ux = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy))
  uy = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy))

  for l in range(0,order):
    for m in range(0,order):
      for k in range(0,nvars):
        ux[k,:,:,:,:] += main.wp[l][:,None,None,None]*main.w[m][None,:,None,None]*a[k,l,m,:,:]
        uy[k,:,:,:,:] += main.w[l][:,None,None,None]*main.wp[m][None,:,None,None]*a[k,l,m,:,:]
  return ux,uy



def reconstructEdgesGeneral(a,main):
  nvars = np.shape(a)[0]
  aU = np.einsum('klmij->klij',a)
  aD= np.einsum('klmij->klij',a*main.altarray[None,None,:,None,None])
  aR = np.einsum('klmij->kmij',a)
  aL = np.einsum('klmij->kmij',a*main.altarray[None,:,None,None,None])
  #aU =  np.sum(a,axis=(2))
  #aD =  np.sum(a*main.altarray[None,None,:,None,None],axis=(2))
  #aR =  np.sum(a,axis=(1))
  #aL =  np.sum(a*main.altarray[None,:,None,None,None],axis=(1))
  uU = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))
  uD = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))
  uL = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))
  uR = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))

  uU[:] = np.einsum('lp,klij->kpij',main.w,aU)
  uD[:] = np.einsum('lp,klij->kpij',main.w,aD)
  uR[:] = np.einsum('lp,klij->kpij',main.w,aR)
  uL[:] = np.einsum('lp,klij->kpij',main.w,aL)

  #for m in range(0,main.order):
  #  for k in range(0,main.nvars):
  #     uU[k,:,:,:] += main.w[m,:,None,None]*aU[k,m,:,:]
  #     uD[k,:,:,:] += main.w[m,:,None,None]*aD[k,m,:,:]
  #     uR[k,:,:,:] += main.w[m,:,None,None]*aR[k,m,:,:]
  #     uL[k,:,:,:] += main.w[m,:,None,None]*aL[k,m,:,:]
  return uR,uL,uU,uD


def reconstructEdges2(main,var):
  var.aU[:] =  np.sum(var.a,axis=(2))
  var.aD[:] =  np.sum(var.a*main.altarray[None,None,:,None,None],axis=(2))
  var.aR[:] =  np.sum(var.a,axis=(1))
  var.aL[:] =  np.sum(var.a*main.altarray[None,:,None,None,None],axis=(1))

  var.uU[:] = 0. 
  var.uD[:] = 0.
  var.uL[:] = 0. 
  var.uR[:] = 0.
  for m in range(0,var.order):
    #for k in range(0,var.nvars):
       var.uU[:,:,:,:] += main.w[None,m,:,None,None]*var.aU[:,m,:,:]
       var.uD[:,:,:,:] += main.w[None,m,:,None,None]*var.aD[:,m,:,:]
       var.uR[:,:,:,:] += main.w[None,m,:,None,None]*var.aR[:,m,:,:]
       var.uL[:,:,:,:] += main.w[None,m,:,None,None]*var.aL[:,m,:,:]


def reconstructEdges(main,var):
  var.aU[:] = np.einsum('klmij->klij',main.a.a)
  var.aD[:] = np.einsum('klmij->klij',main.a.a*main.altarray[None,None,:,None,None])
  var.aR[:] = np.einsum('klmij->kmij',main.a.a)
  var.aL[:] = np.einsum('klmij->kmij',main.a.a*main.altarray[None,:,None,None,None])
  #var.aU[:] =  np.sum(var.a,axis=(2))
  #var.aD[:] =  np.sum(var.a*main.altarray[None,None,:,None,None],axis=(2))
  #var.aR[:] =  np.sum(var.a,axis=(1))
  #var.aL[:] =  np.sum(var.a*main.altarray[None,:,None,None,None],axis=(1))
  var.uU[:] = 0. 
  var.uD[:] = 0.
  var.uL[:] = 0. 
  var.uR[:] = 0.
  var.uU[:] = np.einsum('lp,klij->kpij',main.w,var.aU)
  var.uD[:] = np.einsum('lp,klij->kpij',main.w,var.aD)
  var.uR[:] = np.einsum('lp,klij->kpij',main.w,var.aR)
  var.uL[:] = np.einsum('lp,klij->kpij',main.w,var.aL)

#  for m in range(0,var.order):
#    #for k in range(0,var.nvars):
#       var.uU[:,:,:,:] += main.w[None,m,:,None,None]*var.aU[:,m,:,:]
#       var.uD[:,:,:,:] += main.w[None,m,:,None,None]*var.aD[:,m,:,:]
#       var.uR[:,:,:,:] += main.w[None,m,:,None,None]*var.aR[:,m,:,:]
#       var.uL[:,:,:,:] += main.w[None,m,:,None,None]*var.aL[:,m,:,:]
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



def volIntegrate2(weights,w,u):
  nvars,order,order,Nelx,Nely = np.shape(u)
  integ = np.zeros((nvars,Nelx,Nely))
  integ[:,:,:] =  np.sum(weights[None,:,None,None,None]*weights[None,None,:,None,None]*\
                       w[None,:,:,None,None]*u[:,:,:,:,:],axis=(1,2))
  return integ

def volIntegrate(weights,w,u):
  return  np.einsum('kpqij->kij',weights[None,:,None,None,None]*weights[None,None,:,None,None]*\
                       w[None,:,:,None,None]*u[:,:,:,:,:])

def volIntegrateGen(weights,f):
  return  np.einsum('kpqij->kij',weights[None,:,None,None,None]*weights[None,None,:,None,None]*f[:,:,:,:,:])


def faceIntegrate(weights,w,f):
  return np.einsum('kpij->kij',weights[None,:,None,None]*w[None,:,None,None]*f)


def faceIntegrate2(weights,w,f):
  return np.sum(weights[None,:,None,None]*w[None,:,None,None]*f,axis=1)




def getFlux(main,eqns,schemes):
  # first reconstruct states
  reconstructEdges(main,main.a)
  sendEdgesSlab(main,main.a)
  #eqns.evalFluxX(main.a.uR,main.iFlux.fR)
  #eqns.evalFluxX(main.a.uL,main.iFlux.fL)
  #eqns.evalFluxX(main.a.uR_edge,main.iFlux.fR_edge)
  #eqns.evalFluxX(main.a.uL_edge,main.iFlux.fL_edge)
  #eqns.evalFluxY(main.a.uU,main.iFlux.fU)
  #eqns.evalFluxY(main.a.uD,main.iFlux.fD)
  #eqns.evalFluxY(main.a.uU_edge,main.iFlux.fU_edge)
  #eqns.evalFluxY(main.a.uD_edge,main.iFlux.fD_edge)
  # now construct star state
  #starState(main.uR,main.uL,main.uU,main.uD,main.uU_edge,main.uD_edge,main.uRLS,main.uUDS)
  #eigsRL,eigsUD = eqns.getEigs(uRLS,uUDS)
  #fRS,fLS,fUS,fDS = centralFlux(fR,fL,fU,fD,fU_edge,fD_edge)
  #fRS,fLS,fUS,fDS = rusanovFlux(fR,fL,fU,fD,fU_edge,fD_edge,eigsRL,eigsUD,uR,uL,uU,uD,uU_edge,uD_edge)
  #schemes.inviscidFlux(main,eqns,schemes,main.iFlux,main.a)
  roeFlux(main,eqns,schemes,main.iFlux,main.a)
  # now we need to integrate along the boundary 
  for i in range(0,main.order):
    main.iFlux.fRI[:,i] = faceIntegrate(main.weights,main.w[i],main.iFlux.fRS)
    main.iFlux.fLI[:,i] = faceIntegrate(main.weights,main.w[i],main.iFlux.fLS)
    main.iFlux.fUI[:,i] = faceIntegrate(main.weights,main.w[i],main.iFlux.fUS)
    main.iFlux.fDI[:,i] = faceIntegrate(main.weights,main.w[i],main.iFlux.fDS)


def getRHS_BR1(main,eqns,schemes):
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
      main.RHS[:,i,j] = volIntegrate(main.weights,main.wp[i][:,None]*main.w[j][None,:],main.iFlux.fx - main.mu*main.vFlux2.fx)*2./main.dx + volIntegrate(main.weights,main.w[i][:,None]*main.wp[j][None,:],main.iFlux.fy - main.mu*main.vFlux2.fy )*2./main.dy + (-main.iFlux.fRI[:,j] + main.iFlux.fLI[:,j]*main.altarray[i])*2./main.dx + (-main.iFlux.fUI[:,i] + main.iFlux.fDI[:,i]*main.altarray[j])*2./main.dy \
                  + main.mu*(main.vFlux2.fRI[:,j] - main.vFlux2.fLI[:,j]*main.altarray[i])*2./main.dx + main.mu*(main.vFlux2.fUI[:,i] - main.vFlux2.fDI[:,i]*main.altarray[j])*2./main.dy
      main.RHS[:,i,j] = main.RHS[:,i,j]*(2.*i + 1.)*(2.*j + 1.)/4.






def getRHS_INVISCID(main,eqns,schemes):
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,eqns,schemes)
  # now get viscous flux
  upx,upy = diffU(main.a.a,main)
  upx = upx*2./main.dx
  upy = upy*2./main.dy
  ### Quadratures
  eqns.evalFluxX(main.a.u,main.iFlux.fx)
  eqns.evalFluxY(main.a.u,main.iFlux.fy)
  for i in range(0,main.order):
    for j in range(0,main.order):
      main.RHS[:,i,j] = volIntegrate(main.weights,main.wp[i][:,None]*main.w[j][None,:],main.iFlux.fx)*2./main.dx + volIntegrate(main.weights,main.w[i][:,None]*main.wp[j][None,:],main.iFlux.fy)*2./main.dy + \
                        (-main.iFlux.fRI[:,j] + main.iFlux.fLI[:,j]*main.altarray[i])*2./main.dx + (-main.iFlux.fUI[:,i] + main.iFlux.fDI[:,i]*main.altarray[j])*2./main.dy 

      main.RHS[:,i,j] = main.RHS[:,i,j]*(2.*i + 1.)*(2.*j + 1.)/4.



def getRHS_IP(main,eqns,schemes):
  reconstructU(main,main.a)
  # evaluate inviscid flux
  getFlux(main,eqns,schemes)
  # now get viscous flux
  fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvR2I,fvL2I,fvU2I,fvD2I = getViscousFlux(main,eqns,schemes)
  upx,upy = diffU(main.a.a,main)
  upx = upx*2./main.dx
  upy = upy*2./main.dy
  ### Quadratures
  eqns.evalFluxX(main.a.u,main.iFlux.fx)
  eqns.evalFluxY(main.a.u,main.iFlux.fy)
  fvGX = np.zeros(np.shape(main.a.u))
  fvGY = np.zeros(np.shape(main.a.u))
  G11,G12,G21,G22 = eqns.getGs(main.a.u,main)
  for i in range(0,eqns.nvars):
    for j in range(0,eqns.nvars):
      fvGX[i] += G11[i,j]*upx[j] + G12[i,j]*upy[j]
      fvGY[i] += G21[i,j]*upx[j] + G22[i,j]*upy[j]

  for i in range(0,main.order):
    for j in range(0,main.order):
      main.RHS[:,i,j] = volIntegrate(main.weights,main.wp[i][:,None]*main.w[j][None,:],main.iFlux.fx - fvGX)*2./main.dx + volIntegrate(main.weights,main.w[i][:,None]*main.wp[j][None,:],main.iFlux.fy - fvGY)*2./main.dy + \
                        (-main.iFlux.fRI[:,j] + main.iFlux.fLI[:,j]*main.altarray[i])*2./main.dx + (-main.iFlux.fUI[:,i] + main.iFlux.fDI[:,i]*main.altarray[j])*2./main.dy \
                      + (fvRIG11[:,j]*main.wpedge[i,1] + fvRIG21[:,j]  - (fvLIG11[:,j]*main.wpedge[i,0] + fvLIG21[:,j]*main.altarray[i]) )*2./main.dx + \
                        (fvUIG12[:,i] + fvUIG22[:,i]*main.wpedge[j,1]  - (fvDIG12[:,i]*main.altarray[j] + fvDIG22[:,i]*main.wpedge[j,0]) )*2./main.dy \
                      + (fvR2I[:,j] - fvL2I[:,j]*main.altarray[i])*2./main.dx + (fvU2I[:,i] - fvD2I[:,i]*main.altarray[j])*2./main.dy
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
  for i in range(0,main.b.order):
    for j in range(0,main.b.order):
      main.b.a[:,i,j] = -volIntegrate(main.weights,main.wp[i][:,None]*main.w[j][None,:],main.vFlux.fx)*2./main.dx - volIntegrate(main.weights,main.w[i][:,None]*main.wp[j][None,:],main.vFlux.fy)*2./main.dy + (main.vFlux.fRI[:,j] - main.vFlux.fLI[:,j]*main.altarray[i])*2./main.dx + (main.vFlux.fUI[:,i] - main.vFlux.fDI[:,i]*main.altarray[j])*2./main.dy
      main.b.a[:,i,j] = main.b.a[:,i,j]*(2.*i + 1.)*(2.*j + 1.)/4.
  ## Now reconstruct tau and get edge states for later flux computations
  reconstructU(main,main.b)
  reconstructEdges(main,main.b)
  sendEdgesSlab(main,main.b)
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








def getViscousFlux(main,eqns,schemes):
  gamma = 1.4
  Pr = 0.72
  a = main.a.a
  nvars,order,order,Npx,Npy = np.shape(a)
  fvRG11 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))
  fvLG11 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))

  fvRG21 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))
  fvLG21 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))

  fvUG12 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))
  fvDG12 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))

  fvUG22 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))
  fvDG22 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))

  fvR2 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))
  fvL2 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))
  fvU2 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))
  fvD2 = np.zeros((main.nvars,main.quadpoints,main.Npx,main.Npy))
  uhatR,uhatL,uhatU,uhatD = centralFluxGeneral(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge)

  G11R,G21R = eqns.getGsX(main.a.uR,main)
  G11L,G21L = eqns.getGsX(main.a.uL,main)
  G12U,G22U = eqns.getGsY(main.a.uU,main)
  G12D,G22D = eqns.getGsY(main.a.uD,main)


#  G11R,G12R,G21R,G22R = eqns.getGs(main.a.uR,main)
#  G11L,G12L,G21L,G22L = eqns.getGs(main.a.uL,main)
#  G11U,G12U,G21U,G22U = eqns.getGs(main.a.uU,main)
#  G11D,G12D,G21D,G22D = eqns.getGs(main.a.uD,main)

  for i in range(0,nvars):
    for j in range(0,nvars):
      fvRG11[i] += G11R[i,j]*(main.a.uR[j] - uhatR[j])
      fvLG11[i] += G11L[i,j]*(main.a.uL[j] - uhatL[j])

      fvRG21[i] += G21R[i,j]*(main.a.uR[j] - uhatR[j])
      fvLG21[i] += G21L[i,j]*(main.a.uL[j] - uhatL[j])

      fvUG12[i] += G12U[i,j]*(main.a.uU[j] - uhatU[j])
      fvDG12[i] += G12D[i,j]*(main.a.uD[j] - uhatD[j])

      fvUG22[i] += G22U[i,j]*(main.a.uU[j] - uhatU[j])
      fvDG22[i] += G22D[i,j]*(main.a.uD[j] - uhatD[j])


  apx,apy = diffCoeffs(main.a.a)
  apx = apx*2./main.dx
  apy = apy*2./main.dy
  UxR,UxL,UxU,UxD = reconstructEdgesGeneral(apx,main)
  UyR,UyL,UyU,UyD = reconstructEdgesGeneral(apy,main)
  UxR_edge,UxL_edge,UxU_edge,UxD_edge = sendEdgesGeneralSlab(UxL,UxR,UxD,UxU,main)
  UyR_edge,UyL_edge,UyU_edge,UyD_edge = sendEdgesGeneralSlab(UyL,UyR,UyD,UyU,main)

  fvxR = eqns.evalViscousFluxX(main,main.a.uR,UxR,UyR)
  fvxL = eqns.evalViscousFluxX(main,main.a.uL,UxL,UyL)
  fvxR_edge = eqns.evalViscousFluxX(main,main.a.uR_edge,UxR_edge,UyR_edge)
  fvxL_edge = eqns.evalViscousFluxX(main,main.a.uL_edge,UxL_edge,UyL_edge)

  fvyU = eqns.evalViscousFluxY(main,main.a.uU,UxU,UyU)
  fvyD = eqns.evalViscousFluxY(main,main.a.uD,UxD,UyD)
  fvyU_edge = eqns.evalViscousFluxY(main,main.a.uU_edge,UxU_edge,UyU_edge)
  fvyD_edge = eqns.evalViscousFluxY(main,main.a.uD_edge,UxD_edge,UyD_edge)


  shatR,shatL,shatU,shatD = centralFluxGeneral(fvxR,fvxL,fvyU,fvyD,fvxR_edge,fvxL_edge,fvyU_edge,fvyD_edge)
  jumpR,jumpL,jumpU,jumpD = computeJump(main.a.uR,main.a.uL,main.a.uU,main.a.uD,main.a.uR_edge,main.a.uL_edge,main.a.uU_edge,main.a.uD_edge)
  fvR2[:] = shatR[:] - 2.*main.mu*jumpR[:]*3**2/main.dx
  fvL2[:] = shatL[:] - 2.*main.mu*jumpL[:]*3**2/main.dx
  fvU2[:] = shatU[:] - 2.*main.mu*jumpU[:]*3**2/main.dx
  fvD2[:] = shatD[:] - 2.*main.mu*jumpD[:]*3**2/main.dx

 # now we need to integrate along the boundary 
  fvRIG11 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvLIG11 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))

  fvRIG21 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvLIG21 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))

  fvUIG12 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvDIG12 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))

  fvUIG22 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvDIG22 = np.zeros((main.nvars,main.order,main.Npx,main.Npy))

  fvR2I = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvL2I = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvU2I = np.zeros((main.nvars,main.order,main.Npx,main.Npy))
  fvD2I = np.zeros((main.nvars,main.order,main.Npx,main.Npy))

  for i in range(0,main.order):
    fvRIG11[:,i] = faceIntegrate(main.weights,main.w[i],fvRG11)
    fvLIG11[:,i] = faceIntegrate(main.weights,main.w[i],fvLG11)

    fvRIG21[:,i] = faceIntegrate(main.weights,main.wp[i],fvRG21)
    fvLIG21[:,i] = faceIntegrate(main.weights,main.wp[i],fvLG21)

    fvUIG12[:,i] = faceIntegrate(main.weights,main.wp[i],fvUG12)
    fvDIG12[:,i] = faceIntegrate(main.weights,main.wp[i],fvDG12)

    fvUIG22[:,i] = faceIntegrate(main.weights,main.w[i],fvUG22)
    fvDIG22[:,i] = faceIntegrate(main.weights,main.w[i],fvDG22)

    fvR2I[:,i] = faceIntegrate(main.weights,main.w[i],fvR2)
    fvL2I[:,i] = faceIntegrate(main.weights,main.w[i],fvL2)
    fvU2I[:,i] = faceIntegrate(main.weights,main.w[i],fvU2)
    fvD2I[:,i] = faceIntegrate(main.weights,main.w[i],fvD2)

  return fvRIG11,fvLIG11,fvRIG21,fvLIG21,fvUIG12,fvDIG12,fvUIG22,fvDIG22,fvR2I,fvL2I,fvU2I,fvD2I


def diffCoeffs(a):
  atmp = np.zeros(np.shape(a))
  atmp[:] = a[:]
  nvars,order,order,Nelx,Nely = np.shape(a)
  ax = np.zeros((nvars,order,order,Nelx,Nely))
  ay = np.zeros((nvars,order,order,Nelx,Nely))
  for j in range(order-1,2,-1):
    ax[:,j-1,:] = (2.*j-1)*atmp[:,j,:]
    atmp[:,j-2,:] = atmp[:,j-2,:] + atmp[:,j,:]

  if (order >= 3):
    ax[:,1,:] = 3.*atmp[:,2,:]
    ax[:,0,:] = atmp[:,1,:]
  if (order == 2):
    ax[:,1,:] = 0.
    ax[:,0,:] = atmp[:,1,:]
  if (order == 1):
    ax[:,0,:] = 0.

  atmp[:] = a[:]
  for j in range(order-1,2,-1):
    ay[:,:,j-1] = (2.*j-1)*atmp[:,:,j]
    atmp[:,:,j-2] = atmp[:,:,j-2] + atmp[:,:,j]

  if (order >= 3):
    ay[:,:,1] = 3.*atmp[:,:,2]
    ay[:,:,0] = atmp[:,:,1]
  if (order == 2):
    ay[:,:,1] = 0.
    ay[:,:,0] = atmp[:,:,1]
  if (order == 1):
    ay[:,:,0] = 0.


  return ax,ay



def computeJump(uR,uL,uU,uD,uR_edge,uL_edge,uU_edge,uD_edge):
  nvars,order,Npx,Npy = np.shape(uR)
  jumpR = np.zeros((nvars,order,Npx,Npy))
  jumpL = np.zeros((nvars,order,Npx,Npy))
  jumpU = np.zeros((nvars,order,Npx,Npy))
  jumpD = np.zeros((nvars,order,Npx,Npy))

  jumpR[:,:,0:-1,:] = uR[:,:,0:-1,:] - uL[:,:,1::,:]
  jumpR[:,:,-1   ,:] = uR[:,:,  -1,:] - uR_edge
  jumpL[:,:,1:: ,:] = jumpR[:,:,0:-1,:]
  jumpL[:,:,0   ,:] = uL_edge - uL[:,:,  0,:]
  jumpU[:,:,:,0:-1] = uU[:,:,:,0:-1] - uD[:,:,:,1::]
  jumpU[:,:,:,  -1] = uU[:,:,:,  -1] - uU_edge
  jumpD[:,:,:,1:: ] = jumpU[:,:,:,0:-1]
  jumpD[:,:,:,0   ] = uD_edge - uD[:,:,:,   0]
  return jumpR,jumpL,jumpU,jumpD

